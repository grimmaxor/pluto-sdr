"""
ADALM Pluto Auto-Calibrating Two-Radio Link + PTT Chat
=======================================================
Runs the SAME script on both PCs. One is master, one is slave.

  PC1 (master): python pluto_autolink.py --role master
  PC2 (slave) : python pluto_autolink.py --role slave

Optionally pin the Pluto URI / frequency:
  python pluto_autolink.py --role master --ip ip:pluto.local --freq 433e6

WHAT IT DOES
------------
1. Auto-calibrates RX gain on BOTH radios (starting from low gain, sweeping up)
2. Auto-negotiates TX power (starting high attenuation, reducing until link is solid)
3. Locks in the discovered settings
4. Drops into Push-To-Talk chat (type to talk, Enter to listen)

CALIBRATION IS HANDSHAKE-DRIVEN, NOT TIMER-DRIVEN
-------------------------------------------------
The two radios never advance a phase because "N seconds passed". Every
transition happens only when a radio actually *decodes the expected message*
from the other one. The whole thing is built from three primitives:

  - exchange(send, want) : stop-and-wait. Transmit `send` in short bursts,
                           listen between bursts, return the instant `want`
                           is decoded. Progress depends solely on the partner.
  - hold_tone_until(stop): transmit a continuous calibration tone (long bursts,
                           tiny listen-gaps) so the partner can sweep its gain,
                           and stop ONLY when the partner says it's done.
  - echo_until(...)      : listen, echo every probe heard, leave when the
                           partner advances to the next phase.

The only timing that remains is local and physical, not coordination:
  * a transmit burst must last long enough to fill one RX buffer on the other
    side (TX_BURST_SEC, derived from buffer size / sample rate), and
  * "listening" means grabbing a fixed number of RX *buffers* (the radio's
    rx() call blocks, so the hardware paces it) rather than watching a clock.
A small random jitter is added to each burst to break the lock-step hazard
where two identical half-duplex radios transmit and listen in perfect unison
and therefore never hear each other.

CFO handling: every decode tries three strategies (none / FFT-coarse / PLL)
and accepts the first that yields a valid CRC.
"""

import adi
import numpy as np
import argparse
import threading
import time
import random
import sys
from scipy.signal import firwin, lfilter

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--role', choices=['master', 'slave'], required=True)
parser.add_argument('--ip',   type=str,   default='ip:pluto.local')
parser.add_argument('--freq', type=float, default=433e6)
args = parser.parse_args()

ROLE = args.role

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CENTER_FREQ        = int(args.freq)
SAMPLE_RATE        = int(1e6)
TX_BUFFER_SIZE     = 65536
SAMPLES_PER_SYMBOL = 16
MAX_MSG_LEN        = 180
FRAME_MAGIC        = 0xBE
BARKER_13          = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)

# Calibration sweep ranges
RX_GAIN_SWEEP      = list(range(0, 74, 3))      # 0..73 dB
TX_ATTEN_SWEEP     = list(range(-80, 1, 5))     # -80..0 dB (high atten -> low atten)

# ── Local, physical timing (NOT cross-radio coordination) ──────────────────────
# One RX capture is TX_BUFFER_SIZE / SAMPLE_RATE seconds long. A transmit burst
# has to outlast a capture so the partner can grab a clean buffer of it.
CAPTURE_SEC      = TX_BUFFER_SIZE / SAMPLE_RATE          # ~0.066 s
TX_BURST_SEC     = max(0.22, 3 * CAPTURE_SEC)           # length of one TX burst
JITTER_SEC       = 0.15                                 # symmetry-breaking jitter
CHECK_BUFFERS    = 2                                    # RX buffers grabbed in a listen-gap
LISTEN_BUFFERS   = 5                                    # RX buffers grabbed when waiting on a reply
GAIN_STEP_BUFFERS= 6                                    # RX buffers grabbed per gain step
GAIN_CONFIRM     = 2                                    # decodes needed to accept a gain step
POWER_TRIES      = 8                                    # bursts per atten step before "not heard"
MAX_ROUNDS       = 600                                  # safety cap on any handshake (NOT a timer)

# Special control messages used during calibration
MSG_CAL_TONE   = "__CAL__"          # calibration carrier for gain sweep
MSG_GAIN_OK    = "__GAINOK__"       # "my RX gain is locked"
MSG_SWAP       = "__SWAP__"         # "your turn to transmit the tone"
MSG_PWR_PROBE  = "__PWR__"          # power probe (TX power varied around it)
MSG_PWR_ACK    = "__PWRACK__"       # probe echo ("I heard your probe")
MSG_READY      = "__READY__"        # link-ready beacon
MSG_READY_ACK  = "__READYACK__"     # link-ready acknowledge
MSG_DONE       = "__DONE__"         # "handshake complete, enter chat"

# ─── DSP ──────────────────────────────────────────────────────────────────────
def crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc

def make_filter(sps):
    return firwin(sps * 4 + 1, 1.4 / sps, window='hamming').astype(np.float32)

FILT = make_filter(SAMPLES_PER_SYMBOL)


def encode_fill(message: str) -> np.ndarray:
    """
    Encode a message into BPSK and TILE it to fill the whole TX buffer.
    No silence gaps — any RX capture window contains complete packets.
    """
    message   = message[:MAX_MSG_LEN]
    msg_bytes = message.encode('utf-8')
    payload   = bytes([FRAME_MAGIC, len(msg_bytes)]) + msg_bytes
    full      = payload + bytes([crc8(payload)])

    bbits = ((1 - BARKER_13) / 2).astype(np.uint8)
    pbits = np.unpackbits(np.frombuffer(full, dtype=np.uint8))
    bits  = np.concatenate([bbits, pbits]).astype(np.float32)

    symbols = 1.0 - 2.0 * bits
    up      = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL, dtype=np.float32)
    up[::SAMPLES_PER_SYMBOL] = symbols
    shaped  = lfilter(FILT, 1.0, up).astype(np.float32)
    mx      = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15

    iq      = shaped.astype(np.complex64)
    repeats = int(np.ceil(TX_BUFFER_SIZE / len(iq))) + 1
    iq_full = np.tile(iq, repeats)[:TX_BUFFER_SIZE]
    return iq_full.astype(np.complex64)


# ── Three CFO strategies ───────────────────────────────────────────────────────
def cfo_none(iq):
    return iq, 0.0

def cfo_fft(iq, sr):
    """Squaring + FFT coarse estimate."""
    sq   = (iq ** 2)
    n    = len(sq)
    fv   = np.fft.fft(sq)
    fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/sr)
    pk    = int(np.argmax(np.abs(fv)))
    cfo   = freqs[pk] / 2.0
    ph    = np.angle(fv[pk]) / 2.0
    t     = np.arange(n) / sr
    corr  = np.exp(-1j * (2*np.pi*cfo*t + ph)).astype(np.complex64)
    return (iq * corr).astype(np.complex64), float(cfo)

def cfo_pll(iq):
    """Decision-directed per-symbol PLL."""
    out   = np.zeros_like(iq)
    ph    = 0.0
    fr    = 0.0
    alpha = 0.01
    beta  = 0.0002
    for i in range(len(iq)):
        cs      = iq[i] * np.exp(-1j * ph)
        out[i]  = cs
        d       = 1.0 if cs.real >= 0 else -1.0
        err     = cs.imag * d
        fr     += beta * err
        ph     += alpha * err + fr
        ph      = (ph + np.pi) % (2*np.pi) - np.pi
    return out, 0.0


def _try_extract(sig):
    """Given a filtered real signal, brute-force scan for a valid packet."""
    for polarity in [1, -1]:
        s = sig * polarity
        for toff in range(SAMPLES_PER_SYMBOL):
            sampled = s[toff::SAMPLES_PER_SYMBOL].astype(np.float32)
            bits    = (sampled < 0).astype(np.uint8)
            nb      = len(bits) // 8
            if nb < 8:
                continue
            raw = bytes(np.packbits(bits[:nb * 8]))
            for pos in range(len(raw) - 4):
                if raw[pos] != FRAME_MAGIC:
                    continue
                ml = raw[pos + 1]
                if ml == 0 or ml > MAX_MSG_LEN:
                    continue
                ei = pos + 2 + ml
                if ei + 1 >= len(raw):
                    continue
                if crc8(raw[pos:ei]) != raw[ei]:
                    continue
                try:
                    return raw[pos+2:ei].decode('utf-8')
                except UnicodeDecodeError:
                    continue
    return None


def decode_any(iq):
    """
    Try all 3 CFO strategies, return (message, method, cfo) for first success.
    """
    peak = np.max(np.abs(iq))
    if peak < 5:
        return None, None, 0.0
    iq = (iq / peak * 2**13).astype(np.complex64)

    # Strategy 1: no CFO correction
    s0, _ = cfo_none(iq)
    sig   = lfilter(FILT, 1.0, s0.real).astype(np.float32)
    m     = _try_extract(sig)
    if m is not None:
        return m, "none", 0.0

    # Strategy 2: FFT coarse
    s1, cfo = cfo_fft(iq, SAMPLE_RATE)
    sig     = lfilter(FILT, 1.0, s1.real).astype(np.float32)
    m       = _try_extract(sig)
    if m is not None:
        return m, "fft", cfo

    # Strategy 3: PLL
    s2, _ = cfo_pll(iq)
    sig   = lfilter(FILT, 1.0, s2.real).astype(np.float32)
    m     = _try_extract(sig)
    if m is not None:
        return m, "pll", 0.0

    return None, None, 0.0


# ─── PLUTO ────────────────────────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate             = SAMPLE_RATE
    sdr.rx_lo                   = CENTER_FREQ
    sdr.tx_lo                   = CENTER_FREQ
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.tx_rf_bandwidth         = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = 10
    sdr.tx_hardwaregain_chan0   = -40
    sdr.rx_buffer_size          = TX_BUFFER_SIZE
    sdr.tx_cyclic_buffer        = True

    # Disable DDS
    try:
        import iio
        ctx = iio.Context(args.ip)
        dds = ctx.find_device("cf-ad9361-dds-core-lpc")
        if dds:
            for ch in dds.channels:
                if ch.output:
                    for attr in ["raw", "scale"]:
                        try:
                            ch.attrs[attr].value = "0" if attr == "raw" else "0.0"
                        except Exception:
                            pass
            print("[*] DDS disabled")
    except Exception as e:
        print(f"[*] DDS note: {e}")

    print(f"[\u2713] Connected. RX LO {sdr.rx_lo/1e6:.4f} MHz")
    return sdr


def set_rx_gain(sdr, g):
    sdr.rx_hardwaregain_chan0 = int(g)

def set_tx_atten(sdr, a):
    sdr.tx_hardwaregain_chan0 = int(a)

def tx_silence(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))

def tx_message_cyclic(sdr, msg):
    """Start transmitting msg continuously (cyclic buffer)."""
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    time.sleep(0.05)            # hardware buffer settle (local, not coordination)
    sdr.tx(encode_fill(msg))

def flush_rx(sdr, n=2):
    """Discard a few stale captures so we read fresh air after a TX→RX switch."""
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  HANDSHAKE PRIMITIVES  (the only thing that drives calibration forward)
# ═══════════════════════════════════════════════════════════════════════════════
def _burst(sdr, msg):
    """
    Transmit `msg` cyclically for one burst, then go silent so we can listen.
    The burst length is the physical floor (long enough for the partner to
    capture it); the jitter breaks the half-duplex lock-step hazard.
    """
    tx_message_cyclic(sdr, msg)
    time.sleep(TX_BURST_SEC + random.uniform(0.0, JITTER_SEC))
    tx_silence(sdr)


def listen_burst(sdr, want=None, n_buffers=LISTEN_BUFFERS):
    """
    Grab a FIXED NUMBER of RX buffers and try to decode each.
    rx() blocks until a buffer is ready, so the hardware — not a clock — paces
    this. Returns the first decode matching `want` (or any decode if want=None).
    """
    flush_rx(sdr, 1)
    for _ in range(n_buffers):
        try:
            rx = sdr.rx()
            m, method, cfo = decode_any(rx)
            if m is not None and (want is None or m == want):
                return m, method, cfo
        except Exception:
            pass
    return None, None, 0.0


def exchange(sdr, send_msg, want_msg, max_rounds=MAX_ROUNDS):
    """
    Half-duplex stop-and-wait. Each round: send one burst of `send_msg`, then
    listen for `want_msg`. Returns (method, cfo) the moment `want_msg` is heard,
    else None after `max_rounds` rounds. NOTHING advances this except hearing
    the partner — there is no assumption about how long the partner takes.
    """
    for _ in range(max_rounds):
        _burst(sdr, send_msg)
        m, method, cfo = listen_burst(sdr, want=want_msg, n_buffers=CHECK_BUFFERS)
        if m == want_msg:
            return method, cfo
    return None


def hold_tone_until(sdr, stop_token, tone=MSG_CAL_TONE, max_rounds=MAX_ROUNDS):
    """
    Transmit `tone` in long bursts with tiny listen-gaps so the partner can
    sweep its RX gain against an (almost) continuous carrier. Stop ONLY when we
    actually decode `stop_token` from the partner — i.e. when the partner tells
    us it's finished. Never stops on a timer.
    """
    for _ in range(max_rounds):
        _burst(sdr, tone)
        m, _, _ = listen_burst(sdr, want=stop_token, n_buffers=CHECK_BUFFERS)
        if m == stop_token:
            return True
    return False


def echo_until(sdr, trigger, reply, finish, max_rounds=MAX_ROUNDS):
    """
    Listen continuously. On hearing `trigger`, send one `reply` burst. Return
    True the moment `finish` is heard. Used by the slave during power
    negotiation: every probe it can decode is echoed, and it leaves only when
    the master advances to `finish` — not after some elapsed time.
    """
    for _ in range(max_rounds):
        m, _, _ = listen_burst(sdr, want=None, n_buffers=LISTEN_BUFFERS)
        if m == finish:
            return True
        if m == trigger:
            _burst(sdr, reply)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  RX GAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════
def find_rx_gain_while_other_transmits(sdr, label):
    """
    Sweep our RX gain from low to high while the OTHER radio holds MSG_CAL_TONE.
    The partner keeps toning until WE tell it to stop, so the tone is present
    for the whole sweep regardless of how long the sweep takes. Pick the gain
    that decodes reliably without ADC clipping. Returns chosen gain or None.
    """
    print(f"\n[{label}] Sweeping RX gain against partner's cal tone...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []

    for g in RX_GAIN_SWEEP:
        set_rx_gain(sdr, g)
        time.sleep(0.1)             # hardware gain settle (local, not coordination)
        flush_rx(sdr, 1)

        decodes = 0
        peak_acc = 0
        for _ in range(GAIN_STEP_BUFFERS):
            try:
                rx   = sdr.rx()
                peak = np.max(np.abs(rx))
                peak_acc = max(peak_acc, peak)
                m, _, _ = decode_any(rx)
                if m == MSG_CAL_TONE:
                    decodes += 1
            except Exception:
                pass

        adc = peak_acc / 2896 * 100
        print(f"  {g:>5}  {peak_acc:>6.0f}  {adc:>4.0f}%  {decodes:>7}")

        if adc > 98:                # reject clipping
            continue
        if decodes >= GAIN_CONFIRM:
            score = decodes - (abs(adc - 60) / 100.0)   # prefer ADC near 60%
            candidates.append((score, g, adc, decodes))

    if not candidates:
        print(f"[{label}] No gain decoded the cal tone.")
        return None

    candidates.sort(reverse=True)
    best = candidates[0]
    print(f"[{label}] \u2713 Chosen RX gain = {best[1]} dB "
          f"(ADC {best[2]:.0f}%, {best[3]}/{GAIN_STEP_BUFFERS} decodes)")
    return best[1]


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER CALIBRATION  (purely message-gated)
# ═══════════════════════════════════════════════════════════════════════════════
def master_calibrate(sdr):
    print("\n" + "="*52)
    print("  MASTER — calibration sequence (handshake-driven)")
    print("="*52)

    # ── Phase 1: hold tone so SLAVE can set its RX gain ─────────────────────────
    print("\n[MASTER] Phase 1: holding cal tone until SLAVE locks its RX gain.")
    set_tx_atten(sdr, -30)              # fixed mid power for gain cal
    if hold_tone_until(sdr, stop_token=MSG_GAIN_OK):
        print("[MASTER] \u2713 SLAVE reported RX gain locked.")
    else:
        print("[MASTER] ! Never heard SLAVE gain-lock. Continuing.")

    # ── Phase 2: hand the tone to SLAVE, then sweep OUR RX gain ─────────────────
    print("\n[MASTER] Phase 2: handing tone to SLAVE so I can set MY RX gain.")
    if exchange(sdr, send_msg=MSG_SWAP, want_msg=MSG_CAL_TONE) is None:
        print("[MASTER] ! SLAVE never picked up the tone. Continuing.")
    else:
        print("[MASTER] \u2713 SLAVE is transmitting; sweeping my RX gain.")
    rx_gain = find_rx_gain_while_other_transmits(sdr, "MASTER")
    if rx_gain is None:
        rx_gain = 20
    set_rx_gain(sdr, rx_gain)

    # ── Phase 3: TX power negotiation ───────────────────────────────────────────
    # SLAVE is still holding the tone; the first probe it can decode makes it
    # stop toning and start echoing. We raise power until that echo comes back.
    print("\n[MASTER] Phase 3: negotiating TX power (weak -> strong).")
    chosen_atten = None
    for atten in TX_ATTEN_SWEEP:        # -80 .. 0
        set_tx_atten(sdr, atten)
        res = exchange(sdr, send_msg=MSG_PWR_PROBE, want_msg=MSG_PWR_ACK,
                       max_rounds=POWER_TRIES)
        ok = res is not None
        method = res[0] if ok else ""
        print(f"  TX atten {atten:>4} dB -> "
              f"{'SLAVE echoed (' + method + ')' if ok else 'no echo'}")
        if ok:
            chosen_atten = atten
            break

    if chosen_atten is None:
        chosen_atten = 0
        print("[MASTER] Could not confirm link, using max power (0 dB).")
    else:
        chosen_atten = min(0, chosen_atten + 5)   # small margin for stability
    set_tx_atten(sdr, chosen_atten)
    print(f"[MASTER] \u2713 TX attenuation = {chosen_atten} dB")

    # ── Phase 4: ready handshake ────────────────────────────────────────────────
    print("\n[MASTER] Phase 4: confirming link...")
    if exchange(sdr, send_msg=MSG_READY, want_msg=MSG_READY_ACK) is not None:
        # Tell SLAVE we're done so it stops acking and enters chat.
        for _ in range(4):
            _burst(sdr, MSG_DONE)
        print("[MASTER] \u2713 Link confirmed both ways!")
    else:
        print("[MASTER] ! No ready-ack from SLAVE. Entering chat anyway.")

    return rx_gain, chosen_atten


# ═══════════════════════════════════════════════════════════════════════════════
#  SLAVE CALIBRATION  (purely message-gated)
# ═══════════════════════════════════════════════════════════════════════════════
def slave_calibrate(sdr):
    print("\n" + "="*52)
    print("  SLAVE — calibration sequence (handshake-driven)")
    print("="*52)

    # ── Phase 1: sweep OUR RX gain against master's tone ────────────────────────
    print("\n[SLAVE] Phase 1: sweeping RX gain against MASTER's cal tone.")
    tx_silence(sdr)
    rx_gain = find_rx_gain_while_other_transmits(sdr, "SLAVE")
    if rx_gain is None:
        rx_gain = 20
    set_rx_gain(sdr, rx_gain)

    # Report the lock and wait for the SWAP that hands us the tone.
    print("[SLAVE] Reporting gain lock; waiting for SWAP...")
    set_tx_atten(sdr, -30)
    if exchange(sdr, send_msg=MSG_GAIN_OK, want_msg=MSG_SWAP) is None:
        print("[SLAVE] ! Never heard SWAP. Taking over the tone anyway.")
    else:
        print("[SLAVE] \u2713 Got SWAP — taking over the tone.")

    # ── Phase 2: hold tone so MASTER can set its RX gain ────────────────────────
    # Stop the instant MASTER starts power-probing (first PWR_PROBE) — that probe
    # is our cue to switch into echo mode.
    print("\n[SLAVE] Phase 2: holding cal tone until MASTER starts probing.")
    hold_tone_until(sdr, stop_token=MSG_PWR_PROBE)

    # ── Phase 3: echo every probe; leave when MASTER goes to READY ──────────────
    print("\n[SLAVE] Phase 3: echoing MASTER's power probes.")
    _burst(sdr, MSG_PWR_ACK)        # bootstrap-echo the probe that stopped us
    echo_until(sdr, trigger=MSG_PWR_PROBE, reply=MSG_PWR_ACK, finish=MSG_READY)

    # ── Phase 4: ready handshake ────────────────────────────────────────────────
    print("\n[SLAVE] Phase 4: confirming link...")
    if exchange(sdr, send_msg=MSG_READY_ACK, want_msg=MSG_DONE) is not None:
        print("[SLAVE] \u2713 Link confirmed!")
    else:
        print("[SLAVE] ! No DONE from MASTER. Entering chat anyway.")

    # SLAVE keeps the cal power for its own TX: master swept its RX gain and
    # negotiated using exactly this level, so it is known-good for slave->master.
    return rx_gain, -30


# ═══════════════════════════════════════════════════════════════════════════════
#  PTT CHAT  (same for both roles after calibration)
#  NOTE: the chat's "listen N seconds" is intentionally time-based — it waits on
#  a human to type, which is not radio coordination.
# ═══════════════════════════════════════════════════════════════════════════════
def transmit(sdr, message):
    tx_message_cyclic(sdr, message)
    time.sleep(0.4)                     # brief on-air repeat for reliability
    print(f"[TX \u2713] '{message}'  — listening...\n")
    tx_silence(sdr)

def listen(sdr, timeout=15.0):
    flush_rx(sdr, 4)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rx = sdr.rx()
            m, method, cfo = decode_any(rx)
            if m and not m.startswith("__"):   # ignore leftover control messages
                return m, method, cfo
        except Exception:
            pass
    return None, None, 0.0

def ptt_chat(sdr, rx_gain, tx_atten):
    set_rx_gain(sdr, rx_gain)
    set_tx_atten(sdr, tx_atten)
    tx_silence(sdr)

    print("\n" + "="*52)
    print("  LINK READY — PTT Chat")
    print(f"  RX gain     : {rx_gain} dB")
    print(f"  TX atten    : {tx_atten} dB")
    print(f"  Frequency   : {CENTER_FREQ/1e6:.3f} MHz")
    print("="*52)
    print("\n  Type a message + Enter  -> transmit then listen")
    print("  Press Enter (blank)     -> just listen 15s")
    print("  Type 'quit'             -> exit\n")

    while True:
        try:
            print("-"*40)
            msg = input("You: ").strip()
            if msg.lower() == 'quit':
                sys.exit(0)
            if msg:
                transmit(sdr, msg)
            print("[RX] Listening 15s...", flush=True)
            recv, method, cfo = listen(sdr, timeout=15.0)
            if recv:
                ts = time.strftime('%H:%M:%S')
                print(f"\n  +-[{ts}]----------------------")
                print(f"  | Them: {recv}")
                print(f"  | CFO method: {method}  ({cfo:+.0f} Hz)")
                print(f"  +--------------------------------\n")
            else:
                print("[RX] Nothing received.\n")
        except KeyboardInterrupt:
            print("\n[*] Exiting.")
            break
        except Exception as e:
            print(f"[ERR] {e}")


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    sdr = setup_pluto()

    print(f"\n*** ROLE: {ROLE.upper()} ***")
    print("Make sure the OTHER PC is started too (the other role).")
    print("Calibration begins in 3 seconds...")
    time.sleep(3)

    if ROLE == 'master':
        rx_gain, tx_atten = master_calibrate(sdr)
    else:
        rx_gain, tx_atten = slave_calibrate(sdr)

    print(f"\n[{ROLE.upper()}] Calibration done: "
          f"RX gain {rx_gain} dB, TX atten {tx_atten} dB")

    ptt_chat(sdr, rx_gain, tx_atten)


if __name__ == "__main__":
    main()