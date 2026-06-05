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

The calibration is driven entirely over the radio link itself.
The master always initiates each phase; the slave only responds to what it
hears, so the two never transmit simultaneously and cannot deadlock.

CFO handling: every decode tries three strategies (none / FFT-coarse / PLL)
and accepts the first that yields a valid CRC.
"""

import adi
import numpy as np
import argparse
import threading
import time
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

# Special control messages used during calibration
MSG_CAL_TONE   = "__CAL__"          # master's calibration beacon
MSG_GAIN_OK    = "__GAINOK__"       # slave found its gain
MSG_PWR_PROBE  = "__PWR__"          # power probe (master varies TX power)
MSG_READY      = "__READY__"        # handshake complete

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
    time.sleep(0.05)
    sdr.tx(encode_fill(msg))

def flush_rx(sdr, n=4):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass
        time.sleep(0.02)

def listen_for(sdr, want=None, timeout=5.0):
    """
    Listen up to `timeout` sec. If `want` is given, only return when that
    exact message is decoded. Otherwise return first decoded message.
    Returns (msg, method, cfo) or (None, None, 0).
    """
    flush_rx(sdr, 3)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rx = sdr.rx()
            m, method, cfo = decode_any(rx)
            if m is not None:
                if want is None or m == want:
                    return m, method, cfo
        except Exception as e:
            if "timeout" not in str(e).lower():
                pass
        time.sleep(0.02)
    return None, None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# How many calibration packets must decode at a gain step to accept it
GAIN_CONFIRM = 3

def find_rx_gain_while_other_transmits(sdr, label):
    """
    Sweep our RX gain from low to high while the OTHER radio sends MSG_CAL_TONE.
    Pick the gain that decodes reliably without ADC clipping.
    Returns chosen gain or None.
    """
    print(f"\n[{label}] Sweeping RX gain to hear calibration tone...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []

    for g in RX_GAIN_SWEEP:
        set_rx_gain(sdr, g)
        time.sleep(0.15)
        flush_rx(sdr, 2)

        decodes = 0
        peak_acc = 0
        for _ in range(5):
            try:
                rx   = sdr.rx()
                peak = np.max(np.abs(rx))
                peak_acc = max(peak_acc, peak)
                m, _, _ = decode_any(rx)
                if m == MSG_CAL_TONE:
                    decodes += 1
            except Exception:
                pass
            time.sleep(0.02)

        adc = peak_acc / 2896 * 100
        print(f"  {g:>5}  {peak_acc:>6.0f}  {adc:>4.0f}%  {decodes:>7}")

        # Reject clipping
        if adc > 98:
            continue
        if decodes >= GAIN_CONFIRM:
            # Score = decodes, prefer ADC in 40-85% range
            score = decodes - (abs(adc - 60) / 100.0)
            candidates.append((score, g, adc, decodes))

    if not candidates:
        print(f"[{label}] No gain decoded the cal tone.")
        return None

    candidates.sort(reverse=True)
    best = candidates[0]
    print(f"[{label}] \u2713 Chosen RX gain = {best[1]} dB "
          f"(ADC {best[2]:.0f}%, {best[3]}/5 decodes)")
    return best[1]


def master_calibrate(sdr):
    print("\n" + "="*52)
    print("  MASTER — calibration sequence")
    print("="*52)

    # ── Phase A: slave finds its RX gain while we transmit cal tone ──
    print("\n[MASTER] Phase A: transmitting cal tone so SLAVE can set RX gain.")
    set_tx_atten(sdr, -30)            # fixed mid power for gain cal
    tx_message_cyclic(sdr, MSG_CAL_TONE)
    # Give slave generous time to sweep its gain
    # Slave signals done by sending MSG_GAIN_OK; we listen for it.
    print("[MASTER] Waiting for SLAVE to report gain lock (up to 60s)...")
    # We must stop transmitting to hear the slave, but slave needs our tone
    # to calibrate. So: transmit for a while, then pause and check.
    got_ok = False
    t_end = time.time() + 60
    while time.time() < t_end:
        # transmit cal tone for 2s
        tx_message_cyclic(sdr, MSG_CAL_TONE)
        time.sleep(2.0)
        # pause and listen for 1.5s
        tx_silence(sdr)
        m, _, _ = listen_for(sdr, want=MSG_GAIN_OK, timeout=1.5)
        if m == MSG_GAIN_OK:
            got_ok = True
            break
    if not got_ok:
        print("[MASTER] Did not hear slave gain-OK. Continuing anyway.")
    else:
        print("[MASTER] \u2713 Slave reported RX gain locked.")

    # ── Phase B: master finds its own RX gain while slave transmits ──
    print("\n[MASTER] Phase B: asking SLAVE to transmit so I can set MY RX gain.")
    # Tell slave to start transmitting cal tone
    tx_message_cyclic(sdr, MSG_PWR_PROBE)  # reuse as 'your turn to TX' signal
    time.sleep(2.0)
    tx_silence(sdr)
    # Now slave should be transmitting cal tone; sweep our gain
    rx_gain = find_rx_gain_while_other_transmits(sdr, "MASTER")
    if rx_gain is None:
        rx_gain = 20
    set_rx_gain(sdr, rx_gain)

    # ── Phase C: TX power negotiation ──
    print("\n[MASTER] Phase C: negotiating TX power (high atten -> low atten).")
    chosen_atten = None
    for atten in TX_ATTEN_SWEEP:    # -80 .. 0  (weak -> strong)
        set_tx_atten(sdr, atten)
        tx_message_cyclic(sdr, MSG_PWR_PROBE)
        time.sleep(1.5)
        tx_silence(sdr)
        # slave echoes MSG_PWR_PROBE back if it heard us
        m, method, _ = listen_for(sdr, want=MSG_PWR_PROBE, timeout=2.0)
        ok = (m == MSG_PWR_PROBE)
        print(f"  TX atten {atten:>4} dB -> "
              f"{'slave HEARD (' + (method or '') + ')' if ok else 'no echo'}")
        if ok:
            chosen_atten = atten
            break
    if chosen_atten is None:
        chosen_atten = 0
        print("[MASTER] Could not confirm link, using max power (0 dB).")
    else:
        # add small margin (more power) for stability: reduce atten by 5
        chosen_atten = min(0, chosen_atten + 5)
    set_tx_atten(sdr, chosen_atten)
    print(f"[MASTER] \u2713 TX attenuation = {chosen_atten} dB")

    # ── Phase D: final handshake ──
    print("\n[MASTER] Phase D: confirming link...")
    for _ in range(10):
        tx_message_cyclic(sdr, MSG_READY)
        time.sleep(1.0)
        tx_silence(sdr)
        m, _, _ = listen_for(sdr, want=MSG_READY, timeout=1.5)
        if m == MSG_READY:
            print("[MASTER] \u2713 Link confirmed both ways!")
            break

    return rx_gain, chosen_atten


def slave_calibrate(sdr):
    print("\n" + "="*52)
    print("  SLAVE — calibration sequence")
    print("="*52)

    # ── Phase A: find OUR rx gain while master sends cal tone ──
    print("\n[SLAVE] Phase A: sweeping RX gain against master's cal tone.")
    tx_silence(sdr)
    rx_gain = find_rx_gain_while_other_transmits(sdr, "SLAVE")
    if rx_gain is None:
        rx_gain = 20
    set_rx_gain(sdr, rx_gain)

    # Tell master we locked gain — send MSG_GAIN_OK a few times
    print("[SLAVE] Reporting gain lock to master...")
    set_tx_atten(sdr, -30)
    for _ in range(6):
        tx_message_cyclic(sdr, MSG_GAIN_OK)
        time.sleep(1.0)
        tx_silence(sdr)
        # check if master moved to phase B (asks us to transmit)
        m, _, _ = listen_for(sdr, want=MSG_PWR_PROBE, timeout=1.0)
        if m == MSG_PWR_PROBE:
            break

    # ── Phase B: master sweeps its gain; we transmit cal tone ──
    print("\n[SLAVE] Phase B: transmitting cal tone so master sets its gain.")
    for _ in range(20):
        tx_message_cyclic(sdr, MSG_CAL_TONE)
        time.sleep(1.0)
        tx_silence(sdr)
        # Master will move to power probe; detect it
        m, _, _ = listen_for(sdr, want=MSG_PWR_PROBE, timeout=0.8)
        if m == MSG_PWR_PROBE:
            break

    # ── Phase C: echo power probes back to master ──
    print("\n[SLAVE] Phase C: echoing master's power probes.")
    set_tx_atten(sdr, -30)
    t_end = time.time() + 40
    while time.time() < t_end:
        m, method, _ = listen_for(sdr, want=MSG_PWR_PROBE, timeout=2.0)
        if m == MSG_PWR_PROBE:
            # echo back
            tx_message_cyclic(sdr, MSG_PWR_PROBE)
            time.sleep(1.2)
            tx_silence(sdr)
        # Detect master moving to READY
        m2, _, _ = listen_for(sdr, want=MSG_READY, timeout=0.8)
        if m2 == MSG_READY:
            break

    # ── Phase D: handshake ──
    print("\n[SLAVE] Phase D: confirming link...")
    for _ in range(10):
        m, _, _ = listen_for(sdr, want=MSG_READY, timeout=1.5)
        if m == MSG_READY:
            tx_message_cyclic(sdr, MSG_READY)
            time.sleep(1.0)
            tx_silence(sdr)
            print("[SLAVE] \u2713 Link confirmed!")
            break

    return rx_gain, -30


# ═══════════════════════════════════════════════════════════════════════════════
#  PTT CHAT  (same for both roles after calibration)
# ═══════════════════════════════════════════════════════════════════════════════
def transmit(sdr, message):
    tx_message_cyclic(sdr, message)
    # repeat briefly for reliability
    time.sleep(0.4)
    print(f"[TX \u2713] '{message}'  — listening...\n")
    tx_silence(sdr)

def listen(sdr, timeout=15.0):
    flush_rx(sdr, 4)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rx = sdr.rx()
            m, method, cfo = decode_any(rx)
            # ignore leftover control messages
            if m and not m.startswith("__"):
                return m, method, cfo
        except Exception:
            pass
        time.sleep(0.02)
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
