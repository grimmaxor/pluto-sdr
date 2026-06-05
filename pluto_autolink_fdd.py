"""
ADALM Pluto FDD Two-Radio Link + Full-Duplex Chat
==================================================
Runs the SAME script on both PCs. One is 'A', one is 'B'.

  PC1: python pluto_autolink_fdd.py --role a
  PC2: python pluto_autolink_fdd.py --role b

Optionally pin the Pluto URI / the two frequencies:
  python pluto_autolink_fdd.py --role a --ip ip:pluto.local --freq-a 2412e6 --freq-b 2437e6

WHY FDD
-------
Frequency-Division Duplex puts transmit and receive on DIFFERENT LO
frequencies, so a radio can transmit and receive at the SAME TIME without
colliding with itself or the other radio:

      role A :  TX on FREQ_A   ,  RX on FREQ_B
      role B :  TX on FREQ_B   ,  RX on FREQ_A      (mirror image)

That means there is no turn-taking, no master/slave sequencing, and no
timing-based coordination at all:

  * Both radios calibrate AT THE SAME TIME. Each one continuously transmits a
    beacon on its own TX frequency while independently sweeping its RX gain and
    negotiating its TX power on its RX frequency. The two links (A->B and B->A)
    are completely independent, so neither side waits on the other's clock.
  * Chat is genuinely full-duplex: a background thread receives continuously
    while you type, so both directions can talk simultaneously.

FREQUENCIES (LEGAL NOTE)
------------------------
Defaults are inside the 2.4 GHz ISM band (2400-2483.5 MHz), which is
license-exempt for short-range devices in many jurisdictions (e.g. Singapore
under IMDA TS SRD, EIRP roughly 100 mW-1 W). The Pluto's output is far below
that. EVEN SO: you are responsible for compliance with your local regulator
(power, occupied bandwidth, spurious emissions, duty cycle). The safest way to
develop this is CONDUCTED — connect the two radios over coax with attenuators
rather than radiating into the air:

      PC1 Pluto TX  --[attenuator]-->  PC2 Pluto RX
      PC2 Pluto TX  --[attenuator]-->  PC1 Pluto RX

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
parser.add_argument('--role',   choices=['a', 'b'], required=True,
                    help="'a' = TX on FREQ_A / RX on FREQ_B; 'b' = the mirror")
parser.add_argument('--ip',     type=str,   default='ip:pluto.local')
parser.add_argument('--freq-a', type=float, default=2412e6, dest='freq_a')
parser.add_argument('--freq-b', type=float, default=2437e6, dest='freq_b')
args = parser.parse_args()

ROLE   = args.role
FREQ_A = int(args.freq_a)
FREQ_B = int(args.freq_b)

# FDD frequency assignment — the two roles are mirror images of each other.
if ROLE == 'a':
    TX_FREQ, RX_FREQ = FREQ_A, FREQ_B
else:
    TX_FREQ, RX_FREQ = FREQ_B, FREQ_A

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
TX_BUFFER_SIZE     = 65536
SAMPLES_PER_SYMBOL = 16
MAX_MSG_LEN        = 180
FRAME_MAGIC        = 0xBE
BARKER_13          = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)

# Calibration sweeps
RX_GAIN_SWEEP      = list(range(0, 74, 3))      # 0..73 dB
TX_ATTEN_SWEEP     = list(range(-80, 1, 5))     # -80..0 dB (weak -> strong)
CAL_TX_ATTEN       = -30                         # fixed power used during RX-gain cal

# Tuning
GAIN_STEP_BUFFERS  = 6      # RX buffers captured per gain step
GAIN_CONFIRM       = 2      # decodes needed to accept a gain step
GAIN_SWEEP_RETRIES = 8      # re-sweep this many times if partner not yet heard
LISTEN_BUFFERS     = 5      # RX buffers captured when waiting for a control frame
POWER_ROUNDS       = 6      # listens per atten step before declaring "not heard"
POWER_MARGIN       = 5      # reduce atten by this (more power) for stability
READY_ROUNDS       = 40     # listens while confirming the link is up

# Control / data frame markers
MSG_CAL_TONE = "__CAL__"    # beacon used during the RX-gain sweep
# Status frame:  "S|<rxok>|<atten>|<ready>"
#   rxok  = 1 if THIS radio is currently decoding the other one
#   atten = this radio's current TX attenuation
#   ready = 1 once this radio has finished calibrating
# Chat frame:    "M|<seq>|<text>"

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
    """Encode a message into BPSK and TILE it to fill the whole TX buffer."""
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
    """Try all 3 CFO strategies, return (message, method, cfo) for first success."""
    peak = np.max(np.abs(iq))
    if peak < 5:
        return None, None, 0.0
    iq = (iq / peak * 2**13).astype(np.complex64)

    s0, _ = cfo_none(iq)
    sig   = lfilter(FILT, 1.0, s0.real).astype(np.float32)
    m     = _try_extract(sig)
    if m is not None:
        return m, "none", 0.0

    s1, cfo = cfo_fft(iq, SAMPLE_RATE)
    sig     = lfilter(FILT, 1.0, s1.real).astype(np.float32)
    m       = _try_extract(sig)
    if m is not None:
        return m, "fft", cfo

    s2, _ = cfo_pll(iq)
    sig   = lfilter(FILT, 1.0, s2.real).astype(np.float32)
    m     = _try_extract(sig)
    if m is not None:
        return m, "pll", 0.0

    return None, None, 0.0


# ── Frame helpers ───────────────────────────────────────────────────────────────
def make_status(rxok, atten, ready=0):
    return f"S|{int(rxok)}|{int(atten)}|{int(ready)}"

def parse_status(m):
    if m is None or not m.startswith("S|"):
        return None
    try:
        _, rxok, atten, ready = m.split("|")
        return {'rxok': int(rxok), 'atten': int(atten), 'ready': int(ready)}
    except Exception:
        return None

def make_chat(seq, text):
    return f"M|{seq}|{text[:150]}"

def parse_chat(m):
    if m is None or not m.startswith("M|"):
        return None
    try:
        _, seq, text = m.split("|", 2)
        return {'seq': seq, 'text': text}
    except Exception:
        return None


# ─── PLUTO ────────────────────────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate             = SAMPLE_RATE
    sdr.tx_lo                   = TX_FREQ      # FDD: independent TX LO
    sdr.rx_lo                   = RX_FREQ      # FDD: independent RX LO
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.tx_rf_bandwidth         = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = 10
    sdr.tx_hardwaregain_chan0   = CAL_TX_ATTEN
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

    print(f"[\u2713] Connected.  TX LO {sdr.tx_lo/1e6:.3f} MHz   "
          f"RX LO {sdr.rx_lo/1e6:.3f} MHz")
    return sdr


def set_rx_gain(sdr, g):
    sdr.rx_hardwaregain_chan0 = int(g)

def set_tx_atten(sdr, a):
    sdr.tx_hardwaregain_chan0 = int(a)

def rx_gain_limits(sdr):
    """
    Read the valid manual RX-gain range at the CURRENT rx_lo. The AD9361's gain
    range is frequency-dependent: the minimum is ~0 dB below 1.3 GHz but a few dB
    higher above it, so writing 0 dB at 2.4 GHz raises EINVAL. The device
    advertises the live range in 'hardwaregain_available' = "[min step max]".
    """
    try:
        ch  = sdr._ctrl.find_channel("voltage0", False)   # RX input gain channel
        raw = ch.attrs["hardwaregain_available"].value    # e.g. "[0.0 1.0 71.0]"
        nums = [float(x) for x in raw.strip("[] \t").split()]
        if len(nums) == 3 and nums[2] > nums[0]:
            return nums[0], nums[2]
    except Exception:
        pass
    return 0.0, 71.0      # safe default for the 1.3-4.0 GHz band

def build_gain_sweep(gmin, gmax, step=3):
    lo = int(np.ceil(gmin))
    hi = int(np.floor(gmax))
    if hi < lo:
        lo, hi = 0, 71
    return list(range(lo, hi + 1, step))

def tx_set(sdr, msg):
    """
    Push a cyclic TX buffer. The hardware loops it continuously on TX_FREQ; we
    can keep calling rx() (on RX_FREQ) the whole time — that is the point of FDD.
    Re-call this only when the transmitted message changes.
    """
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(encode_fill(msg))

def tx_idle(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))

def flush_rx(sdr, n=1):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass

def listen_one(sdr):
    """Capture a single RX buffer and decode it (or return None)."""
    try:
        rx = sdr.rx()
        return decode_any(rx)
    except Exception:
        return None, None, 0.0

def listen_for(sdr, predicate, n_buffers=LISTEN_BUFFERS):
    """
    Capture up to n_buffers and return the first decoded message for which
    predicate(msg) is True, else None. Counts BUFFERS, not wall-clock time.
    """
    flush_rx(sdr, 1)
    for _ in range(n_buffers):
        m, method, cfo = listen_one(sdr)
        if m is not None and predicate(m):
            return m, method, cfo
    return None, None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION  (identical on both radios; the two links are independent)
# ═══════════════════════════════════════════════════════════════════════════════
def find_rx_gain(sdr, label, sweep):
    """
    Sweep RX gain while the partner beacons on our RX frequency. Count ANY valid
    decode as a hit (the partner may be sending a cal tone or a status frame).
    Returns the chosen gain or None if the partner wasn't heard.
    """
    print(f"\n[{label}] Sweeping RX gain ...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []

    for g in sweep:
        try:
            set_rx_gain(sdr, g)     # skip any value the device rejects (EINVAL)
        except OSError:
            continue
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
                if m is not None:
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
        return None
    candidates.sort(reverse=True)
    best = candidates[0]
    print(f"[{label}] \u2713 RX gain = {best[1]} dB "
          f"(ADC {best[2]:.0f}%, {best[3]}/{GAIN_STEP_BUFFERS} decodes)")
    return best[1]


def calibrate_rx_gain(sdr, label):
    """Beacon continuously and sweep RX gain, retrying until the partner is heard."""
    set_tx_atten(sdr, CAL_TX_ATTEN)
    tx_set(sdr, MSG_CAL_TONE)               # beacon loops in hardware during the sweep
    gmin, gmax = rx_gain_limits(sdr)
    sweep = build_gain_sweep(gmin, gmax)
    print(f"[{label}] Valid RX gain at this frequency: {gmin:.0f}..{gmax:.0f} dB")
    for attempt in range(GAIN_SWEEP_RETRIES):
        g = find_rx_gain(sdr, label, sweep)
        if g is not None:
            set_rx_gain(sdr, g)
            return g
        print(f"[{label}] partner not heard yet, re-sweeping ({attempt+1}).")
    fallback = int(np.clip(20, np.ceil(gmin), np.floor(gmax)))
    print(f"[{label}] ! Falling back to RX gain {fallback} dB.")
    set_rx_gain(sdr, fallback)
    return fallback


def calibrate_tx_power(sdr, label):
    """
    Find the minimum TX power the partner can decode. We advertise a status
    frame whose rxok bit says whether WE currently decode the partner; we read
    the partner's status frame to learn whether IT decodes US. We raise power
    until the partner reports rxok==1 about our link, then add a margin.
    Fully parallel: the partner is doing the same thing on the other frequency.
    """
    print(f"\n[{label}] Negotiating TX power (weak -> strong) ...")
    my_rxok = 1                              # we just established RX in the gain phase
    advertised = None
    chosen = None

    def advertise(atten):
        nonlocal advertised
        frame = make_status(my_rxok, atten, ready=0)
        if frame != advertised:              # only re-push when it changes
            tx_set(sdr, frame)
            advertised = frame

    for atten in TX_ATTEN_SWEEP:
        set_tx_atten(sdr, atten)
        advertise(atten)
        partner_ok = False
        for _ in range(POWER_ROUNDS):
            m, _, _ = listen_one(sdr)
            st = parse_status(m)
            if st is not None:
                my_rxok = 1                  # we can hear the partner
                advertise(atten)             # let the partner know we hear it
                if st['rxok'] == 1:          # partner can hear US at this power
                    partner_ok = True
                    break
            else:
                my_rxok = 0
                advertise(atten)
        state = "partner decodes us" if partner_ok else "no confirmation"
        print(f"  TX atten {atten:>4} dB -> {state}")
        if partner_ok:
            chosen = atten
            break

    if chosen is None:
        chosen = 0
        print(f"[{label}] Could not confirm; using max power (0 dB).")
    else:
        chosen = min(0, chosen + POWER_MARGIN)
    set_tx_atten(sdr, chosen)
    print(f"[{label}] \u2713 TX attenuation = {chosen} dB")
    return chosen


def confirm_link(sdr, label, tx_atten):
    """
    Announce 'ready' and wait to hear the partner's 'ready'. The ready frame
    still carries rxok=1, so a partner that is finishing its own power
    negotiation keeps getting valid confirmation from us.
    """
    print(f"\n[{label}] Confirming link ...")
    tx_set(sdr, make_status(1, tx_atten, ready=1))
    for _ in range(READY_ROUNDS):
        m, _, _ = listen_one(sdr)
        st = parse_status(m)
        if st is not None and st['ready'] == 1:
            print(f"[{label}] \u2713 Link confirmed both ways!")
            return True
    print(f"[{label}] ! Partner ready not seen. Entering chat anyway.")
    return False


def calibrate(sdr, label):
    print("\n" + "="*52)
    print(f"  ROLE {label} — FDD calibration (runs in parallel)")
    print("="*52)
    rx_gain  = calibrate_rx_gain(sdr, label)
    tx_atten = calibrate_tx_power(sdr, label)
    confirm_link(sdr, label, tx_atten)
    return rx_gain, tx_atten


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL-DUPLEX CHAT
#  A background thread receives continuously on RX_FREQ while the main thread
#  reads the keyboard and transmits on TX_FREQ. Both directions are live at once.
# ═══════════════════════════════════════════════════════════════════════════════
def fdd_chat(sdr, rx_gain, tx_atten):
    set_rx_gain(sdr, rx_gain)
    set_tx_atten(sdr, tx_atten)

    print("\n" + "="*52)
    print("  LINK READY — Full-Duplex Chat")
    print(f"  Role        : {ROLE.upper()}")
    print(f"  TX LO       : {TX_FREQ/1e6:.3f} MHz   (atten {tx_atten} dB)")
    print(f"  RX LO       : {RX_FREQ/1e6:.3f} MHz   (gain  {rx_gain} dB)")
    print("="*52)
    print("\n  Type a message + Enter  -> sent immediately (you can keep typing)")
    print("  Incoming messages print as they arrive")
    print("  Type 'quit'             -> exit\n")

    lock  = threading.Lock()              # serialize libiio access across threads
    stop  = threading.Event()
    seen  = {'seq': None}

    def rx_loop():
        with lock:
            flush_rx(sdr, 2)
        while not stop.is_set():
            with lock:
                try:
                    rx = sdr.rx()
                except Exception:
                    rx = None
            if rx is not None:
                m, method, cfo = decode_any(rx)
                ch = parse_chat(m)
                if ch is not None and ch['seq'] != seen['seq']:
                    seen['seq'] = ch['seq']
                    ts = time.strftime('%H:%M:%S')
                    sys.stdout.write(
                        f"\n  +-[{ts}]----------------------\n"
                        f"  | Them: {ch['text']}\n"
                        f"  | CFO: {method} ({cfo:+.0f} Hz)\n"
                        f"  +--------------------------------\n"
                        f"You: ")
                    sys.stdout.flush()
            else:
                time.sleep(0.005)

    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    rx_thread.start()

    with lock:
        tx_idle(sdr)                      # transmit nothing until we have something to say

    seq = 0
    try:
        while True:
            msg = input("You: ").strip()
            if msg.lower() == 'quit':
                break
            if msg:
                seq += 1
                with lock:
                    tx_set(sdr, make_chat(seq, msg))   # loops on air until the next message
                print(f"[TX \u2713] '{msg}'")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop.set()
        rx_thread.join(timeout=1.0)
        with lock:
            tx_idle(sdr)
        print("\n[*] Exiting.")


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    sdr = setup_pluto()

    print(f"\n*** ROLE: {ROLE.upper()} ***")
    print(f"    TX on {TX_FREQ/1e6:.3f} MHz, RX on {RX_FREQ/1e6:.3f} MHz")
    print("    Start the OTHER PC too (the other role).")
    print("    Calibration begins in 3 seconds (both sides calibrate at once)...")
    time.sleep(3)

    rx_gain, tx_atten = calibrate(sdr, ROLE.upper())

    print(f"\n[{ROLE.upper()}] Calibration done: "
          f"RX gain {rx_gain} dB, TX atten {tx_atten} dB")

    fdd_chat(sdr, rx_gain, tx_atten)


if __name__ == "__main__":
    main()