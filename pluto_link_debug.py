#!/usr/bin/env python3
"""
pluto_link_debug.py — FDD link diagnostic for pluto_image_fdd_raw.py

Run on BOTH PCs simultaneously:
  Linux   (tx role): python pluto_link_debug.py --role tx
  Windows (rx role): python pluto_link_debug.py --role rx --ip ip:192.168.2.1

The script runs two phases back-to-back (both sides in parallel):

  Phase 1 — DATA channel (Linux tx -> Windows rx)
    Linux Pluto transmits at each attenuation level.
    Windows Pluto reports how many it receives.

  Phase 2 — CTRL channel (Windows rx -> Linux tx)   ← LIKELY FAILURE POINT
    Windows Pluto transmits at each attenuation level.
    Linux Pluto reports how many it receives.
    If Phase 2 decode count is 0 or much lower than Phase 1, the Windows
    Pluto TX is too weak. Fix: add  --tx-atten 0  on the Windows side.

Signal is reported even when packets don't decode (amplitude / correlation peak)
so you can distinguish "no RF at all" from "RF present but can't decode".

Use --quick for a 30-second listen-only sniff with no TX (safe to run first
to verify the Pluto is connected and the frequency is right).
"""

import sys, time, struct, zlib, threading, argparse
import numpy as np
from scipy.signal import firwin, lfilter

try:
    import adi
except ImportError:
    print("[!] pyadi-iio not installed: pip install pyadi-iio"); sys.exit(1)

# ─── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                             description=__doc__)
ap.add_argument('--role',      choices=['tx', 'rx'], required=True,
                help='tx=Linux/sender  rx=Windows/receiver')
ap.add_argument('--ip',        default='ip:pluto.local',
                help='Pluto URI (default: ip:pluto.local)')
ap.add_argument('--freq-data', type=float, default=2412e6, dest='freq_data',
                help='DATA channel frequency Hz (must match both sides)')
ap.add_argument('--freq-ctrl', type=float, default=2437e6, dest='freq_ctrl',
                help='CTRL channel frequency Hz (must match both sides)')
ap.add_argument('--sps',       type=int, default=16,
                help='Samples per symbol (default 16, must match both sides)')
ap.add_argument('--rx-gain',   type=int, default=40,
                help='Starting RX gain dB (default 40)')
ap.add_argument('--quick',     action='store_true',
                help='30-second listen-only sniff, no TX — safe first check')
args = ap.parse_args()

ROLE = args.role
# Each role's TX carrier is its *outbound* channel
if ROLE == 'tx':   # Linux: sends image data, receives control
    TX_FREQ, RX_FREQ = int(args.freq_data), int(args.freq_ctrl)
else:              # Windows: sends control, receives image data
    TX_FREQ, RX_FREQ = int(args.freq_ctrl), int(args.freq_data)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SAMPLE_RATE  = int(1e6)
SPS          = args.sps
TX_BUF       = 65536
RX_BUF       = 262144
MAGIC        = 0xA5
PKT_DBG      = 0xDD    # debug-beacon type (not used by normal scripts)

BARKER_13      = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)
PREAMBLE_SIGNS = np.tile(BARKER_13, 3).astype(np.float32)
PREAMBLE_LEN   = len(PREAMBLE_SIGNS)

# Sweep: send this many beacons at each TX attenuation level
ATTEN_LEVELS      = list(range(-40, 1, 5))   # -40 .. 0 dB
BEACONS_PER_LEVEL = 20
BEACON_INTERVAL   = 0.25   # seconds between beacons
PHASE_GAP         = 4.0    # silence between Phase 1 and Phase 2

PHASE_DURATION = len(ATTEN_LEVELS) * (BEACONS_PER_LEVEL * BEACON_INTERVAL + 1.0)

FILT = firwin(SPS * 4 + 1, 1.4 / SPS, window='hamming').astype(np.float32)


# ─── FRAMING (identical to fdd_raw.py) ───────────────────────────────────────
def build_packet(pkt_type, seq, total, payload=b''):
    header = struct.pack('>BBIIH', MAGIC, pkt_type, seq, total, len(payload))
    body   = header + payload
    crc    = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack('>I', crc)


def parse_packet(raw, start):
    HDR = 12
    if start + HDR > len(raw):
        return None
    magic, ptype, seq, total, plen = struct.unpack('>BBIIH', raw[start:start+HDR])
    if magic != MAGIC:
        return None
    if plen > 4096:
        return None
    end = start + HDR + plen + 4
    if end > len(raw):
        return None
    payload  = raw[start+HDR : start+HDR+plen]
    crc_rx   = struct.unpack('>I', raw[start+HDR+plen : end])[0]
    crc_calc = zlib.crc32(raw[start:start+HDR+plen]) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return None
    return {'type': ptype, 'seq': seq, 'total': total, 'payload': payload}


# ─── DSP (identical to fdd_raw.py) ───────────────────────────────────────────
def packet_to_iq(pkt_bytes):
    pbits  = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    payload = (1.0 - 2.0 * pbits.astype(np.float32))
    syms   = np.concatenate([PREAMBLE_SIGNS, payload]).astype(np.complex64)
    up     = np.zeros(len(syms) * SPS, dtype=np.complex64)
    up[::SPS] = syms
    shaped = lfilter(FILT, 1.0, up.real).astype(np.float32).astype(np.complex64)
    mx = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15
    plen = len(shaped)
    if plen >= TX_BUF:
        return shaped[:TX_BUF].astype(np.complex64)
    n_whole = TX_BUF // plen
    body    = np.tile(shaped, n_whole)
    pad     = np.zeros(TX_BUF - len(body), dtype=np.complex64)
    return np.concatenate([body, pad]).astype(np.complex64)


def _cfo_correct(iq):
    """Yield raw IQ, then CFO-corrected version if CFO > 50 Hz."""
    yield iq
    nrm = iq / (np.max(np.abs(iq)) + 1e-9)
    sq  = nrm ** 2
    fv  = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(len(sq), d=1.0 / SAMPLE_RATE)
    cfo   = freqs[int(np.argmax(np.abs(fv)))] / 2
    if abs(cfo) > 50:
        t = np.arange(len(iq)) / SAMPLE_RATE
        yield (iq * np.exp(-1j * 2 * np.pi * cfo * t)).astype(np.complex64)


def _find_packets(syms):
    found = {}
    if len(syms) < PREAMBLE_LEN + 32:
        return found
    signs = np.sign(syms.real).astype(np.float32)
    corr  = np.correlate(signs, PREAMBLE_SIGNS, mode='valid')
    thr   = PREAMBLE_LEN * 0.8
    for c in np.where(np.abs(corr) > thr)[0]:
        inverted   = corr[c] < 0
        data_start = c + PREAMBLE_LEN
        for slip in (0, 1, -1, 2, -2):
            ss = data_start + slip
            if ss < 0 or ss >= len(syms):
                continue
            ds   = syms[ss:]
            bits = ((-ds.real if inverted else ds.real) < 0).astype(np.uint8)
            nb   = len(bits) // 8
            if nb < 16:
                continue
            raw = bytes(np.packbits(bits[:nb * 8]))
            pkt = parse_packet(raw, 0)
            if pkt:
                found[(pkt['type'], pkt['seq'])] = pkt
                break
    return found


def iq_to_packets_diag(iq):
    """
    Demod with diagnostic stats.
    Returns (packets_list, diag_dict) where diag keys:
      raw_amplitude  — max |IQ| before normalisation (proxy for SNR)
      cfo_hz         — coarse CFO estimate
      corr_peak      — best preamble correlation value (max = PREAMBLE_LEN = 39)
      n_candidates   — number of correlation hits above threshold
    """
    raw_amp = float(np.max(np.abs(iq)))
    diag = {'raw_amplitude': raw_amp, 'cfo_hz': 0.0,
            'corr_peak': 0.0, 'n_candidates': 0}

    if raw_amp < 5:
        return [], diag

    iq_norm = (iq / raw_amp).astype(np.complex64)

    # Compute CFO for diagnostic display even if we don't apply it
    sq    = iq_norm ** 2
    fv    = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(len(sq), d=1.0 / SAMPLE_RATE)
    diag['cfo_hz'] = float(freqs[int(np.argmax(np.abs(fv)))] / 2)

    delay = len(FILT) // 2
    found = {}
    for corrected in _cfo_correct(iq_norm):
        filt = lfilter(FILT, 1.0, corrected.real).astype(np.float32).astype(np.complex64)
        for toff in range(SPS):
            start  = (delay + toff) % SPS
            stream = filt[start::SPS]
            if len(stream) < PREAMBLE_LEN + 32:
                continue
            signs = np.sign(stream.real).astype(np.float32)
            corr  = np.correlate(signs, PREAMBLE_SIGNS, mode='valid')
            thr   = PREAMBLE_LEN * 0.8
            cands = np.where(np.abs(corr) > thr)[0]
            if len(cands):
                diag['n_candidates'] += len(cands)
                peak = float(np.max(np.abs(corr[cands])))
                if peak > diag['corr_peak']:
                    diag['corr_peak'] = peak
            for pkt_map in [_find_packets(stream)]:
                found.update(pkt_map)
        if found:
            break

    return list(found.values()), diag


# ─── PLUTO SETUP ──────────────────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting {args.ip}")
    print(f"    TX={TX_FREQ/1e6:.3f} MHz  RX={RX_FREQ/1e6:.3f} MHz  "
          f"SPS={SPS}  RXgain={args.rx_gain}")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate             = SAMPLE_RATE
    sdr.tx_lo                   = TX_FREQ
    sdr.rx_lo                   = RX_FREQ
    sdr.tx_rf_bandwidth         = SAMPLE_RATE
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = int(args.rx_gain)
    sdr.tx_hardwaregain_chan0   = -40   # silent until we're ready
    sdr.rx_buffer_size          = RX_BUF
    sdr.tx_cyclic_buffer        = True

    # Disable DDS tone (critical — otherwise it pollutes TX)
    dds_ok = False
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
            dds_ok = True
    except Exception as e:
        print(f"[!] DDS disable failed: {e}")
        print(f"    If 'iio' is not installed on Windows: pip install pylibiio")

    print(f"[{'✓' if dds_ok else '!'}] DDS tone {'disabled' if dds_ok else 'NOT disabled — TX may be noisy'}")
    print("[✓] Connected\n")
    return sdr


def tx_packet(sdr, pkt_bytes):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(packet_to_iq(pkt_bytes))


def tx_silence(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(np.zeros(TX_BUF, dtype=np.complex64))


def set_atten(sdr, a):
    sdr.tx_hardwaregain_chan0 = int(a)


def set_gain(sdr, g):
    sdr.rx_hardwaregain_chan0 = int(g)


def flush_rx(sdr, n=3):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass


# ─── BACKGROUND LISTENER ──────────────────────────────────────────────────────
class Listener:
    """
    Drains RX continuously in a background thread.
    Call snapshot() to get accumulated stats and clear the buffer.
    """
    def __init__(self, sdr):
        self._sdr   = sdr
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._buf   = []   # list of (pkts, diag)
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw = self._sdr.rx()
                pkts, diag = iq_to_packets_diag(raw)
                with self._lock:
                    self._buf.append((pkts, diag))
            except Exception:
                pass
            time.sleep(0.005)

    def snapshot(self):
        """Return accumulated (pkts, diag) list and reset."""
        with self._lock:
            s = self._buf[:]
            self._buf.clear()
        return s

    def stop(self):
        self._stop.set()


# ─── QUICK SNIFF ──────────────────────────────────────────────────────────────
def quick_sniff(sdr):
    """30-second passive listen. No TX. Shows signal level even without decodes."""
    print("=" * 60)
    print(f"  SNIFF MODE — 30s passive listen on RX={RX_FREQ/1e6:.3f} MHz")
    print("  (No TX — safe to run first)")
    print("=" * 60)
    tx_silence(sdr)
    flush_rx(sdr, 5)

    t0     = time.time()
    n_pkt  = 0
    n_call = 0
    max_amp  = 0.0
    max_corr = 0.0

    print(f"  {'Time':>6}  {'Amp':>6}  {'Corr':>5}  {'CFO':>8}  {'Cands':>5}  Decoded")
    while time.time() - t0 < 30:
        try:
            raw = sdr.rx()
        except Exception:
            continue
        pkts, diag = iq_to_packets_diag(raw)
        n_call += 1
        n_pkt  += len(pkts)
        amp    = diag['raw_amplitude']
        corr   = diag['corr_peak']
        cands  = diag['n_candidates']
        cfo    = diag['cfo_hz']
        if amp > max_amp:
            max_amp = amp
        if corr > max_corr:
            max_corr = corr

        # Print if there's something interesting
        if pkts or amp > 200 or corr > 5:
            elapsed = time.time() - t0
            tag = f"DECODED {len(pkts)}" if pkts else "signal, no decode"
            print(f"  {elapsed:6.1f}s  {amp:6.0f}  {corr:5.1f}  {cfo:+8.0f}Hz"
                  f"  {cands:5}  {tag}")

    print()
    print(f"  Results: {n_pkt} packets decoded / {n_call} rx() calls")
    print(f"  Peak amplitude: {max_amp:.0f}   Peak correlation: {max_corr:.1f}/{PREAMBLE_LEN}")
    print()
    if n_pkt > 0:
        print("  [✓] Packets decoded — link is working on this channel")
    elif max_amp > 2000:
        print("  [!] Strong signal but no packets decoded")
        print("      → Possible CFO mismatch (try different --freq-data/ctrl)")
        print("      → Or wrong SPS (--sps must match both sides)")
        print(f"      → CFO seen: up to ~{max_corr:.0f}  corr peak: {max_corr:.1f}")
    elif max_amp > 200:
        print("  [~] Weak signal — no packets decoded")
        print("      → Try increasing --rx-gain (current: {})".format(args.rx_gain))
        print("      → Or partner TX attenuation is too high")
    else:
        print("  [!] No meaningful signal detected")
        print("      → Check: is the other Pluto transmitting?")
        print("      → Check: --freq-data and --freq-ctrl match on both sides")
        print("      → Check: physical connection / antenna / distance")


# ─── SWEEP PHASE ──────────────────────────────────────────────────────────────
def run_as_beaconer(sdr, phase_label, channel_label):
    """TX role: sweep attenuation levels, send beacons at each."""
    pkt = build_packet(PKT_DBG, 0, 0, f"DBG:{ROLE}".encode())
    print(f"\n  [{phase_label}] Beaconing on {channel_label}")
    print(f"  Sweeping TX atten {ATTEN_LEVELS[0]}..{ATTEN_LEVELS[-1]} dB, "
          f"{BEACONS_PER_LEVEL} packets/level")
    print(f"  {'Atten':>6}  Status")

    for atten in ATTEN_LEVELS:
        set_atten(sdr, atten)
        tx_packet(sdr, pkt)
        print(f"  {atten:>+5}dB  sending...", end='', flush=True)
        for _ in range(BEACONS_PER_LEVEL):
            time.sleep(BEACON_INTERVAL)
        print(f"  done")

    tx_silence(sdr)
    set_atten(sdr, -40)


def run_as_listener(sdr, listener, phase_label, channel_label):
    """
    RX role: listen for PHASE_DURATION + 6s buffer.
    Accumulates decode counts and signal stats second-by-second.
    Returns summary dict.
    """
    listen_dur = PHASE_DURATION + 6.0
    print(f"\n  [{phase_label}] Listening on {channel_label} for {listen_dur:.0f}s")
    print(f"  {'Time':>6}  {'Decoded':>7}  {'Amp':>6}  {'Corr':>5}  {'CFO':>8}")

    listener.snapshot()  # clear old data
    t0        = time.time()
    total_pkt = 0
    max_amp   = 0.0
    max_corr  = 0.0

    while time.time() - t0 < listen_dur:
        time.sleep(1.0)
        stats = listener.snapshot()
        n_pkt  = sum(len(p) for p, _ in stats)
        amps   = [d['raw_amplitude'] for _, d in stats if d['raw_amplitude'] > 5]
        corrs  = [d['corr_peak']     for _, d in stats if d['corr_peak'] > 0]
        cfos   = [d['cfo_hz']        for _, d in stats if d['n_candidates'] > 0]
        total_pkt += n_pkt
        avg_amp    = float(np.mean(amps))  if amps  else 0.0
        avg_corr   = float(np.mean(corrs)) if corrs else 0.0
        avg_cfo    = float(np.mean(cfos))  if cfos  else 0.0
        if avg_amp > max_amp:
            max_amp = avg_amp
        if avg_corr > max_corr:
            max_corr = avg_corr
        elapsed = time.time() - t0
        print(f"  {elapsed:6.1f}s  {total_pkt:7}  {avg_amp:6.0f}  {avg_corr:5.1f}  {avg_cfo:+8.0f}Hz")

    return {'total_decoded': total_pkt, 'max_amplitude': max_amp, 'max_corr': max_corr}


# ─── FULL SWEEP ───────────────────────────────────────────────────────────────
def full_sweep(sdr):
    listener = Listener(sdr)
    results  = {}

    print("=" * 60)
    print("  PHASE 1 — DATA channel: Linux (tx) → Windows (rx)")
    print("  Both sides should start Phase 1 at roughly the same time.")
    print("=" * 60)
    time.sleep(2)  # small sync window so both sides are past the header

    if ROLE == 'tx':
        run_as_beaconer(sdr, "Phase1", f"DATA {TX_FREQ/1e6:.0f}MHz")
        results['phase1_role'] = 'beaconer'
        print(f"\n  Phase 1 done. Waiting {PHASE_GAP:.0f}s before Phase 2 ...")
        tx_silence(sdr)
        time.sleep(PHASE_GAP)
    else:
        res = run_as_listener(sdr, listener, "Phase1", f"DATA {RX_FREQ/1e6:.0f}MHz")
        results['phase1'] = res
        print(f"\n  Phase 1 done. Decoded: {res['total_decoded']}  "
              f"Max amp: {res['max_amplitude']:.0f}  Max corr: {res['max_corr']:.1f}")
        print(f"  Waiting {PHASE_GAP:.0f}s before Phase 2 ...")
        time.sleep(PHASE_GAP)

    print()
    print("=" * 60)
    print("  PHASE 2 — CTRL channel: Windows (rx) → Linux (tx)")
    print("  *** THIS IS THE LIKELY FAILURE POINT ***")
    print("  If Linux decodes 0 here, Windows Pluto TX is too weak.")
    print("=" * 60)
    time.sleep(2)

    if ROLE == 'rx':   # Windows: now the beaconer
        run_as_beaconer(sdr, "Phase2", f"CTRL {TX_FREQ/1e6:.0f}MHz")
        results['phase2_role'] = 'beaconer'
        print(f"\n  Phase 2 done. Waiting {PHASE_GAP:.0f}s ...")
        tx_silence(sdr)
        time.sleep(PHASE_GAP)
    else:              # Linux: now the listener
        res = run_as_listener(sdr, listener, "Phase2", f"CTRL {RX_FREQ/1e6:.0f}MHz")
        results['phase2'] = res
        print(f"\n  Phase 2 done. Decoded: {res['total_decoded']}  "
              f"Max amp: {res['max_amplitude']:.0f}  Max corr: {res['max_corr']:.1f}")
        time.sleep(PHASE_GAP)

    listener.stop()

    # ── DIAGNOSIS ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  DIAGNOSIS  ({ROLE.upper()} side — {args.ip})")
    print("=" * 60)

    if ROLE == 'tx':   # Linux listened for Phase 2
        r = results.get('phase2', {})
        dec  = r.get('total_decoded', 0)
        amp  = r.get('max_amplitude', 0)
        corr = r.get('max_corr', 0)
        expected = len(ATTEN_LEVELS) * BEACONS_PER_LEVEL

        print(f"\n  CTRL channel (Windows → Linux):")
        print(f"    Packets decoded : {dec} / ~{expected} expected")
        print(f"    Peak amplitude  : {amp:.0f}  (signal if > 200)")
        print(f"    Peak correlation: {corr:.1f} / {PREAMBLE_LEN} max")

        if dec == 0 and amp < 200:
            verdict = "NO SIGNAL from Windows Pluto at all"
            fixes   = [
                "Check Windows Pluto is connected and pyadi-iio can reach it",
                "Verify --freq-ctrl matches on both PCs",
                "Make sure Windows script hasn't exited early (check its terminal)",
                "Try a shorter cable / antenna / closer together",
            ]
        elif dec == 0 and amp >= 200:
            verdict = "Signal detected but CANNOT decode — CFO or timing problem"
            fixes   = [
                "Verify --sps is the same on both PCs (current: {})".format(SPS),
                "Large CFO ({:.0f} Hz peak). Try --freq-ctrl slightly shifted (+/- 1e4)".format(corr),
                "DDS tone may not be disabled on Windows — check its terminal output",
            ]
        elif dec < expected // 4:
            verdict = "WEAK CTRL channel — Windows Pluto TX too low-powered"
            fixes   = [
                "Add  --tx-atten 0  on the Windows side (max TX power)",
                "Alternatively  --tx-atten -5  and see if that's enough",
                "Increase Linux RX gain: --rx-gain 60 or --rx-gain 71",
                f"Only {dec}/{expected} decoded → signal is marginal",
            ]
        elif dec < expected // 2:
            verdict = "Marginal CTRL channel — some packets lost"
            fixes   = [
                "Add  --tx-atten -5  on the Windows side for a bit more power",
                "This may explain intermittent failures during real transfers",
            ]
        else:
            verdict = "CTRL channel looks OK"
            fixes   = [
                "If transfers still fail, the issue is elsewhere (ARQ timing, generation counter, etc.)",
                "Try running pluto_image_fdd_raw.py with  --tx-atten 0  on Windows anyway",
            ]

        print(f"\n  Verdict: {verdict}")
        print()
        print("  Recommended fixes:")
        for i, f in enumerate(fixes, 1):
            print(f"    {i}. {f}")

    else:   # Windows listened for Phase 1
        r = results.get('phase1', {})
        dec  = r.get('total_decoded', 0)
        amp  = r.get('max_amplitude', 0)
        corr = r.get('max_corr', 0)
        expected = len(ATTEN_LEVELS) * BEACONS_PER_LEVEL

        print(f"\n  DATA channel (Linux → Windows):")
        print(f"    Packets decoded : {dec} / ~{expected} expected")
        print(f"    Peak amplitude  : {amp:.0f}  (signal if > 200)")
        print(f"    Peak correlation: {corr:.1f} / {PREAMBLE_LEN} max")

        if dec == 0 and amp < 200:
            verdict = "NO SIGNAL from Linux Pluto — DATA channel dead"
            fixes   = [
                "Check this PC can reach the Pluto: ping {}".format(args.ip.replace('ip:', '')),
                "Verify --freq-data matches on both PCs",
                "Try --rx-gain 60 or --rx-gain 71",
                "Physical: antenna connected? Correct Pluto selected?",
            ]
        elif dec == 0 and amp >= 200:
            verdict = "Signal present but cannot decode — possible CFO or SPS mismatch"
            fixes   = [
                "Verify --sps={} matches Linux side".format(SPS),
                "DDS tone not disabled? Check terminal output above",
                "Try --rx-gain 50 (current ADC may be clipping at gain {})".format(args.rx_gain),
            ]
        else:
            verdict = f"DATA channel OK ({dec}/{expected} decoded)"
            fixes   = [
                "If the real transfer still fails, Phase 2 CTRL channel is the culprit",
                "Check Linux terminal: did it decode CTRL packets in Phase 2?",
                "If not: re-run with  --tx-atten 0  on this (Windows) side",
            ]

        print(f"\n  Verdict: {verdict}")
        print()
        print("  Recommended fixes:")
        for i, f in enumerate(fixes, 1):
            print(f"    {i}. {f}")

    print()
    print("  TIP: Compare both terminals. If Phase 1 decoded OK on Windows but")
    print("  Phase 2 decoded poorly on Linux → Pluto unit TX asymmetry confirmed.")
    print("  Fix: run Windows side with  --tx-atten 0  (or -5)")
    print()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print()
    print("pluto_link_debug.py")
    print(f"  role={ROLE}  ip={args.ip}")
    print(f"  freq-data={args.freq_data/1e6:.3f}MHz  freq-ctrl={args.freq_ctrl/1e6:.3f}MHz")
    print(f"  sps={SPS}  rx-gain={args.rx_gain}")
    print()

    sdr = setup_pluto()
    flush_rx(sdr, 5)

    try:
        if args.quick:
            quick_sniff(sdr)
        else:
            full_sweep(sdr)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    finally:
        try:
            tx_silence(sdr)
        except Exception:
            pass


if __name__ == '__main__':
    main()
