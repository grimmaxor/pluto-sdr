"""
ADALM Pluto — FDD RAW Image Transfer (uncompressed, unlimited size, BPSK)
========================================================================
A dedicated one-way image link with a reverse control channel, pure Python.

Canonical source: https://github.com/grimmaxor/pluto-sdr

  Pluto 1 (image SENDER) : python pluto_image_fdd_raw.py --role tx --image photo.jpg
  Pluto 2 (image RECEIVER): python pluto_image_fdd_raw.py --role rx --out-dir ./received

FDD CHANNEL ASSIGNMENT  (this is exactly what was requested)
-----------------------------------------------------------
  FREQ_DATA : Pluto 1 -> Pluto 2   image bytes  (META / DATA / end-of-round)
  FREQ_CTRL : Pluto 2 -> Pluto 1   commands     (GO / requests / block-ack / done)

  role tx :  TX_lo = FREQ_DATA ,  RX_lo = FREQ_CTRL
  role rx :  TX_lo = FREQ_CTRL ,  RX_lo = FREQ_DATA   (mirror)

Because each radio's TX frequency is fixed by its role, image data and control
traffic live on different carriers and never collide — both directions are live
at the same time (no turn-taking).

NO COMPRESSION, NO SIZE LIMIT
-----------------------------
The file is sent as raw bytes, byte-for-byte (the received md5 matches the
source md5). Two techniques borrowed from the GNU/ scripts make arbitrary file
sizes practical:
  * BLOCK WINDOWING (GNU zfec "super-block" idea): the file is split into blocks
    of BLOCK_CHUNKS packets. Each block is fully recovered with selective-repeat
    ARQ before the next begins, so the receiver's "missing list" and the
    sender's retransmit set are always bounded — independent of total file size.
    A 4-byte sequence number means the global packet space is effectively
    unlimited (~4 billion packets).
  * DECOUPLED CAPTURE (GNU catcher/math split): a background listener thread
    drains the SDR into a frame queue so packet capture never blocks on
    demodulation/state-machine work.

MODULATION
----------
BPSK for now (1 bit/symbol, most robust). The framing, ARQ, and threading are
modulation-agnostic, so QPSK/16QAM can be dropped in later by generalising
bits<->symbols and the demod (see pluto_filelink_auto_qpsk.py for a phase-
carrying preamble that resolves QPSK ambiguity).

GAINS
-----
This script does NOT run the over-the-air calibration handshake (the sender and
receiver have fixed asymmetric roles). Set RX gain / TX attenuation manually and
tune live with pluto_gain_control.py if the link is weak:
  --rx-gain 40 --tx-atten -20
Develop CONDUCTED where possible: Pluto1 TX --[attenuator]--> Pluto2 RX, etc.
"""

import adi
import numpy as np
import argparse
import os
import sys
import time
import struct
import zlib
import hashlib
import threading
import queue
from scipy.signal import firwin, lfilter

# ─── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--role', choices=['tx', 'rx'], required=True,
                help="tx = image sender (Pluto 1), rx = image receiver (Pluto 2)")
ap.add_argument('--ip',        type=str,   default='ip:pluto.local')
ap.add_argument('--freq-data', type=float, default=2412e6, dest='freq_data',
                help='carrier for image bytes (Pluto 1 -> Pluto 2)')
ap.add_argument('--freq-ctrl', type=float, default=2437e6, dest='freq_ctrl',
                help='carrier for commands (Pluto 2 -> Pluto 1)')
ap.add_argument('--image',     type=str,   default=None, help='tx: image file to send')
ap.add_argument('--out-dir',   type=str,   default='./received', help='rx: save directory')
ap.add_argument('--sps',       type=int,   default=16, help='samples per symbol')
ap.add_argument('--chunk',     type=int,   default=256, help='raw bytes of file per packet')
ap.add_argument('--rx-gain',   type=int,   default=40,
                help='RX gain (dB) — starting value; used directly with --skip-cal')
ap.add_argument('--tx-atten',  type=int,   default=-20,
                help='TX attenuation (dB) — used directly with --skip-cal')
ap.add_argument('--skip-cal',  action='store_true',
                help='skip over-the-air calibration; use --rx-gain/--tx-atten as-is')
args = ap.parse_args()

ROLE = args.role

# FDD frequency assignment — each role's TX carrier is fixed.
if ROLE == 'tx':                       # image sender
    TX_FREQ, RX_FREQ = int(args.freq_data), int(args.freq_ctrl)
else:                                   # image receiver
    TX_FREQ, RX_FREQ = int(args.freq_ctrl), int(args.freq_data)

# ─── LINK CONSTANTS ───────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
SAMPLES_PER_SYMBOL = args.sps
TX_BUFFER_SIZE     = 65536            # cyclic TX DMA buffer (samples)
RX_BUFFER_SIZE     = 262144          # 4x TX so any capture holds several whole packets
CHUNK_BYTES        = args.chunk

# A packet must fit inside ONE TX buffer (with room to pad). For BPSK at the
# given SPS, that caps the wire size. Guard the user's --chunk choice.
_MAX_PKT_BITS = (TX_BUFFER_SIZE // SAMPLES_PER_SYMBOL)        # 1 bit/sym (BPSK)
_MAX_CHUNK    = (_MAX_PKT_BITS // 8) - 16 - 40                # minus header+crc+preamble margin
if CHUNK_BYTES > _MAX_CHUNK:
    print(f"[!] --chunk {CHUNK_BYTES} too large for SPS={SAMPLES_PER_SYMBOL}; "
          f"clamping to {_MAX_CHUNK}")
    CHUNK_BYTES = _MAX_CHUNK

# Block windowing (the GNU "super-block" technique). One block is fully ARQ'd
# before the next starts, bounding memory and the missing-list size.
BLOCK_CHUNKS       = 200             # packets per block (~BLOCK_CHUNKS*CHUNK_BYTES bytes)

# Reverse-channel request batching: keep each REQ frame small so it decodes well
# and fits the TX buffer.
MAX_REQ_SEQS       = 50

# Calibration (symmetric FDD — both radios run it concurrently before transfer).
CAL_TX_ATTEN       = -30            # TX power while sweeping RX gain
GAIN_STEP_BUFS     = 6             # rx() captures averaged per gain step
GAIN_CONFIRM       = 2             # decodes needed to accept a gain step
GAIN_RETRIES       = 8             # whole-sweep retries if partner not heard yet
POWER_ROUNDS       = 6             # status exchanges per TX-power step
POWER_MARGIN       = 5             # extra dB of power (less atten) for stability
READY_ROUNDS       = 40            # link-confirm exchanges
TX_ATTEN_SWEEP     = list(range(-80, 1, 5))   # weak -> strong

# Timing
TX_PACKET_BURST    = 0.30            # seconds each DATA packet stays on air
EOR_REPEAT         = 4               # how many times to send end-of-round
GO_REPEAT          = 4
ACK_REPEAT         = 3
LISTEN_SLEEP       = 0.010           # gap between rx() calls (yields the sdr lock)
REQ_INTERVAL       = 3.0             # rx: resend a request if this long with no progress
WAIT_TIMEOUT       = 12.0            # tx: give up waiting in a block, re-broadcast

# Frame markers
MAGIC = 0xA5

# Packet types — data channel (sender -> receiver)
PKT_META = 0x01     # size, total chunks, chunk bytes, block size, md5, filename
PKT_DATA = 0x02     # seq = global chunk index, payload = raw file bytes
PKT_EOR  = 0x03     # seq = block id, total = total blocks  (end of round)

# Packet types — control channel (receiver -> sender)
PKT_GO   = 0x10     # "META decoded, start sending"
PKT_REQ  = 0x11     # seq = block id, payload = gen(4) + missing global idxs (4 each)
PKT_BACK = 0x12     # seq = block id  (block fully received, advance)
PKT_DONE = 0x13     # whole file received
PKT_ACK  = 0x14     # neutral keepalive (parks the cyclic TX buffer off REQ/EOR)

# Packet types — calibration (both directions, before transfer)
PKT_CAL_TONE = 0x20  # beacon transmitted while the partner sweeps its RX gain
PKT_CAL_STAT = 0x21  # status: payload = rxok(1) + atten(1) + ready(1)  (signed bytes)

# Sync sequence — Barker-13 repeated x3 = strong, real BPSK preamble.
BARKER_13      = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)
PREAMBLE_SIGNS = np.tile(BARKER_13, 3).astype(np.float32)     # 39 symbols
PREAMBLE_LEN   = len(PREAMBLE_SIGNS)


# ═══════════════════════════════════════════════════════════════════════════════
#  DSP  (BPSK)
# ═══════════════════════════════════════════════════════════════════════════════
def make_filter(sps):
    return firwin(sps * 4 + 1, 1.4 / sps, window='hamming').astype(np.float32)

FILT = make_filter(SAMPLES_PER_SYMBOL)


# ─── PACKET FRAMING (binary, CRC32) ───────────────────────────────────────────
def build_packet(pkt_type, seq, total, payload=b''):
    """[MAGIC:1][type:1][seq:4][total:4][len:2][payload:len][crc32:4]"""
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


# ─── MODULATION: bytes -> IQ (fills the TX buffer) ────────────────────────────
def packet_to_iq(pkt_bytes):
    """
    Real-BPSK preamble (Barker x3) + BPSK payload -> pulse-shaped IQ, tiled with
    WHOLE copies only (+ silence pad) so every repeat in the cyclic buffer is
    intact and the RX scan always finds a clean packet.
    """
    pbits   = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    payload = (1.0 - 2.0 * pbits.astype(np.float32))            # BPSK: 0->+1, 1->-1
    syms    = np.concatenate([PREAMBLE_SIGNS, payload]).astype(np.complex64)

    up = np.zeros(len(syms) * SAMPLES_PER_SYMBOL, dtype=np.complex64)
    up[::SAMPLES_PER_SYMBOL] = syms
    shaped = lfilter(FILT, 1.0, up.real).astype(np.float32).astype(np.complex64)

    mx = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15

    plen = len(shaped)
    if plen >= TX_BUFFER_SIZE:
        return shaped[:TX_BUFFER_SIZE].astype(np.complex64)
    n_whole = TX_BUFFER_SIZE // plen
    body    = np.tile(shaped, n_whole)
    pad     = np.zeros(TX_BUFFER_SIZE - len(body), dtype=np.complex64)
    return np.concatenate([body, pad]).astype(np.complex64)


# ─── DEMODULATION: IQ -> packets ──────────────────────────────────────────────
def _cfo_variants(iq):
    """Yield CFO-corrected versions: none, then a coarse FFT estimate."""
    yield iq

    nrm = iq / (np.max(np.abs(iq)) + 1e-9)
    sq  = nrm ** 2                       # BPSK power-law (square)
    n   = len(sq)
    fv  = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/SAMPLE_RATE)
    cfo = freqs[int(np.argmax(np.abs(fv)))] / 2
    if abs(cfo) > 50:
        t = np.arange(n) / SAMPLE_RATE
        yield (iq * np.exp(-1j * 2*np.pi*cfo*t)).astype(np.complex64)


def _packets_from_symbol_stream(syms):
    """
    Correlate the real rail against the BPSK preamble to find timing AND
    polarity, then read the packet that follows. Returns {(type,seq): pkt}.
    """
    found = {}
    if len(syms) < PREAMBLE_LEN + 32:
        return found

    signs = np.sign(syms.real).astype(np.float32)
    corr  = np.correlate(signs, PREAMBLE_SIGNS, mode='valid')
    thr   = PREAMBLE_LEN * 0.8
    cand  = np.where(np.abs(corr) > thr)[0]

    for c in cand:
        inverted   = corr[c] < 0                 # constellation flipped 180°
        data_start = c + PREAMBLE_LEN
        for slip in (0, 1, -1, 2, -2):
            ss = data_start + slip
            if ss < 0 or ss >= len(syms):
                continue
            ds = syms[ss:]
            bits = ((-ds.real if inverted else ds.real) < 0).astype(np.uint8)
            nb = len(bits) // 8
            if nb < 16:
                continue
            raw = bytes(np.packbits(bits[:nb*8]))
            pkt = parse_packet(raw, 0)
            if pkt:
                found[(pkt['type'], pkt['seq'])] = pkt
                break
    return found


def iq_to_packets(iq):
    peak = np.max(np.abs(iq))
    if peak < 5:
        return []
    iq = (iq / peak).astype(np.complex64)
    found = {}
    delay = len(FILT) // 2

    for corrected in _cfo_variants(iq):
        filt = lfilter(FILT, 1.0, corrected.real).astype(np.float32).astype(np.complex64)
        for toff in range(SAMPLES_PER_SYMBOL):
            start = (delay + toff) % SAMPLES_PER_SYMBOL
            stream = filt[start::SAMPLES_PER_SYMBOL]
            if len(stream) < PREAMBLE_LEN + 32:
                continue
            for k, v in _packets_from_symbol_stream(stream).items():
                found[k] = v
        if found:
            break
    return list(found.values())


# ─── META payload codec ───────────────────────────────────────────────────────
def encode_meta(size, total, chunk_bytes, block_chunks, md5_16, name):
    nb = name.encode('utf-8')
    return struct.pack('>QIHH', size, total, chunk_bytes, block_chunks) + md5_16 + nb


def decode_meta(payload):
    if len(payload) < 8 + 4 + 2 + 2 + 16:
        return None
    size, total, chunk_bytes, block_chunks = struct.unpack('>QIHH', payload[:16])
    md5_16 = payload[16:32]
    name   = payload[32:].decode('utf-8', errors='replace')
    return {'size': size, 'total': total, 'chunk_bytes': chunk_bytes,
            'block_chunks': block_chunks, 'md5': md5_16, 'name': name}


def encode_req(gen, seqs):
    return struct.pack('>I', gen) + struct.pack('>%dI' % len(seqs), *seqs)


def decode_req(payload):
    if len(payload) < 4:
        return None, []
    gen  = struct.unpack('>I', payload[:4])[0]
    rest = payload[4:]
    n    = len(rest) // 4
    seqs = list(struct.unpack('>%dI' % n, rest[:n*4])) if n else []
    return gen, seqs


# ─── Calibration status codec ─────────────────────────────────────────────────
def encode_status(rxok, atten, ready):
    """rxok/ready are 0/1; atten is -80..0 — all fit in signed bytes."""
    return struct.pack('>bbb', int(rxok), int(atten), int(ready))


def decode_status(payload):
    if len(payload) < 3:
        return None
    rxok, atten, ready = struct.unpack('>bbb', payload[:3])
    return {'rxok': rxok, 'atten': atten, 'ready': ready}


# ═══════════════════════════════════════════════════════════════════════════════
#  PLUTO
# ═══════════════════════════════════════════════════════════════════════════════
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate             = SAMPLE_RATE
    sdr.tx_lo                   = TX_FREQ
    sdr.rx_lo                   = RX_FREQ
    sdr.tx_rf_bandwidth         = SAMPLE_RATE
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = int(args.rx_gain)
    sdr.tx_hardwaregain_chan0   = int(args.tx_atten)
    sdr.rx_buffer_size          = RX_BUFFER_SIZE
    sdr.tx_cyclic_buffer        = True

    # Disable the onboard DDS tone (otherwise it leaks into TX).
    try:
        import iio
        dds = iio.Context(args.ip).find_device("cf-ad9361-dds-core-lpc")
        if dds:
            for ch in dds.channels:
                if ch.output:
                    for attr in ["raw", "scale"]:
                        try:
                            ch.attrs[attr].value = "0" if attr == "raw" else "0.0"
                        except Exception:
                            pass
    except Exception:
        pass

    print(f"[✓] Connected.  TX {TX_FREQ/1e6:.3f} MHz   RX {RX_FREQ/1e6:.3f} MHz   "
          f"BPSK SPS={SAMPLES_PER_SYMBOL}  chunk={CHUNK_BYTES}B  "
          f"RXg={args.rx_gain} TXa={args.tx_atten}")
    return sdr


def tx_set(sdr, pkt_bytes):
    """Load a packet into the cyclic TX buffer (keeps repeating until replaced)."""
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
    sdr.tx(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))


def start_listener(sdr, sdr_lock, frame_q, stop):
    """Background thread: drain RX into frame_q, yielding the lock between calls."""
    def _run():
        while not stop.is_set():
            with sdr_lock:
                try:
                    raw = sdr.rx()
                except Exception:
                    raw = None
            if raw is not None:
                for pkt in iq_to_packets(raw):
                    try:
                        frame_q.put_nowait(pkt)
                    except queue.Full:
                        pass
            time.sleep(LISTEN_SLEEP)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def print_progress(done, total, label, extra=""):
    bar = int(32 * done / max(total, 1))
    b   = "=" * bar + (">" if bar < 32 else "") + " " * max(31 - bar, 0)
    pct = 100.0 * done / max(total, 1)
    sys.stdout.write(f"\r[{label}] [{b}] {done}/{total} ({pct:.1f}%) {extra}   ")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-CALIBRATION  (symmetric FDD — both radios run this concurrently)
# ═══════════════════════════════════════════════════════════════════════════════
# Unlike the half-duplex scripts, there is no master/slave sequencing here: the
# A->B and B->A links are on different carriers and independent, so each radio
# beacons on its own TX carrier while sweeping its own RX, and negotiates its own
# TX power against what the partner reports hearing. (Same model as
# pluto_image_fdd.py, adapted to this script's binary framing.)

def set_rx_gain(sdr, g):  sdr.rx_hardwaregain_chan0 = int(g)
def set_tx_atten(sdr, a): sdr.tx_hardwaregain_chan0 = int(a)


def rx_gain_limits(sdr):
    """Query the device for the valid RX gain range at the current LO."""
    try:
        ch = sdr._ctrl.find_channel("voltage0", False)
        nums = [float(x) for x in
                ch.attrs["hardwaregain_available"].value.strip("[] ").split()]
        if len(nums) == 3 and nums[2] > nums[0]:
            return nums[0], nums[2]
    except Exception:
        pass
    return 0.0, 71.0


def flush_rx(sdr, n=1):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass


def cal_rx_packets(sdr):
    """One RX capture -> decoded packets (used only during calibration)."""
    try:
        return iq_to_packets(sdr.rx())
    except Exception:
        return []


def find_rx_gain(sdr, label, sweep):
    """Sweep RX gain against the partner's beacon; pick the best non-clipping step."""
    print(f"\n[{label}] Sweeping RX gain ...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []
    for g in sweep:
        try:
            set_rx_gain(sdr, g)
        except OSError:
            continue
        time.sleep(0.1); flush_rx(sdr, 1)
        dec = pk = 0
        for _ in range(GAIN_STEP_BUFS):
            try:
                rx = sdr.rx(); pk = max(pk, np.max(np.abs(rx)))
                if iq_to_packets(rx):
                    dec += 1
            except Exception:
                pass
        adc = pk / 2896 * 100
        print(f"  {g:>5}  {pk:>6.0f}  {adc:>4.0f}%  {dec:>7}")
        if adc > 98:                       # ADC clipping — reject this step
            continue
        if dec >= GAIN_CONFIRM:
            candidates.append((dec - abs(adc - 60) / 100.0, g, adc, dec))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, g, adc, dec = candidates[0]
    print(f"[{label}] ✓ RX gain = {g} dB  (ADC {adc:.0f}%, {dec}/{GAIN_STEP_BUFS} decodes)")
    return g


def calibrate_rx_gain(sdr, label):
    set_tx_atten(sdr, CAL_TX_ATTEN)
    tx_set(sdr, build_packet(PKT_CAL_TONE, 0, 0))      # beacon for the partner
    lo, hi = rx_gain_limits(sdr)
    sweep  = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1, 3))
    print(f"[{label}] Valid RX gain: {lo:.0f}..{hi:.0f} dB")
    for attempt in range(GAIN_RETRIES):
        g = find_rx_gain(sdr, label, sweep)
        if g is not None:
            set_rx_gain(sdr, g); return g
        print(f"[{label}] Partner not heard — retry {attempt+1}/{GAIN_RETRIES}")
    fb = int(np.clip(args.rx_gain, np.ceil(lo), np.floor(hi)))
    print(f"[{label}] Falling back to RX gain {fb} dB")
    set_rx_gain(sdr, fb); return fb


def calibrate_tx_power(sdr, label):
    print(f"\n[{label}] Negotiating TX power (weak -> strong) ...")
    my_rxok = 1; advertised = None; chosen = None

    def advertise(atten):
        nonlocal advertised
        key = (my_rxok, atten)
        if key != advertised:
            tx_set(sdr, build_packet(PKT_CAL_STAT, 0, 0,
                                     encode_status(my_rxok, atten, 0)))
            advertised = key

    for atten in TX_ATTEN_SWEEP:
        set_tx_atten(sdr, atten); advertise(atten); ok = False
        for _ in range(POWER_ROUNDS):
            st = None
            for pkt in cal_rx_packets(sdr):
                if pkt['type'] == PKT_CAL_STAT:
                    st = decode_status(pkt['payload'])
            if st:
                my_rxok = 1; advertise(atten); ok = (st['rxok'] == 1)
            else:
                my_rxok = 0; advertise(atten)
            if ok:
                break
        print(f"  TX atten {atten:>4} dB -> {'partner hears us' if ok else 'no echo'}")
        if ok:
            chosen = atten; break

    chosen = min(0, (chosen if chosen is not None else 0) + POWER_MARGIN)
    set_tx_atten(sdr, chosen)
    print(f"[{label}] ✓ TX atten = {chosen} dB")
    return chosen


def confirm_link(sdr, label, atten):
    print(f"\n[{label}] Confirming link ...")
    tx_set(sdr, build_packet(PKT_CAL_STAT, 0, 0, encode_status(1, atten, 1)))
    for _ in range(READY_ROUNDS):
        for pkt in cal_rx_packets(sdr):
            if pkt['type'] == PKT_CAL_STAT:
                st = decode_status(pkt['payload'])
                if st and st['ready']:
                    print(f"[{label}] ✓ Link confirmed both ways!")
                    return
    print(f"[{label}] Partner-ready not seen — continuing anyway.")


def calibrate(sdr, label):
    print(f"\n{'='*54}\n  ROLE {label} — FDD auto-calibration (runs on both)\n{'='*54}")
    rx_gain  = calibrate_rx_gain(sdr, label)
    tx_atten = calibrate_tx_power(sdr, label)
    confirm_link(sdr, label, tx_atten)
    print(f"\n[{label}] Calibration result: RX gain {rx_gain} dB, TX atten {tx_atten} dB")
    return rx_gain, tx_atten


# ═══════════════════════════════════════════════════════════════════════════════
#  SENDER  (Pluto 1 — image bytes on the data channel)
# ═══════════════════════════════════════════════════════════════════════════════
def block_range(block_id, total):
    lo = block_id * BLOCK_CHUNKS
    hi = min(lo + BLOCK_CHUNKS, total)
    return lo, hi


def sender_main(sdr):
    if not os.path.isfile(args.image):
        print(f"[TX] File not found: {args.image}")
        return

    with open(args.image, 'rb') as f:
        data = f.read()                                    # RAW bytes, no compression

    fname  = os.path.basename(args.image)
    size   = len(data)
    total  = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    nblk   = (total + BLOCK_CHUNKS - 1) // BLOCK_CHUNKS
    md5_16 = hashlib.md5(data).digest()
    chunks = [data[i*CHUNK_BYTES:(i+1)*CHUNK_BYTES] for i in range(total)]

    print(f"\n[TX] '{fname}'  {size:,} bytes  {total} packets  {nblk} blocks  "
          f"md5:{md5_16.hex()[:8]}")

    sdr_lock = threading.Lock()
    frame_q  = queue.Queue(maxsize=2000)
    stop     = threading.Event()
    start_listener(sdr, sdr_lock, frame_q, stop)

    meta_pkt = build_packet(PKT_META, 0, total,
                            encode_meta(size, total, CHUNK_BYTES, BLOCK_CHUNKS, md5_16, fname))

    def drain_ctrl():
        """Return list of control frames waiting in the queue."""
        out = []
        while True:
            try:
                out.append(frame_q.get_nowait())
            except queue.Empty:
                break
        return out

    # ── ANNOUNCE: broadcast META until the receiver says GO ──
    print("[TX] ANNOUNCE — broadcasting META, waiting for GO ...")
    got_go = False
    t_end  = time.time() + 120
    while not got_go and time.time() < t_end:
        with sdr_lock:
            tx_set(sdr, meta_pkt)
        time.sleep(0.5)
        for f in drain_ctrl():
            if f['type'] == PKT_GO:
                got_go = True
            elif f['type'] == PKT_DONE:        # receiver already had it somehow
                got_go = True
    if not got_go:
        print("[TX] No GO from receiver — aborting.")
        stop.set(); return
    print("[TX] GO received. Sending blocks ...")

    # ── Per-block selective-repeat ARQ ──
    for blk in range(nblk):
        lo, hi = block_range(blk, total)
        to_send = list(range(lo, hi))          # round 0: every chunk in the block
        last_gen = -1
        t_block  = time.time()

        while True:
            # Transmit the current send-set.
            for seq in to_send:
                with sdr_lock:
                    tx_set(sdr, build_packet(PKT_DATA, seq, total, chunks[seq]))
                print_progress(blk + 1, nblk, "TX",
                               f"block pkt {seq-lo+1}/{hi-lo}")
                t0 = time.time()
                while time.time() - t0 < TX_PACKET_BURST:
                    for f in drain_ctrl():
                        if f['type'] == PKT_DONE:
                            print("\n[TX] DONE received — transfer complete!")
                            stop.set(); return
                    time.sleep(0.01)

            # End-of-round marker so the receiver knows to evaluate the block.
            for _ in range(EOR_REPEAT):
                with sdr_lock:
                    tx_set(sdr, build_packet(PKT_EOR, blk, nblk))
                time.sleep(0.12)
            with sdr_lock:
                tx_set(sdr, build_packet(PKT_ACK, 0, 0))   # park off EOR in cyclic buffer

            # Wait for this block's REQ (retransmit) or BACK (advance).
            to_send = []
            advanced = False
            t_wait = time.time()
            while time.time() - t_wait < WAIT_TIMEOUT:
                try:
                    f = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if f['type'] == PKT_DONE:
                    print("\n[TX] DONE received — transfer complete!")
                    stop.set(); return
                if f['type'] == PKT_BACK and f['seq'] == blk:
                    advanced = True
                    break
                if f['type'] == PKT_REQ and f['seq'] == blk:
                    gen, seqs = decode_req(f['payload'])
                    if gen > last_gen:
                        last_gen = gen
                        to_send  = [s for s in seqs if lo <= s < hi]
                    elif gen == last_gen:
                        to_send += [s for s in seqs if lo <= s < hi and s not in to_send]
                    # keep collecting same-gen REQ frames briefly, then resend
                    if to_send:
                        # short grace window to gather multi-frame requests
                        t_g = time.time()
                        while time.time() - t_g < 0.6:
                            try:
                                g2 = frame_q.get(timeout=0.2)
                            except queue.Empty:
                                break
                            if g2['type'] == PKT_REQ and g2['seq'] == blk:
                                gg, ss = decode_req(g2['payload'])
                                if gg >= last_gen:
                                    last_gen = gg
                                    to_send += [s for s in ss if lo <= s < hi and s not in to_send]
                            elif g2['type'] == PKT_BACK and g2['seq'] == blk:
                                advanced = True; break
                            elif g2['type'] == PKT_DONE:
                                print("\n[TX] DONE received — transfer complete!")
                                stop.set(); return
                        break

            if advanced:
                break                          # next block
            if not to_send:
                # No request and no ack heard — re-announce this block's EOR by
                # looping again with an empty send-set (just re-sends EOR), unless
                # we've been stuck too long.
                if time.time() - t_block > 4 * WAIT_TIMEOUT:
                    print(f"\n[TX] Block {blk} stalled — re-broadcasting META.")
                    with sdr_lock:
                        tx_set(sdr, meta_pkt)
                    time.sleep(1.0)
                    t_block = time.time()
                continue

    print("\n[TX] All blocks sent. Waiting for final DONE ...")
    t_end = time.time() + WAIT_TIMEOUT
    while time.time() < t_end:
        try:
            f = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if f['type'] == PKT_DONE:
            print("[TX] ✓ DONE — transfer complete!")
            break
    stop.set()
    tx_silence(sdr)


# ═══════════════════════════════════════════════════════════════════════════════
#  RECEIVER  (Pluto 2 — commands on the control channel)
# ═══════════════════════════════════════════════════════════════════════════════
def receiver_main(sdr):
    os.makedirs(args.out_dir, exist_ok=True)

    sdr_lock = threading.Lock()
    frame_q  = queue.Queue(maxsize=2000)
    stop     = threading.Event()
    start_listener(sdr, sdr_lock, frame_q, stop)

    meta = None
    print("[RX] Waiting for META on the data channel ...")
    while meta is None and not stop.is_set():
        try:
            f = frame_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if f['type'] == PKT_META:
            meta = decode_meta(f['payload'])

    if meta is None:
        print("[RX] No META — aborting."); stop.set(); return

    size  = meta['size']
    total = meta['total']
    nblk  = (total + meta['block_chunks'] - 1) // meta['block_chunks']
    fname = meta['name']
    global BLOCK_CHUNKS
    BLOCK_CHUNKS = meta['block_chunks']        # honour the sender's block size
    print(f"\n[RX] Incoming '{fname}'  {size:,} bytes  {total} packets  {nblk} blocks  "
          f"md5:{meta['md5'].hex()[:8]}")

    # Tell the sender to start.
    for _ in range(GO_REPEAT):
        with sdr_lock:
            tx_set(sdr, build_packet(PKT_GO, 0, total))
        time.sleep(0.15)
    with sdr_lock:
        tx_set(sdr, build_packet(PKT_ACK, 0, 0))

    buf = {}                                   # global seq -> bytes
    gen = 0

    def send_req(blk, missing):
        """Send the missing list for a block, split into small frames."""
        nonlocal gen
        gen += 1
        for i in range(0, len(missing), MAX_REQ_SEQS):
            batch = missing[i:i+MAX_REQ_SEQS]
            with sdr_lock:
                tx_set(sdr, build_packet(PKT_REQ, blk, total, encode_req(gen, batch)))
            time.sleep(0.15)
        with sdr_lock:
            tx_set(sdr, build_packet(PKT_ACK, 0, 0))     # park off REQ in cyclic buffer

    for blk in range(nblk):
        lo, hi = block_range(blk, total)
        last_req = 0.0
        while not stop.is_set():
            try:
                f = frame_q.get(timeout=0.25)
            except queue.Empty:
                f = None

            if f is not None:
                if f['type'] == PKT_DATA and lo <= f['seq'] < hi:
                    if f['seq'] not in buf:
                        buf[f['seq']] = f['payload']
                    print_progress(len(buf), total, "RX",
                                   f"block {blk+1}/{nblk}")
                elif f['type'] == PKT_DATA and f['seq'] not in buf:
                    buf[f['seq']] = f['payload']           # accept stray useful chunks
                elif f['type'] == PKT_EOR and f['seq'] == blk:
                    missing = [s for s in range(lo, hi) if s not in buf]
                    if not missing:
                        for _ in range(ACK_REPEAT):
                            with sdr_lock:
                                tx_set(sdr, build_packet(PKT_BACK, blk, nblk))
                            time.sleep(0.15)
                        with sdr_lock:
                            tx_set(sdr, build_packet(PKT_ACK, 0, 0))
                        break                              # block done -> next block
                    else:
                        send_req(blk, missing)
                        last_req = time.time()
                elif f['type'] == PKT_META and decode_meta(f['payload']):
                    # sender fell back to ANNOUNCE — nudge it forward again
                    for _ in range(2):
                        with sdr_lock:
                            tx_set(sdr, build_packet(PKT_GO, 0, total))
                        time.sleep(0.12)

            # Periodic request if the block is incomplete and quiet for a while.
            if time.time() - last_req >= REQ_INTERVAL:
                missing = [s for s in range(lo, hi) if s not in buf]
                if missing:
                    send_req(blk, missing)
                    last_req = time.time()
                elif len(buf) >= hi - lo and all(s in buf for s in range(lo, hi)):
                    for _ in range(ACK_REPEAT):
                        with sdr_lock:
                            tx_set(sdr, build_packet(PKT_BACK, blk, nblk))
                        time.sleep(0.15)
                    break

    # ── Whole file received: reassemble, verify, save ──
    sys.stdout.write("\n")
    data = b"".join(buf.get(i, b'\x00' * CHUNK_BYTES) for i in range(total))[:size]
    got_md5 = hashlib.md5(data).digest()
    ok = (got_md5 == meta['md5'])

    out_path = os.path.join(args.out_dir, f"rx_{fname}")
    with open(out_path, 'wb') as fh:
        fh.write(data)

    print(f"[RX] Saved {out_path}  ({len(data):,} bytes)")
    print(f"[RX] md5 {'✓ MATCH' if ok else '✗ MISMATCH'}  "
          f"got:{got_md5.hex()[:8]} want:{meta['md5'].hex()[:8]}")

    # Tell the sender we are finished (whether or not md5 matched — we have all seqs).
    for _ in range(8):
        with sdr_lock:
            tx_set(sdr, build_packet(PKT_DONE, 0, total))
        time.sleep(0.2)
    stop.set()
    tx_silence(sdr)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if ROLE == 'tx' and not args.image:
        print("[!] --role tx requires --image <file>")
        return
    sdr = setup_pluto()
    print(f"\n*** ROLE: {ROLE.upper()}  "
          f"({'image SENDER' if ROLE=='tx' else 'image RECEIVER'}) ***")
    print(f"    DATA carrier {int(args.freq_data)/1e6:.3f} MHz  (Pluto1 -> Pluto2)")
    print(f"    CTRL carrier {int(args.freq_ctrl)/1e6:.3f} MHz  (Pluto2 -> Pluto1)")

    # ── Auto-calibration (unless skipped) ──
    if args.skip_cal:
        print(f"[*] Skipping calibration — using RX gain {args.rx_gain} dB, "
              f"TX atten {args.tx_atten} dB.")
        set_rx_gain(sdr, args.rx_gain); set_tx_atten(sdr, args.tx_atten)
    else:
        print("\n*** Start BOTH radios — calibration runs on both at once. "
              "Beginning in 3s ***")
        time.sleep(3)
        calibrate(sdr, ROLE.upper())
        flush_rx(sdr, 3)               # clear stale calibration frames

    time.sleep(1.0)

    try:
        if ROLE == 'tx':
            sender_main(sdr)
        else:
            receiver_main(sdr)
    except KeyboardInterrupt:
        print("\n[*] Interrupted.")
    finally:
        try:
            tx_silence(sdr)
        except Exception:
            pass


if __name__ == "__main__":
    main()
