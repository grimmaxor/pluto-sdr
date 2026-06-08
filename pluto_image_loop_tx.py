"""
ADALM Pluto — Continuous Image TX  (loops until Ctrl+C)
=========================================================
Run on the SENDER (Windows PC). Loads the image once, then sends it
repeatedly over FDD until Ctrl+C.

  First run (auto-calibration):
    python pluto_image_loop_tx.py --image photo.jpg

  Subsequent runs (use the printed values from the first run):
    python pluto_image_loop_tx.py --image photo.jpg --skip-cal --rx-gain 40 --tx-atten -20

FDD channel assignment
  FREQ_DATA  TX (this machine) -> RX   image bytes
  FREQ_CTRL  RX -> TX (this machine)   ACKs / NACKs / DONE

Pair with pluto_image_loop_rx.py on the receiver.
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
ap.add_argument('--ip',        type=str,   default='ip:pluto.local')
ap.add_argument('--freq-data', type=float, default=2412e6, dest='freq_data',
                help='carrier for image bytes (TX -> RX)')
ap.add_argument('--freq-ctrl', type=float, default=2437e6, dest='freq_ctrl',
                help='carrier for control frames (RX -> TX)')
ap.add_argument('--image',     type=str,   required=True, help='image file to send')
ap.add_argument('--sps',       type=int,   default=16,    help='samples per symbol')
ap.add_argument('--chunk',     type=int,   default=256,   help='raw bytes per packet')
ap.add_argument('--rx-gain',   type=int,   default=40,    help='RX gain dB')
ap.add_argument('--tx-atten',  type=int,   default=-20,   help='TX attenuation dB')
ap.add_argument('--skip-cal',  action='store_true',       help='skip auto-calibration')
args = ap.parse_args()

TX_FREQ = int(args.freq_data)
RX_FREQ = int(args.freq_ctrl)

# ─── LINK CONSTANTS ───────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
SAMPLES_PER_SYMBOL = args.sps
TX_BUFFER_SIZE     = 65536
RX_BUFFER_SIZE     = 262144
CHUNK_BYTES        = args.chunk

_MAX_PKT_BITS = TX_BUFFER_SIZE // SAMPLES_PER_SYMBOL
_MAX_CHUNK    = (_MAX_PKT_BITS // 8) - 16 - 40
if CHUNK_BYTES > _MAX_CHUNK:
    print(f"[!] --chunk {CHUNK_BYTES} too large for SPS={SAMPLES_PER_SYMBOL}; "
          f"clamping to {_MAX_CHUNK}")
    CHUNK_BYTES = _MAX_CHUNK

BLOCK_CHUNKS    = 200
MAX_REQ_SEQS    = 50

CAL_TX_ATTEN    = -30
GAIN_STEP_BUFS  = 6
GAIN_CONFIRM    = 2
GAIN_RETRIES    = 8
POWER_ROUNDS    = 6
POWER_MARGIN    = 5
READY_ROUNDS    = 40
TX_ATTEN_SWEEP  = list(range(-80, 1, 5))

TX_PACKET_BURST = 0.30
EOR_REPEAT      = 4
GO_REPEAT       = 4
ACK_REPEAT      = 3
LISTEN_SLEEP    = 0.010
WAIT_TIMEOUT    = 12.0

MAGIC        = 0xA5
PKT_META     = 0x01
PKT_DATA     = 0x02
PKT_EOR      = 0x03
PKT_GO       = 0x10
PKT_REQ      = 0x11
PKT_BACK     = 0x12
PKT_DONE     = 0x13
PKT_ACK      = 0x14
PKT_CAL_TONE = 0x20
PKT_CAL_STAT = 0x21

BARKER_13      = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)
PREAMBLE_SIGNS = np.tile(BARKER_13, 3).astype(np.float32)
PREAMBLE_LEN   = len(PREAMBLE_SIGNS)


# ═══════════════════════════════════════════════════════════════════════════════
#  DSP  (BPSK)
# ═══════════════════════════════════════════════════════════════════════════════
def make_filter(sps):
    return firwin(sps * 4 + 1, 1.4 / sps, window='hamming').astype(np.float32)

FILT = make_filter(SAMPLES_PER_SYMBOL)


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
    crc_rx   = struct.unpack('>I', raw[start+HDR+plen:end])[0]
    crc_calc = zlib.crc32(raw[start:start+HDR+plen]) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return None
    return {'type': ptype, 'seq': seq, 'total': total, 'payload': payload}


def packet_to_iq(pkt_bytes):
    pbits   = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    payload = (1.0 - 2.0 * pbits.astype(np.float32))
    syms    = np.concatenate([PREAMBLE_SIGNS, payload]).astype(np.complex64)
    up      = np.zeros(len(syms) * SAMPLES_PER_SYMBOL, dtype=np.complex64)
    up[::SAMPLES_PER_SYMBOL] = syms
    shaped  = lfilter(FILT, 1.0, up.real).astype(np.float32).astype(np.complex64)
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


def _cfo_variants(iq):
    yield iq
    nrm   = iq / (np.max(np.abs(iq)) + 1e-9)
    sq    = nrm ** 2
    n     = len(sq)
    fv    = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/SAMPLE_RATE)
    cfo   = freqs[int(np.argmax(np.abs(fv)))] / 2
    if abs(cfo) > 50:
        t = np.arange(n) / SAMPLE_RATE
        yield (iq * np.exp(-1j * 2*np.pi*cfo*t)).astype(np.complex64)


def _packets_from_symbol_stream(syms):
    found = {}
    if len(syms) < PREAMBLE_LEN + 32:
        return found
    signs = np.sign(syms.real).astype(np.float32)
    corr  = np.correlate(signs, PREAMBLE_SIGNS, mode='valid')
    thr   = PREAMBLE_LEN * 0.8
    cand  = np.where(np.abs(corr) > thr)[0]
    for c in cand:
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
    iq    = (iq / peak).astype(np.complex64)
    found = {}
    delay = len(FILT) // 2
    for corrected in _cfo_variants(iq):
        filt = lfilter(FILT, 1.0, corrected.real).astype(np.float32).astype(np.complex64)
        for toff in range(SAMPLES_PER_SYMBOL):
            start  = (delay + toff) % SAMPLES_PER_SYMBOL
            stream = filt[start::SAMPLES_PER_SYMBOL]
            if len(stream) < PREAMBLE_LEN + 32:
                continue
            for k, v in _packets_from_symbol_stream(stream).items():
                found[k] = v
        if found:
            break
    return list(found.values())


def encode_meta(size, total, chunk_bytes, block_chunks, md5_16, name):
    nb = name.encode('utf-8')
    return struct.pack('>QIHH', size, total, chunk_bytes, block_chunks) + md5_16 + nb


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


def encode_status(rxok, atten, ready):
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
    print(f"[✓] Connected.  TX {TX_FREQ/1e6:.3f} MHz  RX {RX_FREQ/1e6:.3f} MHz  "
          f"BPSK SPS={SAMPLES_PER_SYMBOL}  chunk={CHUNK_BYTES}B  "
          f"RXg={args.rx_gain} TXa={args.tx_atten}")
    return sdr


def tx_set(sdr, pkt_bytes):
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


def print_progress(done, total, extra=""):
    bar = int(32 * done / max(total, 1))
    b   = "=" * bar + (">" if bar < 32 else "") + " " * max(31 - bar, 0)
    pct = 100.0 * done / max(total, 1)
    sys.stdout.write(f"\r[TX] [{b}] {done}/{total} ({pct:.1f}%) {extra}   ")
    sys.stdout.flush()


def set_rx_gain(sdr, g): sdr.rx_hardwaregain_chan0 = int(g)
def set_tx_atten(sdr, a): sdr.tx_hardwaregain_chan0 = int(a)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-CALIBRATION  (symmetric FDD — run on both radios simultaneously)
# ═══════════════════════════════════════════════════════════════════════════════
def rx_gain_limits(sdr):
    try:
        ch   = sdr._ctrl.find_channel("voltage0", False)
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
    try:
        return iq_to_packets(sdr.rx())
    except Exception:
        return []


def find_rx_gain(sdr, sweep):
    print(f"\n[TX-CAL] Sweeping RX gain ...")
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
        if adc > 98:
            continue
        if dec >= GAIN_CONFIRM:
            candidates.append((dec - abs(adc - 60) / 100.0, g, adc, dec))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, g, adc, dec = candidates[0]
    print(f"[TX-CAL] ✓ RX gain = {g} dB  (ADC {adc:.0f}%, {dec}/{GAIN_STEP_BUFS} decodes)")
    return g


def calibrate_rx_gain(sdr):
    set_tx_atten(sdr, CAL_TX_ATTEN)
    tx_set(sdr, build_packet(PKT_CAL_TONE, 0, 0))
    lo, hi = rx_gain_limits(sdr)
    sweep  = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1, 3))
    print(f"[TX-CAL] Valid RX gain: {lo:.0f}..{hi:.0f} dB")
    for attempt in range(GAIN_RETRIES):
        g = find_rx_gain(sdr, sweep)
        if g is not None:
            set_rx_gain(sdr, g); return g
        print(f"[TX-CAL] Partner not heard — retry {attempt+1}/{GAIN_RETRIES}")
    fb = int(np.clip(args.rx_gain, np.ceil(lo), np.floor(hi)))
    print(f"[TX-CAL] Falling back to RX gain {fb} dB")
    set_rx_gain(sdr, fb); return fb


def calibrate_tx_power(sdr):
    print(f"\n[TX-CAL] Negotiating TX power (weak -> strong) ...")
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
    print(f"[TX-CAL] ✓ TX atten = {chosen} dB")
    return chosen


def confirm_link(sdr, atten):
    print(f"\n[TX-CAL] Confirming link ...")
    tx_set(sdr, build_packet(PKT_CAL_STAT, 0, 0, encode_status(1, atten, 1)))
    for _ in range(READY_ROUNDS):
        for pkt in cal_rx_packets(sdr):
            if pkt['type'] == PKT_CAL_STAT:
                st = decode_status(pkt['payload'])
                if st and st['ready']:
                    print("[TX-CAL] ✓ Link confirmed both ways!")
                    return
    print("[TX-CAL] Partner-ready not seen — continuing anyway.")


def calibrate(sdr):
    print(f"\n{'='*54}\n  TX — FDD auto-calibration (run on both radios at once)\n{'='*54}")
    rx_gain  = calibrate_rx_gain(sdr)
    tx_atten = calibrate_tx_power(sdr)
    confirm_link(sdr, tx_atten)
    print(f"\n{'='*60}")
    print(f"  CALIBRATION COMPLETE — save these for --skip-cal:")
    print(f"  TX:  --skip-cal --rx-gain {rx_gain} --tx-atten {tx_atten}")
    print(f"{'='*60}\n")
    return rx_gain, tx_atten


# ═══════════════════════════════════════════════════════════════════════════════
#  SENDER LOOP
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
        data = f.read()

    fname    = os.path.basename(args.image)
    size     = len(data)
    total    = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    nblk     = (total + BLOCK_CHUNKS - 1) // BLOCK_CHUNKS
    md5_16   = hashlib.md5(data).digest()
    chunks   = [data[i*CHUNK_BYTES:(i+1)*CHUNK_BYTES] for i in range(total)]
    meta_pkt = build_packet(PKT_META, 0, total,
                            encode_meta(size, total, CHUNK_BYTES, BLOCK_CHUNKS, md5_16, fname))

    print(f"[TX] '{fname}'  {size:,} bytes  {total} pkts  {nblk} blocks  "
          f"md5:{md5_16.hex()[:8]}")
    print(f"[TX] Sending continuously — Ctrl+C to stop.")

    sdr_lock = threading.Lock()
    frame_q  = queue.Queue(maxsize=2000)
    stop     = threading.Event()
    start_listener(sdr, sdr_lock, frame_q, stop)

    def drain():
        out = []
        while True:
            try:
                out.append(frame_q.get_nowait())
            except queue.Empty:
                break
        return out

    run = 0
    try:
        while True:
            run += 1
            t_start = time.time()
            print(f"\n[TX] === Run #{run} ===")
            drain()   # discard any stale frames from last run

            # ── ANNOUNCE: broadcast META until GO ──
            print("[TX] ANNOUNCE — broadcasting META, waiting for GO ...")
            got_go = False
            t_ann  = time.time() + 120
            while not got_go and time.time() < t_ann:
                with sdr_lock:
                    tx_set(sdr, meta_pkt)
                time.sleep(0.5)
                for f in drain():
                    if f['type'] in (PKT_GO, PKT_DONE):
                        got_go = True
            if not got_go:
                print("[TX] No GO — RX not responding. Retrying next run.")
                continue

            print("[TX] GO received — sending blocks ...")
            done = False

            for blk in range(nblk):
                if done:
                    break
                lo, hi   = block_range(blk, total)
                to_send  = list(range(lo, hi))
                last_gen = -1
                t_block  = time.time()

                while not done:
                    for seq in to_send:
                        with sdr_lock:
                            tx_set(sdr, build_packet(PKT_DATA, seq, total, chunks[seq]))
                        print_progress(blk + 1, nblk,
                                       f"blk {blk+1}/{nblk} pkt {seq-lo+1}/{hi-lo}")
                        t0 = time.time()
                        while time.time() - t0 < TX_PACKET_BURST:
                            for f in drain():
                                if f['type'] == PKT_DONE:
                                    done = True
                            time.sleep(0.01)
                        if done:
                            break
                    if done:
                        break

                    for _ in range(EOR_REPEAT):
                        with sdr_lock:
                            tx_set(sdr, build_packet(PKT_EOR, blk, nblk))
                        time.sleep(0.12)
                    with sdr_lock:
                        tx_set(sdr, build_packet(PKT_ACK, 0, 0))

                    to_send  = []
                    advanced = False
                    t_wait   = time.time()
                    while time.time() - t_wait < WAIT_TIMEOUT:
                        try:
                            f = frame_q.get(timeout=0.5)
                        except queue.Empty:
                            continue
                        if f['type'] == PKT_DONE:
                            done = True; break
                        if f['type'] == PKT_BACK and f['seq'] == blk:
                            advanced = True; break
                        if f['type'] == PKT_REQ and f['seq'] == blk:
                            gen, seqs = decode_req(f['payload'])
                            if gen > last_gen:
                                last_gen = gen
                                to_send  = [s for s in seqs if lo <= s < hi]
                            elif gen == last_gen:
                                to_send += [s for s in seqs
                                            if lo <= s < hi and s not in to_send]
                            if to_send:
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
                                            to_send += [s for s in ss
                                                        if lo <= s < hi and s not in to_send]
                                    elif g2['type'] == PKT_BACK and g2['seq'] == blk:
                                        advanced = True; break
                                    elif g2['type'] == PKT_DONE:
                                        done = True; break
                                break

                    if done or advanced:
                        break
                    if not to_send:
                        if time.time() - t_block > 4 * WAIT_TIMEOUT:
                            print(f"\n[TX] Block {blk} stalled — re-broadcasting META.")
                            with sdr_lock:
                                tx_set(sdr, meta_pkt)
                            time.sleep(1.0)
                            t_block = time.time()

            if not done:
                print("\n[TX] All blocks sent — waiting for final DONE ...")
                t_end = time.time() + WAIT_TIMEOUT
                while time.time() < t_end:
                    try:
                        f = frame_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if f['type'] == PKT_DONE:
                        done = True; break

            elapsed = time.time() - t_start
            rate    = size / elapsed / 1024 if elapsed > 0 else 0
            sys.stdout.write("\n")
            if done:
                print(f"[TX] Run #{run} DONE  {elapsed:.1f}s  {rate:.1f} KB/s")
            else:
                print(f"[TX] Run #{run} TIMEOUT after {elapsed:.1f}s — restarting")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print(f"\n[TX] Stopped after {run} run(s).")
    finally:
        stop.set()
        try:
            tx_silence(sdr)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    sdr = setup_pluto()
    print(f"\n*** TX — image SENDER (loops until Ctrl+C) ***")
    print(f"    DATA carrier {TX_FREQ/1e6:.3f} MHz  (TX -> RX)")
    print(f"    CTRL carrier {RX_FREQ/1e6:.3f} MHz  (RX -> TX)")

    if args.skip_cal:
        print(f"[*] Skipping calibration — RX gain {args.rx_gain} dB, "
              f"TX atten {args.tx_atten} dB.")
        set_rx_gain(sdr, args.rx_gain)
        set_tx_atten(sdr, args.tx_atten)
    else:
        print("\n*** Start BOTH radios — calibration runs on both at once. "
              "Beginning in 3s ***")
        time.sleep(3)
        calibrate(sdr)
        flush_rx(sdr, 3)

    time.sleep(1.0)
    sender_main(sdr)


if __name__ == "__main__":
    main()
