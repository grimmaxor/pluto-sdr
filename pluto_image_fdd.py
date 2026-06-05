"""
ADALM Pluto — FDD Image Transfer with ARQ
==========================================
Sender  : python pluto_image_fdd.py --role a --image photo.jpg
Receiver: python pluto_image_fdd.py --role b [--save-dir PATH]

Either role can be the sender.  The other role receives automatically.

FDD channel assignment
----------------------
  role a : TX on FREQ_A  RX on FREQ_B
  role b : TX on FREQ_B  RX on FREQ_A   (mirror)

Because TX and RX live on different frequencies both directions are live
simultaneously — no turn-taking, no half-duplex timing assumptions.

Transfer protocol
-----------------
1. Sender broadcasts META (filename / total chunks / byte count).
2. Receiver replies CTRL|GO after decoding META.  Sender starts sending.
3. Sender streams DATA packets; receiver collects them.
4. Receiver sends a NAK (with a generation counter) listing missing seqs
   every NAK_INTERVAL seconds, then switches TX to CTRL|ACK so the sender
   does not keep re-decoding a stale NAK from the cyclic buffer.
5. Sender re-queues NAK'd seqs (using a set to prevent double-entry),
   retransmits, then sends CTRL|PAUSE briefly and waits for NAK or DONE.
6. Receiver replies DONE once all chunks are in hand; image is saved.

Key tuning notes
----------------
CHUNK_BYTES = 48  →  DATA frame = 78 chars.  At ~0.2% BER a 174-char frame
  had only 6% decode probability; 78 chars raises it to ~27% per attempt.
TX_PACKET_BURST = 0.35 s  →  ~4-5 rx() captures per packet.  Combined with
  the improved frame length, first-pass delivery is roughly 75%.
MIN_NAK_INTERVAL = 1.5 s  →  caps NAK send rate even when CTRL|PAUSE floods
  the receiver from the sender's cyclic TX buffer.
"""

import adi
import numpy as np
import argparse, threading, queue, collections
import time, sys, os, base64, hashlib
from scipy.signal import firwin, lfilter

try:
    from PIL import Image as _Pil
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--role',     choices=['a','b'], required=True)
ap.add_argument('--ip',       default='ip:pluto.local')
ap.add_argument('--freq-a',   type=float, default=2412e6, dest='freq_a')
ap.add_argument('--freq-b',   type=float, default=2437e6, dest='freq_b')
ap.add_argument('--image',    default=None,  help="Image to send (omit to receive)")
ap.add_argument('--save-dir', default='.',   dest='save_dir')
args = ap.parse_args()

IS_SENDER          = args.image is not None
TX_FREQ, RX_FREQ   = (int(args.freq_a), int(args.freq_b)) if args.role == 'a' \
                     else (int(args.freq_b), int(args.freq_a))

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
TX_BUFFER_SIZE     = 65536
SAMPLES_PER_SYMBOL = 16
MAX_MSG_LEN        = 180
FRAME_MAGIC        = 0xBE
BARKER_13          = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)

# Calibration
CAL_TX_ATTEN       = -30
GAIN_STEP_BUFS     = 6
GAIN_CONFIRM       = 2
GAIN_RETRIES       = 8
POWER_ROUNDS       = 6
POWER_MARGIN       = 5
READY_ROUNDS       = 40
TX_ATTEN_SWEEP     = list(range(-80, 1, 5))
CAL_TONE           = "__CAL__"

# Transfer
CHUNK_BYTES        = 48      # 48 raw → 64 b64 → 78-char frame  (BER sweet spot)
TX_PACKET_BURST    = 0.35    # seconds each packet stays on air (~4-5 rx() captures)
TX_LISTENER_SLEEP  = 0.010   # gap between rx() calls — yields the sdr lock
NAK_INTERVAL       = 3.0     # receiver sends NAK every N seconds
MIN_NAK_INTERVAL   = 1.5     # floor between consecutive NAK sends (PAUSE flood guard)
PAUSE_TIMEOUT      = 15.0    # sender waits in PAUSE before returning to ANNOUNCE
MAX_IMAGE_BYTES    = 60_000  # auto-resize above this
REPORT_INTERVAL    = 10.0    # periodic stats print

# ─── DSP ──────────────────────────────────────────────────────────────────────
def crc8(data: bytes) -> int:
    crc = 0xFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07 if crc & 0x80 else crc << 1) & 0xFF
    return crc

FILT = firwin(SAMPLES_PER_SYMBOL*4+1, 1.4/SAMPLES_PER_SYMBOL, window='hamming').astype(np.float32)

def encode_fill(msg: str) -> np.ndarray:
    msg   = msg[:MAX_MSG_LEN]
    raw   = msg.encode()
    pay   = bytes([FRAME_MAGIC, len(raw)]) + raw
    full  = pay + bytes([crc8(pay)])
    bbits = ((1 - BARKER_13)/2).astype(np.uint8)
    pbits = np.unpackbits(np.frombuffer(full, np.uint8))
    bits  = np.concatenate([bbits, pbits]).astype(np.float32)
    syms  = 1.0 - 2.0*bits
    up    = np.zeros(len(syms)*SAMPLES_PER_SYMBOL, np.float32)
    up[::SAMPLES_PER_SYMBOL] = syms
    shaped = lfilter(FILT, 1.0, up).astype(np.float32)
    mx = np.max(np.abs(shaped))
    if mx: shaped = shaped / mx * 0.8 * 2**15
    iq = shaped.astype(np.complex64)
    return np.tile(iq, int(np.ceil(TX_BUFFER_SIZE/len(iq)))+1)[:TX_BUFFER_SIZE]

def _try_extract(sig):
    for pol in [1,-1]:
        s = sig*pol
        for t in range(SAMPLES_PER_SYMBOL):
            bits = (s[t::SAMPLES_PER_SYMBOL] < 0).astype(np.uint8)
            nb = len(bits)//8
            if nb < 8: continue
            raw = bytes(np.packbits(bits[:nb*8]))
            for p in range(len(raw)-4):
                if raw[p] != FRAME_MAGIC: continue
                ml = raw[p+1]
                if ml == 0 or ml > MAX_MSG_LEN: continue
                ei = p+2+ml
                if ei+1 >= len(raw): continue
                if crc8(raw[p:ei]) != raw[ei]: continue
                try: return raw[p+2:ei].decode()
                except: continue
    return None

def decode_any(iq):
    pk = np.max(np.abs(iq))
    if pk < 5: return None
    iq = (iq/pk*2**13).astype(np.complex64)
    # strategy 1: no correction
    m = _try_extract(lfilter(FILT,1.0,iq.real).astype(np.float32))
    if m: return m
    # strategy 2: FFT coarse CFO
    sq = iq**2; n = len(sq); fv = np.fft.fft(sq); fv[0] = 0
    fr = np.fft.fftfreq(n, 1.0/SAMPLE_RATE)
    pk2 = int(np.argmax(np.abs(fv)))
    cfo = fr[pk2]/2; ph = np.angle(fv[pk2])/2
    corr = np.exp(-1j*(2*np.pi*cfo*np.arange(n)/SAMPLE_RATE+ph)).astype(np.complex64)
    m = _try_extract(lfilter(FILT,1.0,(iq*corr).real).astype(np.float32))
    if m: return m
    # strategy 3: decision-directed PLL
    out = np.zeros_like(iq); ph2 = fr2 = 0.0
    for i in range(n):
        cs = iq[i]*np.exp(-1j*ph2); out[i] = cs
        d = 1.0 if cs.real >= 0 else -1.0
        e = cs.imag*d; fr2 += 2e-4*e; ph2 += 0.01*e+fr2
        ph2 = (ph2+np.pi)%(2*np.pi)-np.pi
    return _try_extract(lfilter(FILT,1.0,out.real).astype(np.float32))

# ─── FRAME CODEC ──────────────────────────────────────────────────────────────
# Calibration status frames
def make_status(rxok, atten, ready=0): return f"S|{int(rxok)}|{int(atten)}|{int(ready)}"
def parse_status(m):
    if not m or not m.startswith("S|"): return None
    try: _,a,b,c = m.split("|"); return {'rxok':int(a),'atten':int(b),'ready':int(c)}
    except: return None

# Transfer frames
def encode_packet(seq, total, chunk):
    return f"D|{seq:05d}|{total:05d}|{base64.b64encode(chunk).decode()}"

def encode_nak(missing, gen):
    """Split missing-seq list into frames that fit within MAX_MSG_LEN."""
    hdr = f"N|{gen:05d}|"
    frames, buf = [], []
    for s in missing:
        e = str(s)
        if buf and len(hdr+",".join(buf)+","+e) > MAX_MSG_LEN-2:
            frames.append(hdr+",".join(buf)); buf = [e]
        else: buf.append(e)
    if buf: frames.append(hdr+",".join(buf))
    return frames

def decode_frame(m):
    if not m: return None
    if m.startswith("D|"):
        p = m.split("|",3)
        if len(p)==4:
            try: return {'type':'DATA','seq':int(p[1]),'total':int(p[2]),'b64':p[3]}
            except: pass
    elif m.startswith("N|"):
        p = m.split("|",2)
        if len(p)==3:
            try: return {'type':'NAK','gen':int(p[1]),'seqs':[int(s) for s in p[2].split(",") if s.isdigit()]}
            except: pass
    elif m.startswith("META|"):
        p = m.split("|")
        if len(p)==4:
            try: return {'type':'META','fname':p[1],'total':int(p[2]),'size':int(p[3])}
            except: pass
    elif m=="CTRL|GO":    return {'type':'GO'}
    elif m=="CTRL|READY": return {'type':'READY'}
    elif m=="CTRL|ACK":   return {'type':'ACK'}
    elif m=="CTRL|PAUSE": return {'type':'PAUSE'}
    elif m=="CTRL|DONE":  return {'type':'DONE'}
    return None

# ─── PLUTO SETUP ──────────────────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate = SAMPLE_RATE
    sdr.tx_lo = TX_FREQ;  sdr.rx_lo = RX_FREQ
    sdr.rx_rf_bandwidth = SAMPLE_RATE;  sdr.tx_rf_bandwidth = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 10
    sdr.tx_hardwaregain_chan0 = CAL_TX_ATTEN
    sdr.rx_buffer_size = TX_BUFFER_SIZE
    sdr.tx_cyclic_buffer = True
    try:
        import iio
        dds = iio.Context(args.ip).find_device("cf-ad9361-dds-core-lpc")
        if dds:
            for ch in dds.channels:
                if ch.output:
                    for a in ["raw","scale"]:
                        try: ch.attrs[a].value = "0" if a=="raw" else "0.0"
                        except: pass
    except: pass
    print(f"[✓] TX {TX_FREQ/1e6:.3f} MHz   RX {RX_FREQ/1e6:.3f} MHz")
    return sdr

def set_rx_gain(sdr, g): sdr.rx_hardwaregain_chan0 = int(g)
def set_tx_atten(sdr, a): sdr.tx_hardwaregain_chan0 = int(a)

def rx_gain_limits(sdr):
    """Query the device for the valid RX gain range at the current LO frequency."""
    try:
        ch = sdr._ctrl.find_channel("voltage0", False)
        nums = [float(x) for x in ch.attrs["hardwaregain_available"].value.strip("[] ").split()]
        if len(nums)==3 and nums[2]>nums[0]: return nums[0], nums[2]
    except: pass
    return 0.0, 71.0

def tx_set(sdr, msg):
    try: sdr.tx_destroy_buffer()
    except: pass
    sdr.tx(encode_fill(msg))

def flush_rx(sdr, n=1):
    for _ in range(n):
        try: sdr.rx()
        except: pass

# ─── CALIBRATION ──────────────────────────────────────────────────────────────
def find_rx_gain(sdr, label, sweep):
    print(f"\n[{label}] Sweeping RX gain ...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []
    for g in sweep:
        try: set_rx_gain(sdr, g)
        except OSError: continue
        time.sleep(0.1); flush_rx(sdr, 1)
        dec = pk = 0
        for _ in range(GAIN_STEP_BUFS):
            try:
                rx = sdr.rx(); p = np.max(np.abs(rx)); pk = max(pk, p)
                if decode_any(rx) is not None: dec += 1
            except: pass
        adc = pk/2896*100
        print(f"  {g:>5}  {pk:>6.0f}  {adc:>4.0f}%  {dec:>7}")
        if adc > 98: continue
        if dec >= GAIN_CONFIRM: candidates.append((dec - abs(adc-60)/100, g, adc, dec))
    if not candidates: return None
    candidates.sort(reverse=True); _, g, adc, dec = candidates[0]
    print(f"[{label}] ✓ RX gain={g} dB  ADC={adc:.0f}%  {dec}/{GAIN_STEP_BUFS} decodes")
    return g

def calibrate_rx_gain(sdr, label):
    set_tx_atten(sdr, CAL_TX_ATTEN); tx_set(sdr, CAL_TONE)
    lo, hi = rx_gain_limits(sdr)
    sweep = list(range(int(np.ceil(lo)), int(np.floor(hi))+1, 3))
    print(f"[{label}] Valid RX gain: {lo:.0f}..{hi:.0f} dB")
    for attempt in range(GAIN_RETRIES):
        g = find_rx_gain(sdr, label, sweep)
        if g is not None: set_rx_gain(sdr, g); return g
        print(f"[{label}] Partner not heard — retry {attempt+1}")
    fb = int(np.clip(20, np.ceil(lo), np.floor(hi)))
    set_rx_gain(sdr, fb); return fb

def calibrate_tx_power(sdr, label):
    print(f"\n[{label}] Negotiating TX power ...")
    my_rxok = 1; advertised = None; chosen = None
    def advertise(atten):
        nonlocal advertised
        f = make_status(my_rxok, atten)
        if f != advertised: tx_set(sdr, f); advertised = f
    for atten in TX_ATTEN_SWEEP:
        set_tx_atten(sdr, atten); advertise(atten); ok = False
        for _ in range(POWER_ROUNDS):
            rx = decode_any(sdr.rx()) if True else None
            try: rx = decode_any(sdr.rx())
            except: rx = None
            st = parse_status(rx)
            if st: my_rxok = 1; advertise(atten); ok = st['rxok'] == 1
            else:  my_rxok = 0; advertise(atten)
            if ok: break
        print(f"  TX {atten:>4} dB → {'✓' if ok else 'no echo'}")
        if ok: chosen = atten; break
    chosen = min(0, (chosen or 0) + POWER_MARGIN)
    set_tx_atten(sdr, chosen)
    print(f"[{label}] ✓ TX atten={chosen} dB"); return chosen

def confirm_link(sdr, label, atten):
    print(f"\n[{label}] Confirming link ...")
    tx_set(sdr, make_status(1, atten, ready=1))
    for _ in range(READY_ROUNDS):
        try: rx = decode_any(sdr.rx())
        except: rx = None
        st = parse_status(rx)
        if st and st['ready']: print(f"[{label}] ✓ Link confirmed!"); return
    print(f"[{label}] Partner ready not seen — continuing anyway")

def calibrate(sdr, label):
    print(f"\n{'='*52}\n  ROLE {label} — FDD calibration (parallel)\n{'='*52}")
    rx_gain  = calibrate_rx_gain(sdr, label)
    tx_atten = calibrate_tx_power(sdr, label)
    confirm_link(sdr, label, tx_atten)
    return rx_gain, tx_atten

# ─── SHARED INFRASTRUCTURE ────────────────────────────────────────────────────
def start_listener(sdr, sdr_lock, frame_q, stop):
    """Background thread: drain RX into frame_q; yields lock between calls."""
    def _run():
        flush_rx(sdr, 2)
        while not stop.is_set():
            with sdr_lock:
                try: raw = sdr.rx()
                except: raw = None
            if raw is not None:
                m = decode_any(raw)
                f = decode_frame(m)
                if f:
                    try: frame_q.put_nowait(f)
                    except queue.Full: pass
            time.sleep(TX_LISTENER_SLEEP)
    t = threading.Thread(target=_run, daemon=True); t.start(); return t

def print_progress(done, total, label, extra=""):
    bar = int(32*done/max(total,1))
    b   = "="*bar + (">" if bar<32 else "") + " "*max(31-bar,0)
    pct = 100.0*done/max(total,1)
    sys.stdout.write(f"\r[{label}] [{b}] {done}/{total} ({pct:.1f}%) {extra}  ")
    sys.stdout.flush()

def periodic_stats(label, rx_calls, decoded, n_data, n_nak, n_ctrl, t_start, t_last):
    if time.time() - t_last < REPORT_INTERVAL: return t_last
    now = time.time(); elapsed = now - t_start
    rate = decoded/max(rx_calls,1)*100
    print(f"\n[{label}] stats: rx={rx_calls} decoded={decoded} ({rate:.0f}%) "
          f"DATA={n_data} NAK={n_nak} CTRL={n_ctrl} t={elapsed:.0f}s")
    if rate < 20:
        print(f"  ⚠ Low decode rate — check signal / attenuation / distance")
    return now

def prepare_image(path):
    with open(path,'rb') as f: data = f.read()
    fname = os.path.basename(path)
    if len(data) <= MAX_IMAGE_BYTES: return data, fname
    if not PIL_AVAILABLE:
        print(f"[!] {len(data):,} bytes > {MAX_IMAGE_BYTES:,} limit. "
              "Install Pillow for auto-resize or use a smaller file.")
        return data, fname
    from io import BytesIO
    img = _Pil.open(path).convert("RGB"); f = 1.0
    while len(data) > MAX_IMAGE_BYTES and f > 0.1:
        f *= 0.70; w,h = img.size
        sm = img.resize((max(int(w*f),32), max(int(h*f),32)), _Pil.LANCZOS)
        buf = BytesIO(); sm.save(buf,'JPEG',quality=80); data = buf.getvalue()
    fname = f"thumb_{os.path.splitext(fname)[0]}.jpg"
    print(f"[SENDER] Auto-resized → {len(data):,} bytes"); return data, fname

# ═══════════════════════════════════════════════════════════════════════════════
#  SENDER
# ═══════════════════════════════════════════════════════════════════════════════
def sender_main(sdr, image_path, rx_gain, tx_atten):
    data, fname = prepare_image(image_path)
    chunks = [data[i:i+CHUNK_BYTES] for i in range(0,len(data),CHUNK_BYTES)]
    total  = len(chunks)
    digest = hashlib.md5(data).hexdigest()[:8]
    if total > 99_999: print("[SENDER] Too many chunks — use a smaller image"); return

    print(f"\n[SENDER] {fname}  {len(data):,} bytes  {total} packets  md5:{digest}")
    print(f"  First pass: ~{int(total*TX_PACKET_BURST)}s + retransmits")

    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    sdr_lock = threading.Lock()
    frame_q  = queue.Queue(maxsize=1000)
    stop     = threading.Event()
    start_listener(sdr, sdr_lock, frame_q, stop)

    # Deque + set to prevent double-queueing retransmits
    deque       = collections.deque()
    queued      = set()
    unique_sent = set()

    def init_deque():
        deque.clear(); queued.clear(); unique_sent.clear()
        for i in range(total): deque.append(i); queued.add(i)

    def enqueue_nak(seqs):
        added = 0
        for s in seqs:
            if 0 <= s < total and s not in queued:
                queued.add(s); deque.append(s); added += 1
        return added

    meta    = f"META|{fname}|{total}|{len(data)}"
    nak_gen = -1
    pass_n  = 0
    state   = 'ANNOUNCE'

    # Stats for periodic report
    rx_calls = decoded = n_data = n_nak = n_ctrl = 0
    t_start = t_report = time.time()

    print("[SENDER] ANNOUNCE — broadcasting META, waiting for CTRL|GO ...")

    while not stop.is_set():

        # ── ANNOUNCE ─────────────────────────────────────────────────────────
        if state == 'ANNOUNCE':
            with sdr_lock: tx_set(sdr, meta)
            try:
                f = frame_q.get(timeout=1.0)
                t = f['type']
                if t in ('GO', 'ACK'):           # GO = explicit; ACK = implicit (receiver in RECEIVE)
                    init_deque(); nak_gen = -1; pass_n += 1
                    print(f"\n[SENDER] → SEND  pass={pass_n}  {total} packets")
                    state = 'SEND'
                elif t == 'READY':
                    pass  # receiver not ready yet — keep broadcasting
                else:
                    pass  # other frames ignored in ANNOUNCE
            except queue.Empty: pass

        # ── SEND ─────────────────────────────────────────────────────────────
        elif state == 'SEND':
            if not deque:
                state = 'PAUSE'; continue

            seq = deque.popleft(); queued.discard(seq); unique_sent.add(seq)
            with sdr_lock: tx_set(sdr, encode_packet(seq, total, chunks[seq]))
            print_progress(len(unique_sent), total, "SENDER", f"pass={pass_n} pending={len(deque)}")

            t_burst = time.time()
            while time.time() - t_burst < TX_PACKET_BURST:
                try:
                    f = frame_q.get_nowait()
                    if f['type'] == 'NAK' and f['gen'] > nak_gen:
                        nak_gen = f['gen']
                        added   = enqueue_nak(f['seqs'])
                        print(f"\n[SENDER] ← NAK gen={f['gen']}  {len(f['seqs'])} missing  "
                              f"({added} new queued  deque={len(deque)})")
                    elif f['type'] == 'DONE':
                        print(f"\n[SENDER] ← DONE — transfer complete!")
                        stop.set()
                except queue.Empty:
                    time.sleep(0.01)

        # ── PAUSE ─────────────────────────────────────────────────────────────
        elif state == 'PAUSE':
            sys.stdout.write("\n")
            print(f"[SENDER] PAUSE — pass {pass_n} done, {len(unique_sent)} unique sent")

            # Signal pass-complete briefly, then go neutral (prevents receiver
            # from flooding NAKs triggered by the cyclic CTRL|PAUSE buffer)
            for _ in range(5):
                with sdr_lock: tx_set(sdr, "CTRL|PAUSE"); time.sleep(0.15)
            with sdr_lock: tx_set(sdr, "CTRL|ACK")
            print(f"[SENDER] Waiting for NAK or DONE (timeout {PAUSE_TIMEOUT:.0f}s) ...")

            t0 = time.time(); responded = False
            while not stop.is_set() and time.time()-t0 < PAUSE_TIMEOUT:
                try:
                    f = frame_q.get(timeout=0.5)
                    if f['type'] == 'NAK' and f['gen'] > nak_gen:
                        nak_gen = f['gen']
                        enqueue_nak(f['seqs'])
                        print(f"[SENDER] ← NAK gen={f['gen']}  "
                              f"{len(f['seqs'])} missing → retransmitting {len(deque)}")
                        pass_n += 1; unique_sent.clear()
                        state = 'SEND'; responded = True; break
                    elif f['type'] == 'DONE':
                        print("[SENDER] ← DONE — transfer complete!")
                        stop.set(); responded = True; break
                except queue.Empty: pass

            if not stop.is_set() and not responded:
                print("[SENDER] PAUSE timeout — returning to ANNOUNCE")
                state = 'ANNOUNCE'

        # Periodic stats
        t_report = periodic_stats("SENDER", rx_calls, decoded, n_data, n_nak, n_ctrl,
                                  t_start, t_report)

    sys.stdout.write("\n")
    print(f"[SENDER] ✓ Done  md5:{digest}  (verify with receiver)")
    stop.set()


# ═══════════════════════════════════════════════════════════════════════════════
#  RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════
def receiver_main(sdr, save_dir, rx_gain, tx_atten):
    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    sdr_lock = threading.Lock()
    frame_q  = queue.Queue(maxsize=1000)
    stop     = threading.Event()
    start_listener(sdr, sdr_lock, frame_q, stop)

    buf        = {}   # seq → bytes
    total      = None
    fname      = "image"
    nak_gen    = 0
    last_nak   = 0.0   # time of last NAK transmission
    last_count = -1
    state      = 'READY'

    rx_calls = decoded = n_data = n_nak = n_ctrl = 0
    t_start = t_report = time.time()

    print("[RECEIVER] READY — broadcasting CTRL|READY ...")
    with sdr_lock: tx_set(sdr, "CTRL|READY")

    def send_nak():
        nonlocal nak_gen, last_nak
        missing = [i for i in range(total) if i not in buf]
        if not missing:
            # Nothing missing — we're done
            print(f"\n[RECEIVER] All packets confirmed — sending DONE")
            for _ in range(6):
                with sdr_lock: tx_set(sdr, "CTRL|DONE"); time.sleep(0.25)
            stop.set(); return
        nak_gen += 1
        frames = encode_nak(missing, nak_gen)
        print(f"\n[RECEIVER] → NAK gen={nak_gen}  {len(missing)} missing  ({len(frames)} frame(s))")
        for nf in frames:
            with sdr_lock: tx_set(sdr, nf); time.sleep(0.15)
        with sdr_lock: tx_set(sdr, "CTRL|ACK")   # switch away from NAK frame in cyclic buffer
        last_nak = time.time()

    while not stop.is_set():

        # ── READY ─────────────────────────────────────────────────────────────
        if state == 'READY':
            try:
                f = frame_q.get(timeout=1.0)
                if f['type'] == 'META':
                    total, fname = f['total'], f['fname']
                    print(f"\n[RECEIVER] Incoming: '{fname}'  {total} packets  "
                          f"{f['size']:,} bytes  (~{int(total*TX_PACKET_BURST)}s first pass)")
                    for _ in range(4):    # send GO several times for reliability
                        with sdr_lock: tx_set(sdr, "CTRL|GO"); time.sleep(0.15)
                    with sdr_lock: tx_set(sdr, "CTRL|ACK")
                    last_nak = time.time()
                    state = 'RECEIVE'
                    print("[RECEIVER] RECEIVE — collecting packets ...")
            except queue.Empty:
                with sdr_lock: tx_set(sdr, "CTRL|READY")

        # ── RECEIVE ───────────────────────────────────────────────────────────
        elif state == 'RECEIVE':
            try:
                f = frame_q.get(timeout=0.25)
                if f['type'] == 'DATA':
                    if f['seq'] not in buf:
                        try: buf[f['seq']] = base64.b64decode(f['b64'])
                        except: pass
                    count = len(buf)
                    if count != last_count:
                        print_progress(count, total, "RECEIVER", f"missing={total-count}")
                        last_count = count
                    if count == total: state = 'COMPLETE'; continue

                elif f['type'] == 'PAUSE':
                    # Sender finished a pass — trigger immediate NAK if MIN interval ok
                    if time.time() - last_nak >= MIN_NAK_INTERVAL:
                        send_nak()

                elif f['type'] == 'META':
                    # Sender returned to ANNOUNCE (missed our GO) — re-send GO
                    print(f"\n[RECEIVER] ← META in RECEIVE — re-sending GO")
                    for _ in range(3):
                        with sdr_lock: tx_set(sdr, "CTRL|GO"); time.sleep(0.12)
                    with sdr_lock: tx_set(sdr, "CTRL|ACK")

            except queue.Empty: pass

            # Periodic NAK — MIN_NAK_INTERVAL enforced regardless of trigger source
            if total and time.time() - last_nak >= NAK_INTERVAL:
                send_nak()

        # ── COMPLETE ──────────────────────────────────────────────────────────
        elif state == 'COMPLETE':
            sys.stdout.write("\n")
            print("[RECEIVER] All packets received — sending DONE ...")
            for _ in range(8):
                with sdr_lock: tx_set(sdr, "CTRL|DONE"); time.sleep(0.25)
            try:
                os.makedirs(save_dir, exist_ok=True)
                data   = b"".join(buf[i] for i in range(total))
                digest = hashlib.md5(data).hexdigest()[:8]
                path   = os.path.join(save_dir, f"rx_{fname}")
                with open(path,'wb') as fh: fh.write(data)
                print(f"\n[RECEIVER] ✓ Saved: {path}")
                print(f"  Size : {len(data):,} bytes   md5:{digest}  ← compare with sender")
                if PIL_AVAILABLE:
                    try: _Pil.open(path).show()
                    except: pass
            except Exception as e:
                print(f"[RECEIVER] Reassembly error: {e}")
            stop.set()

        t_report = periodic_stats("RECEIVER", rx_calls, decoded, n_data, n_nak, n_ctrl,
                                  t_start, t_report)

    stop.set()
    print("[RECEIVER] Done.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    sdr = setup_pluto()
    mode = "SENDER" if IS_SENDER else "RECEIVER"
    print(f"\n*** ROLE:{args.role.upper()}  MODE:{mode} ***")
    print(f"    TX {TX_FREQ/1e6:.3f} MHz   RX {RX_FREQ/1e6:.3f} MHz")
    print(f"    {'Image: '+args.image if IS_SENDER else 'Save: '+args.save_dir+'/rx_<file>'}")
    print("\n    Calibration in 3 seconds ...")
    time.sleep(3)

    rx_gain, tx_atten = calibrate(sdr, args.role.upper())
    print(f"\n[{args.role.upper()}] Calibration done: RX={rx_gain} dB  TX={tx_atten} dB")

    if IS_SENDER: sender_main(sdr, args.image, rx_gain, tx_atten)
    else:         receiver_main(sdr, args.save_dir, rx_gain, tx_atten)

if __name__ == "__main__":
    main()