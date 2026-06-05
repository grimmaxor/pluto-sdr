"""
ADALM Pluto — FDD Image Transfer, Round-Based Selective-Repeat ARQ
====================================================================
Sender  : python pluto_image_arq.py --role a --image photo.jpg
Receiver: python pluto_image_arq.py --role b [--save-dir PATH]

  --codec auto|h264|jpeg   (default: auto — H.264 if ffmpeg present, else JPEG)

HOW THE TRANSFER WORKS (the round model you asked for)
------------------------------------------------------
  Round 0:  sender transmits ALL packets once
            receiver keeps whatever decodes
            sender marks end-of-round (EOR)
            receiver replies with a REQUEST listing only the packets it is missing
  Round 1:  sender transmits ONLY the requested packets
            receiver updates, sends a new (shorter) request
  ...       repeat until the request is empty → receiver sends DONE.

Because this is FDD, the request channel (receiver→sender) and the data channel
(sender→receiver) are on separate frequencies and never collide.  Each request
carries a generation counter so duplicate/stale request frames are ignored.

CODEC
-----
H.264 is a video codec; here it encodes the image as a single intra (I-)frame,
which compresses ~30% smaller than JPEG at similar quality but needs ffmpeg.
'auto' uses H.264 when ffmpeg is on PATH and falls back to JPEG (Pillow)
otherwise.  The chosen codec travels in the META frame so the receiver decodes
correctly.  Switch to H.265 by changing 'libx264'→'libx265' below.

FDD channel assignment
----------------------
  role a : TX on FREQ_A  RX on FREQ_B        role b : the mirror
"""

import adi
import numpy as np
import argparse, threading, queue, collections
import time, sys, os, base64, hashlib, subprocess, shutil, tempfile
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
ap.add_argument('--image',    default=None, help="Image to send (omit to receive)")
ap.add_argument('--save-dir', default='.',  dest='save_dir')
ap.add_argument('--codec',    choices=['auto','h264','jpeg'], default='auto')
args = ap.parse_args()

IS_SENDER        = args.image is not None
TX_FREQ, RX_FREQ = (int(args.freq_a), int(args.freq_b)) if args.role == 'a' \
                   else (int(args.freq_b), int(args.freq_a))

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
TX_BUFFER_SIZE     = 65536
SAMPLES_PER_SYMBOL = 16
MAX_MSG_LEN        = 180
FRAME_MAGIC        = 0xBE
BARKER_13          = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)

# Calibration
CAL_TX_ATTEN   = -30
GAIN_STEP_BUFS = 6
GAIN_CONFIRM   = 2
GAIN_RETRIES   = 8
POWER_ROUNDS   = 6
POWER_MARGIN   = 5
READY_ROUNDS   = 40
TX_ATTEN_SWEEP = list(range(-80, 1, 5))
CAL_TONE       = "__CAL__"

# Transfer / round ARQ
CHUNK_BYTES      = 48      # 48 raw → 64 b64 → 78-char frame (good decode rate)
TX_PACKET_BURST  = 0.30    # seconds each packet is held on air
TX_LISTEN_SLEEP  = 0.010   # gap between rx() calls (yields the sdr lock)
EOR_REPEATS      = 6       # times the sender repeats the end-of-round marker
EOR_GAP          = 0.12
EOR_RETRIES      = 4       # EOR/request attempts before resending the whole batch
WAIT_REQ_TIMEOUT = 6.0     # sender waits this long for a request per EOR attempt
REQ_COLLECT_SEC  = 2.0     # window to gather all frames of one request
MIN_REQ_INTERVAL = 5     # receiver: min gap between requests it sends
NODATA_TIMEOUT   = 3.0     # receiver: silence this long ⇒ assume round ended
MAX_ROUNDS       = 60
MAX_DIM          = 720     # longest image side after resize
REPORT_INTERVAL  = 10.0

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
    raw  = msg[:MAX_MSG_LEN].encode()
    pay  = bytes([FRAME_MAGIC, len(raw)]) + raw
    full = pay + bytes([crc8(pay)])
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
    m = _try_extract(lfilter(FILT,1.0,iq.real).astype(np.float32))
    if m: return m
    sq = iq**2; n = len(sq); fv = np.fft.fft(sq); fv[0] = 0
    fr = np.fft.fftfreq(n, 1.0/SAMPLE_RATE); k = int(np.argmax(np.abs(fv)))
    corr = np.exp(-1j*(2*np.pi*(fr[k]/2)*np.arange(n)/SAMPLE_RATE + np.angle(fv[k])/2)).astype(np.complex64)
    m = _try_extract(lfilter(FILT,1.0,(iq*corr).real).astype(np.float32))
    if m: return m
    out = np.zeros_like(iq); ph = frq = 0.0
    for i in range(n):
        cs = iq[i]*np.exp(-1j*ph); out[i] = cs
        e = cs.imag*(1.0 if cs.real >= 0 else -1.0)
        frq += 2e-4*e; ph += 0.01*e + frq
        ph = (ph+np.pi)%(2*np.pi)-np.pi
    return _try_extract(lfilter(FILT,1.0,out.real).astype(np.float32))

# ─── FRAME CODEC ──────────────────────────────────────────────────────────────
def make_status(rxok, atten, ready=0): return f"S|{int(rxok)}|{int(atten)}|{int(ready)}"
def parse_status(m):
    if not m or not m.startswith("S|"): return None
    try: _,a,b,c = m.split("|"); return {'rxok':int(a),'atten':int(b),'ready':int(c)}
    except: return None

def encode_packet(seq, total, chunk):
    return f"D|{seq:05d}|{total:05d}|{base64.b64encode(chunk).decode()}"

def encode_request(missing, gen):
    """Split missing list into R|gen|seq,seq,... frames within MAX_MSG_LEN."""
    hdr = f"R|{gen:05d}|"
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
    elif m.startswith("R|"):
        p = m.split("|",2)
        if len(p)==3:
            try: return {'type':'REQ','gen':int(p[1]),'seqs':[int(s) for s in p[2].split(",") if s.isdigit()]}
            except: pass
    elif m.startswith("EOR|"):
        p = m.split("|")
        if len(p)==2:
            try: return {'type':'EOR','round':int(p[1])}
            except: pass
    elif m.startswith("META|"):
        p = m.split("|")
        if len(p)==5:
            try: return {'type':'META','codec':p[1],'fname':p[2],'total':int(p[3]),'size':int(p[4])}
            except: pass
    elif m=="CTRL|GO":    return {'type':'GO'}
    elif m=="CTRL|READY": return {'type':'READY'}
    elif m=="CTRL|ACK":   return {'type':'ACK'}
    elif m=="CTRL|DONE":  return {'type':'DONE'}
    return None

# ─── IMAGE CODEC (H.264 single I-frame, JPEG fallback) ────────────────────────
def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None

def _resized_source(path):
    """Return (src_path, is_temp). Resizes via PIL to <= MAX_DIM if available."""
    if not PIL_AVAILABLE:
        return path, False
    img = _Pil.open(path).convert("RGB")
    if max(img.size) > MAX_DIM:
        s = MAX_DIM / max(img.size)
        w = max(2, (int(img.width*s)//2)*2)      # even dims for yuv420p
        h = max(2, (int(img.height*s)//2)*2)
        img = img.resize((w, h), _Pil.LANCZOS)
    tmp = tempfile.mktemp(suffix=".png")
    img.save(tmp)
    return tmp, True

def encode_image(path, prefer):
    """Return (payload_bytes, codec_tag, out_filename)."""
    base = os.path.splitext(os.path.basename(path))[0]
    src, is_tmp = _resized_source(path)
    try:
        if prefer in ('auto','h264') and _have_ffmpeg():
            out = tempfile.mktemp(suffix=".h264")
            cmd = ['ffmpeg','-y','-loglevel','error','-i',src,
                   '-frames:v','1','-c:v','libx264','-preset','slower',
                   '-crf','30','-pix_fmt','yuv420p','-f','h264',out]
            if not PIL_AVAILABLE:   # let ffmpeg bound the size
                cmd[6:6] = ['-vf', f'scale=min({MAX_DIM}\\,iw):-2']
            subprocess.run(cmd, check=True)
            data = open(out,'rb').read(); os.unlink(out)
            return data, 'h264', base + '.h264'
        if prefer == 'h264' and not _have_ffmpeg():
            print("[codec] ffmpeg not found — falling back to JPEG")
    except Exception as e:
        print(f"[codec] H.264 encode failed ({e}); falling back to JPEG")
    finally:
        if is_tmp and prefer != 'jpeg':
            # keep src for the JPEG path below if needed; clean later
            pass

    # JPEG fallback
    if PIL_AVAILABLE:
        from io import BytesIO
        buf = BytesIO(); _Pil.open(src).save(buf, 'JPEG', quality=80)
        data = buf.getvalue()
        if is_tmp: os.unlink(src)
        return data, 'jpeg', base + '.jpg'
    data = open(path,'rb').read()
    return data, 'raw', os.path.basename(path)

def decode_image(data, codec, fname, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if codec == 'h264':
        if _have_ffmpeg():
            inp = tempfile.mktemp(suffix=".h264")
            open(inp,'wb').write(data)
            out = os.path.join(save_dir, f"rx_{os.path.splitext(fname)[0]}.png")
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',inp,'-frames:v','1',out], check=True)
            os.unlink(inp)
            return out
        out = os.path.join(save_dir, f"rx_{fname}")
        open(out,'wb').write(data)
        print("[codec] ffmpeg not found — saved raw .h264 (install ffmpeg to view)")
        return out
    out = os.path.join(save_dir, f"rx_{fname}")
    open(out,'wb').write(data)
    return out

# ─── PLUTO SETUP / CALIBRATION ────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate = SAMPLE_RATE
    sdr.tx_lo = TX_FREQ; sdr.rx_lo = RX_FREQ
    sdr.rx_rf_bandwidth = SAMPLE_RATE; sdr.tx_rf_bandwidth = SAMPLE_RATE
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

def find_rx_gain(sdr, label, sweep):
    print(f"\n[{label}] Sweeping RX gain ...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Dec':>4}")
    cand = []
    for g in sweep:
        try: set_rx_gain(sdr, g)
        except OSError: continue
        time.sleep(0.1); flush_rx(sdr, 1)
        dec = pk = 0
        for _ in range(GAIN_STEP_BUFS):
            try:
                rx = sdr.rx(); pk = max(pk, np.max(np.abs(rx)))
                if decode_any(rx) is not None: dec += 1
            except: pass
        adc = pk/2896*100
        print(f"  {g:>5}  {pk:>6.0f}  {adc:>4.0f}%  {dec:>4}")
        if adc > 98: continue
        if dec >= GAIN_CONFIRM: cand.append((dec - abs(adc-60)/100, g, adc, dec))
    if not cand: return None
    cand.sort(reverse=True); _, g, adc, dec = cand[0]
    print(f"[{label}] ✓ RX gain={g} dB (ADC {adc:.0f}%, {dec}/{GAIN_STEP_BUFS})")
    return g

def calibrate(sdr, label):
    print(f"\n{'='*52}\n  ROLE {label} — FDD calibration\n{'='*52}")
    # RX gain
    set_tx_atten(sdr, CAL_TX_ATTEN); tx_set(sdr, CAL_TONE)
    lo, hi = rx_gain_limits(sdr)
    sweep = list(range(int(np.ceil(lo)), int(np.floor(hi))+1, 3))
    print(f"[{label}] Valid RX gain: {lo:.0f}..{hi:.0f} dB")
    rx_gain = None
    for attempt in range(GAIN_RETRIES):
        rx_gain = find_rx_gain(sdr, label, sweep)
        if rx_gain is not None: break
        print(f"[{label}] Partner not heard — retry {attempt+1}")
    if rx_gain is None: rx_gain = int(np.clip(20, np.ceil(lo), np.floor(hi)))
    set_rx_gain(sdr, rx_gain)
    # TX power
    print(f"\n[{label}] Negotiating TX power ...")
    my_rxok = 1; adv = None; chosen = None
    def advertise(at):
        nonlocal adv
        f = make_status(my_rxok, at)
        if f != adv: tx_set(sdr, f); adv = f
    for atten in TX_ATTEN_SWEEP:
        set_tx_atten(sdr, atten); advertise(atten); ok = False
        for _ in range(POWER_ROUNDS):
            try: rx = decode_any(sdr.rx())
            except: rx = None
            st = parse_status(rx)
            if st: my_rxok = 1; advertise(atten); ok = (st['rxok'] == 1)
            else:  my_rxok = 0; advertise(atten)
            if ok: break
        print(f"  TX {atten:>4} dB → {'✓' if ok else 'no echo'}")
        if ok: chosen = atten; break
    tx_atten = min(0, (chosen or 0) + POWER_MARGIN)
    set_tx_atten(sdr, tx_atten)
    print(f"[{label}] ✓ TX atten={tx_atten} dB")
    # confirm
    tx_set(sdr, make_status(1, tx_atten, ready=1))
    for _ in range(READY_ROUNDS):
        try: rx = decode_any(sdr.rx())
        except: rx = None
        st = parse_status(rx)
        if st and st['ready']: print(f"[{label}] ✓ Link confirmed!"); break
    return rx_gain, tx_atten

# ─── SHARED ───────────────────────────────────────────────────────────────────
def start_listener(sdr, lock, frame_q, stop):
    def _run():
        flush_rx(sdr, 2)
        while not stop.is_set():
            with lock:
                try: raw = sdr.rx()
                except: raw = None
            if raw is not None:
                f = decode_frame(decode_any(raw))
                if f:
                    try: frame_q.put_nowait(f)
                    except queue.Full: pass
            time.sleep(TX_LISTEN_SLEEP)
    threading.Thread(target=_run, daemon=True).start()

def progress(done, total, label, extra=""):
    bar = int(32*done/max(total,1))
    b = "="*bar + (">" if bar<32 else "") + " "*max(31-bar,0)
    sys.stdout.write(f"\r[{label}] [{b}] {done}/{total} ({100*done/max(total,1):.1f}%) {extra}  ")
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
#  SENDER  (round-based selective repeat)
# ═══════════════════════════════════════════════════════════════════════════════
def sender_main(sdr, image_path, rx_gain, tx_atten):
    data, codec, fname = encode_image(image_path, args.codec)
    chunks = [data[i:i+CHUNK_BYTES] for i in range(0,len(data),CHUNK_BYTES)]
    total  = len(chunks)
    digest = hashlib.md5(data).hexdigest()[:8]
    if total > 99_999: print("[SENDER] Too many chunks — use a smaller image"); return

    print(f"\n[SENDER] {fname}  codec={codec}  {len(data):,} bytes  {total} packets  md5:{digest}")

    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    lock = threading.Lock(); frame_q = queue.Queue(maxsize=1000); stop = threading.Event()
    start_listener(sdr, lock, frame_q, stop)

    def drain_for_done(seconds):
        t = time.time()
        while time.time() - t < seconds:
            try:
                f = frame_q.get_nowait()
                if f['type'] == 'DONE': stop.set(); return
            except queue.Empty:
                time.sleep(0.01)

    # ── ANNOUNCE: broadcast META until receiver sends GO (or ACK) ─────────────
    meta = f"META|{codec}|{fname}|{total}|{len(data)}"
    print("[SENDER] ANNOUNCE — broadcasting META, waiting for GO ...")
    while not stop.is_set():
        with lock: tx_set(sdr, meta)
        try:
            f = frame_q.get(timeout=1.0)
            if f['type'] in ('GO','ACK'): break
        except queue.Empty: pass

    # ── ROUND LOOP ────────────────────────────────────────────────────────────
    to_send  = list(range(total))
    rnd      = 0
    last_gen = -1
    while not stop.is_set() and rnd < MAX_ROUNDS:
        # send every packet in the current batch exactly once
        print(f"\n[SENDER] Round {rnd}: sending {len(to_send)} packet(s)")
        for i, seq in enumerate(to_send):
            with lock: tx_set(sdr, encode_packet(seq, total, chunks[seq]))
            progress(i+1, len(to_send), "SENDER", f"round={rnd}")
            drain_for_done(TX_PACKET_BURST)
            if stop.is_set(): break
        if stop.is_set(): break
        sys.stdout.write("\n")

        # solicit the receiver's request (only the still-missing packets)
        outcome, seqs, gen = _solicit(sdr, lock, frame_q, rnd, last_gen)
        if outcome == 'done':
            print("[SENDER] Receiver confirmed complete!"); break
        elif outcome == 'req':
            if not seqs:
                print("[SENDER] Empty request — assuming complete."); break
            last_gen = gen; to_send = seqs; rnd += 1
            print(f"[SENDER] {len(seqs)} still missing → round {rnd}")
        else:   # timeout — receiver silent; resend the same batch
            print("[SENDER] No request heard — resending batch")

    sys.stdout.write("\n")
    print(f"[SENDER] ✓ Finished.  codec={codec}  md5:{digest}  (compare with receiver)")
    stop.set()

def _solicit(sdr, lock, frame_q, rnd, last_gen):
    """
    Mark end-of-round and collect the receiver's request.
    Returns ('done',None,None) | ('req', sorted_seqs, gen) | ('timeout',None,None).
    """
    for _ in range(EOR_RETRIES):
        for _ in range(EOR_REPEATS):
            with lock: tx_set(sdr, f"EOR|{rnd}")
            time.sleep(EOR_GAP)
        with lock: tx_set(sdr, "CTRL|ACK")

        by_gen = {}; first_t = None
        deadline = time.time() + WAIT_REQ_TIMEOUT
        while time.time() < deadline:
            try: f = frame_q.get(timeout=0.3)
            except queue.Empty: continue
            if f['type'] == 'DONE':
                return ('done', None, None)
            if f['type'] == 'REQ' and f['gen'] > last_gen:
                by_gen.setdefault(f['gen'], set()).update(f['seqs'])
                if first_t is None: first_t = time.time()
            if first_t and time.time() - first_t > REQ_COLLECT_SEC:
                break
        if by_gen:
            g = max(by_gen)
            return ('req', sorted(by_gen[g]), g)
    return ('timeout', None, None)

# ═══════════════════════════════════════════════════════════════════════════════
#  RECEIVER  (round-based selective repeat)
# ═══════════════════════════════════════════════════════════════════════════════
def receiver_main(sdr, save_dir, rx_gain, tx_atten):
    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    lock = threading.Lock(); frame_q = queue.Queue(maxsize=1000); stop = threading.Event()
    start_listener(sdr, lock, frame_q, stop)

    buf = {}; total = None; codec = fname = None
    req_gen = 0; last_req_t = 0.0; last_data_t = time.time(); last_shown = -1

    # ── READY: wait for META, reply GO ────────────────────────────────────────
    print("[RECEIVER] READY — broadcasting CTRL|READY ...")
    with lock: tx_set(sdr, "CTRL|READY")
    while not stop.is_set() and total is None:
        try:
            f = frame_q.get(timeout=1.0)
            if f['type'] == 'META':
                total, codec, fname = f['total'], f['codec'], f['fname']
                print(f"\n[RECEIVER] Incoming '{fname}'  codec={codec}  "
                      f"{total} packets  {f['size']:,} bytes")
                for _ in range(4):
                    with lock: tx_set(sdr, "CTRL|GO"); time.sleep(0.15)
                with lock: tx_set(sdr, "CTRL|ACK")
        except queue.Empty:
            with lock: tx_set(sdr, "CTRL|READY")

    def save_and_finish():
        sys.stdout.write("\n")
        print("[RECEIVER] All packets received — sending DONE ...")
        for _ in range(8):
            with lock: tx_set(sdr, "CTRL|DONE"); time.sleep(0.25)
        raw = b"".join(buf[i] for i in range(total))
        digest = hashlib.md5(raw).hexdigest()[:8]
        out = decode_image(raw, codec, fname, save_dir)
        print(f"[RECEIVER] ✓ Saved: {out}   ({len(raw):,} bytes  md5:{digest})")
        if PIL_AVAILABLE:
            try: _Pil.open(out).show()
            except: pass
        stop.set()

    def send_request():
        nonlocal req_gen, last_req_t
        if time.time() - last_req_t < MIN_REQ_INTERVAL: return
        missing = [i for i in range(total) if i not in buf]
        if not missing:
            save_and_finish(); return
        req_gen += 1
        frames = encode_request(missing, req_gen)
        print(f"\n[RECEIVER] → request gen={req_gen}: {len(missing)} missing ({len(frames)} frame(s))")
        for fr in frames:
            with lock: tx_set(sdr, fr); time.sleep(0.15)
        with lock: tx_set(sdr, "CTRL|ACK")
        last_req_t = time.time()

    # ── RECEIVE LOOP ──────────────────────────────────────────────────────────
    last_data_t = time.time()
    while not stop.is_set():
        try:
            f = frame_q.get(timeout=0.25)
            if f['type'] == 'DATA':
                last_data_t = time.time()
                if f['seq'] not in buf:
                    try: buf[f['seq']] = base64.b64decode(f['b64'])
                    except: pass
                if len(buf) != last_shown:
                    progress(len(buf), total, "RECEIVER", f"missing={total-len(buf)}")
                    last_shown = len(buf)
                if len(buf) == total:
                    save_and_finish()
            elif f['type'] == 'EOR':
                send_request()                 # sender finished a round
            elif f['type'] == 'META':
                for _ in range(3):
                    with lock: tx_set(sdr, "CTRL|GO"); time.sleep(0.12)
                with lock: tx_set(sdr, "CTRL|ACK")
        except queue.Empty:
            pass

        # silence for too long ⇒ assume round ended (EOR lost) and request
        if total and len(buf) < total and time.time() - last_data_t > NODATA_TIMEOUT:
            send_request()
            last_data_t = time.time()

    print("[RECEIVER] Done.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    sdr = setup_pluto()
    mode = "SENDER" if IS_SENDER else "RECEIVER"
    print(f"\n*** ROLE:{args.role.upper()}  MODE:{mode} ***")
    print(f"    TX {TX_FREQ/1e6:.3f} MHz   RX {RX_FREQ/1e6:.3f} MHz")
    if IS_SENDER:
        print(f"    Image: {args.image}   codec={args.codec}"
              f"   ffmpeg={'yes' if _have_ffmpeg() else 'no'}")
    else:
        print(f"    Save : {args.save_dir}/rx_<file>")
    print("\n    Calibration in 3 seconds ...")
    time.sleep(3)
    rx_gain, tx_atten = calibrate(sdr, args.role.upper())
    print(f"\n[{args.role.upper()}] Calibration done: RX={rx_gain} dB  TX={tx_atten} dB")
    if IS_SENDER: sender_main(sdr, args.image, rx_gain, tx_atten)
    else:         receiver_main(sdr, args.save_dir, rx_gain, tx_atten)

if __name__ == "__main__":
    main()
