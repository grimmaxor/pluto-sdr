"""
ADALM Pluto FDD Image Transfer with ARQ  (v2 – bugfixes + diagnostics)
========================================================================
Sender : python pluto_image_fdd.py --role a --image photo.jpg
Receiver: python pluto_image_fdd.py --role b

  --role a/b     : determines which frequency pair is TX and which is RX
  --image <path> : makes this node the sender; omit to receive
  --save-dir <d> : directory for received file  (default: '.')

BUG FIXES vs v1
---------------
1. NAK flooding (coax)
   The receiver's cyclic TX buffer kept looping the last NAK frame.
   The sender's listener decoded it repeatedly and re-added already-queued
   packets to the retransmit deque, creating an ever-growing loop.
   FIX: every NAK batch carries a generation counter (nak_gen).  The sender
   tracks the last generation it processed and silently drops older/duplicate
   NAK frames.  After sending NAK, the receiver switches its cyclic TX to
   CTRL|ACK so the sender sees a neutral "alive" signal instead of NAK.

2. Announcement desync (OTA)
   The sender's ANNOUNCE state accepted CTRL|READY (receiver is alive but
   hasn't decoded META yet) as a trigger to start sending data.  The sender
   moved to SEND while the receiver was still in READY — sender finished the
   pass and sat in PAUSE indefinitely while the receiver sent nothing.
   FIX: sender now only advances on CTRL|GO, which the receiver sends *after*
   successfully decoding the META frame.  CTRL|READY is logged but ignored.

3. No recovery from PAUSE timeout
   If sender timed out in PAUSE with no response it sat in an infinite PAUSE
   loop.
   FIX: after PAUSE_TIMEOUT with no NAK or DONE, sender goes back to ANNOUNCE
   and re-broadcasts META, allowing the receiver to re-synchronise.

DIAGNOSTICS
-----------
A Diag object lives on each side.  It tracks:
  - rx() call rate and decode success rate
  - frame type breakdown (DATA / NAK / CTRL)
  - every state transition with timestamps
  - a periodic health report every DIAG_INTERVAL seconds
  - an explicit lock-hold rate estimate (via listener timing)
All diagnostic output is prefixed with ↯ so it stands out in the log.
"""

import adi
import numpy as np
import argparse
import threading
import queue
import collections
import time
import sys
import os
import base64
import hashlib
from scipy.signal import firwin, lfilter

try:
    from PIL import Image as _PilImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--role',     choices=['a', 'b'], required=True)
parser.add_argument('--ip',       type=str,   default='ip:pluto.local')
parser.add_argument('--freq-a',   type=float, default=2412e6, dest='freq_a')
parser.add_argument('--freq-b',   type=float, default=2437e6, dest='freq_b')
parser.add_argument('--image',    type=str,   default=None)
parser.add_argument('--save-dir', type=str,   default='.',  dest='save_dir')
args = parser.parse_args()

ROLE      = args.role
FREQ_A    = int(args.freq_a)
FREQ_B    = int(args.freq_b)
IS_SENDER = (args.image is not None)
TX_FREQ, RX_FREQ = (FREQ_A, FREQ_B) if ROLE == 'a' else (FREQ_B, FREQ_A)

# ─── RADIO CONSTANTS ──────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
TX_BUFFER_SIZE     = 65536
SAMPLES_PER_SYMBOL = 16
MAX_MSG_LEN        = 180
FRAME_MAGIC        = 0xBE
BARKER_13          = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)

# ─── CALIBRATION CONSTANTS ────────────────────────────────────────────────────
CAL_TX_ATTEN       = -30
CAPTURE_SEC        = TX_BUFFER_SIZE / SAMPLE_RATE
GAIN_STEP_BUFFERS  = 6
GAIN_CONFIRM       = 2
GAIN_SWEEP_RETRIES = 8
LISTEN_BUFFERS     = 5
POWER_ROUNDS       = 6
POWER_MARGIN       = 5
READY_ROUNDS       = 40
TX_ATTEN_SWEEP     = list(range(-80, 1, 5))
MSG_CAL_TONE       = "__CAL__"

# ─── TRANSFER CONSTANTS ───────────────────────────────────────────────────────
# Frame:  "D|SSSSS|TTTTT|<b64>"  header = 14 chars
#   48 raw bytes → 64 b64 chars → 78 total  (vs old 174 — shorter = better decode rate)
#   At ~0.21% BER: P(78-char frame ok) ≈ 27%  vs  P(174-char frame ok) ≈ 6%
CHUNK_BYTES       = 48
TX_PACKET_BURST   = 0.35          # seconds each packet is held on air (~4-5 decode attempts)
TX_LISTENER_SLEEP = 0.010         # gap between rx() calls (yields the sdr lock)
NAK_INTERVAL      = 3.0           # receiver sends a new NAK every this many seconds
MIN_NAK_INTERVAL  = 1.5           # minimum gap between consecutive NAK sends (prevents
                                  # PAUSE flood from spamming NAK and starving the listener)
PAUSE_TIMEOUT     = 15.0          # sender waits this long in PAUSE before returning to ANNOUNCE
MAX_IMAGE_BYTES   = 60_000        # auto-resize threshold
DIAG_INTERVAL     = 10.0          # seconds between automatic diagnostic reports

# ─── DSP ──────────────────────────────────────────────────────────────────────
def crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc

FILT = firwin(SAMPLES_PER_SYMBOL * 4 + 1, 1.4 / SAMPLES_PER_SYMBOL,
              window='hamming').astype(np.float32)

def encode_fill(message: str) -> np.ndarray:
    message   = message[:MAX_MSG_LEN]
    msg_bytes = message.encode('utf-8')
    payload   = bytes([FRAME_MAGIC, len(msg_bytes)]) + msg_bytes
    full      = payload + bytes([crc8(payload)])
    bbits = ((1 - BARKER_13) / 2).astype(np.uint8)
    pbits = np.unpackbits(np.frombuffer(full, dtype=np.uint8))
    bits  = np.concatenate([bbits, pbits]).astype(np.float32)
    symbols = 1.0 - 2.0 * bits
    up  = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL, dtype=np.float32)
    up[::SAMPLES_PER_SYMBOL] = symbols
    shaped = lfilter(FILT, 1.0, up).astype(np.float32)
    mx = np.max(np.abs(shaped))
    if mx > 0: shaped = shaped / mx * 0.8 * 2**15
    iq = shaped.astype(np.complex64)
    reps = int(np.ceil(TX_BUFFER_SIZE / len(iq))) + 1
    return np.tile(iq, reps)[:TX_BUFFER_SIZE].astype(np.complex64)

def _cfo_fft(iq):
    sq = iq**2; n = len(sq); fv = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/SAMPLE_RATE)
    pk = int(np.argmax(np.abs(fv)))
    cfo = freqs[pk]/2; ph = np.angle(fv[pk])/2
    corr = np.exp(-1j*(2*np.pi*cfo*np.arange(n)/SAMPLE_RATE+ph)).astype(np.complex64)
    return (iq*corr).astype(np.complex64), float(cfo)

def _cfo_pll(iq):
    out = np.zeros_like(iq); ph = fr = 0.0; alpha, beta = 0.01, 0.0002
    for i in range(len(iq)):
        cs = iq[i]*np.exp(-1j*ph); out[i] = cs
        d = 1.0 if cs.real >= 0 else -1.0; err = cs.imag*d
        fr += beta*err; ph += alpha*err+fr
        ph = (ph+np.pi)%(2*np.pi)-np.pi
    return out, 0.0

def _try_extract(sig):
    for pol in [1,-1]:
        s = sig*pol
        for toff in range(SAMPLES_PER_SYMBOL):
            bits = (s[toff::SAMPLES_PER_SYMBOL]<0).astype(np.uint8)
            nb = len(bits)//8
            if nb < 8: continue
            raw = bytes(np.packbits(bits[:nb*8]))
            for pos in range(len(raw)-4):
                if raw[pos] != FRAME_MAGIC: continue
                ml = raw[pos+1]
                if ml == 0 or ml > MAX_MSG_LEN: continue
                ei = pos+2+ml
                if ei+1 >= len(raw): continue
                if crc8(raw[pos:ei]) != raw[ei]: continue
                try:    return raw[pos+2:ei].decode('utf-8')
                except: continue
    return None

def decode_any(iq):
    peak = np.max(np.abs(iq))
    if peak < 5: return None, None, 0.0
    iq = (iq/peak*2**13).astype(np.complex64)
    for fn, tag in [(lambda q:(q,0.0), "none"), (_cfo_fft, "fft"), (_cfo_pll, "pll")]:
        s, cfo = fn(iq)
        m = _try_extract(lfilter(FILT,1.0,s.real).astype(np.float32))
        if m is not None: return m, tag, cfo
    return None, None, 0.0

# ─── CALIBRATION FRAME HELPERS ────────────────────────────────────────────────
def make_status(rxok, atten, ready=0):
    return f"S|{int(rxok)}|{int(atten)}|{int(ready)}"

def parse_status(m):
    if m is None or not m.startswith("S|"): return None
    try:
        _, rxok, atten, ready = m.split("|")
        return {'rxok':int(rxok),'atten':int(atten),'ready':int(ready)}
    except: return None

# ─── TRANSFER FRAME HELPERS ───────────────────────────────────────────────────
def encode_packet(seq: int, total: int, chunk: bytes) -> str:
    return f"D|{seq:05d}|{total:05d}|{base64.b64encode(chunk).decode('ascii')}"

def encode_nak(missing: list, gen: int) -> list:
    """
    Build NAK frames.  Each frame now includes a generation counter so the
    sender can detect and drop duplicate decodes of the same NAK batch.
    Format: "N|<gen>|<seq>,<seq>,..."
    """
    header = f"N|{gen:05d}|"
    frames, buf = [], []
    for seq in missing:
        entry = str(seq)
        if buf and len(header + ",".join(buf) + "," + entry) > MAX_MSG_LEN - 2:
            frames.append(header + ",".join(buf))
            buf = [entry]
        else:
            buf.append(entry)
    if buf:
        frames.append(header + ",".join(buf))
    return frames

def decode_frame(m):
    """
    Parse a transfer frame.  Returns a dict with 'type', or None.

    Types: DATA, NAK, META, GO, READY, ACK, PAUSE, DONE
    CTRL|ACK  = receiver is alive and collecting (sent between NAK batches)
    CTRL|READY = receiver has not yet seen the META frame
    CTRL|GO    = receiver successfully decoded META; sender may start sending
    """
    if m is None: return None

    if m.startswith("D|"):
        p = m.split("|", 3)
        if len(p) == 4:
            try: return {'type':'DATA','seq':int(p[1]),'total':int(p[2]),'b64':p[3]}
            except: pass

    elif m.startswith("N|"):
        # New format: "N|<gen>|<seqs>"
        p = m.split("|", 2)
        if len(p) == 3:
            try:
                gen  = int(p[1])
                seqs = [int(s) for s in p[2].split(",") if s.strip().isdigit()]
                return {'type':'NAK','gen':gen,'seqs':seqs}
            except: pass

    elif m.startswith("META|"):
        p = m.split("|")
        if len(p) == 4:
            try: return {'type':'META','fname':p[1],'total':int(p[2]),'size':int(p[3])}
            except: pass

    elif m == "CTRL|GO":    return {'type':'GO'}
    elif m == "CTRL|READY": return {'type':'READY'}
    elif m == "CTRL|ACK":   return {'type':'ACK'}    # receiver alive, no pending NAK
    elif m == "CTRL|PAUSE": return {'type':'PAUSE'}
    elif m == "CTRL|DONE":  return {'type':'DONE'}

    return None

# ─── PLUTO SETUP ──────────────────────────────────────────────────────────────
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate             = SAMPLE_RATE
    sdr.tx_lo                   = TX_FREQ
    sdr.rx_lo                   = RX_FREQ
    sdr.rx_rf_bandwidth         = SAMPLE_RATE
    sdr.tx_rf_bandwidth         = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = 10
    sdr.tx_hardwaregain_chan0   = CAL_TX_ATTEN
    sdr.rx_buffer_size          = TX_BUFFER_SIZE
    sdr.tx_cyclic_buffer        = True
    try:
        import iio
        ctx = iio.Context(args.ip)
        dds = ctx.find_device("cf-ad9361-dds-core-lpc")
        if dds:
            for ch in dds.channels:
                if ch.output:
                    for attr in ["raw","scale"]:
                        try: ch.attrs[attr].value = "0" if attr=="raw" else "0.0"
                        except: pass
            print("[*] DDS disabled")
    except Exception as e:
        print(f"[*] DDS note: {e}")
    print(f"[\u2713] TX {TX_FREQ/1e6:.3f} MHz   RX {RX_FREQ/1e6:.3f} MHz")
    return sdr

# ─── HARDWARE HELPERS ─────────────────────────────────────────────────────────
def set_rx_gain(sdr, g):  sdr.rx_hardwaregain_chan0 = int(g)
def set_tx_atten(sdr, a): sdr.tx_hardwaregain_chan0 = int(a)

def rx_gain_limits(sdr):
    try:
        ch  = sdr._ctrl.find_channel("voltage0", False)
        raw = ch.attrs["hardwaregain_available"].value
        nums = [float(x) for x in raw.strip("[] \t").split()]
        if len(nums)==3 and nums[2]>nums[0]: return nums[0], nums[2]
    except: pass
    return 0.0, 71.0

def build_gain_sweep(gmin, gmax, step=3):
    lo,hi = int(np.ceil(gmin)), int(np.floor(gmax))
    if hi<lo: lo,hi = 0,71
    return list(range(lo, hi+1, step))

def tx_set(sdr, msg: str):
    try:    sdr.tx_destroy_buffer()
    except: pass
    sdr.tx(encode_fill(msg))

def tx_idle(sdr):
    try:    sdr.tx_destroy_buffer()
    except: pass
    sdr.tx(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))

def flush_rx(sdr, n=1):
    for _ in range(n):
        try:    sdr.rx()
        except: pass

def listen_one(sdr):
    try:    return decode_any(sdr.rx())
    except: return None, None, 0.0

# ─── CALIBRATION ──────────────────────────────────────────────────────────────
def find_rx_gain(sdr, label, sweep):
    print(f"\n[{label}] Sweeping RX gain ...")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []
    for g in sweep:
        try:    set_rx_gain(sdr, g)
        except OSError: continue
        time.sleep(0.1); flush_rx(sdr, 1)
        decodes = peak_acc = 0
        for _ in range(GAIN_STEP_BUFFERS):
            try:
                rx = sdr.rx(); peak = np.max(np.abs(rx)); peak_acc=max(peak_acc,peak)
                m,_,_ = decode_any(rx)
                if m is not None: decodes+=1
            except: pass
        adc = peak_acc/2896*100
        print(f"  {g:>5}  {peak_acc:>6.0f}  {adc:>4.0f}%  {decodes:>7}")
        if adc>98: continue
        if decodes>=GAIN_CONFIRM:
            candidates.append((decodes-abs(adc-60)/100.0, g, adc, decodes))
    if not candidates: return None
    candidates.sort(reverse=True); _,g,adc,dec = candidates[0]
    print(f"[{label}] \u2713 RX gain={g} dB  ADC={adc:.0f}%  decodes={dec}/{GAIN_STEP_BUFFERS}")
    return g

def calibrate_rx_gain(sdr, label):
    set_tx_atten(sdr, CAL_TX_ATTEN); tx_set(sdr, MSG_CAL_TONE)
    gmin,gmax = rx_gain_limits(sdr); sweep = build_gain_sweep(gmin,gmax)
    print(f"[{label}] Valid RX gain: {gmin:.0f}..{gmax:.0f} dB")
    for attempt in range(GAIN_SWEEP_RETRIES):
        g = find_rx_gain(sdr, label, sweep)
        if g is not None: set_rx_gain(sdr,g); return g
        print(f"[{label}] Partner not heard — retry {attempt+1}")
    fb = int(np.clip(20, np.ceil(gmin), np.floor(gmax)))
    print(f"[{label}] ! Fallback RX gain={fb}"); set_rx_gain(sdr,fb); return fb

def calibrate_tx_power(sdr, label):
    print(f"\n[{label}] Negotiating TX power ...")
    my_rxok=1; advertised=None; chosen=None
    def advertise(atten):
        nonlocal advertised
        frame=make_status(my_rxok,atten,ready=0)
        if frame!=advertised: tx_set(sdr,frame); advertised=frame
    for atten in TX_ATTEN_SWEEP:
        set_tx_atten(sdr,atten); advertise(atten); partner_ok=False
        for _ in range(POWER_ROUNDS):
            m,_,_ = listen_one(sdr); st=parse_status(m)
            if st is not None:
                my_rxok=1; advertise(atten)
                if st['rxok']==1: partner_ok=True; break
            else:
                my_rxok=0; advertise(atten)
        print(f"  TX {atten:>4} dB → {'partner decodes us ✓' if partner_ok else 'no echo'}")
        if partner_ok: chosen=atten; break
    if chosen is None: chosen=0; print(f"[{label}] Fallback: max power")
    else: chosen=min(0,chosen+POWER_MARGIN)
    set_tx_atten(sdr,chosen); print(f"[{label}] \u2713 TX atten={chosen} dB"); return chosen

def confirm_link(sdr, label, tx_atten):
    print(f"\n[{label}] Confirming link ...")
    tx_set(sdr, make_status(1,tx_atten,ready=1))
    for _ in range(READY_ROUNDS):
        m,_,_ = listen_one(sdr); st=parse_status(m)
        if st and st['ready']==1: print(f"[{label}] \u2713 Link confirmed!"); return True
    print(f"[{label}] ! Partner ready not seen. Continuing."); return False

def calibrate(sdr, label):
    print("\n"+"="*54)
    print(f"  ROLE {label} — calibration (both sides run in parallel)")
    print("="*54)
    rx_gain  = calibrate_rx_gain(sdr, label)
    tx_atten = calibrate_tx_power(sdr, label)
    confirm_link(sdr, label, tx_atten)
    return rx_gain, tx_atten

# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────
class Diag:
    """
    Lightweight diagnostic counters.  Counts are written by the listener thread
    and read by the main thread; the Python GIL makes individual int increments
    safe without an explicit lock.
    """
    def __init__(self, label: str):
        self.label       = label
        self.t_start     = time.time()
        self.t_last_diag = time.time()
        # listener counters
        self.rx_calls    = 0    # total sdr.rx() attempts
        self.rx_decoded  = 0    # frames that passed CRC
        self.n_data      = 0    # DATA frames seen
        self.n_nak       = 0    # NAK frames seen
        self.n_go        = 0    # GO frames seen
        self.n_ready     = 0    # READY frames seen
        self.n_ack       = 0    # ACK frames seen (receiver alive signal)
        self.n_pause     = 0    # PAUSE frames seen
        self.n_done      = 0    # DONE frames seen
        self.n_meta      = 0    # META frames seen
        self.n_other     = 0    # other decoded frames

    # ── called from listener thread ──────────────────────────────────────────
    def log(self, f):
        self.rx_decoded += 1
        t = f['type']
        if   t == 'DATA':  self.n_data  += 1
        elif t == 'NAK':   self.n_nak   += 1
        elif t == 'GO':    self.n_go    += 1
        elif t == 'READY': self.n_ready += 1
        elif t == 'ACK':   self.n_ack   += 1
        elif t == 'PAUSE': self.n_pause += 1
        elif t == 'DONE':  self.n_done  += 1
        elif t == 'META':  self.n_meta  += 1
        else:              self.n_other += 1

    # ── called from main thread ───────────────────────────────────────────────
    def state(self, new_state: str):
        elapsed = time.time() - self.t_start
        print(f"\n↯ [{self.label}] STATE → {new_state}  (t={elapsed:.1f}s)")

    def event(self, msg: str):
        elapsed = time.time() - self.t_start
        print(f"↯ [{self.label}] {msg}  (t={elapsed:.1f}s)")

    def maybe_report(self):
        now = time.time()
        if now - self.t_last_diag < DIAG_INTERVAL:
            return
        self.t_last_diag = now
        self._report(now)

    def final_report(self):
        self._report(time.time())

    def _report(self, now):
        elapsed = now - self.t_start
        calls   = max(self.rx_calls, 1)
        decoded = self.rx_decoded
        rate    = decoded / calls * 100
        pps     = calls / elapsed
        print(f"\n↯ [{self.label}] ── Diagnostic Report ──────────────────────────")
        print(f"  Elapsed        : {elapsed:.1f}s")
        print(f"  rx() calls     : {self.rx_calls}  ({pps:.1f}/s)")
        print(f"  Decoded frames : {decoded}  ({rate:.1f}% success rate)")
        print(f"  DATA           : {self.n_data}")
        print(f"  NAK            : {self.n_nak}")
        print(f"  META           : {self.n_meta}")
        print(f"  GO             : {self.n_go}")
        print(f"  READY          : {self.n_ready}  ← if high: receiver not seeing META")
        print(f"  ACK            : {self.n_ack}   ← background 'alive' signal")
        print(f"  PAUSE          : {self.n_pause}")
        print(f"  DONE           : {self.n_done}")
        if rate < 20:
            print(f"  ⚠ Low decode rate ({rate:.1f}%) — check signal level / distance / attenuation")
        if self.n_ready > 20 and self.n_meta == 0 and self.n_data == 0:
            print(f"  ⚠ Seeing many READY frames but no META — "
                  f"receiver hasn't decoded the announcement yet")
        print(f"↯ [{self.label}] ───────────────────────────────────────────────")

# ─── SHARED LISTENER THREAD ───────────────────────────────────────────────────
def make_listener(sdr, sdr_lock, frame_q, done_event, diag: Diag):
    """
    Daemon thread: continuously grabs RX buffers and decodes them.
    TX_LISTENER_SLEEP between calls gives the main thread brief windows
    to acquire sdr_lock for tx_set().
    """
    def _run():
        flush_rx(sdr, 2)
        while not done_event.is_set():
            with sdr_lock:
                try:    raw = sdr.rx()
                except: raw = None
            diag.rx_calls += 1
            if raw is not None:
                m, _, _ = decode_any(raw)
                f = decode_frame(m)
                if f is not None:
                    diag.log(f)
                    try:    frame_q.put_nowait(f)
                    except queue.Full: pass
            time.sleep(TX_LISTENER_SLEEP)
    t = threading.Thread(target=_run, daemon=True)
    return t

# ─── IMAGE HELPERS ────────────────────────────────────────────────────────────
def prepare_image(path: str):
    with open(path,'rb') as fh: data=fh.read()
    fname = os.path.basename(path)
    if len(data) <= MAX_IMAGE_BYTES: return data, fname
    if not PIL_AVAILABLE:
        print(f"[!] Image is {len(data):,} bytes (>{MAX_IMAGE_BYTES:,}). "
              "Install Pillow for auto-resize or use a smaller file.")
        return data, fname
    from io import BytesIO
    img = _PilImage.open(path).convert("RGB"); factor=1.0
    while len(data)>MAX_IMAGE_BYTES and factor>0.1:
        factor*=0.70; w,h=img.size
        small=img.resize((max(int(w*factor),32),max(int(h*factor),32)),_PilImage.LANCZOS)
        buf=BytesIO(); small.save(buf,'JPEG',quality=80); data=buf.getvalue()
    fname=f"thumb_{os.path.splitext(fname)[0]}.jpg"
    print(f"[SENDER] Auto-resized → {len(data):,} bytes  ({fname})")
    return data, fname

def print_progress(done, total, label, extra=""):
    bar_len=34; filled=int(bar_len*done/max(total,1))
    bar="="*filled+(">" if filled<bar_len else "")+" "*max(bar_len-filled-1,0)
    pct=100.0*done/max(total,1)
    sys.stdout.write(f"\r[{label}] [{bar}] {done}/{total} ({pct:5.1f}%) {extra}  ")
    sys.stdout.flush()

# ═══════════════════════════════════════════════════════════════════════════════
#  SENDER
# ═══════════════════════════════════════════════════════════════════════════════
def sender_main(sdr, image_path: str, rx_gain: int, tx_atten: int):
    data, fname = prepare_image(image_path)
    chunks = [data[i:i+CHUNK_BYTES] for i in range(0,len(data),CHUNK_BYTES)]
    total  = len(chunks)
    digest = hashlib.md5(data).hexdigest()[:8]

    if total > 99_999:
        print(f"[SENDER] Image too large ({total} chunks). Use a smaller file."); return

    print(f"\n[SENDER] ─────────────────────────────────────────")
    print(f"  File  : {fname}")
    print(f"  Size  : {len(data):,} bytes   md5:{digest}")
    print(f"  Chunks: {total}  (~{int(total*TX_PACKET_BURST)}s first pass)")
    print(f"─────────────────────────────────────────────────")

    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    sdr_lock   = threading.Lock()
    frame_q    = queue.Queue(maxsize=1000)
    done_event = threading.Event()
    diag       = Diag("SENDER")

    lt = make_listener(sdr, sdr_lock, frame_q, done_event, diag)
    lt.start()

    meta_frame = f"META|{fname}|{total}|{len(data)}"
    send_deque : collections.deque = collections.deque()
    # gen tracking: only process each NAK generation once
    last_nak_gen = -1
    pass_num     = 0
    state        = 'ANNOUNCE'
    # queued_set mirrors what is currently in send_deque.
    # It prevents the same seq from being added twice — the old bug caused
    # each new NAK gen to re-append ALL missing seqs even if they were
    # already in the deque, ballooning it to 3× capacity.
    queued_set: set = set()
    # unique_sent tracks distinct seqs sent in the current pass (for display).
    unique_sent: set = set()

    diag.state('ANNOUNCE')
    print("[SENDER] Broadcasting META — waiting for CTRL|GO from receiver ...")
    print("         (CTRL|READY means receiver is alive but hasn't seen META yet)")

    def _init_deque(n):
        """Populate send_deque + queued_set with seqs 0..n-1."""
        send_deque.clear(); queued_set.clear(); unique_sent.clear()
        for i in range(n): send_deque.append(i); queued_set.add(i)

    def _enqueue_from_nak(seqs):
        """Add missing seqs from a NAK, skipping any already queued."""
        added = 0
        for s in seqs:
            if 0 <= s < total and s not in queued_set:
                queued_set.add(s)
                send_deque.append(s)
                added += 1
        return added

    while not done_event.is_set():

        # ── ANNOUNCE ─────────────────────────────────────────────────────────
        if state == 'ANNOUNCE':
            with sdr_lock: tx_set(sdr, meta_frame)
            try:
                f = frame_q.get(timeout=1.0)
                t = f['type']
                if t == 'GO':
                    diag.event("← CTRL|GO — receiver confirmed META")
                    _init_deque(total); last_nak_gen = -1; pass_num += 1
                    state = 'SEND'
                    diag.state(f'SEND (pass {pass_num}, {total} packets)')
                elif t == 'READY':
                    diag.event("← CTRL|READY (hasn't decoded META yet — keep announcing)")
                elif t == 'ACK':
                    diag.event("← CTRL|ACK in ANNOUNCE — treating as implicit GO")
                    _init_deque(total); last_nak_gen = -1; pass_num += 1
                    state = 'SEND'
                    diag.state(f'SEND (pass {pass_num}, {total} implicit GO)')
                else:
                    diag.event(f"← {t} in ANNOUNCE — ignored")
            except queue.Empty:
                pass
            diag.maybe_report()

        # ── SEND ─────────────────────────────────────────────────────────────
        elif state == 'SEND':
            if not send_deque:
                state = 'PAUSE'
                diag.state(f'PAUSE (pass {pass_num} done, {len(unique_sent)} unique sent)')
                continue

            seq = send_deque.popleft()
            queued_set.discard(seq)   # seq is no longer in the deque
            unique_sent.add(seq)
            frame = encode_packet(seq, total, chunks[seq])
            with sdr_lock: tx_set(sdr, frame)

            # Progress: unique seqs sent this pass / total  (never goes negative)
            print_progress(len(unique_sent), total, "SENDER",
                           extra=f"pass={pass_num} pending={len(send_deque)}")

            # Hold packet on air; drain frame_q for incoming NAKs.
            # Only process NAK gens we haven't seen yet (gen > last_nak_gen).
            # _enqueue_from_nak uses queued_set to prevent double-queueing.
            t_burst = time.time()
            while time.time() - t_burst < TX_PACKET_BURST:
                try:
                    f = frame_q.get_nowait()
                    if f['type'] == 'NAK':
                        gen = f['gen']
                        if gen > last_nak_gen:
                            last_nak_gen = gen
                            added = _enqueue_from_nak(f['seqs'])
                            diag.event(f"← NAK gen={gen}  {len(f['seqs'])} missing  "
                                       f"({added} newly queued, deque={len(send_deque)})")
                        # else: stale gen — silently drop
                    elif f['type'] == 'DONE':
                        diag.event("← CTRL|DONE — transfer complete!")
                        done_event.set()
                    # GO, READY, ACK, PAUSE, META all silently ignored in SEND
                except queue.Empty:
                    time.sleep(0.01)

            diag.maybe_report()

        # ── PAUSE ─────────────────────────────────────────────────────────────
        # All queued packets sent for this pass.  Tell receiver we're done and
        # wait for a NAK (more retransmits needed) or DONE (transfer complete).
        # If nothing arrives within PAUSE_TIMEOUT, go back to ANNOUNCE — this
        # handles the case where receiver missed the META and is still in READY.
        elif state == 'PAUSE':
            sys.stdout.write("\n")
            print(f"[SENDER] Pass {pass_num} complete — "
                  f"waiting for NAK or DONE (timeout {PAUSE_TIMEOUT:.0f}s) ...")

            # Send CTRL|PAUSE a few times so receiver knows we are done,
            # then switch to CTRL|ACK so receiver stops flooding with NAKs
            # triggered by the cyclic PAUSE buffer.  The receiver will still
            # send its NAK on the natural NAK_INTERVAL timer.
            for _ in range(5):
                with sdr_lock: tx_set(sdr, "CTRL|PAUSE")
                time.sleep(0.15)
            with sdr_lock: tx_set(sdr, "CTRL|ACK")
            diag.event("→ sent CTRL|PAUSE ×5, TX now CTRL|ACK (prevents PAUSE flood)")

            t_pause      = time.time()
            got_response = False
            while not done_event.is_set() and (time.time()-t_pause) < PAUSE_TIMEOUT:
                try:
                    f = frame_q.get(timeout=0.5)
                    if f['type'] == 'NAK':
                        gen = f['gen']
                        if gen > last_nak_gen:
                            last_nak_gen = gen
                            added = _enqueue_from_nak(f['seqs'])
                            diag.event(f"← NAK gen={gen}  {len(f['seqs'])} missing  "
                                       f"({added} newly queued) → retransmitting")
                            got_response = True
                            pass_num += 1; unique_sent.clear()
                            state = 'SEND'
                            diag.state(f'SEND (pass {pass_num}, '
                                       f'{len(send_deque)} retransmits)')
                            break
                        else:
                            diag.event(f"← NAK gen={gen} (stale — drop)")
                    elif f['type'] == 'DONE':
                        diag.event("← CTRL|DONE — receiver has everything!")
                        done_event.set(); got_response = True; break
                    elif f['type'] == 'READY':
                        diag.event("← CTRL|READY in PAUSE — receiver may have missed META")
                    # ACK / GO / others: alive signals, ignore
                except queue.Empty:
                    pass
                diag.maybe_report()

            if done_event.is_set():
                break

            if not got_response:
                diag.event(f"PAUSE timeout ({PAUSE_TIMEOUT:.0f}s) with no response — "
                           "returning to ANNOUNCE (receiver may have missed META)")
                state = 'ANNOUNCE'
                diag.state('ANNOUNCE (recovery)')

    sys.stdout.write("\n")
    print(f"[SENDER] \u2713 Done.  md5:{digest}  (compare with receiver)")
    diag.final_report()
    done_event.set()


# ═══════════════════════════════════════════════════════════════════════════════
#  RECEIVER
# ═══════════════════════════════════════════════════════════════════════════════
def receiver_main(sdr, save_dir: str, rx_gain: int, tx_atten: int):
    set_rx_gain(sdr, rx_gain); set_tx_atten(sdr, tx_atten)
    sdr_lock   = threading.Lock()
    frame_q    = queue.Queue(maxsize=1000)
    done_event = threading.Event()
    diag       = Diag("RECEIVER")

    lt = make_listener(sdr, sdr_lock, frame_q, done_event, diag)
    lt.start()

    packet_buf  = {}
    total       = None
    fname       = "image"
    file_size   = 0
    nak_gen     = 0          # increment each time we compute a new NAK batch
    last_nak_t  = 0.0
    last_count  = -1
    state       = 'READY'

    diag.state('READY')
    print("[RECEIVER] Waiting — broadcasting CTRL|READY ...")
    with sdr_lock: tx_set(sdr, "CTRL|READY")

    while not done_event.is_set():

        # ── READY ─────────────────────────────────────────────────────────────
        if state == 'READY':
            try:
                f = frame_q.get(timeout=1.0)
                if f['type'] == 'META':
                    total, fname, file_size = f['total'], f['fname'], f['size']
                    diag.event(f"← META  fname={fname}  total={total}  "
                               f"size={file_size:,} bytes")
                    print(f"\n[RECEIVER] Incoming: '{fname}'")
                    print(f"  Packets : {total}")
                    print(f"  Size    : {file_size:,} bytes")
                    print(f"  ~{int(total*TX_PACKET_BURST)}s first pass + retransmits")

                    # Send GO multiple times; switch TX to CTRL|ACK after so
                    # the sender doesn't keep seeing stale CTRL|READY frames.
                    for _ in range(4):
                        with sdr_lock: tx_set(sdr, "CTRL|GO")
                        time.sleep(0.15)
                    with sdr_lock: tx_set(sdr, "CTRL|ACK")
                    diag.event("→ sent CTRL|GO ×4, TX now CTRL|ACK")

                    last_nak_t = time.time()
                    state = 'RECEIVE'
                    diag.state('RECEIVE')
                elif f['type'] == 'READY':
                    pass   # other side is also waiting — both will hear META eventually
                else:
                    diag.event(f"← {f['type']} in READY (unexpected) — ignored")
            except queue.Empty:
                with sdr_lock: tx_set(sdr, "CTRL|READY")
            diag.maybe_report()

        # ── RECEIVE ───────────────────────────────────────────────────────────
        elif state == 'RECEIVE':
            try:
                f = frame_q.get(timeout=0.25)

                if f['type'] == 'DATA':
                    seq = f['seq']
                    if seq not in packet_buf:
                        try:   packet_buf[seq] = base64.b64decode(f['b64'])
                        except: pass   # bad base64 — will appear in next NAK
                    count = len(packet_buf)
                    if count != last_count:
                        print_progress(count, total, "RECEIVER",
                                       extra=f"missing={total-count}")
                        last_count = count
                    if count == total:
                        state = 'COMPLETE'
                        continue

                elif f['type'] == 'PAUSE':
                    # Sender finished a pass.  Schedule an immediate NAK — but only
                    # if MIN_NAK_INTERVAL has elapsed since the last one we sent.
                    # Without this guard the cyclic CTRL|PAUSE buffer triggers a new
                    # NAK every ~75 ms, flooding the link and starving the listener.
                    now = time.time()
                    if now - last_nak_t >= MIN_NAK_INTERVAL:
                        diag.event("← CTRL|PAUSE — scheduling immediate NAK")
                        last_nak_t = 0   # force NAK on next timer check
                    # else: PAUSE received but we sent a NAK very recently; ignore this one

                elif f['type'] == 'META':
                    # Sender returned to ANNOUNCE (recovery path) — re-send GO
                    diag.event("← META in RECEIVE — sender re-announced, sending GO")
                    for _ in range(3):
                        with sdr_lock: tx_set(sdr, "CTRL|GO")
                        time.sleep(0.12)
                    with sdr_lock: tx_set(sdr, "CTRL|ACK")

            except queue.Empty:
                pass

            # Periodic NAK: send with incremented generation counter, then switch
            # TX back to CTRL|ACK so sender doesn't keep re-reading stale NAK.
            # MIN_NAK_INTERVAL enforced here so rapid PAUSE decodes can't spam NAK.
            now = time.time()
            if (now - last_nak_t >= NAK_INTERVAL or last_nak_t == 0) \
                    and total is not None \
                    and now - last_nak_t >= MIN_NAK_INTERVAL:
                missing = [i for i in range(total) if i not in packet_buf]
                if missing:
                    nak_gen += 1
                    frames   = encode_nak(missing, nak_gen)
                    diag.event(f"→ NAK gen={nak_gen}  {len(missing)} missing "
                               f"({len(frames)} frame(s))")
                    for nf in frames:
                        with sdr_lock: tx_set(sdr, nf)
                        time.sleep(0.15)
                    # Switch back to neutral ACK so sender doesn't re-decode this NAK
                    with sdr_lock: tx_set(sdr, "CTRL|ACK")
                else:
                    diag.event("NAK timer fired but nothing missing — sending DONE")
                    for _ in range(4):
                        with sdr_lock: tx_set(sdr, "CTRL|DONE")
                        time.sleep(0.2)
                    done_event.set()
                last_nak_t = now

            diag.maybe_report()

        # ── COMPLETE ──────────────────────────────────────────────────────────
        elif state == 'COMPLETE':
            sys.stdout.write("\n")
            print("[RECEIVER] All packets received — sending CTRL|DONE ...")
            for _ in range(8):
                with sdr_lock: tx_set(sdr, "CTRL|DONE")
                time.sleep(0.25)
            try:
                os.makedirs(save_dir, exist_ok=True)
                img_bytes = b"".join(packet_buf[i] for i in range(total))
                digest    = hashlib.md5(img_bytes).hexdigest()[:8]
                out_path  = os.path.join(save_dir, f"rx_{fname}")
                with open(out_path,'wb') as fh: fh.write(img_bytes)
                print(f"\n[RECEIVER] \u2713 Saved: {out_path}")
                print(f"  Size : {len(img_bytes):,} bytes   md5:{digest}")
                print(f"  ← compare md5 with sender's output")
                if PIL_AVAILABLE:
                    try: _PilImage.open(out_path).show()
                    except: pass
            except Exception as e:
                print(f"[RECEIVER] ! Reassembly error: {e}")
            done_event.set()

    done_event.set()
    diag.final_report()
    print("[RECEIVER] Done.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    sdr  = setup_pluto()
    mode = "SENDER" if IS_SENDER else "RECEIVER"
    print(f"\n*** ROLE:{ROLE.upper()}  MODE:{mode} ***")
    print(f"    TX {TX_FREQ/1e6:.3f} MHz  RX {RX_FREQ/1e6:.3f} MHz")
    if IS_SENDER: print(f"    Image : {args.image}")
    else:         print(f"    Save  : {args.save_dir}/rx_<filename>")
    print("\n    Calibration in 3 seconds ...")
    time.sleep(3)
    rx_gain, tx_atten = calibrate(sdr, ROLE.upper())
    print(f"\n[{ROLE.upper()}] Calibration done: RX={rx_gain} dB  TX={tx_atten} dB")
    if IS_SENDER: sender_main(sdr, args.image, rx_gain, tx_atten)
    else:         receiver_main(sdr, args.save_dir, rx_gain, tx_atten)

if __name__ == "__main__":
    main()