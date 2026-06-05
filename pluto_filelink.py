"""
ADALM Pluto File Transfer over SDR  (images first, then video)
==============================================================
Master REQUESTS a file. Slave SENDS it. Master collects packets and
uses selective ARQ (NACK) to recover any that were missed.

Same script on both PCs:
  PC1 (master): python pluto_filelink.py --role master --ip ip:pluto.local
  PC2 (slave) : python pluto_filelink.py --role slave  --ip ip:pluto.local --serve-dir ./to_send

Modulation:
  --mod qpsk   (default, ~2x BPSK throughput, recommended)
  --mod bpsk   (most robust fallback)
  --mod 16qam  (4x throughput, needs high SNR — experimental)

Throughput guide @ 1 Msps:
  BPSK  SPS=16 : ~40 kbit/s usable   (200 KB photo ~ 40 s)
  QPSK  SPS=16 : ~80 kbit/s usable   (200 KB photo ~ 20 s)
  QPSK  SPS=8  : ~160 kbit/s usable  (200 KB photo ~ 10 s)
  16QAM SPS=8  : ~320 kbit/s usable  (needs clean link)

The master shows a live GUI: progress bar + thumbnail as the image fills in.
Video works the same way (any file is just bytes) but expect minutes for
anything beyond a few-second clip — keep clips small & heavily compressed.
"""

import adi
import numpy as np
import argparse
import os
import sys
import time
import struct
import zlib
import threading
from scipy.signal import firwin, lfilter

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--role', choices=['master', 'slave'], required=True)
parser.add_argument('--ip',   type=str, default='ip:pluto.local')
parser.add_argument('--freq', type=float, default=433e6)
parser.add_argument('--mod',  choices=['bpsk', 'qpsk', '16qam'], default='qpsk')
parser.add_argument('--sps',  type=int, default=16, help='samples per symbol')
parser.add_argument('--rx-gain',  type=int, default=None, help='override RX gain (else auto)')
parser.add_argument('--tx-atten', type=int, default=None, help='override TX atten (else auto)')
parser.add_argument('--serve-dir', type=str, default='.', help='slave: folder to serve files from')
parser.add_argument('--out-dir',   type=str, default='./received', help='master: where to save')
parser.add_argument('--request',   type=str, default=None, help='master: filename to request (else prompt)')
parser.add_argument('--no-gui',    action='store_true', help='master: disable GUI, console only')
args = parser.parse_args()

ROLE = args.role

# ─── LINK CONSTANTS ───────────────────────────────────────────────────────────
CENTER_FREQ        = int(args.freq)
SAMPLE_RATE        = int(1e6)
SAMPLES_PER_SYMBOL = args.sps
TX_BUFFER_SIZE     = 65536
MOD                = args.mod

# Bits per symbol for each modulation
BITS_PER_SYMBOL = {'bpsk': 1, 'qpsk': 2, '16qam': 4}[MOD]

# Frame markers
MAGIC = 0xA5

# Packet types
PKT_REQUEST = 0x01   # master -> slave : "send me this file"
PKT_META    = 0x02   # slave -> master : filename, size, num_chunks
PKT_ACK     = 0x03   # master -> slave : "got meta, start blasting"
PKT_DATA    = 0x04   # slave -> master : one data chunk
PKT_NACK    = 0x05   # master -> slave : list of missing seqs
PKT_DONE    = 0x06   # master -> slave : "all received, thanks"
PKT_ERROR   = 0x07   # slave -> master : "file not found" etc

# Chunk payload size (bytes of file data per packet)
# Keep modest so each packet fits comfortably and ARQ is granular
CHUNK_BYTES = 256

# Sync sequence (Barker-13)
BARKER_13 = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)


# ─── DSP: filters ─────────────────────────────────────────────────────────────
def make_filter(sps):
    return firwin(sps * 4 + 1, 1.4 / sps, window='hamming').astype(np.float32)

FILT = make_filter(SAMPLES_PER_SYMBOL)


# ─── MODULATION ───────────────────────────────────────────────────────────────
def bits_to_symbols(bits):
    """Map a bit array to complex constellation symbols for the chosen MOD."""
    if MOD == 'bpsk':
        # 0 -> +1, 1 -> -1  (real only)
        return (1.0 - 2.0 * bits.astype(np.float32)).astype(np.complex64)

    if MOD == 'qpsk':
        # pad to even
        if len(bits) % 2:
            bits = np.concatenate([bits, [0]])
        b = bits.reshape(-1, 2)
        # Gray-coded QPSK: (I,Q) each from one bit
        i = 1.0 - 2.0 * b[:, 0].astype(np.float32)
        q = 1.0 - 2.0 * b[:, 1].astype(np.float32)
        return ((i + 1j * q) / np.sqrt(2)).astype(np.complex64)

    if MOD == '16qam':
        # pad to multiple of 4
        pad = (-len(bits)) % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
        b = bits.reshape(-1, 4)
        # 2 bits -> I level, 2 bits -> Q level. Gray mapping {00,01,11,10}->{-3,-1,1,3}
        gray = {0: -3, 1: -1, 3: 1, 2: 3}
        def lvl(pair):
            return np.array([gray[2*x + y] for x, y in pair], dtype=np.float32)
        i = lvl(list(zip(b[:, 0], b[:, 1])))
        q = lvl(list(zip(b[:, 2], b[:, 3])))
        return ((i + 1j * q) / np.sqrt(10)).astype(np.complex64)

    raise ValueError(MOD)


def symbols_to_bits(syms):
    """Inverse of bits_to_symbols (hard decisions)."""
    if MOD == 'bpsk':
        return (syms.real < 0).astype(np.uint8)

    if MOD == 'qpsk':
        s = syms * np.sqrt(2)
        i = (s.real < 0).astype(np.uint8)
        q = (s.imag < 0).astype(np.uint8)
        return np.column_stack([i, q]).reshape(-1)

    if MOD == '16qam':
        s = syms * np.sqrt(10)
        def demap(v):
            # nearest of {-3,-1,1,3} -> 2 gray bits
            out = np.zeros((len(v), 2), dtype=np.uint8)
            lv = np.clip(np.round((v + 3) / 2) * 2 - 3, -3, 3)
            table = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}
            for k, val in enumerate(lv):
                out[k] = table[int(val)]
            return out
        bi = demap(s.real)
        bq = demap(s.imag)
        return np.column_stack([bi[:, 0], bi[:, 1],
                                bq[:, 0], bq[:, 1]]).reshape(-1)

    raise ValueError(MOD)


# ─── PACKET FRAMING ───────────────────────────────────────────────────────────
def build_packet(pkt_type, seq, total, payload=b''):
    """
    Wire format (before modulation):
      [MAGIC:1][type:1][seq:4][total:4][len:2][payload:len][crc32:4]
    """
    header = struct.pack('>BBIIH', MAGIC, pkt_type, seq, total, len(payload))
    body   = header + payload
    crc    = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack('>I', crc)


def parse_packet(raw, start):
    """
    Try to parse a packet from raw bytes at offset `start`.
    Returns dict or None.
    """
    HDR = 12  # MAGIC+type+seq+total+len
    if start + HDR > len(raw):
        return None
    magic, ptype, seq, total, plen = struct.unpack('>BBIIH', raw[start:start+HDR])
    if magic != MAGIC:
        return None
    if plen > 4096:           # sanity
        return None
    end = start + HDR + plen + 4
    if end > len(raw):
        return None
    payload = raw[start+HDR : start+HDR+plen]
    crc_rx  = struct.unpack('>I', raw[start+HDR+plen : end])[0]
    crc_calc = zlib.crc32(raw[start:start+HDR+plen]) & 0xFFFFFFFF
    if crc_rx != crc_calc:
        return None
    return {'type': ptype, 'seq': seq, 'total': total, 'payload': payload}


# ─── BITS <-> IQ (fills TX buffer) ────────────────────────────────────────────
def packet_to_iq_fill(pkt_bytes):
    """
    Barker sync + packet bits -> symbols -> pulse-shaped IQ,
    tiled to fill the whole TX buffer (no silence gaps).
    """
    pbits = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    # Barker as bits (1->0, -1->1) prepended for sync
    bbits = ((1 - BARKER_13) / 2).astype(np.uint8)
    # tile barker so it's detectable; repeat barker 2x
    sync  = np.tile(bbits, 2)
    bits  = np.concatenate([sync, pbits]).astype(np.uint8)

    syms  = bits_to_symbols(bits)

    up = np.zeros(len(syms) * SAMPLES_PER_SYMBOL, dtype=np.complex64)
    up[::SAMPLES_PER_SYMBOL] = syms
    shaped_i = lfilter(FILT, 1.0, up.real).astype(np.float32)
    shaped_q = lfilter(FILT, 1.0, up.imag).astype(np.float32)
    shaped   = (shaped_i + 1j * shaped_q).astype(np.complex64)

    mx = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15

    repeats = int(np.ceil(TX_BUFFER_SIZE / len(shaped))) + 1
    iq_full = np.tile(shaped, repeats)[:TX_BUFFER_SIZE]
    return iq_full.astype(np.complex64)


def iq_to_packets(iq):
    """
    Demodulate an RX buffer and extract ALL valid packets found.
    Tries CFO none / FFT / PLL and both/all rotations.
    Returns list of packet dicts.
    """
    peak = np.max(np.abs(iq))
    if peak < 5:
        return []
    iq = (iq / peak).astype(np.complex64)

    found = {}

    for corrected in _cfo_variants(iq):
        # matched filter both rails
        fi = lfilter(FILT, 1.0, corrected.real).astype(np.float32)
        fq = lfilter(FILT, 1.0, corrected.imag).astype(np.float32)

        for rot in _rotations():
            ci = fi * rot[0] - fq * rot[1]
            cq = fi * rot[1] + fq * rot[0]
            csig = (ci + 1j * cq).astype(np.complex64)

            for toff in range(SAMPLES_PER_SYMBOL):
                syms = csig[toff::SAMPLES_PER_SYMBOL]
                if len(syms) < 32:
                    continue
                bits = symbols_to_bits(syms)
                nb   = len(bits) // 8
                if nb < 16:
                    continue
                raw = bytes(np.packbits(bits[:nb*8]))
                # scan for MAGIC
                for pos in range(len(raw) - 12):
                    if raw[pos] != MAGIC:
                        continue
                    pkt = parse_packet(raw, pos)
                    if pkt:
                        found[(pkt['type'], pkt['seq'])] = pkt
    return list(found.values())


def _cfo_variants(iq):
    """Yield CFO-corrected versions: none, FFT-coarse, PLL."""
    yield iq  # none

    # FFT squaring (works for BPSK/QPSK power-law; use ^2 for bpsk, ^4 for qpsk)
    power = 2 if MOD == 'bpsk' else 4
    sq    = iq ** power
    n     = len(sq)
    fv    = np.fft.fft(sq)
    fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/SAMPLE_RATE)
    pk    = int(np.argmax(np.abs(fv)))
    cfo   = freqs[pk] / power
    t     = np.arange(n) / SAMPLE_RATE
    yield (iq * np.exp(-1j * 2*np.pi*cfo*t)).astype(np.complex64)

    # PLL (decision-directed, light)
    out = np.zeros_like(iq)
    ph = 0.0; fr = 0.0
    for i in range(len(iq)):
        cs = iq[i] * np.exp(-1j*ph)
        out[i] = cs
        # error for QPSK/BPSK: imag*sign(real)+... keep simple
        err = np.sign(cs.real)*cs.imag - np.sign(cs.imag)*cs.real
        fr += 0.0002 * err
        ph += 0.01 * err + fr
        ph  = (ph + np.pi) % (2*np.pi) - np.pi
    yield out


def _rotations():
    """Constellation rotation hypotheses (phase ambiguity)."""
    if MOD == 'bpsk':
        angles = [0, np.pi]
    else:  # qpsk / 16qam — 4 rotations
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    return [(np.cos(a), np.sin(a)) for a in angles]


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
    sdr.rx_hardwaregain_chan0   = args.rx_gain if args.rx_gain is not None else 20
    sdr.tx_hardwaregain_chan0   = args.tx_atten if args.tx_atten is not None else -30
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
    except Exception:
        pass

    print(f"[\u2713] Connected. {CENTER_FREQ/1e6:.3f} MHz  MOD={MOD.upper()} "
          f"SPS={SAMPLES_PER_SYMBOL}  RXg={sdr.rx_hardwaregain_chan0} "
          f"TXa={sdr.tx_hardwaregain_chan0}")
    return sdr


def tx_silence(sdr):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(np.zeros(TX_BUFFER_SIZE, dtype=np.complex64))

def tx_packet(sdr, pkt_bytes, hold=0.25):
    """Transmit a packet cyclically for `hold` seconds, then silence."""
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    time.sleep(0.03)
    sdr.tx(packet_to_iq_fill(pkt_bytes))
    time.sleep(hold)

def flush_rx(sdr, n=3):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass
        time.sleep(0.01)

def rx_packets(sdr):
    """Grab one RX buffer, return list of decoded packets."""
    try:
        rx = sdr.rx()
    except Exception:
        return []
    return iq_to_packets(rx)

def wait_for_packet(sdr, want_type, timeout=5.0):
    """Listen until a packet of want_type arrives or timeout."""
    flush_rx(sdr, 2)
    deadline = time.time() + timeout
    while time.time() < deadline:
        for pkt in rx_packets(sdr):
            if pkt['type'] == want_type:
                return pkt
        time.sleep(0.01)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SLAVE  (file server)
# ═══════════════════════════════════════════════════════════════════════════════
def run_slave(sdr):
    print("\n" + "="*54)
    print("  SLAVE — file server")
    print(f"  Serving files from: {os.path.abspath(args.serve_dir)}")
    print("="*54)
    tx_silence(sdr)

    while True:
        print("\n[SLAVE] Waiting for a file request...")
        req = wait_for_packet(sdr, PKT_REQUEST, timeout=60)
        if not req:
            continue

        fname = req['payload'].decode('utf-8', errors='replace').strip()
        path  = os.path.join(args.serve_dir, fname)
        print(f"[SLAVE] Master requested: '{fname}'")

        if not os.path.isfile(path):
            print(f"[SLAVE] File not found: {path}")
            tx_packet(sdr, build_packet(PKT_ERROR, 0, 0,
                      f"NOTFOUND:{fname}".encode()), hold=1.0)
            tx_silence(sdr)
            continue

        with open(path, 'rb') as f:
            data = f.read()

        total = (len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES
        chunks = [data[i*CHUNK_BYTES:(i+1)*CHUNK_BYTES] for i in range(total)]
        print(f"[SLAVE] '{fname}'  {len(data)} bytes  -> {total} chunks")

        # ── Send META until master ACKs ──
        meta_payload = struct.pack('>I', len(data)) + fname.encode('utf-8')
        acked = False
        for _ in range(15):
            tx_packet(sdr, build_packet(PKT_META, 0, total, meta_payload), hold=0.4)
            tx_silence(sdr)
            ack = wait_for_packet(sdr, PKT_ACK, timeout=1.5)
            if ack:
                acked = True
                break
        if not acked:
            print("[SLAVE] No ACK from master, aborting this transfer.")
            continue
        print("[SLAVE] Master ACKed. Blasting data...")

        # ── Blast all chunks ──
        def blast(seqs):
            for s in seqs:
                pkt = build_packet(PKT_DATA, s, total, chunks[s])
                tx_packet(sdr, pkt, hold=0.12)
            tx_silence(sdr)

        blast(range(total))

        # ── ARQ loop: respond to NACKs ──
        rounds = 0
        while rounds < 50:
            rounds += 1
            nack = wait_for_packet(sdr, PKT_NACK, timeout=4.0)
            done = wait_for_packet(sdr, PKT_DONE, timeout=0.1)
            if done:
                print("[SLAVE] Master says DONE. Transfer complete.")
                break
            if nack:
                # payload = packed list of missing seqs (>I each)
                miss = list(struct.unpack('>%dI' % (len(nack['payload'])//4),
                                          nack['payload']))
                print(f"[SLAVE] ARQ round {rounds}: resending {len(miss)} chunks")
                blast(miss)
            else:
                # no nack and no done — resend everything once more
                print(f"[SLAVE] No NACK/DONE heard; rebroadcasting all.")
                blast(range(total))
        tx_silence(sdr)


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER  (file requester + collector)
# ═══════════════════════════════════════════════════════════════════════════════
class TransferState:
    def __init__(self):
        self.fname = None
        self.size  = 0
        self.total = 0
        self.chunks = {}          # seq -> bytes
        self.done  = False
        self.lock  = threading.Lock()

    def progress(self):
        if self.total == 0:
            return 0.0
        return len(self.chunks) / self.total

    def missing(self):
        return [s for s in range(self.total) if s not in self.chunks]

    def assemble(self):
        buf = bytearray()
        for s in range(self.total):
            buf += self.chunks.get(s, b'\x00' * CHUNK_BYTES)
        return bytes(buf[:self.size])


def run_master(sdr, gui=None):
    os.makedirs(args.out_dir, exist_ok=True)
    tx_silence(sdr)

    # Which file?
    fname = args.request
    if not fname:
        fname = input("\n[MASTER] Filename to request from slave: ").strip()

    print(f"[MASTER] Requesting '{fname}' ...")
    state = TransferState()
    if gui:
        gui.set_status(f"Requesting {fname}...")

    # ── Send REQUEST until META arrives ──
    meta = None
    for _ in range(20):
        tx_packet(sdr, build_packet(PKT_REQUEST, 0, 0, fname.encode()), hold=0.4)
        tx_silence(sdr)
        pkt = wait_for_packet(sdr, PKT_META, timeout=1.5)
        err = wait_for_packet(sdr, PKT_ERROR, timeout=0.1)
        if err:
            msg = err['payload'].decode('utf-8', errors='replace')
            print(f"[MASTER] Slave error: {msg}")
            if gui: gui.set_status(f"Error: {msg}")
            return
        if pkt:
            meta = pkt
            break
    if not meta:
        print("[MASTER] No META received. Is the slave running & file present?")
        if gui: gui.set_status("No response from slave")
        return

    state.size  = struct.unpack('>I', meta['payload'][:4])[0]
    state.fname = meta['payload'][4:].decode('utf-8', errors='replace')
    state.total = meta['total']
    print(f"[MASTER] Incoming '{state.fname}'  {state.size} bytes  "
          f"{state.total} chunks")
    if gui:
        gui.set_file(state.fname, state.size, state.total)

    # ── ACK and start collecting ──
    for _ in range(3):
        tx_packet(sdr, build_packet(PKT_ACK, 0, state.total), hold=0.3)
    tx_silence(sdr)

    # ── Collect loop with ARQ ──
    stale_rounds = 0
    last_count   = 0
    while not state.done:
        # Listen for a window of data
        t_end = time.time() + 3.0
        while time.time() < t_end:
            for pkt in rx_packets(sdr):
                if pkt['type'] == PKT_DATA and pkt['seq'] < state.total:
                    if pkt['seq'] not in state.chunks:
                        state.chunks[pkt['seq']] = pkt['payload']
                        if gui:
                            gui.update(state)
            if len(state.chunks) >= state.total:
                break
            time.sleep(0.005)

        got = len(state.chunks)
        pct = state.progress() * 100
        print(f"[MASTER] {got}/{state.total} chunks ({pct:.1f}%)")
        if gui:
            gui.update(state)

        if got >= state.total:
            state.done = True
            break

        # Detect stall
        if got == last_count:
            stale_rounds += 1
        else:
            stale_rounds = 0
        last_count = got

        # Send NACK with missing list (cap to keep packet small)
        miss = state.missing()
        chunk_of_miss = miss[:300]   # up to 300 seqs per NACK
        payload = struct.pack('>%dI' % len(chunk_of_miss), *chunk_of_miss)
        for _ in range(2):
            tx_packet(sdr, build_packet(PKT_NACK, 0, state.total, payload), hold=0.3)
        tx_silence(sdr)

        if stale_rounds > 15:
            print("[MASTER] Link stalled — giving up. "
                  f"Got {got}/{state.total}.")
            break

    # ── Done ──
    for _ in range(4):
        tx_packet(sdr, build_packet(PKT_DONE, 0, state.total), hold=0.25)
    tx_silence(sdr)

    if state.done:
        out_path = os.path.join(args.out_dir, state.fname)
        with open(out_path, 'wb') as f:
            f.write(state.assemble())
        print(f"[MASTER] \u2713 Saved {out_path}  ({state.size} bytes)")
        if gui:
            gui.complete(out_path)
    else:
        print("[MASTER] Transfer incomplete.")
        if gui:
            gui.set_status("Transfer incomplete")


# ═══════════════════════════════════════════════════════════════════════════════
#  MASTER GUI  (progress bar + live thumbnail)
# ═══════════════════════════════════════════════════════════════════════════════
class MasterGUI:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk

        self.root = tk.Tk()
        self.root.title("Pluto File Receiver")
        self.root.configure(bg="#0d1117")
        self.root.resizable(False, False)

        BG, FG, ACC = "#0d1117", "#e6edf3", "#2f81f7"

        tk.Label(self.root, text="PLUTO  SDR  FILE  RECEIVER",
                 font=("Consolas", 15, "bold"),
                 bg=BG, fg=ACC).pack(pady=(14, 2))
        self.sub = tk.Label(self.root, text=f"{MOD.upper()}  ·  "
                            f"{CENTER_FREQ/1e6:.1f} MHz  ·  SPS {SAMPLES_PER_SYMBOL}",
                            font=("Consolas", 9), bg=BG, fg="#7d8590")
        self.sub.pack()

        # Thumbnail canvas
        self.canvas = tk.Canvas(self.root, width=360, height=240,
                                bg="#161b22", highlightthickness=1,
                                highlightbackground="#30363d")
        self.canvas.pack(padx=18, pady=14)
        self.canvas_text = self.canvas.create_text(
            180, 120, text="waiting for file…",
            fill="#7d8590", font=("Consolas", 11))

        # Filename / size
        self.info = tk.Label(self.root, text="—",
                             font=("Consolas", 10), bg=BG, fg=FG)
        self.info.pack()

        # Progress bar
        style = ttk.Style()
        style.theme_use("default")
        style.configure("P.Horizontal.TProgressbar",
                        background=ACC, troughcolor="#161b22",
                        bordercolor="#161b22", thickness=18)
        self.bar = ttk.Progressbar(self.root, length=360, maximum=100,
                                   style="P.Horizontal.TProgressbar")
        self.bar.pack(padx=18, pady=(4, 2))

        self.pct = tk.Label(self.root, text="0.0%   0 / 0 chunks",
                            font=("Consolas", 10, "bold"), bg=BG, fg=FG)
        self.pct.pack()

        self.status = tk.Label(self.root, text="Starting…",
                               font=("Consolas", 9), bg=BG, fg="#7d8590")
        self.status.pack(pady=(2, 14))

        self._photo = None
        self._last_thumb = 0

    def set_status(self, txt):
        self.status.config(text=txt)
        self.root.update_idletasks()

    def set_file(self, name, size, total):
        self.info.config(text=f"{name}   ({size:,} bytes, {total} chunks)")
        self.set_status("Receiving…")
        self.root.update_idletasks()

    def update(self, state):
        pct = state.progress() * 100
        self.bar['value'] = pct
        self.pct.config(text=f"{pct:.1f}%   {len(state.chunks)} / {state.total} chunks")
        # Update thumbnail at most ~3x/sec and only for images
        now = time.time()
        if now - self._last_thumb > 0.35:
            self._last_thumb = now
            self._try_thumbnail(state)
        self.root.update_idletasks()

    def _try_thumbnail(self, state):
        # Only attempt for image extensions
        if not state.fname:
            return
        ext = os.path.splitext(state.fname)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
            self.canvas.itemconfig(self.canvas_text,
                                   text=f"receiving {ext or 'file'}…\n(no preview)")
            return
        try:
            from PIL import Image, ImageTk
            import io
            # Assemble whatever we have so far; missing chunks are zeros
            partial = state.assemble()
            img = Image.open(io.BytesIO(partial))
            img.load()                       # may raise if too incomplete
            img.thumbnail((360, 240))
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(180, 120, image=self._photo)
        except Exception:
            # progressive JPEGs may show partially; ignore decode errors
            pass

    def complete(self, path):
        self.set_status(f"✓ Saved: {path}")
        self.bar['value'] = 100
        self._final_image(path)

    def _final_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'):
            self.canvas.itemconfig(self.canvas_text,
                                   text="✓ file received")
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(path)
            img.thumbnail((360, 240))
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(180, 120, image=self._photo)
        except Exception:
            pass

    def run_transfer(self, sdr):
        # Run the transfer in a worker thread so GUI stays responsive
        def worker():
            run_master(sdr, gui=self)
        threading.Thread(target=worker, daemon=True).start()
        self.root.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    sdr = setup_pluto()

    if ROLE == 'slave':
        run_slave(sdr)
    else:
        if args.no_gui:
            run_master(sdr, gui=None)
        else:
            try:
                gui = MasterGUI()
                gui.run_transfer(sdr)
            except Exception as e:
                print(f"[MASTER] GUI unavailable ({e}); falling back to console.")
                run_master(sdr, gui=None)


if __name__ == "__main__":
    main()
