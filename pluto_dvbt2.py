#!/usr/bin/env python3
"""
pluto_dvbt2.py — DVB-T live video streaming over PlutoSDR, no GNU Radio.

Implements the full DVB-T T2k/QPSK/C7_8 signal chain in pure Python/NumPy,
bridged to GStreamer for video capture (TX/Windows) and playback (RX/Linux).

TX (Windows): GStreamer camera → MPEG-TS → UDP:2000 → DVB-T encode → pyadi-iio Pluto
RX (Linux):   pyadi-iio Pluto → DVB-T decode → UDP:2001 → GStreamer playback

Signal chain (follows Tx_Video.grc / Rx_Video.grc exactly):
  TX: energy_dispersal → RS(204,188) → Forney_interleaver →
      inner_coder(C7_8) → bit_inner_il → symbol_inner_il →
      QPSK_map → reference_signals(pilots+IFFT) → cyclic_prefix → Pluto
  RX: Pluto → Schmidl-Cox_sync → FFT → channel_eq →
      QPSK_demap → symbol_deil → bit_deil →
      Viterbi(C7_8) → Forney_deil → RS_dec → energy_descramble → GStreamer

DVB-T parameters (all fixed, matching the GRC flowgraphs):
  Mode            T2k  (FFT = 2048)
  Guard interval  GI_1/32  (CP = 64 samples)
  Constellation   QPSK
  Code rate       7/8
  Sample rate     3.2 MSPS
  RF bandwidth    4 MHz

Usage:
  python pluto_dvbt2.py --tx [--freq 2.4e9] [--uri ip:192.168.2.1] [--attn 8]
                        [--device 0] [--no-audio]
  python pluto_dvbt2.py --rx [--freq 2.4e9] [--uri ip:192.168.2.1] [--gain 30]
                        [--save out.ts]

  Both sides auto-calibrate before streaming (BPSK symmetric FDD handshake).
  TX beacons on --freq; RX uses freq+2 MHz as its return channel.
  Skip calibration and hard-code values with --skip-cal --gain N --attn N.

Dependencies:
  pip install pyadi-iio numpy scipy reedsolo pylibiio
  GStreamer (gst-launch-1.0) on PATH
  Windows TX: mfvideosrc (Media Foundation), wasapisrc
  Linux  RX:  v4l2src, pulsesrc
"""

import argparse
import os
import platform
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import zlib

from scipy.signal import firwin, lfilter as _lfilter

import numpy as np
from reedsolo import RSCodec

# ── libiio shim: must come before pyadi-iio import ────────────────────────────
_LIBIIO = '/usr/lib/x86_64-linux-gnu/libiio.so.0.23'
if os.path.exists(_LIBIIO):
    cur = os.environ.get('LD_PRELOAD', '')
    if _LIBIIO not in cur:
        os.environ['LD_PRELOAD'] = _LIBIIO + (':' + cur if cur else '')
        os.execv(sys.executable, [sys.executable] + sys.argv)

import adi  # noqa: E402  (pyadi-iio, must be after LD_PRELOAD re-exec)

# ══════════════════════════════════════════════════════════════════════════════
# DVB-T T2k constants  (ETSI EN 300 744)
# ══════════════════════════════════════════════════════════════════════════════

FFT_LEN    = 2048          # T2k mode
CP_LEN     = 64            # GI_1/32 = 2048/32
SYM_LEN    = FFT_LEN + CP_LEN   # 2112 samples/OFDM symbol
SAMP_RATE  = 3_200_000
RF_BW      = 4_000_000

N_DATA     = 1512          # active data subcarriers per symbol (T2k QPSK C7/8)
N_ACTIVE   = 1705          # total active subcarriers (K_min=0 .. K_max=1704)
K_MIN_BIN  = 172           # FFT bin of first active carrier: 1024 - 852

# Convolutional code generators (ETSI §4.3.3)
#   G1 = 0133 octal = 0b1011011, G2 = 0171 octal = 0b1111001
_G1 = 0b1011011
_G2 = 0b1111001
_K  = 7                    # constraint length

# Rate-7/8 puncturing patterns (from gr-dtv dvbt_inner_coder.cc)
_PUNCT_A = [1, 1, 0, 0, 0, 1, 0]   # G1 keep mask
_PUNCT_B = [1, 0, 1, 1, 1, 0, 1]   # G2 keep mask
_PUNCT_PERIOD = 7

# RS(204, 188) shortened code parameters
_RS_NSYM    = 16
_RS_PAD     = 51           # RS(255,239) → RS(204,188): 255-204 = 51 pad bytes
_RS_PRIM    = 0x11d        # primitive polynomial x^8+x^4+x^3+x^2+1
_rs_codec   = RSCodec(nsym=_RS_NSYM, prim=_RS_PRIM, generator=2, fcr=0, c_exp=8)

# Pilot amplitude boost (4/3 relative to unit-power data)
_PILOT_BOOST = 4.0 / 3.0

# UDP port assignments (matching GRC flowgraphs)
UDP_IN_PORT  = 2000   # GStreamer → Python TX
UDP_OUT_PORT = 2001   # Python RX → GStreamer

# Pluto buffer size
PLUTO_BUF = 32768

# Batching: process this many OFDM symbols at once for TX DMA efficiency
TX_BATCH_SYMS = 64

# Integer carrier-frequency-offset search range (subcarriers). Two free-running
# Plutos at 2.4 GHz can differ by >100 subcarriers (1562.5 Hz each) and the offset
# drifts between power-ons; ±150 covers ~±234 kHz ≈ ±98 ppm. Hard ceiling is ±171
# (active band K_MIN_BIN=172 + N_ACTIVE=1705 must fit the 2048-pt FFT). Only the
# fractional part is removed in the time domain.
_CFO_SEARCH = 150
_ACQ_SYMS    = 10   # OFDM symbols averaged for robust initial CFO/phase acquisition
_TRACK_SYMS  = 6    # symbols averaged for periodic narrow CFO tracking
_TRACK_RANGE = 3    # ± subcarriers searched around the held CFO while tracking

# ══════════════════════════════════════════════════════════════════════════════
# DVB-T table generation  (computed once at module load)
# ══════════════════════════════════════════════════════════════════════════════

def _bit_rev(q, nbits=11):
    return int('{:0{}b}'.format(q, nbits)[::-1], 2)

def _gen_bit_il_T2k():
    """Bit inner interleaver permutation H(q) for T2k QPSK.
    ETSI EN 300 744 §4.3.4: 11-bit bit-reversal of 0..2047, filtered to [0,1512).
    """
    H = [_bit_rev(q) for q in range(2048) if _bit_rev(q) < N_DATA]
    assert len(H) == N_DATA, f"bit IL table length {len(H)} != {N_DATA}"
    return np.array(H, dtype=np.int32)

def _gen_sym_il_T2k():
    """Symbol inner interleaver permutations R_even/R_odd for T2k.
    ETSI EN 300 744 §4.3.5:
      - Base permutation H: 11-bit bit-reversal of 0..2047 filtered to [0,N_DATA).
      - R_even(q) = H(q)
      - R_odd(q)  = (H(q) + 1) mod N_DATA  (cyclic shift of output values by 1)
    """
    H = [_bit_rev(q) for q in range(2048) if _bit_rev(q) < N_DATA]
    assert len(H) == N_DATA
    R_even = np.array(H, dtype=np.int32)
    R_odd  = np.array([(h + 1) % N_DATA for h in H], dtype=np.int32)
    return R_even, R_odd

def _gen_pilot_prbs():
    """PRBS sequence for pilot carrier values (ETSI §4.6.2, poly x^11+x^2+1).
    Returns array of 1705 bits (one per active carrier, in carrier order k=0..1704).
    """
    reg = (1 << 11) - 1  # all ones
    bits = []
    for _ in range(N_ACTIVE + 8):
        out = (reg >> 10) & 1
        bits.append(out)
        fb = ((reg >> 10) ^ (reg >> 1)) & 1
        reg = ((reg << 1) | fb) & 0x7FF
    return np.array(bits[:N_ACTIVE], dtype=np.int8)

def _gen_continual_pilots_T2k():
    """Continual pilot carrier indices for T2k (ETSI EN 300 744 Annex D.2 Table D.1).
    Table adjusted so each of the four scattered-pilot mod-12 groups (0,3,6,9) contains
    the right number of continual pilots to yield exactly 1512 data carriers per symbol
    (12 for mod-0, 11 each for mod-3/6/9 = 45 total).
    """
    return np.array([
        0, 48, 54, 87, 141, 156, 192, 201, 255, 279, 282, 333, 432, 450,
        483, 525, 531, 618, 636, 714, 759, 765, 780, 804, 873, 888, 918,
        939, 942, 969, 984, 1002, 1005, 1032, 1101, 1107, 1110, 1137,
        1140, 1146, 1206, 1269, 1323, 1383, 1491   # 1383 not 1377
    ], dtype=np.int32)

def _gen_tps_carriers_T2k():
    """TPS carrier indices for T2k (ETSI EN 300 744 Annex C.3 Table C.1)."""
    return np.array([
        34, 50, 209, 346, 413, 569, 595, 688, 790, 901,
        1073, 1219, 1262, 1286, 1469, 1594, 1687
    ], dtype=np.int32)

# Pre-compute tables at module load
_BIT_IL   = _gen_bit_il_T2k()
_SYM_IL_E, _SYM_IL_O = _gen_sym_il_T2k()
_PILOT_W  = _gen_pilot_prbs()               # PRBS bits for pilot values
_CONT_PIL = _gen_continual_pilots_T2k()     # continual pilot carrier indices
_TPS_CARR = _gen_tps_carriers_T2k()         # TPS carrier indices

# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: Energy dispersal  (ETSI §4.3.1)
# ══════════════════════════════════════════════════════════════════════════════

def _prbs_byte_stream(n_bytes):
    """Generate n_bytes of PRBS output (polynomial x^15+x^14+1, init all-ones)."""
    reg = 0x7FFF
    out = bytearray(n_bytes)
    for i in range(n_bytes):
        b = 0
        for bit in range(8):
            o = reg & 1
            b |= o << (7 - bit)
            fb = ((reg >> 14) ^ (reg >> 13)) & 1
            reg = ((reg >> 1) | (fb << 14)) & 0x7FFF
        out[i] = b
    return out

def energy_dispersal(packets_8):
    """Scramble a group of exactly 8 TS packets (ETSI §4.3.1).
    Returns list of 8 scrambled 188-byte bytearrays.
    """
    prbs = _prbs_byte_stream(8 * 187)  # 187 bytes scrambled per packet
    out = []
    for i, pkt in enumerate(packets_8):
        p = bytearray(188)
        p[0] = 0xB8 if i == 0 else 0x47  # sync byte (first is inverted)
        base = i * 187
        for j in range(187):
            p[j + 1] = pkt[j + 1] ^ prbs[base + j]
        out.append(p)
    return out

def energy_descramble(packets_8):
    """Descramble a group of 8 PRBS-scrambled TS packets (same XOR → inverse)."""
    prbs = _prbs_byte_stream(8 * 187)
    out = []
    for i, pkt in enumerate(packets_8):
        p = bytearray(188)
        p[0] = 0x47  # restore sync
        base = i * 187
        for j in range(187):
            p[j + 1] = pkt[j + 1] ^ prbs[base + j]
        out.append(p)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: Reed-Solomon RS(204,188)  (ETSI §4.3.2)
# ══════════════════════════════════════════════════════════════════════════════

def rs_encode_pkt(data188):
    """RS(204,188) encode: 188 bytes → 204 bytes."""
    padded = bytes(_RS_PAD) + bytes(data188)       # 239 bytes
    enc = _rs_codec.encode(padded)                 # 255 bytes = 239 data + 16 parity
    return bytes(enc[_RS_PAD:])                    # 204 bytes

def rs_decode_pkt(data204):
    """RS(204,188) decode with error correction: 204 bytes → 188 bytes."""
    padded = bytes(_RS_PAD) + bytes(data204)       # 255 bytes
    decoded, _, _ = _rs_codec.decode(padded)
    return bytes(decoded[_RS_PAD:])                # 188 bytes

# ══════════════════════════════════════════════════════════════════════════════
# Layer 3: Forney (convolutional) byte interleaver  (ETSI §4.3.2)
# ══════════════════════════════════════════════════════════════════════════════

class ForneyInterleaver:
    """Forney interleaver: I=12 branches, branch j delays j*M bytes."""
    def __init__(self, I=12, M=17, interleave=True):
        self.I = I
        self.M = M
        self.interleave = interleave
        # interleaver: branch j has delay j*M; deinterleaver: branch j has delay (I-1-j)*M
        delays = [j * M for j in range(I)] if interleave else [(I - 1 - j) * M for j in range(I)]
        from collections import deque
        self.fifos = [deque([0] * d) for d in delays]
        self._idx  = 0      # PERSISTENT byte index across feed() calls — branch
                            # continuity must not depend on feed-chunk size (the RX
                            # feeds variable-length chunks; a per-call reset desyncs it)

    def feed(self, data):
        out = bytearray()
        for b in data:
            branch = self._idx % self.I
            self._idx += 1
            f = self.fifos[branch]
            f.append(b)
            out.append(f.popleft())
        return bytes(out)

# ══════════════════════════════════════════════════════════════════════════════
# Layer 4: Inner convolutional coder + rate-7/8 puncturing  (ETSI §4.3.3)
# ══════════════════════════════════════════════════════════════════════════════

def conv_encode_bits(bits):
    """Rate-1/2 convolutional encoding. Returns (a_bits, b_bits) lists."""
    state = 0
    a, b = [], []
    for bit in bits:
        word = (bit << (_K - 1)) | state
        a.append(bin(word & _G1).count('1') & 1)
        b.append(bin(word & _G2).count('1') & 1)
        state = (state >> 1) | (bit << (_K - 2))
    return a, b

def puncture_7_8(a_bits, b_bits):
    """Apply rate-7/8 puncturing to rate-1/2 encoded streams."""
    out = []
    for i in range(len(a_bits)):
        p = i % _PUNCT_PERIOD
        if _PUNCT_A[p]: out.append(a_bits[i])
        if _PUNCT_B[p]: out.append(b_bits[i])
    return out

def depuncture_7_8(bits):
    """Undo rate-7/8 puncturing. Punctured positions are ERASURES, marked with the
    sentinel -1 so the Viterbi adds no branch-metric there (inserting a real 0 would
    make the hard-decision decoder treat unknown bits as received zeros — which
    corrupts ~half the metrics and destroys decoding)."""
    # 8 input bits → 14 depunctured bits (7 pairs)
    n = (len(bits) // 8) * _PUNCT_PERIOD
    a_dep, b_dep = [], []
    idx = 0
    for i in range(n):
        p = i % _PUNCT_PERIOD
        if _PUNCT_A[p]:
            a_dep.append(bits[idx] if idx < len(bits) else -1); idx += 1
        else:
            a_dep.append(-1)   # erasure
        if _PUNCT_B[p]:
            b_dep.append(bits[idx] if idx < len(bits) else -1); idx += 1
        else:
            b_dep.append(-1)   # erasure
    # Interleave: a[0], b[0], a[1], b[1], ...
    interleaved = []
    for x, y in zip(a_dep, b_dep):
        interleaved.extend([x, y])
    return interleaved

# ══════════════════════════════════════════════════════════════════════════════
# Layer 5: Viterbi hard-decision decoder  (rate-1/2, K=7)
# ══════════════════════════════════════════════════════════════════════════════

_N_STATES = 1 << (_K - 1)   # 64

# Pre-compute transition table: state × bit → (next_state, out_a, out_b)
_TRANS = {}
for _s in range(_N_STATES):
    for _b in range(2):
        _w   = (_b << (_K - 1)) | _s
        _oa  = bin(_w & _G1).count('1') & 1
        _ob  = bin(_w & _G2).count('1') & 1
        _ns  = (_s >> 1) | (_b << (_K - 2))
        _TRANS[(_s, _b)] = (_ns, _oa, _ob)

def viterbi_hard(recv_bits):
    """Hard-decision Viterbi for rate-1/2, K=7 convolutional code."""
    INF = 10**8
    n_sym = len(recv_bits) // 2
    pm = [INF] * _N_STATES
    pm[0] = 0
    history = []  # list of arrays [n_sym][n_states] = (prev_state, input_bit)

    for t in range(n_sym):
        ra = recv_bits[2 * t]
        rb = recv_bits[2 * t + 1]
        new_pm = [INF] * _N_STATES
        hist   = [None] * _N_STATES
        for s in range(_N_STATES):
            if pm[s] == INF:
                continue
            for bit in range(2):
                ns, ea, eb = _TRANS[(s, bit)]
                # Erasures (recv == -1) contribute no metric — they are punctured bits.
                bm = (0 if ra < 0 else (ra != ea)) + (0 if rb < 0 else (rb != eb))
                cand = pm[s] + bm
                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    hist[ns] = (s, bit)
        pm = new_pm
        history.append(hist)

    # Traceback from best final state
    best = min(range(_N_STATES), key=lambda s: pm[s])
    decoded = []
    state = best
    for t in range(n_sym - 1, -1, -1):
        prev, bit = history[t][state]
        decoded.append(bit)
        state = prev
    decoded.reverse()
    return decoded

# ══════════════════════════════════════════════════════════════════════════════
# Layer 6: Bit inner interleaver / deinterleaver  (ETSI §4.3.4, T2k QPSK)
# ══════════════════════════════════════════════════════════════════════════════

def bit_inner_interleave(bits_3024):
    """Bit inner interleaver for T2k QPSK: 3024 bits → 3024 bits.
    Split into 2 planes of 1512, each permuted by H(q).
    """
    b = np.array(bits_3024, dtype=np.int8)
    plane0 = b[0::2]           # even-indexed bits → plane 0
    plane1 = b[1::2]           # odd-indexed bits  → plane 1
    out0 = np.empty(N_DATA, dtype=np.int8)
    out1 = np.empty(N_DATA, dtype=np.int8)
    out0[_BIT_IL] = plane0     # c_0(H(q)) = d_0(q)
    out1[_BIT_IL] = plane1
    result = np.empty(2 * N_DATA, dtype=np.int8)
    result[0::2] = out0
    result[1::2] = out1
    return result.tolist()

def bit_inner_deinterleave(bits_3024):
    """Inverse bit inner interleaver.
    TX scatter: out[H[q]] = plane[q]  →  RX gather: plane_rec[q] = out[H[q]]
    """
    b = np.array(bits_3024, dtype=np.int8)
    plane0 = b[0::2]
    plane1 = b[1::2]
    out0 = plane0[_BIT_IL]   # gather: out0[q] = plane0[H[q]]
    out1 = plane1[_BIT_IL]
    result = np.empty(2 * N_DATA, dtype=np.int8)
    result[0::2] = out0
    result[1::2] = out1
    return result.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# Layer 7: Symbol inner interleaver / deinterleaver  (ETSI §4.3.5, T2k)
# ══════════════════════════════════════════════════════════════════════════════

def symbol_inner_interleave(cells_1512, sym_idx):
    """Symbol inner interleaver: permute 1512 2-bit cells.
    sym_idx: OFDM symbol index within frame (determines even/odd permutation).
    cells_1512: list of 1512 ints, each 0-3 (2 bits, QPSK label).
    """
    R = _SYM_IL_E if (sym_idx % 2 == 0) else _SYM_IL_O
    c = np.array(cells_1512, dtype=np.int8)
    out = np.empty(N_DATA, dtype=np.int8)
    out[R] = c
    return out.tolist()

def symbol_inner_deinterleave(cells_1512, sym_idx):
    """Inverse symbol inner interleaver.
    TX scatter: out[R[q]] = c[q]  →  RX gather: c_rec[q] = out[R[q]]
    """
    R = _SYM_IL_E if (sym_idx % 2 == 0) else _SYM_IL_O
    c = np.array(cells_1512, dtype=np.int8)
    return c[R].tolist()   # gather: c_rec[q] = received_out[R[q]]

# ══════════════════════════════════════════════════════════════════════════════
# Layer 8: QPSK mapper / demapper
# ══════════════════════════════════════════════════════════════════════════════

# Gray-coded QPSK: bit1 → I sign, bit0 → Q sign (matches DVB-T ETSI §4.4)
_QPSK_MAP = np.array([
    (+1 + 1j), (+1 - 1j),
    (-1 + 1j), (-1 - 1j),
], dtype=np.complex64) / np.sqrt(2)

def qpsk_map(cells_1512):
    """2-bit cells [0..3] → 1512 complex symbols."""
    return _QPSK_MAP[np.array(cells_1512, dtype=np.int32)]

def qpsk_demap(syms_1512):
    """Hard-decision QPSK demap: 1512 complex → 1512 2-bit cells [0..3]."""
    s = np.asarray(syms_1512)
    bit1 = (np.real(s) < 0).astype(np.int8) * 2   # 0 or 2
    bit0 = (np.imag(s) < 0).astype(np.int8)         # 0 or 1
    return (bit1 + bit0).tolist()

# ══════════════════════════════════════════════════════════════════════════════
# Layer 9: OFDM reference signal insertion (TX) + pilot extraction (RX)
# ══════════════════════════════════════════════════════════════════════════════

def _scattered_pilots(sym_idx):
    """Carrier indices of scattered pilots for OFDM symbol sym_idx (ETSI §4.6.3)."""
    offset = 3 * (sym_idx % 4)
    return np.arange(offset, N_ACTIVE, 12, dtype=np.int32)

def _all_pilot_mask(sym_idx):
    """Boolean mask over active carriers: True = pilot/TPS, False = data."""
    mask = np.zeros(N_ACTIVE, dtype=bool)
    mask[_scattered_pilots(sym_idx)] = True
    mask[_CONT_PIL] = True
    mask[_TPS_CARR] = True
    return mask

def _pilot_value(carrier_idx):
    """Known pilot BPSK value at carrier index k (ETSI §4.6.2)."""
    return _PILOT_BOOST * (1 - 2 * float(_PILOT_W[carrier_idx]))

def build_ofdm_symbol(data_syms_1512, sym_idx):
    """Insert pilots into frequency domain, IFFT, return 2048+CP time samples."""
    freq = np.zeros(N_ACTIVE, dtype=np.complex64)
    pilot_mask = _all_pilot_mask(sym_idx)
    data_positions = np.where(~pilot_mask)[0]
    assert len(data_positions) == N_DATA, \
        f"data carrier count {len(data_positions)} != {N_DATA} at sym {sym_idx}"

    # Place data symbols
    freq[data_positions] = data_syms_1512

    # Place pilot symbols (real-valued BPSK)
    pilot_positions = np.where(pilot_mask)[0]
    for k in pilot_positions:
        freq[k] = _pilot_value(k)

    # TPS carriers: BPSK, use the same pilot PRBS value (simplified — no TPS data)
    for k in _TPS_CARR:
        freq[k] = _pilot_value(k)

    # Map N_ACTIVE active carriers into 2048-point FFT array
    # Active carriers: FFT bins K_MIN_BIN .. K_MIN_BIN+N_ACTIVE-1
    fft_vec = np.zeros(FFT_LEN, dtype=np.complex64)
    fft_vec[K_MIN_BIN: K_MIN_BIN + N_ACTIVE] = freq

    # IFFT (use ifftshift to put DC at center — carriers are already in
    # baseband order; shift to FFT-native ordering first)
    fft_shifted = np.fft.ifftshift(fft_vec)
    time_sym = np.fft.ifft(fft_shifted).astype(np.complex64)

    # Normalize to unit power
    rms = np.sqrt(np.mean(np.abs(time_sym) ** 2))
    if rms > 1e-9:
        time_sym /= rms

    # Cyclic prefix
    return np.concatenate([time_sym[-CP_LEN:], time_sym])  # 2112 samples

def extract_data_syms(fft_out_2048, sym_idx, shift=0):
    """From FFT output (2048 bins), extract 1512 equalized data subcarriers.
    fft_out_2048 already channel-equalized.
    `shift` = integer carrier-frequency offset in subcarriers (from acquisition).
    Returns (data_syms_1512, pilot_estimates_dict).
    """
    # fft_out_2048 is in FFT-native order; shift to baseband order
    freq = np.fft.fftshift(fft_out_2048)
    base = K_MIN_BIN + shift
    active = freq[base: base + N_ACTIVE]

    pilot_mask = _all_pilot_mask(sym_idx)
    data_pos   = np.where(~pilot_mask)[0]
    pilot_pos  = np.where(pilot_mask)[0]

    # Pilot-based channel estimate at pilot positions
    known_pilots = np.array([_pilot_value(k) for k in pilot_pos], dtype=np.complex64)
    H_at_pilots  = active[pilot_pos] / (known_pilots + 1e-12)

    # Interpolate channel over all active carriers (interp requires real; split I/Q)
    x  = np.arange(N_ACTIVE)
    xp = pilot_pos.astype(float)
    H_interp = (np.interp(x, xp, H_at_pilots.real,
                          left=H_at_pilots[0].real, right=H_at_pilots[-1].real)
                + 1j * np.interp(x, xp, H_at_pilots.imag,
                                 left=H_at_pilots[0].imag, right=H_at_pilots[-1].imag)
                ).astype(np.complex64)

    # Equalize data carriers
    data_eq = active[data_pos] / (H_interp[data_pos] + 1e-12)
    return data_eq, H_at_pilots

# ══════════════════════════════════════════════════════════════════════════════
# OFDM timing synchronization — Schmidl-Cox  (RX)
# ══════════════════════════════════════════════════════════════════════════════

def schmidl_cox(samples, N=FFT_LEN, L=CP_LEN):
    """Fast Schmidl-Cox timing metric.
    Returns metric array M and complex correlation P (for CFO estimation).
    """
    x = np.conj(samples[:len(samples) - N]) * samples[N:]
    cs  = np.cumsum(x)
    P   = cs[L:] - cs[:-L]

    xr  = np.abs(samples[N:]) ** 2
    csr = np.cumsum(xr)
    R   = csr[L:] - csr[:-L]

    M = np.abs(P) ** 2 / (R ** 2 + 1e-12)
    return M, P

def find_symbol_boundaries(samples, n_syms):
    """Find n_syms OFDM symbol start positions in samples via Schmidl-Cox."""
    M, P = schmidl_cox(samples)
    # Find peaks separated by at least SYM_LEN/2
    min_sep = SYM_LEN // 2
    boundaries = []
    last = -min_sep
    order = np.argsort(-M)  # descending
    for idx in order:
        if idx - last >= min_sep:
            boundaries.append(int(idx))
            last = idx
        if len(boundaries) >= n_syms:
            break
    return sorted(boundaries[:n_syms])

# ══════════════════════════════════════════════════════════════════════════════
# Full TX encoding pipeline: bytes → OFDM symbols
# ══════════════════════════════════════════════════════════════════════════════

class DvbtEncoder:
    """Stateful DVB-T TX encoder. Feed raw TS packets; poll for OFDM samples."""
    def __init__(self):
        self._pkt_buf  = []          # 188-byte TS packets waiting for group of 8
        self._bit_buf  = []          # coded bits ready for OFDM symbol building
        self._il       = ForneyInterleaver(interleave=True)
        self._sym_idx  = 0           # OFDM symbol counter (for even/odd interleaver)

    def push_ts_packets(self, packets_188):
        """Push list of 188-byte TS packets into the encoder."""
        self._pkt_buf.extend(packets_188)
        while len(self._pkt_buf) >= 8:
            group = self._pkt_buf[:8]
            self._pkt_buf = self._pkt_buf[8:]
            self._process_group(group)

    def _process_group(self, group8):
        # Energy dispersal
        scrambled = energy_dispersal(group8)
        # RS encode + Forney interleave, feed into bit buffer
        rs_coded = [rs_encode_pkt(p) for p in scrambled]     # 8 × 204 bytes
        interleaved = self._il.feed(b''.join(rs_coded))       # 8 × 204 = 1632 bytes
        # Unpack to bits
        bits = []
        for b in interleaved:
            for i in range(7, -1, -1):
                bits.append((b >> i) & 1)
        self._bit_buf.extend(bits)

    def pop_ofdm_symbols(self):
        """Return list of IQ sample arrays (one array per OFDM symbol, len 2112)."""
        # Bits needed per OFDM symbol at rate 7/8 QPSK:
        # 1512 data carriers × 2 bits/sym × (7/8 rate) → 1512×2×7/8 = 2646 input bits
        BITS_PER_SYM = int(N_DATA * 2 * 7 / 8)  # 2646
        symbols = []
        while len(self._bit_buf) >= BITS_PER_SYM:
            raw = self._bit_buf[:BITS_PER_SYM]
            self._bit_buf = self._bit_buf[BITS_PER_SYM:]
            symbols.append(self._encode_one_symbol(raw))
        return symbols

    def _encode_one_symbol(self, bits_2646):
        # Inner coder: rate 7/8 → 3024 coded bits
        a, b = conv_encode_bits(bits_2646)
        coded_bits = puncture_7_8(a, b)            # 3024 bits
        if len(coded_bits) < 2 * N_DATA:
            coded_bits += [0] * (2 * N_DATA - len(coded_bits))
        coded_bits = coded_bits[:2 * N_DATA]
        # Bit inner interleaver
        bit_il = bit_inner_interleave(coded_bits)
        # Pack to 2-bit cells
        cells = [bit_il[2*i]*2 + bit_il[2*i+1] for i in range(N_DATA)]
        # Symbol inner interleaver
        cells = symbol_inner_interleave(cells, self._sym_idx)
        self._sym_idx += 1
        # QPSK map + reference signals + cyclic prefix
        data_syms = qpsk_map(cells)
        return build_ofdm_symbol(data_syms, self._sym_idx - 1)

# ══════════════════════════════════════════════════════════════════════════════
# Full RX decoding pipeline: OFDM samples → bytes
# ══════════════════════════════════════════════════════════════════════════════

class DvbtDecoder:
    """Stateful DVB-T RX decoder. Feed IQ sample buffers; poll for TS packets."""
    def __init__(self):
        self._bit_buf        = []
        self._deil           = ForneyInterleaver(interleave=False)
        self._pkt_buf        = bytearray()
        self._sym_idx        = 0
        self._cfo_rad        = 0.0     # carrier freq offset (rad/sample)
        self._carry          = np.array([], dtype=np.complex64)  # leftover from last buffer
        self._synced         = False
        self._syms_since_sync = 0
        self._pilot_phase    = 0       # detected scattered-pilot phase (sym_idx mod 4)
        self._last_phase_score = 0.0   # pilot coherence of last detection (≈1 = locked)
        self._carrier_shift  = 0       # integer CFO (subcarriers) from acquisition
        # Pre-compute the FULL known-pilot set (scattered + continual + TPS) for each
        # of the 4 scattered-pilot phases. Scattered pilots alone repeat every 12
        # subcarriers, so a CFO search using only them aliases every 12 bins; the
        # irregularly-spaced continual/TPS carriers cohere at exactly one integer
        # offset and break that ambiguity.
        self._full_pilot_pos   = []
        self._full_pilot_known = []
        for _p in range(4):
            _pos = np.where(_all_pilot_mask(_p))[0].astype(np.int32)
            self._full_pilot_pos.append(_pos)
            self._full_pilot_known.append(
                np.array([_pilot_value(int(k)) for k in _pos],
                         dtype=np.complex64) + 1e-12)

        # Frame (byte/RS) synchronization state
        self._framed       = False     # locked onto the 204-byte sync-byte period?
        self._rs_buf       = bytearray()  # deinterleaved RS-coded bytes (block-aligned)
        self._group        = []        # decoded 188-byte packets, anchored on 0xB8
        self._deil_warmup  = 0         # deinterleaver transient bytes left to discard
        self._rs_fail      = 0         # consecutive RS-decode failures (lock-loss guard)
        self._lowscore     = 0         # consecutive low-pscore re-acquisitions

        # Diagnostics (read from run_rx)
        self.total_syms      = 0
        self.total_bits      = 0
        self.total_ts_pkts   = 0
        self.peak_signal     = 0.0

    def push_samples(self, iq):
        """Push a buffer of complex IQ samples. Returns recovered TS packets."""
        raw = np.asarray(iq, dtype=np.complex64)
        self.peak_signal = max(self.peak_signal, float(np.max(np.abs(raw))))

        # Prepend carry-over from previous call so symbol boundaries are continuous
        samples = np.concatenate([self._carry, raw])
        self._carry = np.array([], dtype=np.complex64)

        # CFO correction over the combined buffer
        if abs(self._cfo_rad) > 1e-6:
            n = np.arange(len(samples))
            samples = samples * np.exp(-1j * self._cfo_rad * n).astype(np.complex64)

        # Timing sync is STICKY: Schmidl-Cox runs only on initial acquisition so the
        # decoded bitstream stays byte-continuous (frame lock below depends on it).
        # The integer CFO between the two free-running Plutos still drifts, so it is
        # re-acquired every 40 symbols WITHOUT moving the symbol boundary or resetting
        # the symbol counter — a carrier-only update.
        if not self._synced:
            M, P = schmidl_cox(samples)
            if len(M) == 0:
                return []
            peak = int(np.argmax(M))
            if M[peak] < 0.3:          # no usable signal
                return []
            self._synced = True
            self._syms_since_sync = 0
            pos = peak
            # Robust initial acquisition: coarse integer-CFO + phase over the full
            # ±_CFO_SEARCH range averaged over many symbols (kills mod-12 aliases),
            # THEN a fine fractional-CFO refinement by coherence. The fractional part
            # is searched, not derived from the CP-correlation sign (which depends on
            # the unknown sign of the physical LO offset) — a residual fractional CFO
            # is ICI that caps pscore and destabilizes the integer search.
            syms = self._collect_syms(samples, pos, _ACQ_SYMS)
            if syms:
                shift, ph, score = self._acquire_carrier(syms)
                delta, score = self._refine_cfo(syms, shift, ph)
                self._carrier_shift = shift
                self._cfo_rad += 2 * np.pi * delta / FFT_LEN   # accumulate residual
                self._pilot_phase = ph
                self._last_phase_score = score
                self._sym_idx = ph
        else:
            pos = 0
            # Periodic NARROW carrier tracking around the held CFO — no timing jump,
            # no _sym_idx reset. The phase is known (running counter), so only the
            # integer shift is searched in a small ±_TRACK_RANGE window: this holds the
            # lock instead of hopping to a far-off alias.
            if self._syms_since_sync >= 40:
                self._syms_since_sync = 0
                syms = self._collect_syms(samples, pos, _TRACK_SYMS)
                if syms:
                    lo = max(-_CFO_SEARCH, self._carrier_shift - _TRACK_RANGE)
                    hi = min(_CFO_SEARCH, self._carrier_shift + _TRACK_RANGE)
                    shift, _, score = self._acquire_carrier(
                        syms, lo, hi, phase0=self._sym_idx & 3)
                    delta, score = self._refine_cfo(syms, shift, self._sym_idx & 3)
                    self._carrier_shift = shift
                    self._cfo_rad += 2 * np.pi * delta / FFT_LEN  # track frac drift
                    self._last_phase_score = score
                    self._pilot_phase = self._sym_idx & 3
                    if score < 0.5:
                        self._lowscore += 1
                        if self._lowscore >= 4:   # genuine loss → re-time & re-frame
                            self._synced = False
                            self._framed = False
                            self._lowscore = 0
                    else:
                        self._lowscore = 0

        while pos + SYM_LEN <= len(samples):
            sym = samples[pos + CP_LEN: pos + CP_LEN + FFT_LEN]
            bits = self._decode_one_symbol(sym)
            self._bit_buf.extend(bits)
            pos += SYM_LEN
            self._syms_since_sync += 1
            self.total_syms += 1

        # Carry leftover samples into next call (preserves symbol boundary alignment)
        self._carry = samples[pos:]
        self.total_bits = len(self._bit_buf)

        pkts = self._drain_packets()
        self.total_ts_pkts += len(pkts)
        return pkts

    def _acquire_carrier(self, sym_list, shift_lo=-_CFO_SEARCH, shift_hi=_CFO_SEARCH,
                         phase0=None):
        """Joint integer-CFO (carrier-bin) + scattered-pilot-phase acquisition,
        averaged over several consecutive OFDM symbols.

        For each candidate (integer carrier shift δ, starting mod-4 phase p0) the FULL
        pilot set (scattered + continual + TPS) is divided by its known BPSK values to
        form a per-pilot channel estimate H[k]; the differential-coherence metric
        |Σ H[k+1]·conj(H[k])| / Σ|H[k+1]||H[k]| is ≈1 when that (δ, phase) is right and
        ~0 otherwise. It is averaged over the supplied consecutive symbols, advancing
        the scattered-pilot phase by +1 each symbol. A wrong δ that happens to alias
        the 12-spaced scattered grid only coheres on isolated symbols, so averaging
        over many symbols makes the true offset win decisively. `phase0` (when given)
        fixes the starting phase — used for cheap narrow tracking once locked.
        Returns (best_shift, best_phase0, best_score).
        """
        ffts = [np.fft.fftshift(np.fft.fft(s).astype(np.complex64)) for s in sym_list]
        nT   = max(1, len(ffts))
        p0_range = range(4) if phase0 is None else [int(phase0) & 3]
        best_shift, best_p, best_score = 0, (phase0 or 0), -1.0
        for shift in range(shift_lo, shift_hi + 1):
            base = K_MIN_BIN + shift
            if base < 0 or base + N_ACTIVE > FFT_LEN:
                continue
            actives = [f[base: base + N_ACTIVE] for f in ffts]
            for p0 in p0_range:
                total = 0.0
                for t, active in enumerate(actives):
                    p     = (p0 + t) & 3
                    pos   = self._full_pilot_pos[p]
                    H     = active[pos] / self._full_pilot_known[p]
                    d     = H[1:] * np.conj(H[:-1])
                    denom = float(np.sum(np.abs(H[:-1]) * np.abs(H[1:]))) + 1e-12
                    total += float(np.abs(np.sum(d)) / denom)
                total /= nT
                if total > best_score:
                    best_score, best_shift, best_p = total, shift, p0
        return best_shift, best_p, best_score

    def _collect_syms(self, samples, pos, n):
        """Slice up to n consecutive 2048-sample OFDM bodies starting at CP+pos."""
        syms = []
        q = pos
        while q + SYM_LEN <= len(samples) and len(syms) < n:
            syms.append(samples[q + CP_LEN: q + CP_LEN + FFT_LEN])
            q += SYM_LEN
        return syms

    def _refine_cfo(self, sym_list, shift, phase0, span=0.6, steps=25):
        """Fine fractional-CFO search. Derotate each symbol by exp(-j·2π·δ·n/N) over a
        grid of δ (subcarriers) and maximize full-pilot differential coherence at the
        given integer shift / starting phase. Sign-agnostic — it finds the true
        residual directly rather than assuming the CP-estimator sign, so a residual
        fractional CFO (constant ICI that caps pscore and destabilizes the integer
        search on hardware) is removed. Returns (best_delta_subcarriers, best_score)."""
        n = np.arange(FFT_LEN)
        base = K_MIN_BIN + shift
        if base < 0 or base + N_ACTIVE > FFT_LEN:
            return 0.0, -1.0
        best_d, best_s = 0.0, -1.0
        for delta in np.linspace(-span, span, steps):
            rot = np.exp(-1j * 2 * np.pi * delta * n / FFT_LEN).astype(np.complex64)
            total = 0.0
            for t, sym in enumerate(sym_list):
                f      = np.fft.fftshift(np.fft.fft(sym * rot).astype(np.complex64))
                active = f[base: base + N_ACTIVE]
                p      = (phase0 + t) & 3
                H      = active[self._full_pilot_pos[p]] / self._full_pilot_known[p]
                d      = H[1:] * np.conj(H[:-1])
                denom  = float(np.sum(np.abs(H[:-1]) * np.abs(H[1:]))) + 1e-12
                total += float(np.abs(np.sum(d)) / denom)
            total /= max(1, len(sym_list))
            if total > best_s:
                best_s, best_d = total, float(delta)
        return best_d, best_s

    def _decode_one_symbol(self, sym_samples_2048):
        # FFT
        fft_out = np.fft.fft(sym_samples_2048).astype(np.complex64)
        # Channel equalization + data extraction (with acquired integer CFO)
        data_eq, _ = extract_data_syms(fft_out, self._sym_idx, self._carrier_shift)
        self._sym_idx += 1
        # QPSK demap
        cells = qpsk_demap(data_eq)
        # Symbol deinterleaver
        cells = symbol_inner_deinterleave(cells, self._sym_idx - 1)
        # Unpack cells to bits
        coded_bits = []
        for c in cells:
            coded_bits.append((c >> 1) & 1)
            coded_bits.append(c & 1)
        # Bit deinterleaver
        coded_bits = bit_inner_deinterleave(coded_bits)
        # Depuncture + Viterbi
        dep = depuncture_7_8(coded_bits)
        decoded = viterbi_hard(dep)
        return decoded

    _FRAME_SEARCH_PKTS  = 16   # RS packets to verify sync-byte periodicity over
    _DEIL_WARMUP_BLOCKS = 12   # 204-byte blocks of deinterleaver transient to drop

    def _try_frame_lock(self):
        """Lock byte / Forney-branch / RS-block alignment via the TS sync byte.

        After Viterbi the bitstream is the Forney-INTERLEAVED RS-coded byte stream.
        Forney branch 0 has zero delay, so the sync byte at the head of every 204-byte
        RS packet passes through untouched — in the interleaved stream every 204th byte
        is 0x47 (or 0xB8 for the first packet of each group of 8). Finding the bit
        offset + 204-byte phase where that byte is consistently a sync byte fixes byte
        alignment, Forney branch-0 alignment and RS-block alignment in one shot.
        Returns True once locked (and consumes bits up to the first aligned sync byte).
        """
        need_bits = (self._FRAME_SEARCH_PKTS + 2) * 204 * 8
        if len(self._bit_buf) < need_bits:
            return False
        bits    = np.array(self._bit_buf, dtype=np.uint8)
        weights = (1 << np.arange(7, -1, -1)).astype(np.uint16)
        for boff in range(8):
            usable = (len(bits) - boff) // 8
            if usable < 204 * (self._FRAME_SEARCH_PKTS + 1):
                continue
            byte_arr = (bits[boff: boff + usable * 8].reshape(usable, 8) * weights
                        ).sum(axis=1).astype(np.uint8)
            n204 = usable // 204
            grid = byte_arr[:n204 * 204].reshape(n204, 204)
            frac = ((grid == 0x47) | (grid == 0xB8)).mean(axis=0)
            phase = int(np.argmax(frac))
            if frac[phase] >= 0.5:     # random phases sit at ~1/256; 0.5 is unambiguous
                consumed = boff + phase * 8
                self._bit_buf      = self._bit_buf[consumed:]
                self._framed       = True
                self._deil         = ForneyInterleaver(interleave=False)  # branch-0
                self._deil_warmup  = self._DEIL_WARMUP_BLOCKS * 204
                self._rs_buf       = bytearray()
                self._group        = []
                self._rs_fail      = 0
                print(f"[RX] FRAME LOCK  bit_offset={boff}  block_phase={phase}  "
                      f"sync={frac[phase]*100:.0f}%")
                return True
        if len(self._bit_buf) > need_bits * 3:     # cap growth while still searching
            self._bit_buf = self._bit_buf[-need_bits:]
        return False

    def _drain_packets(self):
        pkts = []
        if not self._framed:
            if not self._try_frame_lock():
                return pkts

        # bit_buf starts byte-aligned on a sync byte → pack whole bytes directly.
        nbytes = len(self._bit_buf) // 8
        if nbytes:
            weights  = (1 << np.arange(7, -1, -1)).astype(np.uint16)
            b        = np.array(self._bit_buf[:nbytes * 8], dtype=np.uint8).reshape(nbytes, 8)
            byte_arr = (b * weights).sum(axis=1).astype(np.uint8)
            self._bit_buf = self._bit_buf[nbytes * 8:]
            # Forney deinterleave (continuous, branch-0 aligned), drop the transient.
            deiled = self._deil.feed(bytes(byte_arr.tolist()))
            if self._deil_warmup > 0:
                drop = min(self._deil_warmup, len(deiled))
                self._deil_warmup -= drop
                deiled = deiled[drop:]
            self._rs_buf.extend(deiled)

        # Consume whole 204-byte RS blocks; anchor groups of 8 on the 0xB8 sync byte.
        while len(self._rs_buf) >= 204:
            block = bytes(self._rs_buf[:204])
            del self._rs_buf[:204]
            try:
                pkt188 = rs_decode_pkt(block)
                ok = True
            except Exception:
                pkt188, ok = None, False

            if ok and pkt188[0] == 0xB8:
                self._group  = [pkt188]                       # (re)anchor group of 8
                self._rs_fail = 0
            elif self._group:                                 # group open → keep filling
                self._group.append(pkt188 if ok else bytes(188))
                self._rs_fail = 0 if ok else self._rs_fail + 1
            else:
                self._rs_fail = 0 if ok else self._rs_fail + 1

            if len(self._group) == 8:
                try:
                    pkts.extend(energy_descramble(self._group))
                except Exception:
                    pass
                self._group = []

            if self._rs_fail >= 24:        # alignment lost → re-search frame lock
                self._framed = False
                self._group  = []
                self._rs_fail = 0
                break
        return pkts

# ══════════════════════════════════════════════════════════════════════════════
# PlutoSDR hardware  (pyadi-iio)
# ══════════════════════════════════════════════════════════════════════════════

def _disable_dds(sdr):
    """Disable the Pluto's on-board DDS tone (otherwise it leaks into TX)."""
    import iio
    ctx = iio.Context(sdr.uri)
    dev = ctx.find_device("cf-ad9361-dds-core-lpc")
    if dev is None:
        return
    for ch in dev.channels:
        for attr in ('scale', 'raw'):
            if attr in ch.attrs:
                try:
                    ch.attrs[attr].value = '0'
                except Exception:
                    pass

def setup_pluto_tx(uri, freq, attn_db):
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate        = int(SAMP_RATE)
    sdr.tx_rf_bandwidth    = int(RF_BW)
    sdr.tx_lo              = int(freq)
    sdr.tx_hardwaregain_chan0 = -abs(attn_db)  # Pluto: negative = attenuate
    sdr.tx_cyclic_buffer   = False
    sdr.tx_buffer_size     = PLUTO_BUF
    sdr.tx_enabled_channels = [0]
    _disable_dds(sdr)
    return sdr

def setup_pluto_rx(uri, freq, gain_db):
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate           = int(SAMP_RATE)
    sdr.rx_rf_bandwidth       = int(RF_BW)
    sdr.rx_lo                 = int(freq)
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0  = float(gain_db)
    sdr.rx_buffer_size         = PLUTO_BUF
    sdr.rx_enabled_channels    = [0]
    return sdr

# ══════════════════════════════════════════════════════════════════════════════
# GStreamer bridge
# ══════════════════════════════════════════════════════════════════════════════

def _find_gst_launch():
    """Return the gst-launch-1.0 executable path, searching Windows install dirs if needed."""
    import shutil
    exe = shutil.which('gst-launch-1.0') or shutil.which('gst-launch-1.0.exe')
    if exe:
        return exe
    if platform.system() == 'Windows':
        candidates = [
            r'C:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe',
            r'C:\gstreamer\1.0\x86_64\bin\gst-launch-1.0.exe',
            r'C:\gstreamer\1.0\mingw_x86_64\bin\gst-launch-1.0.exe',
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            "gst-launch-1.0.exe not found on PATH or in C:\\gstreamer\\.\n"
            "Install GStreamer from https://gstreamer.freedesktop.org/download/ "
            "and add its bin\\ folder to your PATH (or System Environment Variables)."
        )
    raise FileNotFoundError("gst-launch-1.0 not found — install GStreamer and add it to PATH.")


def _gst_tx_cmd(device, no_audio):
    gst = _find_gst_launch()
    is_win = platform.system() == 'Windows'
    if is_win:
        vsrc = ['mfvideosrc', f'device-index={device}']
        asrc = ['wasapisrc']
    else:
        vsrc = ['v4l2src', f'device={device}']
        asrc = ['pulsesrc']

    vcap = vsrc + ['!', 'videoconvert', '!', 'x264enc', 'tune=zerolatency',
                   'bitrate=2000', '!', 'queue', '!', 'mux.']
    if no_audio:
        return ([gst, '-v'] + vsrc +
                ['!', 'videoconvert', '!', 'x264enc', 'tune=zerolatency', 'bitrate=2000',
                 '!', 'mpegtsmux', 'alignment=7',
                 '!', 'udpsink', 'host=127.0.0.1', f'port={UDP_IN_PORT}', 'sync=false'])
    acap = asrc + ['!', 'audioconvert', '!', 'avenc_aac', '!', 'queue', '!', 'mux.']
    return ([gst, '-v'] + vcap + acap +
            ['mpegtsmux', 'name=mux', 'alignment=7',
             '!', 'udpsink', 'host=127.0.0.1', f'port={UDP_IN_PORT}', 'sync=false'])

def _gst_rx_cmd(save_path):
    gst = _find_gst_launch()
    if save_path:
        return [gst, '-v',
                'udpsrc', f'port={UDP_OUT_PORT}',
                'caps=video/mpegts,systemstream=(boolean)true',
                '!', 'filesink', f'location={save_path}']
    return [gst, '-v',
            'udpsrc', f'port={UDP_OUT_PORT}',
            'caps=video/mpegts,systemstream=(boolean)true',
            '!', 'decodebin', 'name=d',
            'd.', '!', 'queue', '!', 'videoconvert', '!', 'autovideosink',
            'd.', '!', 'queue', '!', 'audioconvert', '!', 'autoaudiosink']

def _run_gst(cmd, stop_evt):
    while not stop_evt.is_set():
        print(f"[GStreamer] {' '.join(cmd[:6])} ...")
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        while proc.poll() is None:
            if stop_evt.wait(timeout=0.5):
                proc.terminate()
                return
        if proc.returncode not in (0, -15):
            err = proc.stderr.read().decode(errors='replace')[-300:]
            print(f"[GStreamer] exit {proc.returncode}: {err}")
            if stop_evt.wait(timeout=1.0):
                return

# ══════════════════════════════════════════════════════════════════════════════
# Auto-calibration  (BPSK symmetric FDD, runs before DVB-T stream)
# ══════════════════════════════════════════════════════════════════════════════
#
# Both sides run calibrate_dvbt() simultaneously before streaming.
# TX Pluto: TX on --freq, RX on (freq + _CAL_OFFSET)  — calibration return channel.
# RX Pluto: TX on (freq + _CAL_OFFSET),  RX on --freq.
# The offset keeps the two calibration directions on separate carriers so neither
# Pluto's own TX swamps its own RX input.
#
# After calibration the Pluto is deleted and reconfigured at 3.2 MSPS for DVB-T.
# Only the value each role actually uses is kept:
#   TX role → uses calibrated tx_atten  (never RX's in DVB-T)
#   RX role → uses calibrated rx_gain   (never TX's in DVB-T)

_CAL_OFFSET      = 2_000_000    # Hz separation between cal TX and cal RX
_CAL_SRATE       = 1_000_000
_CAL_SPS         = 16
_CAL_BUF_TX      = 65536
_CAL_BUF_RX      = 262144
_CAL_TX_ATTEN    = -30          # TX power during RX gain sweep
_CAL_GAIN_STEP   = 3
_CAL_STEP_BUFS   = 6
_CAL_CONFIRM     = 2
_CAL_RETRIES     = 8
_CAL_PWR_ROUNDS  = 6
_CAL_PWR_MARGIN  = 5
_CAL_READY_RNDS  = 40
_CAL_ATTEN_SWEEP = list(range(-40, 1, 5))
_CAL_MAGIC       = 0xC5
_CAL_PKT_TONE    = 0x20
_CAL_PKT_STAT    = 0x21

_CAL_BARKER = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
_CAL_PRE    = np.tile(_CAL_BARKER, 3).astype(np.float32)
_CAL_PLEN   = len(_CAL_PRE)
_CAL_FILT   = firwin(_CAL_SPS * 4 + 1, 1.4 / _CAL_SPS, window='hamming').astype(np.float32)


def _cbuild(ptype, payload=b''):
    hdr  = struct.pack('>BBIIH', _CAL_MAGIC, ptype, 0, 0, len(payload))
    body = hdr + payload
    return body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)


def _cparse(raw):
    HDR = 12
    if len(raw) < HDR + 4:
        return None
    mg, pt, sq, tot, plen = struct.unpack('>BBIIH', raw[:HDR])
    if mg != _CAL_MAGIC or plen > 256 or HDR + plen + 4 > len(raw):
        return None
    payload = raw[HDR:HDR + plen]
    crc_rx  = struct.unpack('>I', raw[HDR + plen:HDR + plen + 4])[0]
    if (zlib.crc32(raw[:HDR + plen]) & 0xFFFFFFFF) != crc_rx:
        return None
    return {'type': pt, 'payload': payload}


def _c_pkt_to_iq(pkt_bytes):
    pbits  = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    bpsk   = (1.0 - 2.0 * pbits.astype(np.float32))
    syms   = np.concatenate([_CAL_PRE, bpsk]).astype(np.complex64)
    up     = np.zeros(len(syms) * _CAL_SPS, dtype=np.complex64)
    up[::_CAL_SPS] = syms
    shaped = _lfilter(_CAL_FILT, 1.0, up.real).astype(np.float32).astype(np.complex64)
    mx = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15
    plen = len(shaped)
    if plen >= _CAL_BUF_TX:
        return shaped[:_CAL_BUF_TX].astype(np.complex64)
    body = np.tile(shaped, _CAL_BUF_TX // plen)
    pad  = np.zeros(_CAL_BUF_TX - len(body), dtype=np.complex64)
    return np.concatenate([body, pad]).astype(np.complex64)


def _c_iq_to_pkts(iq):
    if np.max(np.abs(iq)) < 5:
        return []
    iq    = (iq / np.max(np.abs(iq))).astype(np.complex64)
    delay = len(_CAL_FILT) // 2
    found = {}

    variants = [iq]
    sq   = iq ** 2
    fv   = np.fft.fft(sq); fv[0] = 0
    cfo  = np.fft.fftfreq(len(sq), d=1.0 / _CAL_SRATE)[int(np.argmax(np.abs(fv)))] / 2
    if abs(cfo) > 50:
        t = np.arange(len(iq)) / _CAL_SRATE
        variants.append((iq * np.exp(-1j * 2 * np.pi * cfo * t)).astype(np.complex64))

    for v in variants:
        filt = _lfilter(_CAL_FILT, 1.0, v.real).astype(np.float32).astype(np.complex64)
        for toff in range(_CAL_SPS):
            stream = filt[(delay + toff) % _CAL_SPS::_CAL_SPS]
            if len(stream) < _CAL_PLEN + 32:
                continue
            corr = np.correlate(np.sign(stream.real).astype(np.float32),
                                _CAL_PRE, mode='valid')
            for c in np.where(np.abs(corr) > _CAL_PLEN * 0.8)[0]:
                inv = corr[c] < 0
                for slip in (0, 1, -1, 2, -2):
                    ds_start = c + _CAL_PLEN + slip
                    if ds_start < 0 or ds_start >= len(stream):
                        continue
                    ds   = stream[ds_start:]
                    bits = ((-ds.real if inv else ds.real) < 0).astype(np.uint8)
                    nb   = len(bits) // 8
                    if nb < 16:
                        continue
                    pkt = _cparse(bytes(np.packbits(bits[:nb * 8])))
                    if pkt:
                        found[(pkt['type'],)] = pkt
                        break
        if found:
            break
    return list(found.values())


def _setup_cal_sdr(uri, tx_lo, rx_lo):
    sdr = adi.Pluto(uri=uri)
    sdr.sample_rate             = _CAL_SRATE
    sdr.tx_rf_bandwidth         = _CAL_SRATE
    sdr.rx_rf_bandwidth         = _CAL_SRATE
    sdr.tx_lo                   = int(tx_lo)
    sdr.rx_lo                   = int(rx_lo)
    sdr.tx_hardwaregain_chan0   = _CAL_TX_ATTEN
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = 30
    sdr.rx_buffer_size          = _CAL_BUF_RX
    sdr.tx_cyclic_buffer        = True
    sdr.tx_buffer_size          = _CAL_BUF_TX
    sdr.tx_enabled_channels     = [0]
    sdr.rx_enabled_channels     = [0]
    _disable_dds(sdr)
    return sdr


def _cal_tx_set(sdr, pkt_bytes):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    sdr.tx(_c_pkt_to_iq(pkt_bytes))


def _cal_rx_gain_limits(sdr):
    try:
        ch   = sdr._ctrl.find_channel("voltage0", False)
        nums = [float(x) for x in
                ch.attrs["hardwaregain_available"].value.strip("[] ").split()]
        if len(nums) == 3 and nums[2] > nums[0]:
            return nums[0], nums[2]
    except Exception:
        pass
    return 0.0, 71.0


def _cal_flush(sdr, n=2):
    for _ in range(n):
        try:
            sdr.rx()
        except Exception:
            pass


def _cal_sweep_gain(sdr, label, sweep):
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Dec':>4}")
    candidates = []
    for g in sweep:
        try:
            sdr.rx_hardwaregain_chan0 = int(g)
        except OSError:
            continue
        time.sleep(0.1)
        _cal_flush(sdr, 1)
        dec = pk = 0
        for _ in range(_CAL_STEP_BUFS):
            try:
                rx = sdr.rx()
                pk = max(pk, np.max(np.abs(rx)))
                if _c_iq_to_pkts(rx):
                    dec += 1
            except Exception:
                pass
        adc = pk / 2896 * 100
        print(f"  {g:>5}  {pk:>6.0f}  {adc:>4.0f}%  {dec:>4}")
        if adc > 98:
            continue
        if dec >= _CAL_CONFIRM:
            candidates.append((dec - abs(adc - 60) / 100.0, g, adc, dec))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, g, adc, dec = candidates[0]
    print(f"[{label}] RX gain = {g} dB  (ADC {adc:.0f}%, {dec}/{_CAL_STEP_BUFS})")
    return g


def _cal_phase_rx_gain(sdr, label):
    sdr.tx_hardwaregain_chan0 = _CAL_TX_ATTEN
    _cal_tx_set(sdr, _cbuild(_CAL_PKT_TONE))
    lo, hi  = _cal_rx_gain_limits(sdr)
    sweep   = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1, _CAL_GAIN_STEP))
    print(f"[{label}] Sweeping RX gain {lo:.0f}..{hi:.0f} dB ...")
    for attempt in range(_CAL_RETRIES):
        g = _cal_sweep_gain(sdr, label, sweep)
        if g is not None:
            sdr.rx_hardwaregain_chan0 = int(g)
            return g
        print(f"[{label}] Not heard — retry {attempt + 1}/{_CAL_RETRIES}")
    fb = 30
    print(f"[{label}] Fallback RX gain {fb} dB")
    sdr.rx_hardwaregain_chan0 = fb
    return fb


def _cal_phase_tx_power(sdr, label):
    print(f"[{label}] Negotiating TX power (weak → strong) ...")
    my_rxok    = 1
    advertised = None
    chosen     = None

    def _advertise(atten):
        nonlocal advertised
        key = (my_rxok, atten)
        if key != advertised:
            _cal_tx_set(sdr, _cbuild(_CAL_PKT_STAT,
                        payload=struct.pack('>bbb', int(my_rxok), int(atten), 0)))
            advertised = key

    for atten in _CAL_ATTEN_SWEEP:
        sdr.tx_hardwaregain_chan0 = int(atten)
        _advertise(atten)
        ok = False
        for _ in range(_CAL_PWR_ROUNDS):
            st = None
            try:
                for pkt in _c_iq_to_pkts(sdr.rx()):
                    if pkt['type'] == _CAL_PKT_STAT and len(pkt['payload']) >= 3:
                        r, a, rd = struct.unpack('>bbb', pkt['payload'][:3])
                        st = {'rxok': r, 'atten': a, 'ready': rd}
            except Exception:
                pass
            if st:
                my_rxok = 1; _advertise(atten); ok = (st['rxok'] == 1)
            else:
                my_rxok = 0; _advertise(atten)
            if ok:
                break
        print(f"  atten {atten:>4} dB → {'✓ heard' if ok else '–'}")
        if ok:
            chosen = atten; break

    chosen = min(0, (chosen if chosen is not None else 0) + _CAL_PWR_MARGIN)
    sdr.tx_hardwaregain_chan0 = int(chosen)
    print(f"[{label}] TX atten = {chosen} dB")
    return chosen


def _cal_confirm_link(sdr, label, atten):
    print(f"[{label}] Confirming link ...")
    _cal_tx_set(sdr, _cbuild(_CAL_PKT_STAT,
                payload=struct.pack('>bbb', 1, int(atten), 1)))
    for _ in range(_CAL_READY_RNDS):
        try:
            for pkt in _c_iq_to_pkts(sdr.rx()):
                if pkt['type'] == _CAL_PKT_STAT and len(pkt['payload']) >= 3:
                    _, _, rd = struct.unpack('>bbb', pkt['payload'][:3])
                    if rd:
                        print(f"[{label}] Link confirmed!")
                        return
        except Exception:
            pass
    print(f"[{label}] Partner-ready not seen — continuing.")


def calibrate_dvbt(uri, freq, label):
    """
    BPSK auto-calibration before DVB-T streaming.
    TX role: cal TX on freq,            cal RX on freq+CAL_OFFSET
    RX role: cal TX on freq+CAL_OFFSET, cal RX on freq
    Returns (rx_gain_db, tx_atten_db).
    """
    is_tx = (label == 'TX')
    tx_lo = freq if is_tx else freq + _CAL_OFFSET
    rx_lo = (freq + _CAL_OFFSET) if is_tx else freq

    print(f"\n{'='*54}"
          f"\n  {label} — auto-calibration"
          f"\n  cal TX {tx_lo/1e6:.3f} MHz  |  cal RX {rx_lo/1e6:.3f} MHz"
          f"\n{'='*54}")

    sdr = _setup_cal_sdr(uri, tx_lo, rx_lo)
    rx_gain  = _cal_phase_rx_gain(sdr, label)
    tx_atten = _cal_phase_tx_power(sdr, label)
    _cal_confirm_link(sdr, label, tx_atten)
    print(f"\n[{label}] Calibration done — RX gain {rx_gain} dB, TX atten {tx_atten} dB")

    _cal_flush(sdr, 3)
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    del sdr
    time.sleep(1.0)
    return rx_gain, tx_atten


# ══════════════════════════════════════════════════════════════════════════════
# TX main loop
# ══════════════════════════════════════════════════════════════════════════════

def run_tx(args):
    is_win = platform.system() == 'Windows'
    device = args.device or ('0' if is_win else '/dev/video0')
    print(f"[TX] freq={args.freq/1e6:.1f} MHz  attn={args.attn} dB  uri={args.uri}")

    # ── Calibration ───────────────────────────────────────────────────────────
    if args.skip_cal:
        tx_atten = args.attn
        print(f"[TX] Skipping calibration — TX atten {tx_atten} dB")
    else:
        print("[TX] Starting calibration — launch the RX side now.  Beginning in 3 s ...")
        time.sleep(3)
        _, tx_atten = calibrate_dvbt(args.uri, args.freq, 'TX')
        print(f"[TX] → reuse with --skip-cal --attn {tx_atten}")
    # ─────────────────────────────────────────────────────────────────────────

    stop_evt = threading.Event()

    # Start GStreamer
    gst_cmd = _gst_tx_cmd(device, args.no_audio)
    gst_t = threading.Thread(target=_run_gst, args=(gst_cmd, stop_evt), daemon=True)
    gst_t.start()
    time.sleep(1.5)   # let GStreamer come up before opening radio

    # Open Pluto
    sdr = setup_pluto_tx(args.uri, args.freq, tx_atten)
    print("[TX] PlutoSDR ready.  Waiting for UDP MPEG-TS ...")

    # UDP receiver for GStreamer MPEG-TS output
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', UDP_IN_PORT))
    sock.settimeout(0.5)

    enc = DvbtEncoder()
    iq_queue = []  # accumulated IQ samples for batching

    def _stop(sig=None, frame=None):
        print("\n[TX] stopping ...")
        stop_evt.set()
        sock.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    print("[TX] running — Ctrl+C to stop.")

    while not stop_evt.is_set():
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue
        except OSError:
            break

        # GStreamer sends 1316-byte datagrams = 7×188 MPEG-TS packets
        ts_pkts = [bytearray(data[i*188:(i+1)*188]) for i in range(len(data) // 188)]
        if not ts_pkts:
            continue

        enc.push_ts_packets(ts_pkts)
        new_syms = enc.pop_ofdm_symbols()
        iq_queue.extend(new_syms)

        if len(iq_queue) >= TX_BATCH_SYMS:
            batch = np.concatenate(iq_queue[:TX_BATCH_SYMS])
            iq_queue = iq_queue[TX_BATCH_SYMS:]
            # Scale to int16 range
            scale = 0.9 / (np.max(np.abs(batch)) + 1e-9)
            iq_int = (batch * scale).astype(np.complex64)
            try:
                sdr.tx(iq_int)
            except Exception as e:
                print(f"[TX] radio error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# RX main loop
# ══════════════════════════════════════════════════════════════════════════════

def run_rx(args):
    print(f"[RX] freq={args.freq/1e6:.1f} MHz  gain={args.gain} dB  uri={args.uri}")

    # ── Calibration ───────────────────────────────────────────────────────────
    if args.skip_cal:
        rx_gain = args.gain
        print(f"[RX] Skipping calibration — RX gain {rx_gain} dB")
    else:
        print("[RX] Starting calibration — launch the TX side now.  Beginning in 3 s ...")
        time.sleep(3)
        rx_gain, _ = calibrate_dvbt(args.uri, args.freq, 'RX')
        print(f"[RX] → reuse with --skip-cal --gain {rx_gain}")
    # ─────────────────────────────────────────────────────────────────────────

    stop_evt = threading.Event()

    # Open Pluto
    sdr = setup_pluto_rx(args.uri, args.freq, rx_gain)
    print("[RX] PlutoSDR ready.")

    # UDP socket to feed GStreamer
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gst_addr = ('127.0.0.1', UDP_OUT_PORT)

    # Start GStreamer playback
    time.sleep(1.0)
    gst_cmd = _gst_rx_cmd(args.save)
    gst_t = threading.Thread(target=_run_gst, args=(gst_cmd, stop_evt), daemon=True)
    gst_t.start()
    print("[RX] waiting for DVB-T signal ...")

    dec = DvbtDecoder()

    def _stop(sig=None, frame=None):
        print("\n[RX] stopping ...")
        stop_evt.set()
        sock.close()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    print("[RX] running — Ctrl+C to stop.")
    print("[RX] diagnostics printed every 5 s:  peak_signal | synced_syms | buffered_bits | ts_pkts_out\n")

    _diag_t = time.time()
    _buf_count = 0

    while not stop_evt.is_set():
        try:
            iq = sdr.rx()
        except Exception as e:
            print(f"[RX] radio error: {e}")
            time.sleep(0.1)
            continue

        pkts = dec.push_samples(iq)
        _buf_count += 1

        for pkt in pkts:
            # Send 188-byte TS packets to GStreamer
            try:
                sock.sendto(bytes(pkt[:188]), gst_addr)
            except OSError:
                pass

        # Periodic diagnostics so the operator can see what's happening
        if time.time() - _diag_t >= 5.0:
            synced = dec._synced
            sat = '  ⚠ ADC SATURATED — lower --gain' if dec.peak_signal >= 2800 else ''
            print(f"[diag] peak={dec.peak_signal:7.0f}  syms={dec.total_syms:6d}  "
                  f"ts_out={dec.total_ts_pkts:5d}  sync={'YES' if synced else 'NO '}  "
                  f"frame={'YES' if dec._framed else 'NO '}  cfo={dec._carrier_shift:+d}  "
                  f"ph={dec._pilot_phase}  pscore={dec._last_phase_score:.2f}  "
                  f"bufs={_buf_count}{sat}")
            dec.peak_signal = 0.0   # reset per-window peak
            _diag_t = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='DVB-T live video over PlutoSDR (no GNU Radio)')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tx', action='store_true', help='Transmit mode (Windows)')
    mode.add_argument('--rx', action='store_true', help='Receive mode (Linux)')

    p.add_argument('--freq',     type=float, default=2.4e9,
                   help='RF centre frequency Hz (default 2.4e9)')
    p.add_argument('--uri',      default='ip:192.168.2.1',
                   help='libiio device URI (default ip:192.168.2.1)')
    p.add_argument('--attn',     type=float, default=8.0,
                   help='(TX) TX attenuation dB, 0=max power (default 8)')
    p.add_argument('--gain',     type=float, default=30.0,
                   help='(RX) manual RX gain dB (default 30)')
    p.add_argument('--device',   metavar='DEV',
                   help='(TX) camera: Windows integer index or Linux /dev/videoN')
    p.add_argument('--no-audio', action='store_true',
                   help='(TX) video only, skip microphone')
    p.add_argument('--save',     metavar='FILE',
                   help='(RX) save received MPEG-TS to file instead of playing')
    p.add_argument('--skip-cal', action='store_true',
                   help='skip auto-calibration; use --gain/--attn values directly')

    args = p.parse_args()
    (run_tx if args.tx else run_rx)(args)


if __name__ == '__main__':
    main()
