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
_rs_codec   = RSCodec(nsym=_RS_NSYM, nroots=_RS_NSYM, prim=_RS_PRIM,
                      generator=2, fcr=0, c_exp=8)

# Pilot amplitude boost (4/3 relative to unit-power data)
_PILOT_BOOST = 4.0 / 3.0

# UDP port assignments (matching GRC flowgraphs)
UDP_IN_PORT  = 2000   # GStreamer → Python TX
UDP_OUT_PORT = 2001   # Python RX → GStreamer

# Pluto buffer size
PLUTO_BUF = 32768

# Batching: process this many OFDM symbols at once for TX DMA efficiency
TX_BATCH_SYMS = 64

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

    def feed(self, data):
        out = bytearray()
        for idx, b in enumerate(data):
            branch = idx % self.I
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
    """Undo rate-7/8 puncturing: insert 0 (erasure) for punctured positions."""
    # 8 input bits → 14 depunctured bits (7 pairs)
    n = (len(bits) // 8) * _PUNCT_PERIOD
    a_dep, b_dep = [], []
    idx = 0
    for i in range(n):
        p = i % _PUNCT_PERIOD
        if _PUNCT_A[p]:
            a_dep.append(bits[idx] if idx < len(bits) else 0); idx += 1
        else:
            a_dep.append(0)
        if _PUNCT_B[p]:
            b_dep.append(bits[idx] if idx < len(bits) else 0); idx += 1
        else:
            b_dep.append(0)
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
                bm = (ra != ea) + (rb != eb)
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
    """Inverse bit inner interleaver."""
    b = np.array(bits_3024, dtype=np.int8)
    plane0 = b[0::2]
    plane1 = b[1::2]
    inv_H = np.empty(N_DATA, dtype=np.int32)
    inv_H[_BIT_IL] = np.arange(N_DATA, dtype=np.int32)
    out0 = plane0[inv_H]
    out1 = plane1[inv_H]
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
    """Inverse symbol inner interleaver."""
    R = _SYM_IL_E if (sym_idx % 2 == 0) else _SYM_IL_O
    inv_R = np.empty(N_DATA, dtype=np.int32)
    inv_R[R] = np.arange(N_DATA, dtype=np.int32)
    c = np.array(cells_1512, dtype=np.int8)
    return c[inv_R].tolist()

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

def extract_data_syms(fft_out_2048, sym_idx):
    """From FFT output (2048 bins), extract 1512 equalized data subcarriers.
    fft_out_2048 already channel-equalized.
    Returns (data_syms_1512, pilot_estimates_dict).
    """
    # fft_out_2048 is in FFT-native order; shift to baseband order
    freq = np.fft.fftshift(fft_out_2048)
    active = freq[K_MIN_BIN: K_MIN_BIN + N_ACTIVE]

    pilot_mask = _all_pilot_mask(sym_idx)
    data_pos   = np.where(~pilot_mask)[0]
    pilot_pos  = np.where(pilot_mask)[0]

    # Pilot-based channel estimate at pilot positions
    known_pilots = np.array([_pilot_value(k) for k in pilot_pos], dtype=np.complex64)
    H_at_pilots  = active[pilot_pos] / (known_pilots + 1e-12)

    # Interpolate channel over all active carriers
    H_interp = np.interp(
        np.arange(N_ACTIVE),
        pilot_pos.astype(float),
        H_at_pilots,
        left=H_at_pilots[0], right=H_at_pilots[-1]
    )

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
        self._bit_buf   = []
        self._deil      = ForneyInterleaver(interleave=False)
        self._pkt_buf   = bytearray()
        self._sym_idx   = 0
        self._cfo_rad   = 0.0          # carrier freq offset estimate (radians/sample)

    def push_samples(self, iq):
        """Push a buffer of complex IQ samples. Returns recovered TS packets."""
        pkts = []
        samples = np.asarray(iq, dtype=np.complex64)

        # CFO correction
        if abs(self._cfo_rad) > 1e-6:
            n = np.arange(len(samples))
            samples = samples * np.exp(-1j * self._cfo_rad * n).astype(np.complex64)

        # Find OFDM symbol boundaries in this buffer
        n_syms = max(1, len(samples) // SYM_LEN)
        M, P = schmidl_cox(samples)
        if len(M) == 0:
            return pkts

        # Simple approach: slide through buffer, grab symbols at detected peaks
        start = int(np.argmax(M))
        # Update CFO estimate from CP correlation phase
        if abs(P[start]) > 1e-6:
            self._cfo_rad = -np.angle(P[start]) / FFT_LEN

        pos = start
        while pos + SYM_LEN <= len(samples):
            sym = samples[pos + CP_LEN: pos + CP_LEN + FFT_LEN]
            bits = self._decode_one_symbol(sym)
            self._bit_buf.extend(bits)
            pos += SYM_LEN

        # Try to extract TS packets from bit buffer
        pkts = self._drain_packets()
        return pkts

    def _decode_one_symbol(self, sym_samples_2048):
        # FFT
        fft_out = np.fft.fft(sym_samples_2048).astype(np.complex64)
        # Channel equalization + data extraction
        data_eq, _ = extract_data_syms(fft_out, self._sym_idx)
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

    def _drain_packets(self):
        BITS_PER_CODED_PKT = 204 * 8  # post-RS bytes × 8
        pkts = []
        # Convert bits to RS-coded bytes
        while len(self._bit_buf) >= BITS_PER_CODED_PKT:
            raw = self._bit_buf[:BITS_PER_CODED_PKT]
            self._bit_buf = self._bit_buf[BITS_PER_CODED_PKT:]
            b = 0
            coded_bytes = bytearray()
            for i, bit in enumerate(raw):
                b = (b << 1) | bit
                if (i + 1) % 8 == 0:
                    coded_bytes.append(b & 0xFF)
                    b = 0
            # Forney deinterleave
            deiled = self._deil.feed(bytes(coded_bytes))
            # Accumulate 8 RS-coded packets before energy descramble
            self._pkt_buf.extend(deiled)
            while len(self._pkt_buf) >= 8 * 204:
                group = [self._pkt_buf[i*204:(i+1)*204] for i in range(8)]
                self._pkt_buf = self._pkt_buf[8*204:]
                try:
                    decoded = [rs_decode_pkt(bytes(p)) for p in group]
                    descrambled = energy_descramble(decoded)
                    pkts.extend(descrambled)
                except Exception:
                    pass  # uncorrectable RS errors: drop group
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

def _gst_tx_cmd(device, no_audio):
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
        return (['gst-launch-1.0', '-v'] + vsrc +
                ['!', 'videoconvert', '!', 'x264enc', 'tune=zerolatency', 'bitrate=2000',
                 '!', 'mpegtsmux', 'alignment=7',
                 '!', 'udpsink', 'host=127.0.0.1', f'port={UDP_IN_PORT}', 'sync=false'])
    acap = asrc + ['!', 'audioconvert', '!', 'avenc_aac', '!', 'queue', '!', 'mux.']
    return (['gst-launch-1.0', '-v'] + vcap + acap +
            ['mpegtsmux', 'name=mux', 'alignment=7',
             '!', 'udpsink', 'host=127.0.0.1', f'port={UDP_IN_PORT}', 'sync=false'])

def _gst_rx_cmd(save_path):
    if save_path:
        return ['gst-launch-1.0', '-v',
                'udpsrc', f'port={UDP_OUT_PORT}',
                'caps=video/mpegts,systemstream=(boolean)true',
                '!', 'filesink', f'location={save_path}']
    return ['gst-launch-1.0', '-v',
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
# TX main loop
# ══════════════════════════════════════════════════════════════════════════════

def run_tx(args):
    is_win = platform.system() == 'Windows'
    device = args.device or ('0' if is_win else '/dev/video0')
    print(f"[TX] freq={args.freq/1e6:.1f} MHz  attn={args.attn} dB  uri={args.uri}")

    stop_evt = threading.Event()

    # Start GStreamer
    gst_cmd = _gst_tx_cmd(device, args.no_audio)
    gst_t = threading.Thread(target=_run_gst, args=(gst_cmd, stop_evt), daemon=True)
    gst_t.start()
    time.sleep(1.5)   # let GStreamer come up before opening radio

    # Open Pluto
    sdr = setup_pluto_tx(args.uri, args.freq, args.attn)
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

    stop_evt = threading.Event()

    # Open Pluto
    sdr = setup_pluto_rx(args.uri, args.freq, args.gain)
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

    while not stop_evt.is_set():
        try:
            iq = sdr.rx()
        except Exception as e:
            print(f"[RX] radio error: {e}")
            time.sleep(0.1)
            continue

        pkts = dec.push_samples(iq)
        for pkt in pkts:
            # Send 188-byte TS packets to GStreamer
            try:
                sock.sendto(bytes(pkt[:188]), gst_addr)
            except OSError:
                pass

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

    args = p.parse_args()
    (run_tx if args.tx else run_rx)(args)


if __name__ == '__main__':
    main()
