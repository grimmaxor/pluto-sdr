#!/usr/bin/env python3
"""Capture raw RX IQ from the Pluto and analyze it offline to find the TRUE limit on
pilot coherence (vs CFO and vs symbol index) and detect TX gaps / SCO."""
import numpy as np
import pluto_dvbt2 as d

URI, FREQ, GAIN = 'ip:192.168.2.1', int(2.4e9), 64.0
sdr = d.setup_pluto_rx(URI, FREQ, GAIN)
print(f"capturing @ {FREQ/1e6} MHz gain {GAIN} ...")

bufs = []
for _ in range(8):
    bufs.append(np.asarray(sdr.rx(), dtype=np.complex64))
del sdr

# Pick the strongest buffer.
buf = max(bufs, key=lambda b: np.max(np.abs(b)))
print(f"buffer peak={np.max(np.abs(buf)):.0f}  len={len(buf)}")

# Per-buffer peak (TX-gap detector)
print("per-buffer peaks:", [int(np.max(np.abs(b))) for b in bufs])

# Schmidl-Cox timing
M, P = d.schmidl_cox(buf)
pos = int(np.argmax(M))
print(f"SC peak pos={pos}  M={M[pos]:.3f}")

# Collect consecutive symbols
syms = []
q = pos
while q + d.SYM_LEN <= len(buf) and len(syms) < 14:
    syms.append(buf[q + d.CP_LEN: q + d.CP_LEN + d.FFT_LEN])
    q += d.SYM_LEN
print(f"collected {len(syms)} symbols")

# Precompute pilot tables (mirror DvbtDecoder)
full_pos, full_known = [], []
for p in range(4):
    pp = np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    full_pos.append(pp)
    full_known.append(np.array([d._pilot_value(int(k)) for k in pp], dtype=np.complex64) + 1e-12)

def coherence(active, p):
    H = active[full_pos[p]] / full_known[p]
    dd = H[1:] * np.conj(H[:-1])
    return float(np.abs(np.sum(dd)) / (np.sum(np.abs(H[:-1]) * np.abs(H[1:])) + 1e-12))

# Fine TOTAL-CFO sweep (integer+fractional) via time-domain derotation, averaged.
n = np.arange(d.FFT_LEN)
best = (0.0, 0, -1.0)
for nu in np.arange(-110, 110.01, 0.25):
    rot = np.exp(-1j * 2 * np.pi * nu * n / d.FFT_LEN).astype(np.complex64)
    # nu (time-domain derotation) handles the full integer+fractional shift, so the
    # active band is always read at K_MIN_BIN.
    acts = [np.fft.fftshift(np.fft.fft(s * rot))[d.K_MIN_BIN: d.K_MIN_BIN + d.N_ACTIVE]
            for s in syms]
    for p0 in range(4):
        tot = sum(coherence(acts[t], (p0 + t) & 3) for t in range(len(syms))) / len(syms)
        if tot > best[2]:
            best = (nu, p0, tot)
print(f"\nBEST total CFO ν={best[0]:.2f} subcarriers  phase0={best[1]}  coherence={best[2]:.3f}")

# Coherence per symbol index at the best CFO (SCO signature: drops for later symbols)
nu, p0, _ = best
rot = np.exp(-1j * 2 * np.pi * nu * n / d.FFT_LEN).astype(np.complex64)
percoh = []
for t, s in enumerate(syms):
    active = np.fft.fftshift(np.fft.fft(s * rot))[d.K_MIN_BIN: d.K_MIN_BIN + d.N_ACTIVE]
    percoh.append(coherence(active, (p0 + t) & 3))
print("coherence per symbol:", " ".join(f"{c:.2f}" for c in percoh))
