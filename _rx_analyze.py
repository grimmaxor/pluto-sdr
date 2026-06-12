#!/usr/bin/env python3
"""Deep offline analysis of a capture: find the per-symbol coherence ceiling,
CFO drift (SCO), and whether a per-subcarrier SCO correction lifts coherence."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr = np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N, CP, SYM, KMIN, NA = d.FFT_LEN, d.CP_LEN, d.SYM_LEN, d.K_MIN_BIN, d.N_ACTIVE

full_pos, full_known = [], []
for p in range(4):
    pp = np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    full_pos.append(pp)
    full_known.append(np.array([d._pilot_value(int(k)) for k in pp], dtype=np.complex64)+1e-12)
def coh(active, p):
    H = active[full_pos[p]]/full_known[p]
    dd = H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))

# stable timing from the whole capture: SC on first 64k
M,P = d.schmidl_cox(arr[:65536])
pos0 = int(np.argmax(M))
print(f"SC pos0={pos0} M={M[pos0]:.2f}")

# march symbols
nsym = (len(arr)-pos0)//SYM
nsym = min(nsym, 120)
nrange = np.arange(N)
def fft_at(s, nu):
    rot = np.exp(-1j*2*np.pi*nu*nrange/N).astype(np.complex64)
    return np.fft.fftshift(np.fft.fft(s*rot))[KMIN:KMIN+NA]

# 1) Per-symbol best fractional CFO (around int -59..-62) + coherence ceiling
print("\n[1] per-symbol best-CFO coherence ceiling (phase from running idx):")
# first find global phase0 and coarse int via 10-sym average
syms=[arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N] for i in range(min(nsym,20))]
best=(-1.0,0,0)
for nu in np.arange(-63,-57,0.25):
    for p0 in range(4):
        t=sum(coh(fft_at(syms[i],nu),(p0+i)&3) for i in range(len(syms)))/len(syms)
        if t>best[0]: best=(t,nu,p0)
print(f"  10-sym avg best: coh={best[0]:.3f} nu={best[1]:.2f} phase0={best[2]}")
nu0,p0=best[1],best[2]

# per-symbol ceiling with its OWN best nu
ceils=[]; bestnus=[]
for i in range(nsym):
    s=arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N]
    p=(p0+i)&3
    bc=(-1.0,0.0)
    for nu in np.arange(nu0-2,nu0+2,0.1):
        c=coh(fft_at(s,nu),p)
        if c>bc[0]: bc=(c,nu)
    ceils.append(bc[0]); bestnus.append(bc[1])
ceils=np.array(ceils); bestnus=np.array(bestnus)
print(f"  per-symbol OWN-CFO coherence: median={np.median(ceils):.3f} "
      f"max={ceils.max():.3f} min={ceils.min():.3f}")
print(f"  best-nu drift across {nsym} syms: start~{np.mean(bestnus[:10]):.2f} "
      f"end~{np.mean(bestnus[-10:]):.2f}  (SCO if drifting)")

# 2) Fixed-CFO coherence vs symbol index (does it fall = SCO timing walk?)
fixed=[coh(fft_at(arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N],nu0),(p0+i)&3) for i in range(nsym)]
fixed=np.array(fixed)
print(f"\n[2] fixed-CFO coh vs sym idx: first10={np.mean(fixed[:10]):.3f} "
      f"mid={np.mean(fixed[nsym//2-5:nsym//2+5]):.3f} last10={np.mean(fixed[-10:]):.3f}")

# 3) SCO test: re-time each symbol by tracking SC locally (per-symbol window)
print("\n[3] per-symbol re-timed coherence (kills timing walk):")
retimed=[]
for i in range(nsym):
    c0=pos0+i*SYM
    # search small window for best CP-energy alignment
    seg=arr[max(0,c0-20):c0+SYM+20]
    Ml,_=d.schmidl_cox(seg) if len(seg)>=N+CP+2 else (np.array([0]),0)
    off=int(np.argmax(Ml))-min(20,c0)
    st=c0+off
    if st<0 or st+CP+N>len(arr): retimed.append(np.nan); continue
    s=arr[st+CP:st+CP+N]
    p=(p0+i)&3
    bc=max(coh(fft_at(s,nu),p) for nu in np.arange(nu0-2,nu0+2,0.2))
    retimed.append(bc)
retimed=np.array(retimed)
print(f"  re-timed per-sym coh: median={np.nanmedian(retimed):.3f} max={np.nanmax(retimed):.3f}")
