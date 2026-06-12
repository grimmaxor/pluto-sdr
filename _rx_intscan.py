#!/usr/bin/env python3
"""Sweep integer carrier offset; continual-pilot coherence (phase-independent).
A genuine pluto_dvbt2 signal MUST peak ~0.95+ at one offset. Flat ~0.5 => not our signal."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr=np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,CP,SYM,KMIN,NA=d.FFT_LEN,d.CP_LEN,d.SYM_LEN,d.K_MIN_BIN,d.N_ACTIVE
cont=d._CONT_PIL; cont_k=(d._PILOT_BOOST*(1-2*d._PILOT_W[cont].astype(np.float32))).astype(np.complex64)+1e-12
M,_=d.schmidl_cox(arr[:65536]); pos0=int(np.argmax(M)); print(f"SC pos0={pos0} M={M[pos0]:.2f}")
syms=[np.fft.fftshift(np.fft.fft(arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N])) for i in range(16)]
def coh(active):
    H=active[cont]/cont_k; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))
best=(-1,0); res=[]
for shift in range(-120,121):
    base=KMIN+shift
    if base<0 or base+NA>N: continue
    t=np.mean([coh(f[base:base+NA]) for f in syms])
    res.append((shift,t))
    if t>best[0]: best=(t,shift)
res.sort(key=lambda x:-x[1])
print("top integer offsets by continual-pilot coherence:")
for sh,t in res[:8]:
    print(f"   shift={sh:+4d}  contcoh={t:.3f}")
print(f"\nBEST continual coherence = {best[0]:.3f} at shift={best[1]:+d}")
print("VERDICT:", "clean DVB-T signal present" if best[0]>0.9 else
      ("WEAK/PARTIAL - not a clean pluto_dvbt2 signal" if best[0]>0.6 else
       "NO coherent DVB-T continual pilots - signal is not our TX (or malformed)"))
