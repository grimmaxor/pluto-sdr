#!/usr/bin/env python3
"""For an air symbol, measure scattered-only vs continual-only differential
coherence for each of the 4 scattered-pilot phase hypotheses. Reveals whether
the scattered phase is identifiable / being mis-locked."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr=np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,CP,SYM,KMIN,NA=d.FFT_LEN,d.CP_LEN,d.SYM_LEN,d.K_MIN_BIN,d.N_ACTIVE
_n=np.arange(N)
def diffcoh(active,pos,known):
    H=active[pos]/known; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))
cont=d._CONT_PIL; cont_k=d._PILOT_BOOST*(1-2*d._PILOT_W[cont].astype(np.float32))
M,_=d.schmidl_cox(arr[:65536]); pos0=int(np.argmax(M))
syms=[arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N] for i in range(12)]
# best global CFO via continual-only (phase-independent) over 12 syms
best=(-1,0)
for nu in np.arange(-63,-57,0.1):
    rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
    t=0
    for s in syms:
        act=np.fft.fftshift(np.fft.fft(s*rot))[KMIN:KMIN+NA]
        t+=diffcoh(act,cont,cont_k)
    t/=len(syms)
    if t>best[0]: best=(t,nu)
contcoh,nu=best
rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
print(f"continual-only coh (phase-independent) = {contcoh:.3f}  at nu={nu:.2f}\n")
print("  scattered-only coherence per assumed phase, per symbol:")
print("   sym | phase0  phase1  phase2  phase3 | best")
for i,s in enumerate(syms[:8]):
    act=np.fft.fftshift(np.fft.fft(s*rot))[KMIN:KMIN+NA]
    cs=[]
    for ph in range(4):
        sp=d._scattered_pilots(ph)
        kn=d._PILOT_BOOST*(1-2*d._PILOT_W[sp].astype(np.float32))
        cs.append(diffcoh(act,sp,kn))
    bph=int(np.argmax(cs))
    print(f"   {i:3d} | "+"  ".join(f"{c:.3f}" for c in cs)+f" | ph{bph}={cs[bph]:.3f}")
