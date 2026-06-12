#!/usr/bin/env python3
"""Sweep a continuous freq-domain phase slope (fractional timing) and measure data
EVM with the standard interpolation. Also report H phase behaviour across pilots."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr=np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,CP,SYM,KMIN,NA=d.FFT_LEN,d.CP_LEN,d.SYM_LEN,d.K_MIN_BIN,d.N_ACTIVE
_n=np.arange(N); ka=np.arange(NA)
_pos=[]; _known=[]
for p in range(4):
    pp=np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    _pos.append(pp); _known.append(np.array([d._pilot_value(int(k)) for k in pp],dtype=np.complex64)+1e-12)
def evm_active(active,p):
    freq=np.zeros(N,dtype=np.complex64); freq[KMIN:KMIN+NA]=active
    data_eq,_=d.extract_data_syms(np.fft.ifftshift(freq),p,0)
    s=data_eq/(np.sqrt(np.mean(np.abs(data_eq)**2))+1e-9)
    dmin=np.min(np.abs(s[:,None]-d._QPSK_MAP[None,:]),axis=1)
    return float(np.sqrt(np.mean(dmin**2)))
M,_=d.schmidl_cox(arr[:65536]); pos0=int(np.argmax(M))
syms=[arr[pos0+i*SYM:pos0+i*SYM+SYM] for i in range(10)]
# global CFO+phase
def coh(active,p):
    H=active[_pos[p]]/_known[p]; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))
best=(-1,0,0)
for nu in np.arange(-63,-57,0.1):
    rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
    acts=[np.fft.fftshift(np.fft.fft(s[CP:CP+N]*rot))[KMIN:KMIN+NA] for s in syms]
    for p0 in range(4):
        t=sum(coh(acts[i],(p0+i)&3) for i in range(10))/10
        if t>best[0]: best=(t,nu,p0)
co,nu,p0=best
rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
print(f"global coh={co:.3f} nu={nu:.2f} phase0={p0}\n")
# Inspect H phase slope on sym1 (scattered pilots only, regular 12 spacing)
si=1; act=np.fft.fftshift(np.fft.fft(syms[si][CP:CP+N]*rot))[KMIN:KMIN+NA]
sp=d._scattered_pilots((p0+si)&3)
Hs=act[sp]/(d._PILOT_BOOST*(1-2*d._PILOT_W[sp].astype(np.float32)))
dphi=np.angle(Hs[1:]*np.conj(Hs[:-1]))   # phase step per 12 carriers
print(f"sym1 scattered-pilot H: |H| mean={np.mean(np.abs(Hs)):.2f}  "
      f"phase-step/12car: mean={np.mean(dphi):+.2f} rad std={np.std(dphi):.2f}")
print(f"   => per-carrier slope ~{np.mean(dphi)/12:+.3f} rad  (|>0.26| aliases linear interp over 12)\n")
print("  slope(rad/car)  EVM(sym1)  EVM(sym3)")
for slope in np.arange(-0.30,0.301,0.03):
    ramp=np.exp(-1j*slope*ka).astype(np.complex64)
    e1=evm_active(np.fft.fftshift(np.fft.fft(syms[1][CP:CP+N]*rot))[KMIN:KMIN+NA]*ramp,(p0+1)&3)
    e3=evm_active(np.fft.fftshift(np.fft.fft(syms[3][CP:CP+N]*rot))[KMIN:KMIN+NA]*ramp,(p0+3)&3)
    print(f"   {slope:+.3f}        {e1:.3f}     {e3:.3f}")
