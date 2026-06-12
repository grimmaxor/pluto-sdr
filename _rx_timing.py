#!/usr/bin/env python3
"""Sweep FFT-window sample offset; measure pilot coherence vs data EVM.
If data EVM minimizes sharply at some offset while coherence is flat -> the
real/imag pilot interpolation is being broken by a timing phase-ramp."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr=np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,CP,SYM,KMIN,NA=d.FFT_LEN,d.CP_LEN,d.SYM_LEN,d.K_MIN_BIN,d.N_ACTIVE
_n=np.arange(N)
_pos=[]; _known=[]
for p in range(4):
    pp=np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    _pos.append(pp); _known.append(np.array([d._pilot_value(int(k)) for k in pp],dtype=np.complex64)+1e-12)
def coh(active,p):
    H=active[_pos[p]]/_known[p]; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))
def evm(active,p):
    freq=np.zeros(N,dtype=np.complex64); freq[KMIN:KMIN+NA]=active
    data_eq,_=d.extract_data_syms(np.fft.ifftshift(freq),p,0)
    s=data_eq/(np.sqrt(np.mean(np.abs(data_eq)**2))+1e-9)
    dmin=np.min(np.abs(s[:,None]-d._QPSK_MAP[None,:]),axis=1)
    return float(np.sqrt(np.mean(dmin**2)))

M,_=d.schmidl_cox(arr[:65536]); pos0=int(np.argmax(M))
# fine global CFO + phase over 10 syms
syms=[arr[pos0+i*SYM:pos0+i*SYM+SYM] for i in range(10)]
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
print("  win_off  coh(sym1)  EVM(sym1)  coh(sym3)  EVM(sym3)")
for off in range(-24,25,3):
    row=[]
    for si in (1,3):
        st=pos0+si*SYM+CP+off
        body=arr[st:st+N]*rot
        act=np.fft.fftshift(np.fft.fft(body))[KMIN:KMIN+NA]
        row.append((coh(act,(p0+si)&3),evm(act,(p0+si)&3)))
    print(f"  {off:+5d}   {row[0][0]:.3f}     {row[0][1]:.3f}     {row[1][0]:.3f}     {row[1][1]:.3f}")
