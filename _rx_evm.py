#!/usr/bin/env python3
"""Measure equalized QPSK data EVM on air vs the pilot coherence, and test for
spectral inversion. Uses the module's own extract_data_syms."""
import sys
import numpy as np
import pluto_dvbt2 as d
arr = np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,CP,SYM,KMIN,NA = d.FFT_LEN,d.CP_LEN,d.SYM_LEN,d.K_MIN_BIN,d.N_ACTIVE
_n=np.arange(N)
_pos=[]; _known=[]
for p in range(4):
    pp=np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    _pos.append(pp); _known.append(np.array([d._pilot_value(int(k)) for k in pp],dtype=np.complex64)+1e-12)
def coh(active,p):
    H=active[_pos[p]]/_known[p]; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))

M,_=d.schmidl_cox(arr[:65536]); pos0=int(np.argmax(M))
# find global phase0 + fine CFO over 10 syms
syms=[arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N] for i in range(10)]
best=(-1,0,0)
for nu in np.arange(-63,-57,0.1):
    rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
    acts=[np.fft.fftshift(np.fft.fft(s*rot))[KMIN:KMIN+NA] for s in syms]
    for p0 in range(4):
        t=sum(coh(acts[i],(p0+i)&3) for i in range(10))/10
        if t>best[0]: best=(t,nu,p0)
co,nu,p0=best
print(f"global: coh={co:.3f} nu={nu:.2f} phase0={p0}")

def evm_of(active, sym_idx, invert=False):
    if invert: active = active[::-1]
    # build fft_out_2048 in native order so extract_data_syms works
    freq=np.zeros(N,dtype=np.complex64); freq[KMIN:KMIN+NA]=active
    fft_native=np.fft.ifftshift(freq)
    data_eq,_=d.extract_data_syms(fft_native, sym_idx, 0)
    # normalize and measure distance to nearest QPSK point
    s=data_eq/ (np.sqrt(np.mean(np.abs(data_eq)**2))+1e-9) * (1/np.sqrt(2))*np.sqrt(2)
    # nearest of (±1±1)/sqrt2
    qp=d._QPSK_MAP
    dmin=np.min(np.abs(s[:,None]-qp[None,:]),axis=1)
    return float(np.sqrt(np.mean(dmin**2)))

# pick the best-coherence symbol among first 20
cohs=[]
for i in range(20):
    rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
    act=np.fft.fftshift(np.fft.fft(syms[i] if i<10 else arr[pos0+i*SYM+CP:pos0+i*SYM+CP+N]*0)[ :] )[KMIN:KMIN+NA] if i<10 else None
for i in range(10):
    rot=np.exp(-1j*2*np.pi*nu*_n/N).astype(np.complex64)
    act=np.fft.fftshift(np.fft.fft(syms[i]*rot))[KMIN:KMIN+NA]
    c=coh(act,(p0+i)&3)
    e=evm_of(act,(p0+i)&3,invert=False)
    ei=evm_of(act,(p0+i)&3,invert=True)
    print(f"  sym{i}: coh={c:.3f}  EVM={e:.3f}  EVM_inv={ei:.3f}")
