#!/usr/bin/env python3
"""Control: run the EVM harness on LOOPBACK data (known to decode perfectly).
If EVM~1.0 here too, the harness is the bug. If EVM~0.1, air data is genuinely noise."""
import numpy as np
import pluto_dvbt2 as d
N,CP,KMIN,NA=d.FFT_LEN,d.CP_LEN,d.K_MIN_BIN,d.N_ACTIVE
# build loopback symbols
pkts=[]
for i in range(80):
    p=bytearray(188); p[0]=0x47; p[1]=(i>>8)&0xFF; p[2]=i&0xFF
    for j in range(3,188): p[j]=(i*31+j*17)&0xFF
    pkts.append(p)
enc=d.DvbtEncoder(); enc.push_ts_packets(pkts); syms=enc.pop_ofdm_symbols()
print(f"loopback symbols={len(syms)}")
def evm(active,p):
    freq=np.zeros(N,dtype=np.complex64); freq[KMIN:KMIN+NA]=active
    data_eq,_=d.extract_data_syms(np.fft.ifftshift(freq),p,0)
    s=data_eq/(np.sqrt(np.mean(np.abs(data_eq)**2))+1e-9)
    dmin=np.min(np.abs(s[:,None]-d._QPSK_MAP[None,:]),axis=1)
    return float(np.sqrt(np.mean(dmin**2)))
_pos=[]; _known=[]
for p in range(4):
    pp=np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    _pos.append(pp); _known.append(np.array([d._pilot_value(int(k)) for k in pp],dtype=np.complex64)+1e-12)
def coh(active,p):
    H=active[_pos[p]]/_known[p]; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))
for i in range(6):
    body=syms[i][CP:CP+N]                # strip CP
    act=np.fft.fftshift(np.fft.fft(body))[KMIN:KMIN+NA]
    print(f"  loop sym{i}: coh={coh(act,i&3):.3f}  EVM={evm(act,i&3):.3f}")
