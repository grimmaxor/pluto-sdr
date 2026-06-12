#!/usr/bin/env python3
"""Test hypothesis: per-symbol fine CFO correction lifts pscore -> frame lock.
Monkeypatch DvbtDecoder._decode_one_symbol to derotate each 2048-sample body by
its own best fractional CFO (pilot-coherence max) before the existing decode."""
import sys, time
import numpy as np
import pluto_dvbt2 as d
arr = np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
N,KMIN,NA = d.FFT_LEN, d.K_MIN_BIN, d.N_ACTIVE
_n = np.arange(N)

# pilot tables per phase
_pos=[]; _known=[]
for p in range(4):
    pp=np.where(d._all_pilot_mask(p))[0].astype(np.int32)
    _pos.append(pp); _known.append(np.array([d._pilot_value(int(k)) for k in pp],dtype=np.complex64)+1e-12)
def _coh(active,p):
    H=active[_pos[p]]/_known[p]; dd=H[1:]*np.conj(H[:-1])
    return float(np.abs(np.sum(dd))/(np.sum(np.abs(H[:-1])*np.abs(H[1:]))+1e-12))

_orig = d.DvbtDecoder._decode_one_symbol
GRID = np.arange(-0.6, 0.601, 0.12)   # ± fractional subcarrier, fine
def _patched(self, sym2048):
    p = self._sym_idx & 3
    base = KMIN + self._carrier_shift
    if 0 <= base and base+NA <= N:
        best=(-1.0, 0.0)
        for delta in GRID:
            rot=np.exp(-1j*2*np.pi*delta*_n/N).astype(np.complex64)
            act=np.fft.fftshift(np.fft.fft(sym2048*rot))[base:base+NA]
            c=_coh(act,p)
            if c>best[0]: best=(c,delta)
        if best[1]!=0.0:
            sym2048=sym2048*np.exp(-1j*2*np.pi*best[1]*_n/N).astype(np.complex64)
        self._last_phase_score = best[0]
    return _orig(self, sym2048)
d.DvbtDecoder._decode_one_symbol = _patched

dec=d.DvbtDecoder()
t0=time.time(); ts=0
for off in range(0,len(arr),d.PLUTO_BUF):
    ts += len(dec.push_samples(arr[off:off+d.PLUTO_BUF]))
print(f"[persym] synced={dec._synced} framed={dec._framed} cfo={dec._carrier_shift:+d} "
      f"pscore={dec._last_phase_score:.3f} syms={dec.total_syms} ts_out={dec.total_ts_pkts} "
      f"({time.time()-t0:.0f}s)")
