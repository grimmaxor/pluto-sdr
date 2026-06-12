#!/usr/bin/env python3
"""Pre-derotate the whole capture by a global CFO, then run the STOCK decoder.
If it decodes -> the decoder's CFO acquisition is the bug, not SNR."""
import sys, time
import numpy as np
import pluto_dvbt2 as d
arr = np.load(sys.argv[1] if len(sys.argv)>1 else 'cap.npy')
nu  = float(sys.argv[2]) if len(sys.argv)>2 else -59.75   # subcarriers
n = np.arange(len(arr))
arr = (arr * np.exp(-1j*2*np.pi*nu*n/d.FFT_LEN)).astype(np.complex64)
print(f"[derot] pre-derotated by nu={nu} subcarriers")
dec=d.DvbtDecoder()
t0=time.time(); ts=0
for off in range(0,len(arr),d.PLUTO_BUF):
    ts += len(dec.push_samples(arr[off:off+d.PLUTO_BUF]))
print(f"[derot] synced={dec._synced} framed={dec._framed} cfo={dec._carrier_shift:+d} "
      f"ph={dec._pilot_phase} pscore={dec._last_phase_score:.3f} syms={dec.total_syms} "
      f"ts_out={dec.total_ts_pkts} ({time.time()-t0:.0f}s)")
