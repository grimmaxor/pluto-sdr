#!/usr/bin/env python3
"""Decode a captured .npy through the real DvbtDecoder, chunked like run_rx."""
import sys, time
import numpy as np
import pluto_dvbt2 as d
cap = sys.argv[1] if len(sys.argv) > 1 else 'cap.npy'
arr = np.load(cap)
print(f"[offline] {cap}: {len(arr)} samples ({len(arr)/d.SAMP_RATE*1000:.0f}ms)")
dec = d.DvbtDecoder()
CHUNK = d.PLUTO_BUF
t0 = time.time()
ts = 0
for off in range(0, len(arr), CHUNK):
    pk = dec.push_samples(arr[off:off+CHUNK])
    ts += len(pk)
print(f"[offline] synced={dec._synced} framed={dec._framed} cfo={dec._carrier_shift:+d} "
      f"ph={dec._pilot_phase} pscore={dec._last_phase_score:.3f} "
      f"syms={dec.total_syms} ts_out={dec.total_ts_pkts}  ({time.time()-t0:.1f}s)")
