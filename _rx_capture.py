#!/usr/bin/env python3
"""Fast raw-IQ capture to .npy (no decode). Then decode offline."""
import sys, time
import numpy as np
import pluto_dvbt2 as d
gain = float(sys.argv[1]) if len(sys.argv) > 1 else 64.0
freq = float(sys.argv[2]) if len(sys.argv) > 2 else 2.4e9
nbuf = int(sys.argv[3])   if len(sys.argv) > 3 else 40
out  = sys.argv[4]        if len(sys.argv) > 4 else 'cap.npy'
sdr = d.setup_pluto_rx('ip:192.168.2.1', freq, gain)
bufs = []
t0 = time.time()
for _ in range(nbuf):
    bufs.append(np.asarray(sdr.rx(), dtype=np.complex64))
dt = time.time() - t0
arr = np.concatenate(bufs)
np.save(out, arr)
pk = float(np.max(np.abs(arr)))
print(f"[cap] {nbuf} bufs {len(arr)} samples ({len(arr)/d.SAMP_RATE*1000:.0f}ms air) "
      f"in {dt:.1f}s  peak={pk:.0f} adc={pk/2896*100:.0f}%  -> {out}")
