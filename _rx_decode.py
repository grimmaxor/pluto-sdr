#!/usr/bin/env python3
"""Run the real DvbtDecoder against live RF (no GStreamer). Watch frame lock."""
import sys, time
import numpy as np
import pluto_dvbt2 as d
gain = float(sys.argv[1]) if len(sys.argv) > 1 else 64.0
freq = float(sys.argv[2]) if len(sys.argv) > 2 else 2.4e9
nbuf = int(sys.argv[3])   if len(sys.argv) > 3 else 200
sdr = d.setup_pluto_rx('ip:192.168.2.1', freq, gain)
print(f"[decode] RX {freq/1e6:.1f} MHz gain={gain} buffers={nbuf}")
dec = d.DvbtDecoder()
t0 = time.time()
for i in range(nbuf):
    iq = sdr.rx()
    dec.push_samples(iq)
    if (i+1) % 25 == 0:
        print(f"  buf {i+1:3d}  peak={dec.peak_signal:7.0f}  synced={dec._synced}  "
              f"framed={dec._framed}  cfo={dec._carrier_shift:+d}  ph={dec._pilot_phase}  "
              f"pscore={dec._last_phase_score:.3f}  syms={dec.total_syms}  ts_out={dec.total_ts_pkts}")
        dec.peak_signal = 0.0
print(f"\n[decode] FINAL synced={dec._synced} framed={dec._framed} "
      f"pscore={dec._last_phase_score:.3f} syms={dec.total_syms} ts_out={dec.total_ts_pkts} "
      f"in {time.time()-t0:.1f}s")
