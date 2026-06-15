# `pluto_video_stream_16qam_stable.py` — Project Notes

Simplex (one-way) live video streaming over 16QAM using two ADALM-Pluto radios.
TX encodes video with ffmpeg → modulates into 16QAM IQ → transmits over the air.
RX receives IQ → demodulates → RS FEC decodes → pipes MPEG-TS to ffplay.
No ARQ, no retransmissions — lost packets cause brief glitches, standard broadcast behavior.

---

## How to Run

```bash
# Environment (always needed on this Linux box)
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libiio.so.0.23
PYTHON=.venv/bin/python3.11

# TX (Pluto 1, other PC):
$PYTHON pluto_video_stream_16qam_stable.py --role tx --input my_video.mp4

# TX with webcam:
$PYTHON pluto_video_stream_16qam_stable.py --role tx --input /dev/video0

# RX (Pluto 2, this PC):
$PYTHON pluto_video_stream_16qam_stable.py --role rx

# RX skipping auto-calibration (use known-good gain):
$PYTHON pluto_video_stream_16qam_stable.py --role rx --skip-cal --rx-gain 45

# Offline self-test (no hardware needed):
$PYTHON pluto_video_stream_16qam_stable.py --role test
```

**Start TX first, then RX.** Calibration needs the TX already transmitting to sweep RX gain.

The received MPEG-TS is also saved to `live_video_stable.ts` in the working directory.
If ffplay is too glitchy, play that file afterward: `ffplay live_video_stable.ts`

---

## CLI Flags

| Flag | Default | Notes |
|---|---|---|
| `--role` | required | `tx`, `rx`, or `test` |
| `--ip` | `ip:pluto.local` | Use `ip:192.168.2.1` for USB |
| `--freq` | `2412e6` | Carrier frequency (Hz) |
| `--input` | — | TX only: video file, `/dev/video0`, or `video=<dshow name>` |
| `--bitrate` | `120k` | TX ffmpeg video bitrate |
| `--sps` | `16` | Samples per symbol |
| `--chunk` | `376` | Payload bytes per RF packet (376 = 188×2, MPEG-TS aligned) |
| `--rx-gain` | `40` | RX hardware gain dB (used only with `--skip-cal`) |
| `--tx-atten` | `-20` | TX attenuation dB (negative = weaker) |
| `--skip-cal` | off | Skip RX auto-calibration; use `--rx-gain` directly |

---

## Architecture / DSP Chain

### TX path

```
ffmpeg → MPEG-TS chunks (376 bytes each)
       → build_packet: [MAGIC:1][seq:4][plen:2] + RS-encode(payload+CRC32)
       → bits_to_symbols: Gray-coded 16QAM (4 bits/symbol), scale /√10
       → prepend BPSK Barker-13×3 preamble (39 symbols, real-only)
       → upsample by SPS=16, apply RRC filter (α=0.35, 193 taps)
       → normalize to 0.8×2^15 (DAC scale)
       → sdr.tx(iq)  [non-cyclic, one burst per packet]
```

### RX path

```
sdr.rx()  [131072 samples = 0.131s]
  → iq_to_packets:
      normalize by peak
      for each of 4 CFO variants (none / FFT-4th-power / CFO±15Hz):
          RRC matched filter via oaconvolve (complex, fast)
          for each of 16 timing phases (0..SPS-1):
              correlate symbol stream vs BPSK preamble → magnitude peaks
              find_peaks(height=0.60×39, distance=19) — one candidate per packet
              for each candidate:
                  phi = angle(corr peak) → derot = exp(-j·phi)  [phase recovery]
                  chan_gain = |corr| / PREAMBLE_LEN             [amplitude recovery]
                  fast-abort: decode 1 block, check MAGIC byte + extract plen
                  exact-length decode: only run LS equalizer over this packet's symbols
                  LS equalizer: every 64 symbols, update derot+gain via Σ(rx·conj(ideal))
                  RS decode + CRC verify
                  store in found{seq} (best CRC wins dedup)
  → sorted decoded packets → write to ffplay stdin + live_video_stable.ts
```

### Key design choices

**BPSK preamble on a 16QAM payload**: The Barker-13×3 preamble is always real-valued BPSK regardless of the payload modulation. Correlating the complex received stream against the real reference gives a complex peak; its angle is the residual carrier phase. This directly derotates the payload without any rotation search — resolves 16QAM's phase ambiguity for free.

**RRC filter (α=0.35, span=12×SPS)**: Provides zero ISI at symbol sampling points. Applied at both TX (shaping) and RX (matched filter via `oaconvolve`). Both halves together give a raised-cosine response.

**RS FEC (RS_PARITY=32)**: Reed-Solomon with 32 parity bytes per ≤223-byte RS block. Corrects up to 16 byte errors per block. Payload+CRC32 is RS-encoded; the 7-byte header is unencoded (parsed first for fast-abort). CRC32 is a final sanity check after RS decode.

**Exact-length decode**: After reading `plen` from the header, compute exactly how many symbols this packet needs: `ceil((7 + rs_encoded_len(plen+4)) × 8 / 4) + 8`. Run the LS equalizer only over those symbols. Prevents the equalizer from running over the entire buffer tail for every candidate.

**`find_peaks` de-sidelobe**: The preamble correlation lobe spans ~5 adjacent samples all above threshold. `find_peaks(distance=PREAMBLE_LEN//2=19)` collapses each lobe to one candidate instead of ~5, cutting per-buffer work by 5×.

**4 CFO variants**: Raw IQ + FFT-4th-power CFO estimate + CFO±15Hz micro-offsets. The ±15 Hz sweep catches packets that drifted slightly during transmission. The 4th-power method works because 16QAM raised to the 4th power concentrates energy at 4×carrier.

**CHUNK_BYTES=376 (188×2)**: Each RF packet carries exactly 2 MPEG-TS packets (188 bytes each). A dropped RF packet causes a clean 376-byte gap — ffplay resyncs on the next 0x47 sync byte immediately. With 512-byte chunks (not divisible by 188), a drop misaligns every subsequent TS packet boundary, causing cascading decoder failures.

**`-g 30 -x264-params repeat-headers=1`**: IDR (keyframe) every 30 frames (~1 second). `repeat-headers=1` forces SPS+PPS before every IDR. Without this the H.264 decoder receives SPS/PPS only once at stream start. If the RX misses the first IDR (e.g., calibration delay), it can **never decode video** — `non-existing PPS 0 referenced` on every packet forever. With repeat-headers, recovery happens within 1 second of receiving anything.

---

## Packet Wire Format

```
[MAGIC=0xA5 : 1 byte]  ← unencoded, checked first
[seq         : 4 bytes big-endian uint32]
[plen        : 2 bytes big-endian uint16]  ← payload length
[RS-encode(payload + CRC32(payload))]      ← RS_PARITY=32 per ≤223-byte block
```

Total RF packet size for default CHUNK_BYTES=376:
- payload = 376 bytes
- RS encode(376+4=380 bytes): 2 blocks → 380 + 2×32 = 444 bytes
- Header: 7 bytes
- Wire: 451 bytes = 3608 bits = 902 16QAM symbols + 39 preamble = 941 symbols
- At SPS=16: 15,056 samples = ~15ms of air time at 1 MSPS

---

## Calibration

Two-phase auto-calibration (RX only, TX must already be transmitting):

**Phase 1 — fast ADC sweep**: Sweeps RX gain 0→70 dB in 5 dB steps. For each: set gain, flush 2 buffers, read 1 buffer. Check ADC peak only — no packet decode. Accepts gains where 5% < ADC_utilization < 95% (not noise floor, not clipping). Fast (~7s).

**Phase 2 — targeted packet decode**: Takes the accepted gains, tries them highest-first. For each: flush 2 buffers, attempt 2 packet decodes (`iq_to_packets`). If any decode succeeds, the gain is a candidate. Picks the highest candidate. Falls back to highest non-clipping gain if no decodes succeed. (~8s, dominated by `iq_to_packets` processing time).

Calibration result from first real-hardware run (2026-06-15):
- Gain=45: ADC=92% (scan), ADC=97% (phase 2, signal louder), 2/2 decodes → selected
- Gain=40: ADC=96%, 2/2 decodes (would also work)
- All gains 20–45 decoded 2/2 during phase 2 (strong signal)

To skip calibration after first run: `--skip-cal --rx-gain 45`

> The above (RX-only, TX must already be transmitting at a fixed `--tx-atten`) is what
> `pluto_video_stream_16qam_stable.py` does. For bringing the link up **on air**, where the
> TX power also has to be negotiated, use `pluto_video_stream_16qam_stable_auto.py` (below).

### `pluto_video_stream_16qam_stable_auto.py` — over-the-air TX↔RX auto-calibration

Same DSP/streaming as the stable file, but a half-duplex handshake runs on the streaming
frequency **before** streaming, negotiating *both* RX gain and TX power so neither needs
hand-tuning on air. The video stream stays simplex (TX→RX); the reverse channel (RX→TX) is
used **only** during the handshake for acknowledgements.

- **Phase A (RX gain)**: TX beacons `CAL_TONE` at a fixed weak cal power (`CAL_TX_ATTEN=-30`
  dB); RX sweeps gain 0→71 to decode the beacon, locks the best non-clipping gain, then
  reports `CAL_GAIN_OK`.
- **Phase B (TX power)**: TX sweeps attenuation weak→strong (`-40..0` dB), sending `CAL_PWR`
  probes that carry the current atten; RX (at locked gain) echoes the first probe it decodes;
  TX locks that atten `+ CAL_PWR_MARGIN(5)` dB for headroom.
- **Phase C (READY)**: bidirectional `CAL_READY` handshake, then both sides drop into
  one-way streaming.

Control frames reuse the normal packet path; the control type is encoded in the high `seq`
range (`CTRL_SEQ_BASE=0xCA000000`) so it never collides with video seqs (which count from 1).
Beacons are sent as repeated **non-cyclic** bursts (no TX-buffer mode switch vs streaming),
with small random hold jitter to break the half-duplex lock-step hazard. RX uses
`CAL_ACK_ATTEN=-10` dB for its handshake acks (reverse link is uncalibrated → fairly strong).

Run both sides within ~30 s of each other. `--skip-cal --rx-gain N --tx-atten N` bypasses the
handshake on both roles. Offline-verified: self-test PASS, and all 4 control frames round-trip
through a noisy channel (incl. the PWR value) with no cross-type false matches. **On-air
handshake not yet hardware-tested.**

---

## Bugs Found and Fixed (2026-06-15 debug session)

### Bug 1 — DDS tone missing from setup_pluto (TX corruption)

**Symptom**: TX signal corrupted by a carrier tone leaking from the Pluto's onboard DDS.
**Cause**: `cf-ad9361-dds-core-lpc` DDS channels were not zeroed out. Every other script in the repo does this, but both video stream scripts were missing it.
**Fix**: Added the standard DDS disable block in `setup_pluto`:
```python
dds = iio.Context(args.ip).find_device("cf-ad9361-dds-core-lpc")
for ch in dds.channels:
    if ch.output:
        ch.attrs["raw"].value = "0"
        ch.attrs["scale"].value = "0.0"
```

### Bug 2 — Preamble correlation threshold 0.75 too strict for real hardware

**Symptom**: On real hardware with channel impairments, correlation peaks are lower than in the offline loopback. Packets that exist in the buffer aren't being detected.
**Fix**: Lowered `THR = PREAMBLE_LEN * 0.75` → `PREAMBLE_LEN * 0.60`. False positives are harmless (MAGIC byte + RS + CRC rejects them). Only correct packets get through.

### Bug 3 — H.264 SPS/PPS sent only once, missed during calibration (video never displayed)

**Symptom**: `non-existing PPS 0 referenced` / `decode_slice_header error` / `no frame!` on every single decoded packet across the entire 4-minute run. 63 packets received, 0 frames displayed. Log showed `Lost 4065 packets` in the first streaming buffer.
**Root cause**: libx264 emits SPS+PPS only in the very first IDR frame. The RX calibration phase takes ~15 seconds, during which the TX sends hundreds of packets including that first IDR. By the time the RX main loop starts, SPS+PPS are long gone. Every subsequent P-frame is undecipherable. The decoder never recovers.
**Fix**: Added `-g 30 -x264-params repeat-headers=1` to the TX ffmpeg command. IDR every ~1 second, SPS+PPS in every IDR. Decoder can resync from any point mid-stream.

### Bug 4 — CHUNK_BYTES=512 not MPEG-TS aligned (cascading sync failures)

**Symptom**: Dropped RF packets broke TS packet alignment for all subsequent data; ffplay took a long time to resync.
**Fix**: Changed default `CHUNK_BYTES` from 512 to 376 (= 188×2). Each dropped RF packet now causes a clean 2-TS-packet gap; ffplay resyncs on the very next 0x47 sync byte.

### Bug 5 — RX buffer too large (large temporal gaps in video)

**Symptom**: Video played in ~2-second bursts then jumped forward ~5 seconds (classic "missing chunks" look).
**Cause**: `rx_buffer_size = 524288` (0.524s per buffer) took ~1.7s to process (3.3× real-time). The kernel ring holds 4 buffers (2.1s). After draining 4 consecutive buffers, the ring had been overwritten 3× over, causing ~4.7s gaps.
**Fix**: Reduced `rx_buffer_size` to 131072 (0.131s per buffer). Processing ~0.4s per buffer. Same 4-buffer drain pattern but gaps shrink from ~4.7s to ~1.2s. With `-g 30` (1s I-frame), decoder recovers within each gap.

### Bug 6 — ffplay latency flags fragile for slow/intermittent data

**Symptom**: `-fflags nobuffer -flags low_delay -strict experimental` caused ffplay to give up when data arrived in slow bursts.
**Fix**: Replaced with `-probesize 32 -analyzeduration 0 -framedrop`. Starts quickly with minimal probe data; allows normal buffering; drops late frames to maintain sync.

### Bug 7 — ffplay h264 error flood masking useful stats in console

**Symptom**: `[h264 @ ...] non-existing PPS 0 referenced` spammed every line, burying the packet stats.
**Fix**: Changed ffplay `-loglevel error` to `-loglevel fatal`. Once `repeat-headers=1` is in place this error disappears entirely; `fatal` still shows real problems.

---

## Current Status (2026-06-15)

- Offline self-test: **PASS** — RS decode, CRC, payload byte-exact match (~40ms)
- Real hardware calibration: **working** — gain=45 selected, 2/2 decode during cal
- Real hardware video display: **not yet confirmed** — `repeat-headers=1` fix was applied after the 4-minute run; needs a re-test with the updated TX script on the other PC
- RX processing speed: **RESOLVED** — `iq_to_packets` now 0.087s/buffer (was 0.79s), under the 0.131s real-time budget (see Performance section below)

**Next test**: Pull updated script on the TX PC, restart TX, run RX. With the speed fix the RX should now keep up with the air in real time (no more whole-buffer drops / 90–140-packet gaps). The calibrated gain is 45 dB (`--skip-cal --rx-gain 45` to skip calibration delay on subsequent runs).

---

### Bug 8 — Garbage seq numbers from unprotected header (huge "Lost N" spikes)

**Symptom**: occasional `[RX] Warning: Lost 67093257 packets` / `Lost 32748` / `Lost 15589` — wildly implausible sequence gaps amid otherwise-normal reception.
**Root cause**: the 7-byte header `[MAGIC][seq][plen]` is sent **unencoded** (only payload+CRC32 is RS-protected). A bit-flip in the `seq` field produces a packet whose payload still passes RS+CRC but carries a garbage sequence number. That garbage value poisons `last_seq`.
**Fix (RX-side guard)**: in `receiver_main`, reject any packet whose `seq - last_seq > 100000` so a corrupted header can't poison the ordering/loss accounting. (A fuller fix would move `seq` inside the RS-protected region, but that changes the wire format on both sides.)

---

## Performance — RESOLVED (2026-06-15, session 2)

`iq_to_packets` went from **0.79s → 0.087s** per 131072-sample buffer — now **under** the 0.131s real-time budget, so the kernel RX ring no longer overflows and whole-buffer drops (the dominant cause of the 90–140-packet "Lost" gaps) are eliminated.

**The real bottleneck was NOT `np.correlate`/`oaconvolve`** (the prior guess). cProfile on a full buffer of real packets showed `symbols_to_bits`/`demap_axis` called **16,689×/buffer = 0.95s cumulative**. Root reason: RRC shaping makes each packet correlate at *every* SPS timing phase, so each of ~8 packets was fast-aborted and fully RS-decoded ~16× per CFO variant.

**Three fixes (all in `iq_to_packets` / `symbols_to_bits`):**
1. **Position-dedup** — record the absolute preamble sample position of every CRC-OK packet; skip any later candidate within ±(SPS+2) samples (same physical packet across toffs *and* CFO variants). Also `break`s the slip loop on success. Cut RS-decodes 51→15. (0.79→0.66s)
2. **Allocation-light `symbols_to_bits`** — replaced the per-call nested `demap_axis` (`np.round`/`np.clip`/`np.stack`/`column_stack`) with four strided boolean writes into a preallocated `uint8` array. Bit order identical. This was the single biggest win. (0.66→0.30s)
3. **Coarser timing sweep** — `for toff in range(0, SPS, SPS//4)` (phases {0,4,8,12}) instead of all 16; the existing ±2 `slip` search tiles the gaps, so all 16 phases are still covered at ¼ the per-toff cost. Verified 8/8 decode under random packet-phase jitter. (0.30→0.087s)

Self-test still PASS (RS decode, CRC, payload byte-exact). Benchmark harness: `/tmp/prof_stable.py` (full synthetic buffer + cProfile) — recreate from session 2 if lost.

### RX debug prints are verbose

The `[RX] Requesting buffer from SDR...` and `[RX] Got buffer! Size: 131072` lines were added for debugging and print on every iteration. Fine for debugging, noisy for normal use. Can be removed once stable.

---

## Development Environment

```bash
# Python venv (has pyadi-iio, numpy, scipy, reedsolo)
/home/naveen/Desktop/SDR/claude/solo_pyth/.venv/bin/python3.11

# libiio shim (required — system libiio and pyadi-iio's bundled one conflict)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libiio.so.0.23

# Run anything:
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libiio.so.0.23 .venv/bin/python3.11 pluto_video_stream_16qam_stable.py <args>
```

Dependencies: `pip install pyadi-iio numpy scipy reedsolo`
System: `apt install ffmpeg` (for ffmpeg + ffplay)

---

## Related Files

| File | Notes |
|---|---|
| `pluto_video_stream_16qam.py` | Baseline — no FEC, no RRC, Hamming FIR. CHUNK_BYTES=1024. Has a bug: the `for toff` loop body is de-indented, only the last toff phase is ever used. |
| `16qam_low_errorrate.py` | Adds LS equalizer to the basic version, no FEC. |
| `pluto_video_stream_16qam_stable.py` | **This script** — RRC + RS FEC + LS equalizer + exact-length decode. Most robust. |
| `pluto_video_stream_256qam.py` | Same architecture as basic 16QAM but 256QAM (8 bits/symbol). Best SNR links only. |
