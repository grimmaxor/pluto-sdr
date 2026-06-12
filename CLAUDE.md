# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

This is a software-defined radio (SDR) project for the ADALM-Pluto. Per `readme.txt`, the
end goal is to **merge the two approaches below into a single optimized, Python-only
codebase** — i.e. fold the GNU Radio DSP into the pure-Python pipeline so GNU Radio is no
longer required.

- **`GNU/`** — the original approach. Thin Python scripts that do file I/O, FEC, and ARQ,
  but hand the actual modulation + RF off to a separate **GNU Radio flowgraph** (`.grc`, not
  in this repo). Python ↔ GNU Radio communicate only via local UDP sockets.
- **`solo_pyth/`** — the target approach. Pure Python that owns the *entire* chain — bits →
  modulation → pulse-shaping → RF → demod → ARQ — talking to the Pluto directly through
  `pyadi-iio`. No GNU Radio. **New work goes here.**

There is **no build system, test suite, linter config, or `requirements.txt`** — every file
is a standalone runnable script. "Testing" means running a script against real Pluto
radios (two of them, one per role, for an end-to-end run).

## Dependencies

```bash
pip install pyadi-iio numpy scipy pylibiio reedsolo zfec Pillow
# tkinter is a system package (e.g. apt install python3-tk) — used by the GUIs
# pluto_image_arq.py optionally shells out to ffmpeg for H.264 encoding
```

- `solo_pyth/`: `adi` (pyadi-iio), `numpy`, `scipy`, `iio`; plus `Pillow`+`tkinter` for the
  GUI scripts.
- `pluto_dvbt_video.py` (in `solo_pyth/`): requires **GNU Radio** (`gnuradio`, `gnuradio-dtv`,
  `gnuradio-iio`) and **GStreamer** with `gst-launch-1.0` on PATH. Windows TX needs
  `mfvideosrc` (Media Foundation) + `wasapisrc`; Linux RX needs `v4l2src` + `pulsesrc`. Also
  applies a libiio shim (`LD_PRELOAD`) automatically if `/usr/lib/x86_64-linux-gnu/libiio.so.0.23`
  exists — needed when system libiio and GR's bundled libiio conflict.
- `GNU/`: only stdlib `socket` plus the FEC lib for that script — `reedsolo` for `*_fec`/
  `*_arq_qpsk`, `zfec` for `*_zfec`. Also needs the matching GNU Radio flowgraph running.

## Running

`solo_pyth` scripts run the **same file on both radios**, differentiated by role. Two PCs
each with a Pluto are required. `--role master/slave` is half-duplex (shared frequency);
`--role a/b` is FDD (separate TX/RX frequencies).

```bash
# Pure-Python file transfer, half-duplex (master requests, slave serves) — GUI on master
python solo_pyth/pluto_filelink.py --role master --ip ip:pluto.local --request photo.jpg
python solo_pyth/pluto_filelink.py --role slave  --ip ip:pluto.local --serve-dir ./to_send

# Same, but auto-calibrating the link first (recommended of the filelink family)
python solo_pyth/pluto_filelink_auto_qpsk.py --role master --request photo.jpg
python solo_pyth/pluto_filelink_auto_qpsk.py --role slave  --serve-dir ./to_send
#   skip calibration with fixed RF:  --skip-cal --rx-gain 10 --tx-atten -30

# Auto-calibrating link + chat
python solo_pyth/pluto_chat_auto.py     --role master      # half-duplex PTT chat
python solo_pyth/pluto_autolink_fdd.py  --role a           # FDD full-duplex chat

# FDD image transfer (full-duplex, separate TX/RX frequencies)
python solo_pyth/pluto_image_arq.py --role a --image photo.jpg   # round-based ARQ + codec
python solo_pyth/pluto_image_fdd_diag.py --role b --save-dir ./received

# FDD RAW image transfer — uncompressed, no size limit, BPSK (fixed sender/receiver roles)
python solo_pyth/pluto_image_fdd_raw.py --role tx --image photo.jpg          # Pluto 1
python solo_pyth/pluto_image_fdd_raw.py --role rx --out-dir ./received       # Pluto 2
#   data carrier (tx→rx) = --freq-data, control carrier (rx→tx) = --freq-ctrl
#   skip auto-cal:  --skip-cal --rx-gain 40 --tx-atten -20

# FDD RAW v2 — bug-fixed version (use this when Windows PC is --role rx)
python solo_pyth/pluto_image_fdd_raw2.py --role tx --image photo.jpg
python solo_pyth/pluto_image_fdd_raw2.py --role rx --out-dir ./received      # Windows

# FDD RAW v2 QPSK — selectable --mod {bpsk,qpsk} (default qpsk, ~2x throughput vs BPSK)
python solo_pyth/pluto_image_fdd_raw2_qpsk.py --role tx --image photo.jpg --mod qpsk
python solo_pyth/pluto_image_fdd_raw2_qpsk.py --role rx --out-dir ./received --mod qpsk
#   TX and RX --mod MUST match; skip auto-cal:  --skip-cal --rx-gain 40 --tx-atten -20
#   blast mode (handshake-free decode test, isolates demod from ARQ):
#     TX: --blast --image test.jpg --skip-cal --rx-gain 36 --tx-atten -40
#     RX: --blast --skip-cal --rx-gain 36 --tx-atten -40   (prints stats every 2 s)

# FDD RAW v2 16QAM — selectable --mod {bpsk,qpsk,16qam} (default 16qam, ~4x throughput vs BPSK)
python solo_pyth/pluto_image_fdd_raw2_16qam.py --role tx --image photo.jpg --mod 16qam
python solo_pyth/pluto_image_fdd_raw2_16qam.py --role rx --out-dir ./received --mod 16qam
#   TX and RX --mod MUST match; skip auto-cal:  --skip-cal --rx-gain 40 --tx-atten -20
#   NOTE (2026-06-12): offline self-test failing due to tail ISI at tight 16QAM copy boundaries
#   — use --mod qpsk on this file as a fallback, or use pluto_image_fdd_raw2_qpsk.py

# FDD link diagnostic — run on both PCs simultaneously to identify which direction fails
python solo_pyth/pluto_link_debug.py --role tx                               # Linux
python solo_pyth/pluto_link_debug.py --role rx --ip ip:192.168.2.1          # Windows
#   --quick for a 30-second passive sniff with no TX

# Continuous loop image transfer — TX loops until Ctrl+C, RX verifies MD5 each run
# Split files: TX runs on Windows PC, RX runs on Linux machine
python solo_pyth/pluto_image_loop_tx.py --image photo.jpg                    # Windows (sender)
python solo_pyth/pluto_image_loop_rx.py                                      # Linux  (receiver)
#   first run: auto-calibration prints --skip-cal values to hardcode for subsequent runs
#   skip auto-cal:  --skip-cal --rx-gain 40 --tx-atten -20

# Standalone live RF tuning GUI (run alongside any link to tweak gain/atten/AGC)
python solo_pyth/pluto_gain_control.py --ip ip:pluto.local

# DVB-T live video streaming — GR Python bindings inline + GStreamer (NOT pure pyadi-iio)
python solo_pyth/pluto_dvbt_video.py --tx [--device 0] [--no-audio] [--freq 2.4e9] [--uri ip:192.168.2.1] [--attn 8]   # Windows
python solo_pyth/pluto_dvbt_video.py --rx [--freq 2.4e9] [--uri ip:192.168.2.1] [--gain 30] [--save out.ts]             # Linux

# DVB-T live video streaming — pure Python/NumPy, no GNU Radio (preferred over pluto_dvbt_video.py)
# Both sides auto-calibrate over-the-air before streaming; use --skip-cal to reuse known values.
python solo_pyth/pluto_dvbt2.py --tx [--freq 2.4e9] [--uri ip:192.168.2.1] [--attn 8] [--device 0] [--no-audio]  # Windows
python solo_pyth/pluto_dvbt2.py --rx [--freq 2.4e9] [--uri ip:192.168.2.1] [--gain 30] [--save out.ts]            # Linux
#   skip calibration:  --skip-cal --gain 71 --attn 0
#   RX prints [diag] every 5 s: peak | syms | bits_buf | ts_out | sync — ts_out>0 means packets decoded
#   RX reedsolo must be installed in the correct venv: .venv/bin/python3.11 -m pip install reedsolo

# GNU Radio path — start the matching .grc flowgraph first, then:
python GNU/tx_arq_fec.py      # sends file as UDP to the flowgraph (RS FEC + ARQ)
python GNU/rx_arq_fec.py      # receives demodulated UDP from the flowgraph
```

Common `solo_pyth` flags: `--ip` (default `ip:pluto.local`; USB radios are usually
`ip:192.168.2.1`), `--freq` (half-duplex, default `433e6`), `--freq-a`/`--freq-b` (FDD,
default `2412e6`/`2437e6`), `--mod {bpsk,qpsk,16qam}`, `--sps` (samples per symbol, default
16). Auto-calibrating scripts also take `--skip-cal --rx-gain N --tx-atten N`.

## `solo_pyth/` — the shared DSP pipeline

Every `solo_pyth` script is a self-contained reimplementation of the same layered stack.
**There is no shared library** — the DSP core is copy-pasted across all files, so a fix in
one (e.g. the demodulator) must be replicated in the others. The layers, top to bottom:

1. **Hardware** (`setup_pluto`/`connect_pluto`) — `adi.Pluto`, 1 MHz sample rate, LO, RF
   bandwidth, gains, 65536-sample cyclic TX buffer, and **disables the Pluto's onboard DDS
   tone** via raw `iio` attribute writes (`cf-ad9361-dds-core-lpc`) — otherwise it
   interferes with TX. This DDS-disable block is in every script.
2. **Modulation** (`bits_to_symbols`/`symbols_to_bits`) — Gray-coded BPSK/QPSK/16QAM,
   1/2/4 bits per symbol, `complex64`.
3. **Pulse-shaping & framing** (`packet_to_iq_fill`/`iq_to_packets`) — prepends a Barker-13
   sync, upsamples by SPS, applies a Hamming-windowed FIR (`firwin`/`lfilter`), normalizes,
   and **tiles whole packet copies to fill the TX buffer** for continuous cyclic DMA. The
   newer scripts tile only *whole* copies + silence pad (never cut mid-packet); the original
   `pluto_filelink.py` tiles-and-truncates.
4. **Packet format** (`build_packet`/`parse_packet`) — wire layout
   `[MAGIC:1][type:1][seq:4][total:4][len:2][payload:N][crc32:4]`. CRC32 mismatch → drop.
   `MAGIC` differs per family (`0xA5` filelink, `0xBE` autolink/chat).
5. **CFO + sync recovery** — the key robustness machinery, and the main axis of evolution
   between variants (see below).
6. **ARQ** — selective repeat. Receiver reports missing sequence numbers (NACK/REQUEST),
   sender retransmits only those, looping until complete or stalled.

### Two demodulator generations (important)

- **Brute-force search** (`pluto_filelink.py`, `pluto_filelink_auto.py`, autolink/chat):
  every RX buffer is demod'd under **3 CFO hypotheses (none / FFT power-law / decision-
  directed PLL) × all constellation rotations × every symbol-timing offset**, keeping any
  combination that yields a valid CRC. Robust but expensive; QPSK's 4-fold phase ambiguity
  is handled by trying all 4 rotations.
- **Self-referencing preamble** (`pluto_filelink_auto_qpsk.py` — the most refined filelink):
  the preamble is **always real BPSK (Barker-13 ×3)** regardless of payload MOD. Correlating
  the complex symbol stream against it yields *both* timing *and* residual carrier phase, so
  the payload is derotated directly — resolving phase ambiguity for free and dropping the
  rotation/PLL search. Prefer this pattern when unifying the codebase.

### Half-duplex vs FDD

- **Half-duplex** (`pluto_filelink*`, `pluto_autolink`, `pluto_chat_auto`) share one
  frequency and **take turns**: TX a short cyclic burst, go silent, listen. Turn-taking is
  **handshake-driven, not timer-driven** — a phase advances only when the expected message
  is *decoded*. `pluto_chat_auto.py` formalizes this into three primitives
  (`exchange` / `hold_tone_until` / `echo_until`) and adds random burst jitter to break the
  lock-step hazard where two identical radios TX/listen in unison and never hear each other.
- **FDD** (`*_fdd`, `pluto_image_arq`) put TX and RX on **separate frequencies**
  (`--freq-a`/`--freq-b`), so both directions run simultaneously (background RX thread). The
  receiver's cyclic TX buffer keeps looping its last control frame, so control frames carry
  a **generation counter** and the sender drops stale/duplicate frames (the central bug the
  `_diag` rewrite fixes — see below).

### Auto-calibration

`*_auto*`, `pluto_autolink*`, and `pluto_chat_auto` calibrate the link **over the air before
data transfer**. Half-duplex runs a master-driven 4-phase sequence (A: master beacons so
slave sets RX gain → B: slave beacons so master sweeps its RX gain → C: TX-power negotiation,
sweeping attenuation weak→strong until the partner echoes → D: bidirectional READY
handshake). Calibration packet types are `0x10`–`0x14`. FDD calibrates both directions
simultaneously since the links are independent. Look here, not at fixed gain constants, when
debugging link reliability.

## `solo_pyth/` file map

| File | Duplex | Purpose / what's distinct |
|---|---|---|
| `pluto_filelink.py` | half | Baseline file transfer + ARQ + Tk GUI (progress bar & live thumbnail). No calibration. Brute-force demod, tile-and-truncate TX. |
| `pluto_filelink_auto.py` | half | filelink + 4-phase auto-calibration. Larger 256 KiB RX buffer, Barker bit-correlation sync, whole-packet tiling. `CHUNK_BYTES=256`. |
| `pluto_filelink_auto_qpsk.py` | half | **Most refined filelink.** Real-BPSK preamble (Barker ×3) that carries phase → direct QPSK derotation, no rotation search. |
| `pluto_filelink_auto_dep.py` | half | Deprecated (`_dep`). Smaller `CHUNK_BYTES=64` so more whole copies fit per RX buffer; superseded by the preamble approach. |
| `pluto_autolink.py` | half | Auto-calibration + push-to-talk chat. `MAGIC=0xBE`. |
| `pluto_chat_auto.py` | half | Refined autolink: handshake primitives + burst jitter. |
| `pluto_autolink_fdd.py` | FDD | Full-duplex chat over two frequencies (2.4 GHz ISM defaults); both sides calibrate at once. |
| `pluto_image_arq.py` | FDD | Round-based selective-repeat image transfer. `--codec auto/h264/jpeg` (H.264 single I-frame via ffmpeg, else JPEG via Pillow); codec travels in META. |
| `pluto_image_fdd.py` | FDD | FDD image transfer, state-machine ARQ (META→GO→DATA→NAK→DONE). `CHUNK_BYTES=48`. |
| `pluto_image_fdd_diag.py` | FDD | **v2 of image_fdd** — fixes NAK flooding (generation counter + switch to CTRL\|ACK), ANNOUNCE/GO desync, and PAUSE-timeout deadlock; adds a `Diag` health-report object (output prefixed `↯`). |
| `pluto_image_fdd_raw.py` | FDD | **Raw (uncompressed) image transfer, BPSK.** Fixed roles: `--role tx` (Pluto 1) sends image bytes on `--freq-data`, `--role rx` (Pluto 2) sends control on `--freq-ctrl`. **Binary** framing (not base64), 4-byte seq + 8-byte size → no practical size limit; **block-windowed** selective-repeat ARQ (GNU "super-block" idea) keeps per-round NAK lists bounded. md5-verified. Symmetric FDD auto-cal (`--skip-cal` to use `--rx-gain`/`--tx-atten`). **Cal tuning:** RX gain sweep starts at `max(30, lo)` dB (skips low-gain noise floor), TX atten sweep starts at -40 dB (not -80) for faster negotiation. **ARQ fix:** when no REQ or BACK is heard after `WAIT_TIMEOUT`, the sender re-queues the full block for retransmission instead of looping on EOR-only (was silently stalling block 0). **Has three known bugs fixed in `_raw2`** (see below). |
| `pluto_image_fdd_raw2.py` | FDD | **Bug-fixed version of `pluto_image_fdd_raw.py`.** Three fixes: **(1) TX/RX block desync deadlock** — when the BACK packet for block N is dropped, TX times out and retransmits block N while RX has already advanced to block N+1; TX ignores RX's REQ(N+1) and RX ignores TX's EOR(N), permanent deadlock. Fixed: TX treats REQ(seq > current_blk) as implicit BACK; RX re-sends BACK for any EOR from an already-completed block. **(2) Premature REQ flood** — `last_req = 0.0` caused the periodic-REQ timer to fire on the very first loop iteration of every block, requesting all 200 chunks before any DATA arrived. Fixed: `last_req = time.time()`. **(3) Stale META infinite loop** — Fix 3a drained stale META from `frame_q` using a loop that put non-META packets back and re-read them; if any DATA packets were already in the queue the loop never terminated (`queue.Empty` was never raised), hanging the receiver before it ever entered the block loop. Fixed: single-pass snapshot drain (drain all → filter → put non-META back). **Use this instead of `_raw.py` for Windows receivers.** |
| `pluto_image_fdd_raw2_qpsk.py` | FDD | **QPSK variant of `pluto_image_fdd_raw2.py`** (separate file; original left BPSK-only). Selectable `--mod {bpsk,qpsk}`, default **qpsk** (~2× BPSK throughput). Complex correlation against the always-real-BPSK Barker×3 preamble → timing + phase → direct QPSK derotation (no rotation search). `calibrate()` forces BPSK for the cal handshake (QPSK needs ~3 dB more SNR at `CAL_TX_ATTEN=-30 dB`). `--blast` mode for handshake-free decode debugging (TX loops forever, RX prints stats every 2 s). Offline round-trip verified at noise σ=60. On-air over coax: DATA channel (Windows→Linux) confirmed working; CTRL channel (Linux→Windows) needs adequate TX power — let calibration run or increase `--tx-atten` on the RX role. |
| `pluto_image_fdd_raw2_16qam.py` | FDD | **16QAM variant of `pluto_image_fdd_raw2.py`** (separate file). Selectable `--mod {bpsk,qpsk,16qam}`, default **16qam** (~4× BPSK throughput). Two key demod differences vs the QPSK file: **(1) amplitude normalization** — preamble correlation peak magnitude = `channel_gain × PREAMBLE_LEN`, so `derot = exp(-jφ) / amp_est` normalizes both phase and amplitude in one step, giving 16QAM the reference it needs; **(2) exact-length symbol decode** — after decoding the 12-byte header we know `plen`, so we read exactly `ceil((12+plen+4)×8 / BITS_PER_SYMBOL)` symbols (no tail ISI beyond the packet boundary). `calibrate()` forces BPSK. `--blast` mode included. **Status (2026-06-12): offline round-trip test failing** due to tail ISI at tight 16QAM copy boundaries — 16QAM packs 4× more bits/symbol so 7 copies fit in the TX buffer (vs 1 for BPSK), and the filter ISI from copy N bleeds into the CRC of copy N-1; exact-length decode not yet sufficient to avoid it. Root cause and fix under investigation. Use `--mod qpsk` on this file, or `pluto_image_fdd_raw2_qpsk.py`, as fallback. |
| `pluto_image_fdd_raw_cal.py` | FDD | **Option-B calibration rewrite of `pluto_image_fdd_raw.py`.** Keeps the original two-phase structure so both radios stay in the same phase simultaneously (critical for symmetric FDD). **Phase 1** runs the RX gain sweep with escalating TX power: starts at `CAL_TX_ATTEN` (-30 dB), and if the partner isn't heard, increases TX power by `CAL_POWER_STEP` (5 dB) and retries — up to 0 dB. **Phase 2** negotiates TX power weak→strong as before, but now with the correct RX gain already set. Fixes the original silent failure (fixed -30 dB beacon too weak → wrong fallback gain → Phase 2 also fails) without the sync issue of Option C (where fast-failing power levels completed near-instantly, causing the two radios to drift out of phase). |
| `pluto_image_fdd_raw_seq.py` | FDD | **Sequential-calibration variant of `pluto_image_fdd_raw.py`.** Data-transfer logic is identical (including the block re-queue ARQ fix); only calibration differs. Instead of calibrating both directions simultaneously, it configures one frequency channel completely before the other. **Phase 1** (DATA channel): `tx` beacons on `FREQ_DATA`, `rx` sweeps gain, then `tx` negotiates TX power until `rx` confirms. **Phase 2** (CTRL channel): `rx` beacons on `FREQ_CTRL`, `tx` sweeps gain, then `rx` negotiates TX power. Phases are separated by a `PKT_CAL_NEXT` (`0x22`) handshake so both radios advance together. The `total` field of every CAL packet encodes the current phase (1 or 2) to prevent cross-phase packet confusion. The sweeper transmits its rxok feedback at **max TX power (0 dB)** because the feedback channel is uncalibrated at that point. Easier to diagnose asymmetric link problems than the symmetric approach. |
| `pluto_image_fdd_raw_rxcal.py` | FDD | **RX-only calibration variant of `pluto_image_fdd_raw.py`.** Removes TX power negotiation and confirm-link phases entirely; TX attenuation is fixed at `TX_ATTEN_FINAL` (-10 dB) after the RX gain sweep. **Known bug (not yet fixed):** `calibrate_rx_gain` starts the sweep at `max(30, lo)` dB — borrowed from the original where the beacon TX power is -30 dB. With TX atten fixed at -10 dB the signal is much stronger, so the RX ADC saturates before the sweep finds a decodable gain. Fix: start the sweep from 0 dB when TX power is not at the weak calibration level. |
| `pluto_image_loop_tx.py` | FDD | **Continuous TX loop** — split-file companion to `pluto_image_loop_rx.py`. Runs on the Windows PC (sender). Loads the image once, transmits it repeatedly until Ctrl+C, printing KB/s per run. Auto-cal on first run prints `--skip-cal` values; subsequent runs use `--skip-cal --rx-gain N --tx-atten N`. |
| `pluto_image_loop_rx.py` | FDD | **Continuous RX loop** — runs on the Linux machine (receiver). Receives each transfer, verifies byte-exact MD5, saves with timestamp (`rx_YYYYMMDD_HHMMSS_fname`), prints `PASS`/`FAIL` with missing-chunk diagnostics, then loops back for the next run. Session pass/fail totals on exit. |
| `pluto_gain_control.py` | — | Standalone Tk GUI to live-tune TX atten / RX gain / AGC mode + RSSI. Run alongside any link. |
| `pluto_link_debug.py` | FDD | **FDD link diagnostic tool.** Run on both PCs simultaneously (`--role tx` on Linux, `--role rx` on Windows). Phase 1 tests the DATA channel (Linux→Windows); Phase 2 tests the CTRL channel (Windows→Linux — the likely failure point). Sweeps TX attenuation at each level, counts decoded packets, and prints a diagnosis with recommended fixes. Also has `--quick` mode: 30-second passive sniff with no TX, reports raw signal amplitude and correlation peak even when packets don't decode (distinguishes "no RF" from "RF present but can't decode"). |
| `pluto_dvbt_video.py` | FDD | **DVB-T live video streaming (GNU Radio variant).** Third architectural pattern — **not** pure pyadi-iio. Uses **GNU Radio Python bindings inline** (`gr.top_block` subclasses). TX (Windows): GStreamer → UDP:2000 → GR DVB-T encode → `iio.fmcomms2_sink_fc32`. RX (Linux): `iio.fmcomms2_source_fc32` → GR DVB-T decode → UDP:2001 → GStreamer. T2k/QPSK/C7_8, 3.2 MSPS. Requires `gnuradio`, `gnuradio-dtv`. |
| `pluto_dvbt2.py` | FDD | **DVB-T live video streaming (no GNU Radio).** Same protocol as `pluto_dvbt_video.py` but implements the **full DVB-T T2k/QPSK/C7_8 chain in pure Python/NumPy** — no `gnuradio` import. TX: GStreamer (UDP:2000) → energy dispersal → RS(204,188) → Forney interleaver → rate-7/8 punctured conv encoder → bit/symbol inner interleavers → QPSK map → pilot insertion + IFFT → cyclic prefix → pyadi-iio Pluto. RX: pyadi-iio → Schmidl-Cox OFDM timing+CFO → FFT → pilot-based channel eq → QPSK demap → symbol/bit deinterleavers → Viterbi(C7/8) → Forney deinterleaver → RS decode → energy descramble → UDP:2001 → GStreamer. All DVB-T tables generated from ETSI EN 300 744. Dependencies: `pyadi-iio`, `numpy`, `reedsolo`, GStreamer. **Includes BPSK symmetric FDD auto-calibration** (`calibrate_dvbt()`, `--skip-cal` to skip). **Bugs fixed this session:** (1) `RSCodec(nroots=...)` → `RSCodec(nsym=...)` — reedsolo 1.7.0 dropped the `nroots` kwarg; (2) `np.interp` on complex channel estimates → split real/imaginary interpolation; (3) interleaver direction — both `symbol_inner_deinterleave` and `bit_inner_deinterleave` used inverse-permutation scatter instead of forward gather (`c[R]` not `out[R]=c`); (4) inter-buffer sample loss — `DvbtDecoder._carry` preserves leftover samples across `push_samples` calls so OFDM symbol boundaries stay aligned; (5) Windows GStreamer not on PATH — `_find_gst_launch()` searches `C:\gstreamer\1.0\...\bin\`. **Status (2026-06-12):** loopback perfect (`pscore=0.999`, byte-exact); on air `sync=YES` but `frame=NO`, `pscore≈0.85`, `ts_out=0`. Narrowed offline to the **integer-CFO acquisition locking a wrong alias** — continual-pilot coherence truly peaks at `shift≈+83` but the decoder locks `≈−60`, where pilots partially cohere (`pscore≈0.85`) yet equalized data EVM ≈ 1.0 (noise). See the root `CLAUDE.md` dvbt2 deep-dive for the full diagnosis, the `_rx_*` offline harness, and the next step. Real-time playback also blocked separately by the slow pure-Python Viterbi. |

When unifying toward the Python-only goal, the `_auto_qpsk` filelink and `_fdd_diag` image
transfer are the most evolved endpoints of their respective lineages. `pluto_dvbt2.py` is the
no-GNU-Radio DVB-T video path; `pluto_dvbt_video.py` is retained as a reference.

## `GNU/` — the UDP bridge to a GNU Radio flowgraph

These are **not DSP scripts** — all modulation lives in an external `.grc` flowgraph. The
Python side does only file I/O, FEC, framing, and ARQ over **localhost UDP**:

- **Port 2000** — TX script → flowgraph (data to modulate & transmit)
- **Port 2001** — flowgraph → RX script (demodulated data)
- **Port 2002** — reverse channel for `NACK`/`DONE` control (ARQ variants only)

Three FEC strategies (each a tx/rx pair sharing constants):

| Pair | FEC | ARQ | Notes |
|---|---|---|---|
| `tx/rx_arq_fec.py` | Reed-Solomon `RSCodec(16)` | yes | Header `JPG!`, `CHUNK_SIZE=948`. Single RS encode per chunk. |
| `tx/rx_arq_qpsk.py` | Reed-Solomon, **split 4×237 B** | yes | Slices the 948 B chunk into four ≤255 B pieces to fit RS's 255-byte limit; tighter pacing for a QPSK flowgraph. |
| `tx/rx_qpsk_zfec.py` | **ZFEC erasure code** | no | One-way. Super-blocks of `K_MAX=200` chunks, `m=k*1.25` (25% redundancy); header `ZF`. Pure erasure recovery, no retransmit. |

## Development environment

### Obsidian MCP (broken on this machine — 2026-06-08)

The `mcp__MCP_DOCKER__obsidian_*` tools fail with `[Errno 101] Network is unreachable`.
Two compounding root causes:

1. **Docker Desktop uses gVisor networking** (`NetworkType: gvisor` in
   `~/.docker/desktop/settings-store.json`). gVisor's isolated kernel returns
   `ENETUNREACH` on any attempt to reach host interfaces from inside a container.
2. **Obsidian's Local REST API plugin only binds to `127.0.0.1:27124`** (HTTPS).
   Even with normal Docker networking the container (running inside Docker Desktop's
   VM) cannot reach the host's loopback. The `mcp/obsidian` image has no env vars to
   change protocol or port — only `OBSIDIAN_HOST` and `OBSIDIAN_API_KEY` are exposed.

**Recommended fix:** run `mcp-obsidian` directly on the host (not in Docker) by
changing `~/.claude.json` from `docker mcp gateway run --profile obs` to a local
`uvx mcp-obsidian` or `pip install mcp-obsidian` invocation. This lets the process
connect to `localhost:27124` natively and bypasses all VM networking issues.

Obsidian plugin config: `~/Desktop/Nav/.obsidian/plugins/obsidian-local-rest-api/data.json`
API key: `f8a1a984ae0b0df5c04cd935a9b98dfc7a6cbe5b21998ea3443e88700b2e3cb0`

## Things to watch

- **Hardcoded Windows paths** remain in `GNU/` scripts
  (`IMAGE_FILE = "C:/Users/Nkannan4/Downloads/Pluto/test.jpg"`). `test.jpg` in the repo root
  is the sample payload — repoint these before running on this Linux machine.
- File suffixes encode lineage, not just options: `_auto` (over-the-air calibration),
  `_qpsk` (the BPSK-preamble demod / QPSK-tuned GNU variant), `_dep` (deprecated),
  `_diag` (bugfixed v2 with diagnostics). Check the docstring at the top of each file — they
  are detailed and explain the specific tuning trade-offs (chunk sizes, burst timing, RX
  buffer sizing) behind each variant.
- Best practice noted in the FDD docstrings: develop **conducted** (Pluto TX → attenuator →
  Pluto RX over coax) rather than radiating, and mind local RF regulations.
- **Pluto unit asymmetry (confirmed):** the two physical Pluto radios have different TX output
  strength. In FDD the `--role rx` unit must TX on the control channel (FREQ_CTRL) to deliver
  GO/REQ/BACK frames back to the sender — if that unit is the weaker one, control frames don't
  make it back and the sender stalls. Rule: **assign the stronger Pluto to `--role rx`** (it
  handles CTRL TX). The weaker Pluto as `--role tx` is fine because calibration adjusts for it.
  Symptom of getting this wrong: Windows→Linux works, Linux→Windows always fails (or vice
  versa). Fix without swapping: add `--tx-atten -5` or `--tx-atten 0` on the weaker unit when
  it is in the RX role.
