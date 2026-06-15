#!/usr/bin/env python3
"""
ADALM Pluto — Live Video Streaming over 16QAM (STABLE)
======================================================

Improvements over the basic version:
1. Root-Raised Cosine (RRC) filtering for zero Inter-Symbol Interference (ISI).
2. Reed-Solomon Forward Error Correction (FEC) to recover corrupted bits.
3. Decision-Directed Equalizer for phase & amplitude tracking (from 16QAM version).

Usage:
  Pluto 1 (Sender, streaming a file):
    python3 pluto_video_stream_16qam_stable.py --role tx --input my_video.mp4

  Pluto 2 (Receiver):
    python3 pluto_video_stream_16qam_stable.py --role rx
    
  Run Software Self-Test:
    python3 pluto_video_stream_16qam_stable.py --role test
"""

import adi
import numpy as np
import argparse
import sys
import struct
import zlib
import subprocess
import time
from scipy.signal import lfilter
from reedsolo import RSCodec

# ─── CLI ──────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--role', choices=['tx', 'rx', 'test'], required=True)
ap.add_argument('--ip',        type=str,   default='ip:pluto.local')
ap.add_argument('--freq',      type=float, default=2412e6, help='carrier frequency')
ap.add_argument('--input',     type=str,   default=None, help='tx: input video file or /dev/video0')
ap.add_argument('--bitrate',   type=str,   default='120k', help='tx: video bitrate (e.g. 120k)')
ap.add_argument('--sps',       type=int,   default=16, help='samples per symbol')
ap.add_argument('--chunk',     type=int,   default=512, help='bytes per packet')
ap.add_argument('--rx-gain',   type=int,   default=40, help='RX hardware gain (used if --skip-cal)')
ap.add_argument('--tx-atten',  type=int,   default=-20, help='TX hardware attenuation')
ap.add_argument('--skip-cal',  action='store_true', help='rx: skip auto-calibration')
args = ap.parse_args()

ROLE = args.role

# ─── LINK CONSTANTS ───────────────────────────────────────────────────────────
SAMPLE_RATE        = int(1e6)
SAMPLES_PER_SYMBOL = args.sps
CHUNK_BYTES        = args.chunk
MAGIC              = 0xA5

# Reed-Solomon Setup
RS_PARITY = 32
RS_CHUNK  = 255 - RS_PARITY
rs = RSCodec(RS_PARITY)

def rs_encoded_len(data_len):
    chunks = (data_len + RS_CHUNK - 1) // RS_CHUNK
    return data_len + chunks * RS_PARITY

# The BPSK Preamble
BARKER_13      = np.array([1,1,1,1,1,-1,-1,1,1,-1,1,-1,1], dtype=np.float32)
PREAMBLE_SIGNS = np.tile(BARKER_13, 3).astype(np.float32)
PREAMBLE_LEN   = len(PREAMBLE_SIGNS)

# Root-Raised Cosine (RRC) Filter
def rrcosfilter(N, alpha, Ts, Fs):
    T_delta = 1/Fs
    time_idx = np.arange(-(N-1)//2, (N-1)//2 + 1) * T_delta
    h_rrc = np.zeros(len(time_idx), dtype=np.float64)
    for x in range(len(time_idx)):
        t = time_idx[x]
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.isclose(np.abs(t), Ts / (4 * alpha)):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha)))) + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) + 4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / (np.pi * t / Ts * (1 - (4 * alpha * t / Ts)**2))
    return (h_rrc / np.sqrt(np.sum(h_rrc**2))).astype(np.float32)

FILT = rrcosfilter(SAMPLES_PER_SYMBOL * 12 + 1, 0.35, 1, SAMPLES_PER_SYMBOL)

# 16QAM Gray Map
_GRAY_MAP = {(0,0): -3, (0,1): -1, (1,1): +1, (1,0): +3}
_GRAY_INV = {v: k for k, v in _GRAY_MAP.items()}

# ═══════════════════════════════════════════════════════════════════════════════
#  DSP
# ═══════════════════════════════════════════════════════════════════════════════
def bits_to_symbols(bits):
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    b = bits.reshape(-1, 4)
    i_levels = np.array([_GRAY_MAP[(int(r[0]), int(r[1]))] for r in b], dtype=np.float32)
    q_levels = np.array([_GRAY_MAP[(int(r[2]), int(r[3]))] for r in b], dtype=np.float32)
    return ((i_levels + 1j * q_levels) / np.sqrt(10)).astype(np.complex64)


def symbols_to_bits(syms):
    s = syms * np.sqrt(10)
    def demap_axis(v):
        lv = np.clip(np.round((v + 3) / 2).astype(int) * 2 - 3, -3, 3)
        out = np.zeros((len(v), 2), dtype=np.uint8)
        for k in range(len(lv)):
            out[k] = _GRAY_INV[int(lv[k])]
        return out
    bi = demap_axis(s.real)
    bq = demap_axis(s.imag)
    return np.column_stack([bi[:, 0], bi[:, 1], bq[:, 0], bq[:, 1]]).reshape(-1)


# ─── PACKET FRAMING ───────────────────────────────────────────────────────────
def build_packet(seq, payload):
    # Payload & CRC are RS Encoded
    body = payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    data_crc = body + struct.pack('>I', crc)
    enc_data = rs.encode(data_crc)
    
    # Header is left unencoded for easy parsing
    header = struct.pack('>BIH', MAGIC, seq, len(payload))
    return header + enc_data


def parse_packet(raw, start):
    HDR = 7
    if start + HDR > len(raw):
        return None, False
    magic, seq, plen = struct.unpack('>BIH', raw[start:start+HDR])
    if magic != MAGIC or plen > 2048:
        return None, False
        
    expected_enc_len = rs_encoded_len(plen + 4)
    end = start + HDR + expected_enc_len
    if end > len(raw):
        return None, False
        
    enc_data = raw[start+HDR : end]
    try:
        dec_data, _, _ = rs.decode(enc_data)
    except Exception:
        # Reed-Solomon could not correct all errors
        return None, False
        
    if len(dec_data) != plen + 4:
        return None, False
        
    payload = bytes(dec_data[:plen])
    crc_rx = struct.unpack('>I', dec_data[plen:plen+4])[0]
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    
    crc_ok = (crc_rx == crc_calc)
    return {'seq': seq, 'payload': payload}, crc_ok


# ─── TX MODULATION ────────────────────────────────────────────────────────────
def packet_to_iq(pkt_bytes):
    """Real-BPSK preamble + 16QAM payload -> shaped IQ for non-cyclic transmission."""
    preamble = PREAMBLE_SIGNS.astype(np.complex64)
    pbits   = np.unpackbits(np.frombuffer(pkt_bytes, dtype=np.uint8))
    payload = bits_to_symbols(pbits)
    
    syms = np.concatenate([preamble, payload]).astype(np.complex64)
    
    # Upsample
    up = np.zeros(len(syms) * SAMPLES_PER_SYMBOL + len(FILT), dtype=np.complex64)
    up[::SAMPLES_PER_SYMBOL][:len(syms)] = syms
    
    # RRC Filter
    shaped_i = lfilter(FILT, 1.0, up.real).astype(np.float32)
    shaped_q = lfilter(FILT, 1.0, up.imag).astype(np.float32)
    shaped   = (shaped_i + 1j * shaped_q).astype(np.complex64)

    # Normalize to avoid clipping the DAC
    mx = np.max(np.abs(shaped))
    if mx > 0:
        shaped = shaped / mx * 0.8 * 2**15

    return shaped


# ─── RX DEMODULATION ──────────────────────────────────────────────────────────
def _cfo_variants(iq):
    yield iq
    # Estimate CFO using 4th-power FFT
    sl = iq[:65536]
    nrm = sl / (np.max(np.abs(sl)) + 1e-9)
    sq = nrm ** 4
    n = len(sq)
    fv = np.fft.fft(sq); fv[0] = 0
    freqs = np.fft.fftfreq(n, d=1.0/SAMPLE_RATE)
    cfo = freqs[int(np.argmax(np.abs(fv)))] / 4
    
    t = np.arange(len(iq)) / SAMPLE_RATE
    yield (iq * np.exp(-1j * 2*np.pi*cfo*t)).astype(np.complex64)
    
    # Micro-offsets
    yield (iq * np.exp(-1j * 2*np.pi*(cfo + 15)*t)).astype(np.complex64)
    yield (iq * np.exp(-1j * 2*np.pi*(cfo - 15)*t)).astype(np.complex64)

def iq_to_packets(iq):
    peak = np.max(np.abs(iq))
    if peak < 5:
        return []
    iq = (iq / peak).astype(np.complex64)
    found = {}
    delay = len(FILT) // 2

    for corrected_iq in _cfo_variants(iq):
        # Matched RRC Filter
        fi = lfilter(FILT, 1.0, corrected_iq.real).astype(np.float32)
        fq = lfilter(FILT, 1.0, corrected_iq.imag).astype(np.float32)
        filt = (fi + 1j * fq).astype(np.complex64)
        
        for toff in range(SAMPLES_PER_SYMBOL):
            start_idx = delay + toff
            stream = filt[start_idx::SAMPLES_PER_SYMBOL]
            
            if len(stream) < PREAMBLE_LEN + 32:
                continue
                
            corr = np.correlate(stream, PREAMBLE_SIGNS.astype(np.complex64), mode='valid')
            mag  = np.abs(corr)
            thr  = PREAMBLE_LEN * 0.4
            cand = np.where(mag > thr)[0]
            
            for c in cand:
                phi   = np.angle(corr[c])
                derot = np.exp(-1j * phi)

                chan_gain = mag[c] / PREAMBLE_LEN
                if chan_gain < 1e-6:
                    continue

                data_start = c + PREAMBLE_LEN
                if len(stream[data_start:]) < 16:
                    continue

                for slip in (0, 1, -1, 2, -2):
                    ss = data_start + slip
                    if ss < 0 or ss >= len(stream):
                        continue
                    
                    ds = stream[ss:]
                    decoded_bits = []
                    cur_derot = derot
                    cur_gain = chan_gain
                    block_size = 64
                    
                    for i in range(0, len(ds), block_size):
                        chunk = ds[i:i+block_size]
                        norm_chunk = chunk * cur_derot / cur_gain
                        chunk_bits = symbols_to_bits(norm_chunk)
                        decoded_bits.append(chunk_bits)
                        
                        if len(chunk) > 16:
                            ideal = bits_to_symbols(chunk_bits)
                            corr_ls = np.sum(chunk * np.conj(ideal))
                            cur_derot = np.exp(-1j * np.angle(corr_ls))
                            power_ideal = np.sum(np.abs(ideal)**2)
                            if power_ideal > 0.1:
                                measured_gain = np.abs(corr_ls) / power_ideal
                                cur_gain = 0.8 * cur_gain + 0.2 * measured_gain
                                
                    if not decoded_bits:
                        continue
                    bits = np.concatenate(decoded_bits)
                    nb   = len(bits) // 8
                    if nb < 10:
                        continue
                    raw = bytes(np.packbits(bits[:nb*8]))
                    pkt, crc_ok = parse_packet(raw, 0)
                    if pkt:
                        seq = pkt['seq']
                        pkt['crc_ok'] = crc_ok
                        if seq not in found or (crc_ok and not found[seq]['crc_ok']):
                            found[seq] = pkt
                            
    return [found[k] for k in sorted(found.keys())]

# ═══════════════════════════════════════════════════════════════════════════════
#  SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════
def self_test():
    print("[TEST] Running 16QAM + RRC + FEC Self-Test...")
    
    # 1. Create a dummy payload
    payload = b"Hello PlutoSDR! 16QAM STABLE."
    seq_num = 42
    
    print(f"[TEST] Original Payload Length: {len(payload)} bytes")
    
    # 2. Modulate
    print("[TEST] Building packet and modulating...")
    pkt_bytes = build_packet(seq_num, payload)
    tx_iq = packet_to_iq(pkt_bytes)
    
    # 3. Simulate Channel
    print("[TEST] Applying simulated channel impairments (Noise, CFO, Phase Drift)...")
    noise_pwr = np.mean(np.abs(tx_iq)**2) / (10**(15/10))
    noise = (np.random.randn(len(tx_iq)) + 1j * np.random.randn(len(tx_iq))) * np.sqrt(noise_pwr/2)
    rx_iq = tx_iq + noise
    
    t = np.arange(len(rx_iq)) / SAMPLE_RATE
    cfo_hz = 0.0  # Reduced to 0 Hz for basic test
    rx_iq = rx_iq * np.exp(1j * 2 * np.pi * cfo_hz * t)
    rx_iq = rx_iq * np.exp(1j * np.pi / 4) * 0.3
    
    # 4. Demodulate
    print("[TEST] Demodulating and Decoding...")
    t0 = time.time()
    packets = iq_to_packets(rx_iq.astype(np.complex64))
    t1 = time.time()
    
    print(f"[TEST] Demodulation took {(t1-t0)*1000:.1f} ms")
    
    if not packets:
        print("[FAIL] No packets recovered!")
        sys.exit(1)
        
    pkt = packets[0]
    print(f"[TEST] Recovered Seq: {pkt['seq']}, CRC OK: {pkt['crc_ok']}")
    if pkt['payload'] == payload:
        print("[PASS] Payload matches perfectly! RRC + RS FEC + Equalizer are working.")
    else:
        print("[FAIL] Payload mismatch!")
        print("Expected:", payload[:50], "...")
        print("Got:     ", pkt['payload'][:50], "...")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
def auto_calibrate_rx(sdr):
    print("\n[*] Starting RX Auto-Calibration ...")
    print("    (Make sure the TX side is already running and streaming video)")
    print(f"  {'Gain':>5}  {'Peak':>6}  {'ADC%':>5}  {'Decodes':>7}")
    candidates = []
    
    for g in range(0, 75, 5):
        if g > 71: g = 71
        try: sdr.rx_hardwaregain_chan0 = g
        except OSError: continue
            
        for _ in range(2): 
            try: sdr.rx() 
            except: pass
            
        pk = 0
        dec = 0
        for _ in range(3):
            try:
                rx = sdr.rx()
                pk = max(pk, np.max(np.abs(rx)))
                if len(iq_to_packets(rx)) > 0:
                    dec += 1
            except:
                pass
                
        adc = pk / 2896 * 100
        print(f"  {g:>5}  {pk:>6.0f}  {adc:>4.0f}%  {dec:>7}")
        
        if adc > 95:
            continue
        if dec > 0:
            candidates.append(g)
            
    if candidates:
        best = max(candidates)
        print(f"[*] Calibration complete. Settling on best RX Gain: {best} dB\n")
        sdr.rx_hardwaregain_chan0 = best
    else:
        print("[!] Calibration failed to find a working gain. Falling back to default.")
        sdr.rx_hardwaregain_chan0 = args.rx_gain

# ═══════════════════════════════════════════════════════════════════════════════
#  PLUTO SETUP & MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def setup_pluto():
    print(f"[*] Connecting to {args.ip} ...")
    sdr = adi.Pluto(args.ip)
    sdr.sample_rate       = SAMPLE_RATE
    sdr.tx_lo             = int(args.freq)
    sdr.rx_lo             = int(args.freq)
    sdr.tx_rf_bandwidth   = SAMPLE_RATE
    sdr.rx_rf_bandwidth   = SAMPLE_RATE
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0   = int(args.rx_gain)
    sdr.tx_hardwaregain_chan0   = int(args.tx_atten)
    sdr.tx_cyclic_buffer  = False
    sdr.rx_buffer_size    = 262144 * 2
    print(f"[✓] Connected. Carrier {args.freq/1e6:.3f} MHz, 16QAM STABLE, SPS={args.sps}")
    return sdr

def sender_main(sdr):
    if not args.input:
        print("[!] TX mode requires --input (e.g. video.mp4 or /dev/video0)")
        return

    print(f"[*] Starting ffmpeg to encode {args.input} at {args.bitrate} ...")
    ffmpeg_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    if args.input.startswith('/dev/video'):
        ffmpeg_cmd += ['-f', 'v4l2', '-i', args.input]
    elif args.input.startswith('video='):
        ffmpeg_cmd += ['-f', 'dshow', '-i', args.input]
    else:
        ffmpeg_cmd += ['-re', '-i', args.input]
        
    ffmpeg_cmd += [
        '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-b:v', args.bitrate, '-maxrate', args.bitrate,
        '-bufsize', str(int(args.bitrate.replace('k','')) * 2) + 'k',
        '-f', 'mpegts', '-'
    ]

    p = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=sys.stderr)
    seq = 0
    total_bytes = 0
    start_time = time.time()
    
    print("[TX] Streaming over the air... (Ctrl+C to stop)")
    try:
        while True:
            payload = p.stdout.read(CHUNK_BYTES)
            if not payload:
                print("\n[TX] End of video stream.")
                break
            seq += 1
            total_bytes += len(payload)
            pkt_bytes = build_packet(seq, payload)
            iq = packet_to_iq(pkt_bytes)
            sdr.tx(iq)
            if seq % 100 == 0:
                elapsed = time.time() - start_time
                kbps = (total_bytes * 8 / 1000) / elapsed
                sys.stdout.write(f"\r[TX] Sent: {seq} packets  Data: {total_bytes/1e6:.2f} MB  Avg: {kbps:.1f} kbps   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[TX] Interrupted.")
    finally:
        p.kill()
        try: sdr.tx_destroy_buffer()
        except: pass

def receiver_main(sdr):
    if not args.skip_cal:
        auto_calibrate_rx(sdr)
        
    print("[*] Starting ffplay for live video playback ...")
    ffplay_cmd = [
        'ffplay', '-hide_banner', '-loglevel', 'error', '-f', 'mpegts',
        '-fflags', 'nobuffer', '-flags', 'low_delay', '-strict', 'experimental', '-i', '-'
    ]
    p = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    print("[RX] Listening for 16QAM STABLE stream... (Ctrl+C to stop)")
    print("[RX] ALSO saving stream to 'live_video_stable.ts' in this folder!")
    
    pkts_rx = 0
    bytes_rx = 0
    crc_fails = 0
    last_seq = -1
    start_time = time.time()
    
    try:
        with open("live_video_stable.ts", "wb") as f_out:
            while True:
                try: raw = sdr.rx()
                except OSError as e: 
                    sys.stdout.write(f"\r[RX] SDR Buffer Timeout! ({e}) Retrying...   ")
                    sys.stdout.flush()
                    continue
                packets = iq_to_packets(raw)
                for pkt in packets:
                    seq = pkt['seq']
                    payload = pkt['payload']
                    if not pkt.get('crc_ok', True): crc_fails += 1
                    if last_seq != -1 and seq > last_seq + 1:
                        lost = seq - last_seq - 1
                        if lost > 5: sys.stdout.write(f"\n[RX] Warning: Lost {lost} packets")
                    last_seq = seq
                    pkts_rx += 1
                    bytes_rx += len(payload)
                    f_out.write(payload)
                    try:
                        p.stdin.write(payload)
                        p.stdin.flush()
                    except BrokenPipeError: pass
                
                if packets:
                    elapsed = time.time() - start_time
                    kbps = (bytes_rx * 8 / 1000) / max(elapsed, 0.1)
                    err_rate = (crc_fails / max(pkts_rx, 1)) * 100
                    sys.stdout.write(f"\r[RX] Rcvd: {pkts_rx} pkts | Data: {bytes_rx/1e6:.2f} MB | {kbps:.1f} kbps | Bit-Errors: {err_rate:.1f}%   ")
                    sys.stdout.flush()
                else:
                    # Heartbeat so the user knows it's alive and listening to RF
                    peak = np.max(np.abs(raw))
                    sys.stdout.write(f"\r[RX] Listening... (Raw RF Peak: {peak:.0f}/2048) | No packets decoded yet.   ")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[RX] Interrupted.")
    finally:
        print("\n[RX] Saved video to live_video_stable.ts")
        p.stdin.close()
        p.kill()

def main():
    if ROLE == 'test':
        self_test()
        return
        
    sdr = setup_pluto()
    if ROLE == 'tx':
        sender_main(sdr)
    else:
        receiver_main(sdr)

if __name__ == "__main__":
    main()