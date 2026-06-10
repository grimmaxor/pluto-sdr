#!/usr/bin/env python3
"""
dvbt_video.py — DVB-T video streaming over PlutoSDR / FMCOMMS2

TX: reads a video file, pipes it as MPEG-TS over UDP into a DVB-T GNU Radio
    transmitter, and sends it out over the air.

RX: runs a DVB-T GNU Radio receiver, pipes decoded MPEG-TS out over UDP, and
    hands it to GStreamer for playback or saving.

Usage:
  python3 dvbt_video.py --tx [--file FILE] [--freq 2.4e9] [--uri ip:192.168.2.1] [--attn 8]
  python3 dvbt_video.py --rx [--freq 2.4e9] [--uri ip:192.168.2.1] [--gain 30] [--save out.ts]
"""

import os
import sys

# ── libiio shim fix: must happen before importing gnuradio/iio ─────────────
_LIBIIO_COMPAT = '/usr/lib/x86_64-linux-gnu/libiio.so.0.23'
if os.path.exists(_LIBIIO_COMPAT):
    current = os.environ.get('LD_PRELOAD', '')
    if _LIBIIO_COMPAT not in current:
        os.environ['LD_PRELOAD'] = _LIBIIO_COMPAT + ((':' + current) if current else '')
        os.execv(sys.executable, [sys.executable] + sys.argv)
# ───────────────────────────────────────────────────────────────────────────

import argparse
import signal
import subprocess
import threading
import time

# ── DVB-T / OFDM constants (T2k mode, GI=1/32, QPSK, rate 7/8) ────────────
FFT_LEN   = 2048
CP_LEN    = 64          # GI_1_32 = 2048/32
DATA_CARR = 1512        # active data subcarriers in T2k
OCC_TONES = 6817        # occupied tones for symbol acquisition
SAMP_RATE = 3_200_000
RF_BW     = 4_000_000

# UDP ports bridging GStreamer ↔ GNU Radio
GR_UDP_IN_PORT  = 2000   # GStreamer → GR TX
GR_UDP_OUT_PORT = 2001   # GR RX → GStreamer


# ── File picker ──────────────────────────────────────────────────────────────

def pick_file():
    """Open a GUI file dialog, or fall back to stdin."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            title="Select video file to transmit",
            filetypes=[
                ("Video / TS files", "*.ts *.mp4 *.mkv *.avi *.mov *.m2ts"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path.strip()
    except Exception:
        return input("Path to video file: ").strip()


# ── GStreamer pipeline builders ──────────────────────────────────────────────

def _gst_tx_cmd(filepath):
    """Return gst-launch command list to stream FILE → UDP port 2000."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.ts', '.m2ts', '.mpg', '.mpeg'):
        # Already MPEG-TS: parse and send directly.
        # alignment=7 → 7×188 = 1316 B per UDP packet, matching GR payloadsize.
        return [
            'gst-launch-1.0', '-v',
            'filesrc', f'location={filepath}',
            '!', 'tsparse', 'alignment=7',
            '!', 'udpsink', 'host=127.0.0.1', f'port={GR_UDP_IN_PORT}', 'sync=false',
        ]
    else:
        # Re-encode to H.264/AAC MPEG-TS at 2 Mbit/s (fits in DVB-T T2k/QPSK capacity).
        return [
            'gst-launch-1.0', '-v',
            'filesrc', f'location={filepath}',
            '!', 'decodebin', 'name=d',
            'd.',
            '!', 'queue',
            '!', 'videoconvert',
            '!', 'x264enc', 'tune=zerolatency', 'bitrate=2000',
            '!', 'queue', '!', 'mux.',
            'd.',
            '!', 'queue',
            '!', 'audioconvert',
            '!', 'avenc_aac',
            '!', 'queue', '!', 'mux.',
            'mpegtsmux', 'name=mux', 'alignment=7',
            '!', 'udpsink', 'host=127.0.0.1', f'port={GR_UDP_IN_PORT}', 'sync=false',
        ]


def _gst_rx_cmd(save_path=None):
    """Return gst-launch command list to receive from UDP port 2001."""
    if save_path:
        return [
            'gst-launch-1.0', '-v',
            'udpsrc', f'port={GR_UDP_OUT_PORT}',
            'caps=video/mpegts,systemstream=(boolean)true',
            '!', 'filesink', f'location={save_path}',
        ]
    else:
        # Decode and play; decodebin handles H.264+AAC or any other codec.
        return [
            'gst-launch-1.0', '-v',
            'udpsrc', f'port={GR_UDP_OUT_PORT}',
            'caps=video/mpegts,systemstream=(boolean)true',
            '!', 'decodebin', 'name=d',
            'd.', '!', 'queue', '!', 'videoconvert', '!', 'autovideosink',
            'd.', '!', 'queue', '!', 'audioconvert', '!', 'autoaudiosink',
        ]


# ── GNU Radio flowgraphs ─────────────────────────────────────────────────────

from gnuradio import gr, dtv, digital, blocks
from gnuradio import fft as gr_fft
from gnuradio.fft import window
from gnuradio import iio, network


class DVBTTransmitter(gr.top_block):
    """
    UDP source (port 2000, MPEG-TS)
      → DVB-T encode chain (RS + conv interleave + inner code + QPSK map + pilots + IFFT)
      → OFDM cyclic prefix
      → PlutoSDR / FMCOMMS2 TX
    """
    def __init__(self, uri, freq, attn):
        super().__init__("DVB-T TX", catch_exceptions=True)

        # ── Source: MPEG-TS bytes from GStreamer ──────────────────────────
        # payloadsize=1316 = 7 × 188 B MPEG-TS packets per UDP datagram
        self.udp_src = network.udp_source(
            gr.sizeof_char, 1, GR_UDP_IN_PORT, 0, 1316, False, False, False)

        # ── DVB-T encode chain ────────────────────────────────────────────
        self.energy_disp = dtv.dvbt_energy_dispersal(1)
        self.rs_enc      = dtv.dvbt_reed_solomon_enc(2, 8, 0x11d, 255, 239, 8, 51, 8)
        self.conv_il     = dtv.dvbt_convolutional_interleaver(136, 12, 17)
        self.inner_coder = dtv.dvbt_inner_coder(1, DATA_CARR, dtv.MOD_QPSK, dtv.NH, dtv.C7_8)
        self.bit_il      = dtv.dvbt_bit_inner_interleaver(DATA_CARR, dtv.MOD_QPSK, dtv.NH, dtv.T2k)
        self.sym_il      = dtv.dvbt_symbol_inner_interleaver(DATA_CARR, dtv.T2k, 1)
        self.mapper      = dtv.dvbt_map(DATA_CARR, dtv.MOD_QPSK, dtv.NH, dtv.T2k, 1)
        # reference_signals: inserts pilots + IFFT → outputs 2048 time-domain complex vectors
        self.ref_sigs    = dtv.dvbt_reference_signals(
                               gr.sizeof_gr_complex, DATA_CARR, FFT_LEN,
                               dtv.MOD_QPSK, dtv.NH, dtv.C7_8, dtv.C7_8,
                               dtv.GI_1_32, dtv.T2k, 1, 0)
        self.cyclic_pfx  = digital.ofdm_cyclic_prefixer(FFT_LEN, FFT_LEN + CP_LEN, 0, '')

        # ── Sink: radio ───────────────────────────────────────────────────
        self.radio = iio.fmcomms2_sink_fc32(
            uri if uri else iio.get_pluto_uri(), [True, False], 32768, False)
        self.radio.set_bandwidth(RF_BW)
        self.radio.set_frequency(int(freq))
        self.radio.set_samplerate(int(SAMP_RATE))
        self.radio.set_attenuation(0, float(attn))
        self.radio.set_filter_params('Auto', '', 0, 0)

        # ── Connect ───────────────────────────────────────────────────────
        self.connect(
            self.udp_src,
            self.energy_disp,
            self.rs_enc,
            self.conv_il,
            self.inner_coder,
            self.bit_il,
            self.sym_il,
            self.mapper,
            self.ref_sigs,
            self.cyclic_pfx,
            self.radio,
        )


class DVBTReceiver(gr.top_block):
    """
    PlutoSDR / FMCOMMS2 RX
      → OFDM symbol acquisition + FFT
      → DVB-T decode chain (channel eq + demap + deinterleave + Viterbi + RS)
      → UDP sink (port 2001, MPEG-TS)
    """
    def __init__(self, uri, freq, gain):
        super().__init__("DVB-T RX", catch_exceptions=True)

        # ── Source: radio ─────────────────────────────────────────────────
        self.radio = iio.fmcomms2_source_fc32(
            uri if uri else iio.get_pluto_uri(), [True, True], 32768)
        self.radio.set_len_tag_key('packet_len')
        self.radio.set_frequency(int(freq))
        self.radio.set_samplerate(int(SAMP_RATE))
        self.radio.set_gain_mode(0, 'manual')
        self.radio.set_gain(0, float(gain))
        self.radio.set_quadrature(True)
        self.radio.set_rfdc(True)
        self.radio.set_bbdc(True)
        self.radio.set_filter_params('Auto', '', 0, 0)

        # ── DVB-T decode chain ────────────────────────────────────────────
        # Symbol timing recovery + CP removal → 2048-sample vectors
        self.ofdm_acq  = dtv.dvbt_ofdm_sym_acquisition(1, FFT_LEN, OCC_TONES, CP_LEN, 10)
        # FFT: time-domain → frequency-domain
        self.fft_blk   = gr_fft.fft_vcc(FFT_LEN, True, window.rectangular(FFT_LEN), True, 1)
        # Channel estimation via pilots, output 1512 data-subcarrier vectors
        self.demod_ref = dtv.dvbt_demod_reference_signals(
                             gr.sizeof_gr_complex, FFT_LEN, DATA_CARR,
                             dtv.MOD_QPSK, dtv.NH, dtv.C7_8, dtv.C7_8,
                             dtv.GI_1_32, dtv.T2k, 1, 0)
        self.demapper  = dtv.dvbt_demap(DATA_CARR, dtv.MOD_QPSK, dtv.NH, dtv.T2k, 1)
        self.sym_deil  = dtv.dvbt_symbol_inner_interleaver(DATA_CARR, dtv.T2k, 0)
        self.bit_deil  = dtv.dvbt_bit_inner_deinterleaver(DATA_CARR, dtv.MOD_QPSK, dtv.NH, dtv.T2k)
        self.v2s       = blocks.vector_to_stream(gr.sizeof_char, DATA_CARR)
        self.viterbi   = dtv.dvbt_viterbi_decoder(dtv.MOD_QPSK, dtv.NH, dtv.C7_8, 768)
        self.conv_deil = dtv.dvbt_convolutional_deinterleaver(136, 12, 17)
        self.rs_dec    = dtv.dvbt_reed_solomon_dec(2, 8, 0x11d, 255, 239, 8, 51, 8)
        self.energy_ds = dtv.dvbt_energy_descramble(8)

        # ── Sink: UDP to GStreamer (188 B = 1 MPEG-TS packet per datagram) ─
        self.udp_snk = network.udp_sink(
            gr.sizeof_char, 1, '127.0.0.1', GR_UDP_OUT_PORT, 0, 188, True)

        # ── Connect ───────────────────────────────────────────────────────
        self.connect(
            self.radio,
            self.ofdm_acq,
            self.fft_blk,
            self.demod_ref,
            self.demapper,
            self.sym_deil,
            self.bit_deil,
            self.v2s,
            self.viterbi,
            self.conv_deil,
            self.rs_dec,
            self.energy_ds,
            self.udp_snk,
        )


# ── Mode runners ─────────────────────────────────────────────────────────────

def _run_gst_loop(cmd, stop_evt):
    """Run a GStreamer pipeline, restarting it when the file ends, until stop_evt."""
    while not stop_evt.is_set():
        print(f"[GStreamer] {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        while proc.poll() is None:
            if stop_evt.wait(timeout=0.5):
                proc.terminate()
                return
        if proc.returncode not in (0, -15):
            err = proc.stderr.read().decode(errors='replace')[-400:]
            print(f"[GStreamer] exited with {proc.returncode}:\n{err}")
            if not stop_evt.wait(timeout=1.0):
                continue
        # normal end (file finished) → loop back to start


def run_tx(args):
    filepath = args.file or pick_file()
    if not filepath or not os.path.exists(filepath):
        sys.exit(f"[TX] File not found: {filepath!r}")

    print(f"[TX] File : {filepath}")
    print(f"[TX] Freq : {args.freq/1e6:.1f} MHz   Attn: {args.attn} dB   URI: {args.uri}")

    stop_evt = threading.Event()
    gst_cmd  = _gst_tx_cmd(filepath)
    gst_thread = threading.Thread(target=_run_gst_loop, args=(gst_cmd, stop_evt), daemon=True)
    gst_thread.start()

    # Let GStreamer come up before opening the radio
    time.sleep(1.5)

    print("[TX] Starting DVB-T GNU Radio transmitter …")
    tb = DVBTTransmitter(args.uri, args.freq, args.attn)
    tb.start()

    def _stop(sig=None, frame=None):
        print("\n[TX] Stopping …")
        stop_evt.set()
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    print("[TX] Running.  Ctrl+C to stop.")
    tb.wait()
    _stop()


def run_rx(args):
    print(f"[RX] Freq : {args.freq/1e6:.1f} MHz   Gain: {args.gain} dB   URI: {args.uri}")
    if args.save:
        print(f"[RX] Saving MPEG-TS to: {args.save}")
    else:
        print("[RX] Playing decoded video via GStreamer.")

    stop_evt = threading.Event()

    print("[RX] Starting DVB-T GNU Radio receiver …")
    tb = DVBTReceiver(args.uri, args.freq, args.gain)
    tb.start()

    # Give GR time to initialise the radio before GStreamer tries to connect
    time.sleep(1.5)
    print("[RX] Waiting for DVB-T signal lock (OFDM acquisition may take a few seconds) …")

    gst_cmd    = _gst_rx_cmd(args.save)
    gst_thread = threading.Thread(
        target=_run_gst_loop, args=(gst_cmd, stop_evt), daemon=True)
    gst_thread.start()

    def _stop(sig=None, frame=None):
        print("\n[RX] Stopping …")
        stop_evt.set()
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)
    print("[RX] Running.  Ctrl+C to stop.")
    tb.wait()
    _stop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="DVB-T video streaming over PlutoSDR / FMCOMMS2")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--tx', action='store_true', help="Transmit mode")
    mode.add_argument('--rx', action='store_true', help="Receive mode")

    p.add_argument('--file', metavar='FILE',
                   help="(TX) video file to stream; if omitted a file-picker opens")
    p.add_argument('--freq', type=float, default=2.4e9,
                   help="RF centre frequency in Hz (default: 2.4e9)")
    p.add_argument('--uri',  default='ip:192.168.2.1',
                   help="libiio device URI (default: ip:192.168.2.1)")
    p.add_argument('--attn', type=float, default=8.0,
                   help="(TX) TX attenuation in dB, 0=max power (default: 8)")
    p.add_argument('--gain', type=float, default=30.0,
                   help="(RX) manual RX gain in dB (default: 30)")
    p.add_argument('--save', metavar='FILE',
                   help="(RX) save received MPEG-TS to file instead of playing it")

    args = p.parse_args()
    (run_tx if args.tx else run_rx)(args)


if __name__ == '__main__':
    main()
