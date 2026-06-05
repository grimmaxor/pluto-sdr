"""
Pluto SDR Interactive Gain/Attenuation Control
===============================================
Run alongside your chat script to tune RF parameters live.
Connects to the Pluto and lets you adjust:
  - TX Attenuation  (-89 to 0 dB)
  - RX Gain         (0 to 73 dB)
  - AGC Mode        (manual / slow_attack / fast_attack)

Usage:
  python pluto_gain_control.py
  python pluto_gain_control.py --ip ip:192.168.2.1
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import argparse
import sys

# ─── ARGS ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ip',   type=str, default='ip:pluto.local')
parser.add_argument('--freq', type=float, default=433e6)
args = parser.parse_args()

# ─── PLUTO CONNECTION ─────────────────────────────────────────────────────────
sdr = None
ctx = None
dds = None

def connect_pluto():
    global sdr, ctx, dds
    try:
        import adi, iio
        sdr = adi.Pluto(args.ip)
        sdr.sample_rate             = int(1e6)
        sdr.rx_lo                   = int(args.freq)
        sdr.tx_lo                   = int(args.freq)
        sdr.rx_rf_bandwidth         = int(1e6)
        sdr.tx_rf_bandwidth         = int(1e6)
        sdr.rx_buffer_size          = 4096

        ctx = iio.Context(args.ip)
        dds = ctx.find_device("cf-ad9361-dds-core-lpc")

        # Disable DDS on connect
        if dds:
            for ch in dds.channels:
                if ch.output:
                    for attr in ["raw", "scale"]:
                        try:
                            ch.attrs[attr].value = "0" if attr == "raw" else "0.0"
                        except Exception:
                            pass
        return True, ""
    except Exception as e:
        return False, str(e)


# ─── GUI ──────────────────────────────────────────────────────────────────────
class PlutoGainControl:
    def __init__(self, root):
        self.root = root
        self.root.title("Pluto SDR — Gain Control")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e2e")

        # Colours
        self.BG      = "#1e1e2e"
        self.PANEL   = "#2a2a3e"
        self.ACCENT  = "#7c3aed"
        self.GREEN   = "#22c55e"
        self.RED     = "#ef4444"
        self.AMBER   = "#f59e0b"
        self.TEXT    = "#e2e8f0"
        self.SUBTEXT = "#94a3b8"
        self.SLIDER  = "#4f46e5"

        self.connected    = False
        self.rssi_running = False

        self._build_ui()
        self._connect()

    # ── BUILD ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root

        # ── Header ──
        hdr = tk.Frame(r, bg=self.ACCENT, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="ADALM Pluto SDR  —  Gain Control",
                 font=("Courier", 14, "bold"),
                 bg=self.ACCENT, fg="white").pack()
        tk.Label(hdr, text=f"  {args.ip}   |   {args.freq/1e6:.1f} MHz  ",
                 font=("Courier", 9),
                 bg=self.ACCENT, fg="#c4b5fd").pack()

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Connecting...")
        status_bar = tk.Frame(r, bg=self.PANEL, pady=4)
        status_bar.pack(fill="x", padx=0)
        self.status_dot = tk.Label(status_bar, text="●", font=("Courier", 12),
                                   bg=self.PANEL, fg=self.AMBER)
        self.status_dot.pack(side="left", padx=(12, 4))
        tk.Label(status_bar, textvariable=self.status_var,
                 font=("Courier", 9), bg=self.PANEL, fg=self.TEXT).pack(side="left")

        body = tk.Frame(r, bg=self.BG, padx=20, pady=16)
        body.pack(fill="both")

        # ── TX Attenuation ──
        self._make_slider_panel(
            body,
            title        = "TX Attenuation",
            unit         = "dB",
            var_name     = "tx_atten",
            from_        = -89,
            to           = 0,
            default      = -50,
            callback     = self._on_tx_atten,
            fmt          = lambda v: f"{v:.0f} dB",
            color_fn     = self._tx_color,
            hint         = "More negative = less power   |   0 = max power",
            row          = 0,
        )

        # ── RX Gain ──
        self._make_slider_panel(
            body,
            title        = "RX Gain",
            unit         = "dB",
            var_name     = "rx_gain",
            from_        = 0,
            to           = 73,
            default      = 30,
            callback     = self._on_rx_gain,
            fmt          = lambda v: f"{v:.0f} dB",
            color_fn     = self._rx_color,
            hint         = "Only active in manual AGC mode",
            row          = 1,
        )

        # ── AGC Mode ──
        agc_frame = tk.LabelFrame(body, text="  AGC Mode  ",
                                  font=("Courier", 10, "bold"),
                                  bg=self.BG, fg=self.SUBTEXT,
                                  bd=1, relief="groove",
                                  labelanchor="nw")
        agc_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        body.columnconfigure(0, weight=1)

        self.agc_var = tk.StringVar(value="slow_attack")
        modes = [
            ("Manual  — RX gain slider is active",      "manual"),
            ("Slow Attack  — stable, good for chat",     "slow_attack"),
            ("Fast Attack  — quick lock, may fluctuate", "fast_attack"),
        ]
        for label, val in modes:
            rb = tk.Radiobutton(
                agc_frame, text=label, variable=self.agc_var, value=val,
                command=self._on_agc_change,
                font=("Courier", 9), bg=self.BG, fg=self.TEXT,
                selectcolor=self.PANEL, activebackground=self.BG,
                activeforeground=self.TEXT
            )
            rb.pack(anchor="w", padx=12, pady=2)

        # ── RSSI Display ──
        rssi_frame = tk.LabelFrame(body, text="  Signal Strength (RSSI)  ",
                                   font=("Courier", 10, "bold"),
                                   bg=self.BG, fg=self.SUBTEXT,
                                   bd=1, relief="groove", labelanchor="nw")
        rssi_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        rssi_inner = tk.Frame(rssi_frame, bg=self.BG)
        rssi_inner.pack(fill="x", padx=12, pady=8)

        self.rssi_var  = tk.StringVar(value="-- dBm")
        self.rssi_bar_var = tk.DoubleVar(value=0)

        tk.Label(rssi_inner, textvariable=self.rssi_var,
                 font=("Courier", 22, "bold"),
                 bg=self.BG, fg=self.GREEN).pack()

        self.rssi_bar = ttk.Progressbar(rssi_inner, orient="horizontal",
                                        length=380, mode="determinate",
                                        maximum=100)
        self.rssi_bar.pack(pady=(6, 2))

        self.rssi_label = tk.Label(rssi_inner, text="",
                                   font=("Courier", 8),
                                   bg=self.BG, fg=self.SUBTEXT)
        self.rssi_label.pack()

        # ── DDS Control ──
        dds_frame = tk.LabelFrame(body, text="  DDS Tone Generator  ",
                                  font=("Courier", 10, "bold"),
                                  bg=self.BG, fg=self.SUBTEXT,
                                  bd=1, relief="groove", labelanchor="nw")
        dds_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))

        dds_inner = tk.Frame(dds_frame, bg=self.BG)
        dds_inner.pack(fill="x", padx=12, pady=8)

        self.dds_state = tk.StringVar(value="DISABLED ✓")
        tk.Label(dds_inner, textvariable=self.dds_state,
                 font=("Courier", 11, "bold"),
                 bg=self.BG, fg=self.GREEN).pack(side="left")

        tk.Button(dds_inner, text="Disable DDS",
                  command=self._disable_dds,
                  font=("Courier", 9, "bold"),
                  bg=self.ACCENT, fg="white",
                  activebackground="#6d28d9",
                  relief="flat", padx=10, pady=4).pack(side="right")

        # ── Buttons ──
        btn_frame = tk.Frame(body, bg=self.BG)
        btn_frame.grid(row=5, column=0, sticky="ew", pady=(16, 0))

        tk.Button(btn_frame, text="Apply All",
                  command=self._apply_all,
                  font=("Courier", 10, "bold"),
                  bg=self.GREEN, fg="#052e16",
                  activebackground="#16a34a",
                  relief="flat", padx=16, pady=6).pack(side="left", padx=(0, 8))

        tk.Button(btn_frame, text="Reset Defaults",
                  command=self._reset_defaults,
                  font=("Courier", 10),
                  bg=self.PANEL, fg=self.TEXT,
                  activebackground="#3a3a5e",
                  relief="flat", padx=16, pady=6).pack(side="left")

        tk.Button(btn_frame, text="Reconnect",
                  command=self._connect,
                  font=("Courier", 10),
                  bg=self.PANEL, fg=self.TEXT,
                  activebackground="#3a3a5e",
                  relief="flat", padx=16, pady=6).pack(side="right")

        # ── Log ──
        log_frame = tk.LabelFrame(body, text="  Log  ",
                                  font=("Courier", 9),
                                  bg=self.BG, fg=self.SUBTEXT,
                                  bd=1, relief="groove", labelanchor="nw")
        log_frame.grid(row=6, column=0, sticky="ew", pady=(12, 0))

        self.log = tk.Text(log_frame, height=5, font=("Courier", 8),
                           bg=self.PANEL, fg=self.TEXT,
                           insertbackground=self.TEXT,
                           relief="flat", state="disabled")
        self.log.pack(fill="x", padx=4, pady=4)

    def _make_slider_panel(self, parent, title, unit, var_name,
                           from_, to, default, callback, fmt,
                           color_fn, hint, row):
        frame = tk.LabelFrame(parent, text=f"  {title}  ",
                              font=("Courier", 10, "bold"),
                              bg=self.BG, fg=self.SUBTEXT,
                              bd=1, relief="groove", labelanchor="nw")
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 12))

        inner = tk.Frame(frame, bg=self.BG)
        inner.pack(fill="x", padx=12, pady=8)

        var = tk.DoubleVar(value=default)
        setattr(self, var_name + "_var", var)

        val_label = tk.Label(inner, text=fmt(default),
                             font=("Courier", 20, "bold"),
                             bg=self.BG, fg=color_fn(default))
        val_label.pack()

        slider = tk.Scale(
            inner, from_=from_, to=to,
            orient="horizontal", length=400,
            variable=var, resolution=1,
            showvalue=False,
            bg=self.BG, fg=self.TEXT,
            troughcolor=self.PANEL,
            activebackground=self.SLIDER,
            highlightthickness=0, bd=0,
            command=lambda v, lbl=val_label, fn=fmt,
                           cfn=color_fn, cb=callback:
                self._slider_moved(v, lbl, fn, cfn, cb)
        )
        slider.pack(fill="x", pady=(4, 0))

        range_frame = tk.Frame(inner, bg=self.BG)
        range_frame.pack(fill="x")
        tk.Label(range_frame, text=str(from_),
                 font=("Courier", 8), bg=self.BG,
                 fg=self.SUBTEXT).pack(side="left")
        tk.Label(range_frame, text=str(to),
                 font=("Courier", 8), bg=self.BG,
                 fg=self.SUBTEXT).pack(side="right")

        tk.Label(inner, text=hint, font=("Courier", 8),
                 bg=self.BG, fg=self.SUBTEXT).pack(pady=(2, 0))

    def _slider_moved(self, val, label, fmt, color_fn, callback):
        v = float(val)
        label.config(text=fmt(v), fg=color_fn(v))
        if self.connected:
            try:
                callback(v)
            except Exception as e:
                self._log(f"ERR: {e}")

    # ── COLOUR FUNCTIONS ───────────────────────────────────────────────────────
    def _tx_color(self, v):
        # More negative = less power = green; close to 0 = red
        if v < -60:   return self.GREEN
        elif v < -30: return self.AMBER
        else:         return self.RED

    def _rx_color(self, v):
        if v < 20:    return self.SUBTEXT
        elif v < 50:  return self.GREEN
        elif v < 65:  return self.AMBER
        else:         return self.RED

    # ── CALLBACKS ─────────────────────────────────────────────────────────────
    def _on_tx_atten(self, val):
        if sdr:
            sdr.tx_hardwaregain_chan0 = int(val)
            self._log(f"TX attenuation → {int(val)} dB")

    def _on_rx_gain(self, val):
        if sdr and self.agc_var.get() == "manual":
            sdr.rx_hardwaregain_chan0 = int(val)
            self._log(f"RX gain → {int(val)} dB")

    def _on_agc_change(self):
        if not sdr:
            return
        mode = self.agc_var.get()
        try:
            sdr.gain_control_mode_chan0 = mode
            self._log(f"AGC mode → {mode}")
            # If manual, apply current slider value immediately
            if mode == "manual":
                sdr.rx_hardwaregain_chan0 = int(self.rx_gain_var.get())
        except Exception as e:
            self._log(f"AGC ERR: {e}")

    def _apply_all(self):
        if not self.connected:
            self._log("Not connected — cannot apply")
            return
        try:
            sdr.tx_hardwaregain_chan0   = int(self.tx_atten_var.get())
            sdr.gain_control_mode_chan0 = self.agc_var.get()
            if self.agc_var.get() == "manual":
                sdr.rx_hardwaregain_chan0 = int(self.rx_gain_var.get())
            self._log("✓ All settings applied")
        except Exception as e:
            self._log(f"Apply ERR: {e}")

    def _reset_defaults(self):
        self.tx_atten_var.set(-50)
        self.rx_gain_var.set(30)
        self.agc_var.set("slow_attack")
        self._apply_all()
        self._log("Reset to defaults")

    def _disable_dds(self):
        if not dds:
            self._log("DDS device not found")
            return
        try:
            for ch in dds.channels:
                if ch.output:
                    for attr in ["raw", "scale"]:
                        try:
                            ch.attrs[attr].value = "0" if attr == "raw" else "0.0"
                        except Exception:
                            pass
            self.dds_state.set("DISABLED ✓")
            self._log("✓ DDS disabled")
        except Exception as e:
            self._log(f"DDS ERR: {e}")

    # ── RSSI POLLING ──────────────────────────────────────────────────────────
    def _start_rssi(self):
        self.rssi_running = True
        t = threading.Thread(target=self._rssi_loop, daemon=True)
        t.start()

    def _rssi_loop(self):
        while self.rssi_running and self.connected:
            try:
                import iio
                phy = ctx.find_device("ad9361-phy")
                for ch in phy.channels:
                    if ch.id == "voltage0" and not ch.output:
                        rssi_str = ch.attrs["rssi"].value   # e.g. "42.50 dB"
                        rssi_val = float(rssi_str.split()[0])
                        # Map RSSI (0–90 dB) to signal strength %
                        strength = min(100, max(0, rssi_val / 90 * 100))
                        self.root.after(0, self._update_rssi, rssi_str, strength)
                        break
            except Exception:
                pass
            time.sleep(0.5)

    def _update_rssi(self, rssi_str, strength):
        self.rssi_var.set(rssi_str)
        self.rssi_bar["value"] = strength
        if strength > 70:
            label = "Strong signal ✓"
        elif strength > 40:
            label = "Moderate signal"
        elif strength > 10:
            label = "Weak signal"
        else:
            label = "Very weak / no signal"
        self.rssi_label.config(text=label)

    # ── CONNECT ───────────────────────────────────────────────────────────────
    def _connect(self):
        self.connected = False
        self._set_status("Connecting...", self.AMBER)
        self._log(f"Connecting to {args.ip}...")

        def do_connect():
            ok, err = connect_pluto()
            self.root.after(0, self._on_connect_result, ok, err)

        threading.Thread(target=do_connect, daemon=True).start()

    def _on_connect_result(self, ok, err):
        if ok:
            self.connected = True
            self._set_status(f"Connected  —  {args.ip}", self.GREEN)
            self._log(f"✓ Connected to {args.ip}")
            self._apply_all()
            self._disable_dds()
            self._start_rssi()
        else:
            self.connected = False
            self._set_status(f"Failed: {err[:60]}", self.RED)
            self._log(f"✗ Connection failed: {err}")

    def _set_status(self, msg, color):
        self.status_var.set(msg)
        self.status_dot.config(fg=color)

    # ── LOG ───────────────────────────────────────────────────────────────────
    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log.config(state="normal")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")
        self.log.config(state="disabled")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = PlutoGainControl(root)

    # Style the ttk progressbar
    style = ttk.Style()
    style.theme_use("default")
    style.configure("Horizontal.TProgressbar",
                    background="#22c55e",
                    troughcolor="#2a2a3e",
                    bordercolor="#2a2a3e",
                    lightcolor="#22c55e",
                    darkcolor="#22c55e")

    root.mainloop()


if __name__ == "__main__":
    main()
