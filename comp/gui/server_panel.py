"""OmniVoice server control strip — always-visible bar above the studio tabs.

One-service version of F:\\OS\\control-panel.pyw: status dot, Start/Stop for
server/app.py on the configured port, an auto-start toggle, and a settings
dialog (OmniVoice folder, port, ref audio). Process helpers (port probe,
netstat PID lookup, taskkill) are adapted from control-panel.pyw, which lives
outside this repo and can't be imported.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).parent))
import appconfig  # noqa: E402

ROOT = Path(__file__).parent.parent
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

DOT_COLORS = {"running": "#2e7d32", "stopped": "#c62828", "busy": "#f9a825"}


# --------------------------------------------------------------------------- #
# process helpers (from control-panel.pyw)
# --------------------------------------------------------------------------- #
def is_port_open(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return False


def find_pid_on_port(port: int) -> int | None:
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                return int(line.strip().split()[-1])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def kill_pid(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=10,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        pass


# --------------------------------------------------------------------------- #
# ServerBar
# --------------------------------------------------------------------------- #
class ServerBar(ttk.Frame):
    """Slim strip: dot | 'OmniVoice Server :port' | status | Start Stop | auto | gear."""

    def __init__(self, parent):
        super().__init__(parent, padding=(8, 4))
        self.cfg = appconfig.load_config()
        self.proc: subprocess.Popen | None = None
        self.transition: str | None = None  # "Starting" / "Stopping" / None
        self.autostart_var = tk.BooleanVar(value=bool(self.cfg["server_autostart"]))

        self.dot = tk.Label(
            self, text="⬤", fg=DOT_COLORS["stopped"], font=("Segoe UI", 10)
        )
        self.dot.pack(side="left")
        self.name_lbl = ttk.Label(self, font=("Segoe UI", 10, "bold"))
        self.name_lbl.pack(side="left", padx=(6, 4))
        self.status_lbl = ttk.Label(
            self, text="Stopped", foreground=DOT_COLORS["stopped"], width=9
        )
        self.status_lbl.pack(side="left")

        self.start_btn = ttk.Button(self, text="Start", width=7, command=self.start)
        self.start_btn.pack(side="left", padx=(8, 2))
        self.stop_btn = ttk.Button(
            self, text="Stop", width=7, command=self.stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=2)

        ttk.Button(self, text="⚙", width=3, command=self.open_settings).pack(
            side="right"
        )
        ttk.Checkbutton(
            self,
            text="auto-start",
            variable=self.autostart_var,
            command=self._save_autostart,
        ).pack(side="right", padx=(0, 8))

        self._refresh_name()
        self._poll()
        if self.autostart_var.get() and not is_port_open(self.port):
            self.start()

    # ------------------------------------------------------------ properties
    @property
    def port(self) -> int:
        return int(self.cfg["server_port"])

    def _refresh_name(self) -> None:
        self.name_lbl.config(text=f"OmniVoice Server :{self.port}")

    # ------------------------------------------------------------ actions
    def start(self) -> None:
        if is_port_open(self.port):
            return
        py = ROOT / ".venv" / "Scripts" / "python.exe"
        if not py.exists():
            messagebox.showerror("No venv", f"Interpreter not found:\n{py}")
            return
        cmd = [str(py), "server/app.py", "--host", "0.0.0.0", "--port", str(self.port)]
        ref = self.cfg["server_ref_audio"] or appconfig.default_ref_audio(ROOT)
        if ref:
            cmd += ["--ref-audio", ref]
        try:
            flags = _CREATE_NO_WINDOW
            if sys.platform == "win32":
                flags |= subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            self._set_transition("Starting")
        except OSError as e:
            messagebox.showerror("Start failed", str(e))

    def stop(self) -> None:
        # Kill by port too, so servers started elsewhere (control panel, bat)
        # are also stopped. Run off the UI thread: netstat+taskkill can block.
        proc, self.proc = self.proc, None
        self._set_transition("Stopping")

        def _worker():
            if proc:
                try:
                    proc.terminate()
                    proc.kill()
                except OSError:
                    pass
            pid = find_pid_on_port(self.port)
            if pid:
                kill_pid(pid)

        threading.Thread(target=_worker, daemon=True).start()

    def _save_autostart(self) -> None:
        self.cfg["server_autostart"] = bool(self.autostart_var.get())
        appconfig.save_config(self.cfg)

    # ------------------------------------------------------------ polling
    def _set_transition(self, label: str) -> None:
        self.transition = label
        self.status_lbl.config(text=label, foreground=DOT_COLORS["busy"])
        self.dot.config(fg=DOT_COLORS["busy"])
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")

    def _poll(self) -> None:
        alive = is_port_open(self.port)
        if alive:
            self.transition = None
            self.status_lbl.config(text="Running", foreground=DOT_COLORS["running"])
            self.dot.config(fg=DOT_COLORS["running"])
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
        elif self.transition == "Starting":
            pass  # model still loading (~10-20s); keep the yellow state
        else:
            self.transition = None
            self.status_lbl.config(text="Stopped", foreground=DOT_COLORS["stopped"])
            self.dot.config(fg=DOT_COLORS["stopped"])
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        self.after(2000, self._poll)

    # ------------------------------------------------------------ settings
    def open_settings(self) -> None:
        top = self.winfo_toplevel()
        win = tk.Toplevel(top)
        win.title(f"{appconfig.APP_NAME} — Settings")
        win.geometry("560x230")
        win.transient(top)
        win.grab_set()

        dir_var = tk.StringVar(value=self.cfg["omnivoice_dir"] or str(ROOT))
        port_var = tk.IntVar(value=self.port)
        ref_var = tk.StringVar(value=self.cfg["server_ref_audio"])
        problem_var = tk.StringVar()

        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        def _validate(*_):
            probs = appconfig.validate_omnivoice_dir(dir_var.get().strip())
            problem_var.set(
                "✓ folder looks good" if not probs else "✗ " + "; ".join(probs)
            )

        dir_var.trace_add("write", _validate)

        ttk.Label(body, text="OmniVoice folder:").grid(
            row=0, column=0, sticky="w", pady=3
        )
        ttk.Entry(body, textvariable=dir_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(
            body,
            text="Browse",
            command=lambda: dir_var.set(
                filedialog.askdirectory(initialdir=dir_var.get()) or dir_var.get()
            ),
        ).grid(row=0, column=2)
        ttk.Label(body, textvariable=problem_var, foreground="gray").grid(
            row=1, column=1, columnspan=2, sticky="w", padx=4
        )
        ttk.Label(
            body,
            text="(used by the launcher exe — takes effect next launch)",
            foreground="gray",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=4, pady=(0, 6))

        ttk.Label(body, text="Server port:").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Spinbox(body, from_=1024, to=65535, textvariable=port_var, width=8).grid(
            row=3, column=1, sticky="w", padx=4
        )

        ttk.Label(body, text="Server ref audio:").grid(
            row=4, column=0, sticky="w", pady=3
        )
        ttk.Entry(body, textvariable=ref_var).grid(row=4, column=1, sticky="ew", padx=4)
        ttk.Button(
            body,
            text="Browse",
            command=lambda: ref_var.set(
                filedialog.askopenfilename(
                    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                    initialdir=str(ROOT / "ref-output"),
                )
                or ref_var.get()
            ),
        ).grid(row=4, column=2)
        ttk.Label(
            body,
            text="(leave empty to use ref\\default_ref.wav; restart server to apply)",
            foreground="gray",
        ).grid(row=5, column=1, columnspan=2, sticky="w", padx=4)

        def _save():
            self.cfg["omnivoice_dir"] = dir_var.get().strip()
            try:
                self.cfg["server_port"] = int(port_var.get())
            except (tk.TclError, ValueError):
                pass
            self.cfg["server_ref_audio"] = ref_var.get().strip()
            appconfig.save_config(self.cfg)
            self._refresh_name()
            win.destroy()

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        ttk.Button(btns, text="Save", command=_save).pack(side="right")
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=6)
        _validate()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("ServerBar test")
    ServerBar(root).pack(fill="x")
    root.mainloop()
