"""OmniVoice Studio launcher — source for the installed .exe.

The installed app is deliberately thin: the studio, demucs, torch, and the TTS
server all run out of the OmniVoice folder's own .venv (~7 GB with CUDA), so
freezing them is pointless. This exe just remembers WHERE that folder is
(%APPDATA%\\OmniVoiceStudio\\config.json), validates it, and launches the
studio from it with the venv's pythonw — no console window, exe exits after
spawning.

Run with --reconfigure to re-open the folder picker.

Build (see installer/build.bat):
    pyinstaller --onefile --windowed --paths gui installer/launcher.py
The --paths gui makes `import appconfig` freeze the shared config module in.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import appconfig

SUGGESTED_DIR = ""  # no machine-specific guess; the picker opens at the user's home
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _clean_env() -> dict:
    """os.environ minus everything the PyInstaller bootloader injected.

    The frozen exe sets TCL_LIBRARY/TK_LIBRARY (and _MEI/_PYI vars) pointing at
    its own temp extraction dir, which is deleted the moment this launcher
    exits — the spawned studio must not inherit them or its tkinter dies.
    """
    drop_exact = {
        "TCL_LIBRARY",
        "TK_LIBRARY",
        "TIX_LIBRARY",
        "PYTHONPATH",
        "PYTHONHOME",
    }
    return {
        k: v
        for k, v in os.environ.items()
        if k not in drop_exact and not k.startswith(("_MEI", "_PYI"))
    }


def launch_studio(root_dir: Path) -> None:
    """Spawn the studio from the OmniVoice folder's venv and detach.

    CREATE_BREAKAWAY_FROM_JOB is required: the onefile bootloader wraps this
    process in a kill-on-close job object, so without breakaway the studio is
    killed as soon as the launcher exits. Falls back to no-breakaway (e.g. if
    a parent job forbids it) rather than failing outright.
    """
    pythonw = root_dir / ".venv" / "Scripts" / "pythonw.exe"
    studio = root_dir / "gui" / "studio.pyw"
    base = dict(
        args=[str(pythonw), str(studio)],
        cwd=str(root_dir),
        env=_clean_env(),
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    flags = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(creationflags=flags | _CREATE_BREAKAWAY_FROM_JOB, **base)
    except OSError:
        subprocess.Popen(creationflags=flags, **base)


def pick_folder_dialog(cfg: dict) -> Path | None:
    """First-run / --reconfigure dialog. Returns the validated folder, or None."""
    result: dict = {"dir": None}

    win = tk.Tk()
    win.title(f"{appconfig.APP_NAME} v{appconfig.APP_VERSION} — Setup")
    win.geometry("560x200")
    win.resizable(False, False)

    initial = cfg["omnivoice_dir"]
    if (
        not initial
        and SUGGESTED_DIR
        and appconfig.validate_omnivoice_dir(SUGGESTED_DIR)
    ):
        initial = SUGGESTED_DIR
    dir_var = tk.StringVar(value=initial)
    problem_var = tk.StringVar()

    body = ttk.Frame(win, padding=14)
    body.pack(fill="both", expand=True)
    body.columnconfigure(0, weight=1)

    ttk.Label(
        body,
        text="Where is your OmniVoice folder?\n"
        "(the one containing .venv, gui, server, ref-output)",
        justify="left",
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

    entry = ttk.Entry(body, textvariable=dir_var)
    entry.grid(row=1, column=0, sticky="ew")
    ttk.Button(
        body,
        text="Browse",
        command=lambda: dir_var.set(
            filedialog.askdirectory(initialdir=dir_var.get() or "C:\\") or dir_var.get()
        ),
    ).grid(row=1, column=1, padx=(6, 0))

    problem_lbl = ttk.Label(body, textvariable=problem_var, wraplength=520)
    problem_lbl.grid(row=2, column=0, columnspan=2, sticky="w", pady=6)

    launch_btn = ttk.Button(body, text="Save && Launch")
    launch_btn.grid(row=3, column=0, columnspan=2, pady=(4, 0))

    def _validate(*_):
        probs = appconfig.validate_omnivoice_dir(dir_var.get().strip())
        if probs:
            problem_var.set("✗ " + "\n✗ ".join(probs))
            problem_lbl.config(foreground="#c62828")
            launch_btn.config(state="disabled")
        else:
            problem_var.set("✓ .venv found    ✓ studio found")
            problem_lbl.config(foreground="#2e7d32")
            launch_btn.config(state="normal")

    def _go():
        result["dir"] = Path(dir_var.get().strip())
        win.destroy()

    launch_btn.config(command=_go)
    dir_var.trace_add("write", _validate)
    _validate()
    entry.focus_set()
    win.bind("<Return>", lambda _e: launch_btn.instate(("!disabled",)) and _go())
    win.mainloop()
    return result["dir"]


def main() -> None:
    reconfigure = "--reconfigure" in sys.argv[1:]
    cfg = appconfig.load_config()

    root_dir: Path | None = None
    if not reconfigure and not appconfig.validate_omnivoice_dir(cfg["omnivoice_dir"]):
        root_dir = Path(cfg["omnivoice_dir"])
    else:
        root_dir = pick_folder_dialog(cfg)
        if root_dir is None:
            return  # user closed the dialog
        cfg["omnivoice_dir"] = str(root_dir)
        if not cfg["server_ref_audio"]:
            cfg["server_ref_audio"] = appconfig.default_ref_audio(root_dir)
        appconfig.save_config(cfg)

    try:
        launch_studio(root_dir)
    except OSError as e:
        # Broken venv since last run, etc. — surface it and offer reconfigure.
        hidden = tk.Tk()
        hidden.withdraw()
        retry = messagebox.askretrycancel(
            appconfig.APP_NAME,
            f"Could not launch the studio from:\n{root_dir}\n\n{e}\n\n"
            "Retry opens the folder picker.",
        )
        hidden.destroy()
        if retry:
            picked = pick_folder_dialog(cfg)
            if picked:
                cfg["omnivoice_dir"] = str(picked)
                appconfig.save_config(cfg)
                launch_studio(picked)


if __name__ == "__main__":
    main()
