"""OmniVoice Auto Cut & Merge — embeddable tab frame.

For game/visual-novel rips where the character speaks in short bursts between
unvoiced MC lines and SFX (e.g. Gakuen Idolmaster). Demucs strips the BGM/SFX so
the gaps become real silence, then every voiced span is detected and stitched
into one continuous reference clip — no manual cut-and-merge.

Flow:  pick dirty rip -> Cut & Merge -> clean ref saved to ref-output.

Used standalone via gui/merge.pyw and as the "Auto Cut & Merge" tab in
gui/studio.pyw.
"""

import os
import sys
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).parent))
import refmaker_core as core  # noqa: E402
import enhance  # noqa: E402  — lazy heavy deps (speechbrain/silero); cheap to import
import tempcleanup  # noqa: E402

ROOT = Path(__file__).parent.parent
REF_PREPROCESSED = ROOT / "ref-input"  # dirty long rips (input, gitignored contents)
REF_OUT = ROOT / "ref-output"  # clean refs (output, gitignored contents)
DEFAULT_OUTDIR = str(REF_OUT)
INPUT_TYPES = [
    ("Media files", "*.mp4 *.mkv *.webm *.mov *.avi *.mp3 *.m4a *.flac *.wav *.ogg"),
    ("All files", "*.*"),
]


class MergeTab(ttk.Frame):
    """Auto cut-and-merge reference maker. Embeds into any Tk container."""

    def __init__(self, parent):
        super().__init__(parent)
        self.root = self

        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=DEFAULT_OUTDIR)
        self.isolate_var = tk.BooleanVar(value=True)
        self.clean_var = tk.BooleanVar(value=True)
        self.target_var = tk.DoubleVar(value=25.0)
        self.full_var = tk.BooleanVar(value=False)
        self.sr_var = tk.IntVar(value=core.DEFAULT_TARGET_SR)
        self.status_var = tk.StringVar(
            value="Ready. Pick a long voice rip; speech gets cut and merged into one clean ref."
        )

        self.last_out: Path | None = None
        self.busy = False

        # Reap orphaned workdirs from crashed / force-quit past sessions.
        tempcleanup.sweep_stale("merge_")

        self._build()

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        pad = {"padx": 8, "pady": 3}

        frm_in = ttk.Frame(self.root)
        frm_in.pack(fill="x", **pad)
        ttk.Label(frm_in, text="Input:", width=8).pack(side="left")
        ttk.Entry(frm_in, textvariable=self.input_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(frm_in, text="Browse", command=self.browse_input).pack(side="left")

        frm_out = ttk.Frame(self.root)
        frm_out.pack(fill="x", **pad)
        ttk.Label(frm_out, text="Output:", width=8).pack(side="left")
        ttk.Entry(frm_out, textvariable=self.outdir_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(frm_out, text="Browse", command=self.browse_outdir).pack(side="left")

        frm_opt = ttk.LabelFrame(self.root, text="Options", padding=6)
        frm_opt.pack(fill="x", **pad)
        ttk.Checkbutton(
            frm_opt,
            text="Isolate vocals (demucs) — removes BGM/SFX",
            variable=self.isolate_var,
        ).pack(side="left")
        ttk.Checkbutton(
            frm_opt,
            text="Deep clean (removes blips overlapping speech)",
            variable=self.clean_var,
        ).pack(side="left", padx=(12, 0))
        ttk.Label(frm_opt, text="Output SR:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(
            frm_opt,
            from_=16000,
            to=48000,
            increment=8000,
            width=7,
            textvariable=self.sr_var,
        ).pack(side="left")

        frm_len = ttk.LabelFrame(self.root, text="Merged length", padding=6)
        frm_len.pack(fill="x", **pad)
        ttk.Label(frm_len, text="Target (s):").pack(side="left")
        self.len_lbl = ttk.Label(frm_len, text="25", width=4)
        ttk.Scale(
            frm_len,
            from_=10.0,
            to=60.0,
            orient="horizontal",
            variable=self.target_var,
            command=lambda v: self.len_lbl.config(text=f"{float(v):.0f}"),
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.len_lbl.pack(side="left", padx=4)
        ttk.Checkbutton(
            frm_len, text="Also save full merge", variable=self.full_var
        ).pack(side="left", padx=(12, 0))

        frm_run = ttk.Frame(self.root)
        frm_run.pack(fill="x", **pad)
        self.run_btn = ttk.Button(frm_run, text="Cut & Merge", command=self.run)
        self.run_btn.pack(side="left")
        self.preview_btn = ttk.Button(
            frm_run, text="Preview", command=self.preview, state="disabled"
        )
        self.preview_btn.pack(side="left", padx=4)
        ttk.Button(frm_run, text="Stop", command=self.stop).pack(side="left")
        ttk.Button(frm_run, text="Open output folder", command=self.open_folder).pack(
            side="right"
        )

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(2, 2))

        info = (
            "How it works:  ffmpeg -> demucs (strip BGM/SFX) -> Silero VAD (find her speech) -> "
            "stitch with a tiny crossfade -> SepFormer deep clean (removes blips that overlap her "
            "voice).  The capped target stops once enough speech is gathered; 'full merge' keeps "
            "everything.  Deep clean needs the GPU and adds a few seconds."
        )
        ttk.Label(self.root, text=info, wraplength=820, foreground="gray").pack(
            anchor="w", padx=8, pady=(2, 0)
        )

        ttk.Separator(self.root, orient="horizontal").pack(
            fill="x", padx=8, pady=(8, 0)
        )
        ttk.Label(self.root, textvariable=self.status_var, foreground="gray").pack(
            anchor="w", padx=8, pady=6
        )

    # -------------------------------------------------------------- helpers
    def browse_input(self) -> None:
        initial = str(REF_PREPROCESSED if REF_PREPROCESSED.exists() else ROOT)
        path = filedialog.askopenfilename(
            title="Pick a long voice rip", filetypes=INPUT_TYPES, initialdir=initial
        )
        if path:
            self.input_var.set(path)

    def browse_outdir(self) -> None:
        path = filedialog.askdirectory(
            title="Pick output folder", initialdir=self.outdir_var.get()
        )
        if path:
            self.outdir_var.set(path)

    def set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.run_btn.config(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status is not None:
            self.status_var.set(status)

    # -------------------------------------------------------------- run
    def run(self) -> None:
        if self.busy:
            return
        inp = self.input_var.get().strip()
        if not inp or not Path(inp).exists():
            messagebox.showerror("No input", "Pick a valid video or audio file first.")
            return
        if self.isolate_var.get() and not core.check_demucs_available():
            messagebox.showerror(
                "demucs missing",
                "demucs isn't installed in this venv.\n"
                "Run:  .venv\\Scripts\\pip install demucs\n"
                "...or untick 'Isolate vocals' to skip separation.",
            )
            return
        if self.clean_var.get() and not enhance.is_available():
            messagebox.showerror(
                "Deep clean unavailable",
                "Deep clean needs speechbrain + silero-vad in this venv.\n"
                "Run:  .venv\\Scripts\\pip install speechbrain silero-vad\n"
                "...or untick 'Deep clean' to skip it.",
            )
            return
        outdir = Path(self.outdir_var.get().strip() or DEFAULT_OUTDIR)
        target = float(self.target_var.get())
        sr = int(self.sr_var.get())
        isolate = bool(self.isolate_var.get())
        clean = bool(self.clean_var.get())
        also_full = bool(self.full_var.get())
        self.set_busy(
            True, "Working... (extract -> separate -> detect -> merge -> clean)"
        )
        self.preview_btn.config(state="disabled")
        threading.Thread(
            target=self._run_worker,
            args=(inp, outdir, target, sr, isolate, clean, also_full),
            daemon=True,
        ).start()

    def _status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _make_one(self, source, out_path, sr, target, full, intervals, clean) -> float:
        """Merge (optionally VAD-detected) then optionally deep-clean. Returns length_s."""
        _, length = core.merge_speech(
            source,
            out_path,
            target_sr=sr,
            target_len_s=target,
            full=full,
            intervals=intervals,
        )
        if clean:
            self._status(f"Deep cleaning {out_path.name} (SepFormer)...")
            _, length = enhance.deep_clean(out_path, out_path, target_sr=sr)
        return length

    def _run_worker(self, inp, outdir, target, sr, isolate, clean, also_full) -> None:
        workdir = None
        try:
            stem = Path(inp).stem
            workdir = tempcleanup.new_workdir("merge_")
            self._status("Extracting audio (ffmpeg)...")
            audio = core.extract_audio(inp, workdir=workdir)

            if isolate:
                self._status("Isolating vocals (demucs)... this is the slow part")
                source = core.isolate_vocals(audio, workdir=workdir / "separated")
            else:
                source = audio

            # Prefer Silero VAD over energy gating when deep-clean deps are present.
            intervals = None
            if enhance.is_available():
                self._status("Detecting speech (Silero VAD)...")
                intervals = enhance.detect_speech(source)

            self._status("Merging speech...")
            out_path = outdir / f"{stem}.wav"
            length = self._make_one(
                source, out_path, sr, target, False, intervals, clean
            )
            msg = f"Saved -> {out_path}  ({length:.1f}s, {sr}Hz mono)"

            if also_full:
                self._status("Building full merge...")
                full_path = outdir / f"{stem}_full.wav"
                full_len = self._make_one(
                    source, full_path, sr, target, True, intervals, clean
                )
                msg += f"  +  {full_path.name} ({full_len:.1f}s)"

            self.last_out = out_path
            self.root.after(0, lambda: self._on_done(msg))
        except Exception as e:  # noqa: BLE001
            err = str(e)
            self.root.after(0, lambda: self._on_error(err))
        finally:
            # Output is already written to outdir; the workdir (full WAV + demucs
            # vocals) is pure scratch — always reap it, success or failure.
            tempcleanup.remove_workdir(workdir)

    def _on_done(self, msg: str) -> None:
        self.set_busy(False, msg)
        self.preview_btn.config(state="normal")

    def _on_error(self, err: str) -> None:
        self.set_busy(False)
        self.status_var.set(f"Error: {err.splitlines()[0] if err else 'unknown'}")
        messagebox.showerror("Failed", err)

    # ----------------------------------------------------- preview / output
    def preview(self) -> None:
        if not self.last_out or not self.last_out.exists():
            messagebox.showwarning("Nothing yet", "Run Cut & Merge first.")
            return
        winsound.PlaySound(
            str(self.last_out), winsound.SND_FILENAME | winsound.SND_ASYNC
        )
        self.status_var.set(f"Playing {self.last_out.name}")

    def stop(self) -> None:
        winsound.PlaySound(None, winsound.SND_PURGE)

    def open_folder(self) -> None:
        outdir = Path(self.outdir_var.get().strip() or DEFAULT_OUTDIR)
        outdir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(outdir))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("OmniVoice — Auto Cut & Merge")
    root.geometry("860x560")
    root.minsize(680, 460)
    MergeTab(root).pack(fill="both", expand=True)
    root.mainloop()
