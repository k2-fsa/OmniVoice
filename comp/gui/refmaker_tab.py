"""OmniVoice Reference Clip Maker — embeddable tab frame.

Turns a long video/audio file into clean short reference clips for zero-shot
voice cloning by picking the single best window. Opposite of generate_tab: this
PRODUCES the ref_audio clips that the TTS generator clones from.

Flow:  pick file -> Analyze -> pick candidates from the list -> trim -> Save clip.
Saved clips are mono WAVs in the output folder, ready to drop into the
generator's "Ref audio" field.

Used standalone via gui/refmaker.pyw and as the "Ref Clip Maker" tab in
gui/studio.pyw.
"""

import os
import sys
import tempfile
import threading
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).parent))
import refmaker_core as core  # noqa: E402
import tempcleanup  # noqa: E402

ROOT = Path(__file__).parent.parent
REF_PREPROCESSED = ROOT / "ref-input"  # dirty long rips (input, gitignored contents)
REF_OUT = ROOT / "ref-output"  # clean refs (output, gitignored contents)
DEFAULT_OUTDIR = str(REF_OUT)
INPUT_TYPES = [
    ("Media files", "*.mp4 *.mkv *.webm *.mov *.avi *.mp3 *.m4a *.flac *.wav *.ogg"),
    ("All files", "*.*"),
]


class RefMakerTab(ttk.Frame):
    """Single-best-window reference clip maker. Embeds into any Tk container."""

    def __init__(self, parent):
        super().__init__(parent)
        # self.root points at this frame so the original `self.root.*` widget /
        # scheduling calls work whether we're standalone or inside a tab.
        self.root = self

        self.input_var = tk.StringVar()
        self.outdir_var = tk.StringVar(value=DEFAULT_OUTDIR)
        self.isolate_var = tk.BooleanVar(value=True)
        self.cliplen_var = tk.DoubleVar(value=core.DEFAULT_CLIP_LEN_S)
        self.sr_var = tk.IntVar(value=core.DEFAULT_TARGET_SR)
        self.start_var = tk.DoubleVar(value=0.0)
        self.end_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(
            value="Ready. Pick a video or audio file to start."
        )

        # Path actually analyzed/exported from (demucs vocals, or extracted audio).
        self.source_path: Path | None = None
        self.source_stem: str = "clip"
        self.candidates: list[core.Candidate] = []
        self.busy = False
        # Current run's temp workdir. source_path points INSIDE it, so it lives
        # until the next Analyze (or window close) reaps it — see _analyze_worker.
        self._workdir: Path | None = None

        # Reap orphaned workdirs from crashed / force-quit past sessions.
        tempcleanup.sweep_stale("refmaker_")

        self._build()
        self._bind_cleanup()

    def _bind_cleanup(self) -> None:
        """Delete our workdir when this frame is destroyed (window close / tab teardown)."""
        self.bind("<Destroy>", self._on_destroy)

    def _on_destroy(self, event) -> None:
        # <Destroy> bubbles up from child widgets too; only act for our own frame.
        if event.widget is self:
            tempcleanup.remove_workdir(self._workdir)

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
            frm_opt, text="Isolate vocals (demucs)", variable=self.isolate_var
        ).pack(side="left")
        ttk.Label(frm_opt, text="Clip length (s):").pack(side="left", padx=(12, 2))
        ttk.Spinbox(
            frm_opt,
            from_=4.0,
            to=15.0,
            increment=0.5,
            width=5,
            textvariable=self.cliplen_var,
        ).pack(side="left")
        ttk.Label(frm_opt, text="Output SR:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(
            frm_opt,
            from_=16000,
            to=48000,
            increment=8000,
            width=7,
            textvariable=self.sr_var,
        ).pack(side="left")
        self.analyze_btn = ttk.Button(frm_opt, text="Analyze", command=self.analyze)
        self.analyze_btn.pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill="x", padx=8, pady=(0, 2))

        frm_list = ttk.LabelFrame(
            self.root, text="Candidate clips (best first)", padding=6
        )
        frm_list.pack(fill="both", expand=True, **pad)
        cols = ("idx", "range", "len", "score")
        self.tree = ttk.Treeview(frm_list, columns=cols, show="headings", height=10)
        for c, txt, w, anchor in (
            ("idx", "#", 40, "center"),
            ("range", "Time range", 220, "center"),
            ("len", "Length", 80, "center"),
            ("score", "Score", 80, "center"),
        ):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor=anchor)
        self.tree.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frm_list, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        frm_trim = ttk.LabelFrame(self.root, text="Trim & save selected", padding=6)
        frm_trim.pack(fill="x", **pad)
        ttk.Label(frm_trim, text="Start (s):").pack(side="left")
        ttk.Spinbox(
            frm_trim,
            from_=0.0,
            to=99999.0,
            increment=0.1,
            width=9,
            textvariable=self.start_var,
            command=self.update_duration,
        ).pack(side="left", padx=2)
        ttk.Label(frm_trim, text="End (s):").pack(side="left", padx=(8, 0))
        ttk.Spinbox(
            frm_trim,
            from_=0.0,
            to=99999.0,
            increment=0.1,
            width=9,
            textvariable=self.end_var,
            command=self.update_duration,
        ).pack(side="left", padx=2)
        self.dur_lbl = ttk.Label(frm_trim, text="0.0s", width=10, foreground="gray")
        self.dur_lbl.pack(side="left", padx=6)
        ttk.Button(frm_trim, text="Preview", command=self.preview).pack(
            side="left", padx=4
        )
        ttk.Button(frm_trim, text="Stop", command=self.stop).pack(side="left")
        self.save_btn = ttk.Button(frm_trim, text="Save clip", command=self.save_clip)
        self.save_btn.pack(side="left", padx=8)
        ttk.Button(frm_trim, text="Open output folder", command=self.open_folder).pack(
            side="right"
        )

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=8)
        ttk.Label(self.root, textvariable=self.status_var, foreground="gray").pack(
            anchor="w", padx=8, pady=6
        )

    # -------------------------------------------------------------- helpers
    def browse_input(self) -> None:
        initial = str(REF_PREPROCESSED if REF_PREPROCESSED.exists() else ROOT)
        path = filedialog.askopenfilename(
            title="Pick video or audio", filetypes=INPUT_TYPES, initialdir=initial
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
        self.analyze_btn.config(state=state)
        self.save_btn.config(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status is not None:
            self.status_var.set(status)

    def update_duration(self) -> None:
        try:
            dur = float(self.end_var.get()) - float(self.start_var.get())
        except (tk.TclError, ValueError):
            dur = 0.0
        self.dur_lbl.config(text=f"{max(0.0, dur):.1f}s")

    # -------------------------------------------------------------- analyze
    def analyze(self) -> None:
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
        self.set_busy(True, "Analyzing... (extract -> separate -> detect)")
        threading.Thread(target=self._analyze_worker, args=(inp,), daemon=True).start()

    def _analyze_worker(self, inp: str) -> None:
        try:
            # Free the PREVIOUS run's workdir (its source_path is about to be
            # replaced). We can't delete the new one here — preview/save read
            # from source_path inside it until the next Analyze or window close.
            tempcleanup.remove_workdir(self._workdir)
            workdir = tempcleanup.new_workdir("refmaker_")
            self._workdir = workdir
            self.root.after(
                0, lambda: self.status_var.set("Extracting audio (ffmpeg)...")
            )
            audio = core.extract_audio(inp, workdir=workdir)

            if self.isolate_var.get():
                self.root.after(
                    0,
                    lambda: self.status_var.set(
                        "Isolating vocals (demucs)... this is the slow part"
                    ),
                )
                source = core.isolate_vocals(audio, workdir=workdir / "separated")
            else:
                source = audio

            self.root.after(
                0, lambda: self.status_var.set("Detecting clean speech segments...")
            )
            clip_len = float(self.cliplen_var.get())
            cands = core.find_candidates(source, clip_len_s=clip_len)
            self.root.after(0, lambda: self._on_analyzed(Path(inp).stem, source, cands))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.root.after(0, lambda: self._on_error(msg))

    def _on_analyzed(
        self, stem: str, source: Path, cands: list[core.Candidate]
    ) -> None:
        self.source_stem = stem
        self.source_path = source
        self.candidates = cands
        self.tree.delete(*self.tree.get_children())
        for i, c in enumerate(cands, 1):
            self.tree.insert(
                "",
                "end",
                iid=str(i - 1),
                values=(
                    i,
                    f"{core.format_ts(c.start_s)} – {core.format_ts(c.end_s)}",
                    f"{c.length_s:.1f}s",
                    f"{c.score:.2f}",
                ),
            )
        self.set_busy(False)
        if cands:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.on_select()
            self.status_var.set(
                f"Found {len(cands)} candidates. Preview and Save the good ones."
            )
        else:
            self.status_var.set(
                "No clean segments found. Try a longer source or untick demucs."
            )

    def _on_error(self, err: str) -> None:
        self.set_busy(False)
        self.status_var.set(f"Error: {err.splitlines()[0] if err else 'unknown'}")
        messagebox.showerror("Failed", err)

    # ------------------------------------------------------- select / trim
    def on_select(self, _e=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.candidates):
            c = self.candidates[idx]
            self.start_var.set(round(c.start_s, 2))
            self.end_var.set(round(c.end_s, 2))
            self.update_duration()

    # ----------------------------------------------------- preview / save
    def _current_range(self) -> tuple[float, float] | None:
        if not self.source_path:
            messagebox.showwarning("Nothing analyzed", "Analyze a file first.")
            return None
        try:
            a, b = float(self.start_var.get()), float(self.end_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Bad trim", "Start/End must be numbers.")
            return None
        if b <= a:
            messagebox.showerror("Bad trim", "End must be after Start.")
            return None
        return a, b

    def preview(self) -> None:
        rng = self._current_range()
        if not rng:
            return
        a, b = rng
        self.status_var.set("Rendering preview...")
        threading.Thread(target=self._preview_worker, args=(a, b), daemon=True).start()

    def _preview_worker(self, a: float, b: float) -> None:
        try:
            tmp = Path(tempfile.gettempdir()) / "refmaker_preview.wav"
            core.export_clip(
                self.source_path, a, b, tmp, target_sr=int(self.sr_var.get())
            )
            winsound.PlaySound(str(tmp), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.root.after(0, lambda: self.status_var.set(f"Preview: {b - a:.1f}s"))
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Preview error: {msg}"))

    def stop(self) -> None:
        winsound.PlaySound(None, winsound.SND_PURGE)

    def save_clip(self) -> None:
        rng = self._current_range()
        if not rng:
            return
        a, b = rng
        outdir = Path(self.outdir_var.get().strip() or DEFAULT_OUTDIR)
        n = self._next_index(outdir)
        out = outdir / f"{self.source_stem}_{n:02d}.wav"
        self.set_busy(True, f"Saving {out.name}...")
        threading.Thread(
            target=self._save_worker, args=(a, b, out), daemon=True
        ).start()

    def _save_worker(self, a: float, b: float, out: Path) -> None:
        try:
            core.export_clip(
                self.source_path, a, b, out, target_sr=int(self.sr_var.get())
            )
            self.root.after(
                0,
                lambda: self.set_busy(
                    False, f"Saved -> {out}  ({b - a:.1f}s, {self.sr_var.get()}Hz mono)"
                ),
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            self.root.after(0, lambda: self._on_error(msg))

    def _next_index(self, outdir: Path) -> int:
        if not outdir.exists():
            return 1
        existing = list(outdir.glob(f"{self.source_stem}_*.wav"))
        nums = []
        for p in existing:
            tail = p.stem.rsplit("_", 1)[-1]
            if tail.isdigit():
                nums.append(int(tail))
        return (max(nums) + 1) if nums else 1

    def open_folder(self) -> None:
        outdir = Path(self.outdir_var.get().strip() or DEFAULT_OUTDIR)
        outdir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(outdir))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("OmniVoice — Reference Clip Maker")
    root.geometry("860x620")
    root.minsize(680, 520)
    RefMakerTab(root).pack(fill="both", expand=True)
    root.mainloop()
