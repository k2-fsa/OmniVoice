"""OmniVoice TTS generator — embeddable tab frame.

Talks to server/app.py over HTTP. Server must be running first:
  cd <omnivoice-dir> && .venv/Scripts/python.exe server/app.py

Used standalone via gui/app.pyw and as the "Generate" tab in gui/studio.pyw.
"""

import os
import sys
import threading
import time
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    import httpx
except ImportError:
    import tkinter.messagebox as _mb

    _mb.showerror(
        "Missing dep",
        "httpx not installed in the venv.\nRun: .venv/Scripts/pip install httpx",
    )
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
import translate  # noqa: E402  — EN->JP via Claude CLI

ROOT = Path(__file__).parent.parent
DEFAULT_REF = ""  # empty = the server's startup default voice
REF_OUT = (
    ROOT / "ref-output"
)  # clean refs the merger/refmaker produce (gitignored contents)
DEFAULT_SERVER = "http://localhost:9192"
OUT_DIR = Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 13 official tags + the two unofficial-but-working ones from probe_official_tags.py
TAGS = [
    "(none)",
    "[laughter]",
    "[sigh]",
    "[happy]",
    "[sad]",
    "[dissatisfaction-hnn]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[confirmation-en]",
]

TAG_DOCS = [
    ("[laughter]", "suffix", "Laugh after the line. e.g.  That's hilarious [laughter]"),
    ("[sigh]", "prefix", "Sigh before the line. e.g.  [sigh] Fine, whatever"),
    ("[happy]", "prefix", "Unofficial but works. Adds upbeat tone."),
    ("[sad]", "prefix", "Unofficial but works. Adds downcast tone."),
    ("[dissatisfaction-hnn]", "prefix", "Closest tag to annoyed/angry."),
    ("[surprise-ah]", "prefix", "Surprise gasp 'ah!'"),
    ("[surprise-oh]", "prefix", "Surprise 'oh!'"),
    ("[surprise-wa]", "prefix", "Surprise 'wa!' (more Japanese-style)."),
    ("[surprise-yo]", "prefix", "Surprise 'yo!'"),
    ("[question-en]", "suffix", "Question intonation, English-style."),
    ("[question-ah]", "suffix", "Question 'ah?'"),
    ("[question-oh]", "suffix", "Question 'oh?'"),
    ("[question-ei]", "suffix", "Question 'ei?'"),
    ("[question-yi]", "suffix", "Question 'yi?'"),
    ("[confirmation-en]", "suffix", "Affirmative 'mhm/yeah' confirmation."),
]

INSTRUCT_PRESETS = [
    "(none)",
    "high pitch",
    "very high pitch",
    "low pitch",
    "very low pitch",
    "whisper",
    "young adult, high pitch",
    "whisper, low pitch",
    "child",
    "teenager",
    "young adult",
    "middle-aged",
    "elderly",
    "female",
    "male",
    "japanese accent",
    "british accent",
    "american accent",
]


class GenerateTab(ttk.Frame):
    """TTS generator UI. Embeds into any Tk container (toplevel or Notebook)."""

    def __init__(self, parent):
        super().__init__(parent)
        # self.root points at this frame so the original `self.root.*` widget /
        # scheduling calls work whether we're standalone or inside a tab.
        self.root = self

        self.last_wav: str | None = None
        self.server_var = tk.StringVar(value=DEFAULT_SERVER)
        self.ref_var = tk.StringVar(value=DEFAULT_REF)
        self.tag_var = tk.StringVar(value=TAGS[0])
        self.pos_var = tk.StringVar(value="suffix")
        self.instruct_var = tk.StringVar(value=INSTRUCT_PRESETS[0])
        self.guidance_var = tk.DoubleVar(value=2.0)
        self.asmr_var = tk.StringVar(value="(none)")  # server-side asmr_fx preset
        self.translate_var = tk.BooleanVar(value=False)
        self.jp_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Server status: ?")

        self._build()
        self.after(200, self.check_health)

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 3}

        frm_server = ttk.Frame(self.root)
        frm_server.pack(fill="x", **pad)
        ttk.Label(frm_server, text="Server:").pack(side="left")
        ttk.Entry(frm_server, textvariable=self.server_var, width=32).pack(
            side="left", padx=4
        )
        ttk.Button(frm_server, text="Health", command=self.check_health).pack(
            side="left", padx=4
        )
        self.health_lbl = ttk.Label(frm_server, text="?", width=8)
        self.health_lbl.pack(side="left", padx=4)

        frm_ref = ttk.Frame(self.root)
        frm_ref.pack(fill="x", **pad)
        ttk.Label(frm_ref, text="Ref audio:").pack(side="left")
        ttk.Entry(frm_ref, textvariable=self.ref_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(frm_ref, text="Browse", command=self.browse_ref).pack(side="left")
        ttk.Button(
            frm_ref, text="Default", command=lambda: self.ref_var.set(DEFAULT_REF)
        ).pack(side="left", padx=4)

        frm_tag = ttk.LabelFrame(self.root, text="Non-verbal tag", padding=6)
        frm_tag.pack(fill="x", **pad)
        ttk.Label(frm_tag, text="Tag:").pack(side="left")
        ttk.Combobox(
            frm_tag, textvariable=self.tag_var, values=TAGS, width=24, state="readonly"
        ).pack(side="left", padx=4)
        ttk.Label(frm_tag, text="Position:").pack(side="left", padx=(8, 0))
        ttk.Combobox(
            frm_tag,
            textvariable=self.pos_var,
            values=["prefix", "suffix"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=4)
        ttk.Button(frm_tag, text="Insert into text", command=self.insert_tag).pack(
            side="left", padx=4
        )
        ttk.Button(frm_tag, text="Show cheatsheet", command=self.show_cheatsheet).pack(
            side="right"
        )

        frm_inst = ttk.LabelFrame(self.root, text="Instruct (style hint)", padding=6)
        frm_inst.pack(fill="x", **pad)
        ttk.Label(frm_inst, text="Preset:").pack(side="left")
        cb = ttk.Combobox(
            frm_inst, textvariable=self.instruct_var, values=INSTRUCT_PRESETS, width=28
        )
        cb.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(
            frm_inst, text="Clear", command=lambda: self.instruct_var.set("(none)")
        ).pack(side="left", padx=4)
        ttk.Label(frm_inst, text="ASMR fx:").pack(side="left", padx=(10, 0))
        ttk.Combobox(
            frm_inst,
            textvariable=self.asmr_var,
            values=["(none)", "close", "room", "drift"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=4)

        frm_g = ttk.LabelFrame(self.root, text="Guidance scale", padding=6)
        frm_g.pack(fill="x", **pad)
        self.g_lbl = ttk.Label(frm_g, text="2.00", width=6)
        ttk.Scale(
            frm_g,
            from_=1.0,
            to=5.0,
            orient="horizontal",
            variable=self.guidance_var,
            command=lambda v: self.g_lbl.config(text=f"{float(v):.2f}"),
        ).pack(side="left", fill="x", expand=True, padx=4)
        self.g_lbl.pack(side="left", padx=4)
        ttk.Button(
            frm_g,
            text="Reset 2.0",
            command=lambda: (
                self.guidance_var.set(2.0),
                self.g_lbl.config(text="2.00"),
            ),
        ).pack(side="left", padx=4)

        frm_txt = ttk.Frame(self.root)
        frm_txt.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Label(frm_txt, text="Text:").pack(side="left")
        ttk.Checkbutton(
            frm_txt,
            text="Translate to Japanese (Claude CLI, ~10s)",
            variable=self.translate_var,
        ).pack(side="left", padx=12)
        self.text_box = tk.Text(
            self.root, height=10, wrap="word", font=("Segoe UI", 10)
        )
        self.text_box.pack(fill="both", expand=True, padx=8, pady=4)
        self.text_box.focus_set()

        # Shows the Japanese actually sent to the TTS when translation is on.
        self.jp_lbl = ttk.Label(
            self.root, textvariable=self.jp_var, foreground="#1565c0", wraplength=780
        )
        self.jp_lbl.pack(anchor="w", padx=8)

        frm_btns = ttk.Frame(self.root)
        frm_btns.pack(fill="x", **pad)
        self.gen_btn = ttk.Button(
            frm_btns, text="Generate (Ctrl+Enter)", command=self.generate
        )
        self.gen_btn.pack(side="left")
        self.play_btn = ttk.Button(
            frm_btns, text="Play", command=self.play, state="disabled"
        )
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(frm_btns, text="Stop", command=self.stop).pack(side="left", padx=4)
        self.save_btn = ttk.Button(
            frm_btns, text="Save As...", command=self.save_as, state="disabled"
        )
        self.save_btn.pack(side="left", padx=4)
        ttk.Button(frm_btns, text="Open out folder", command=self.open_folder).pack(
            side="left", padx=4
        )
        ttk.Button(
            frm_btns,
            text="Clear text",
            command=lambda: self.text_box.delete("1.0", "end"),
        ).pack(side="right")

        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=8)
        ttk.Label(self.root, textvariable=self.status_var, foreground="gray").pack(
            anchor="w", padx=8, pady=6
        )

        # Bind on the text box (tab-scoped) so it fires regardless of which tab
        # has focus; "break" stops the newline from being inserted.
        self.text_box.bind("<Control-Return>", lambda _e: (self.generate(), "break")[1])

    def browse_ref(self) -> None:
        cur = self.ref_var.get()
        parent = Path(cur).parent if cur else None
        initial = (
            str(parent)
            if parent and parent.exists()
            else str(REF_OUT if REF_OUT.exists() else ROOT)
        )
        path = filedialog.askopenfilename(
            title="Pick reference WAV",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialdir=initial,
        )
        if path:
            self.ref_var.set(path)

    def insert_tag(self) -> None:
        tag = self.tag_var.get()
        if tag == "(none)" or not tag:
            return
        existing = self.text_box.get("1.0", "end").rstrip()
        self.text_box.delete("1.0", "end")
        if self.pos_var.get() == "prefix":
            self.text_box.insert("1.0", f"{tag} {existing}".strip())
        else:
            self.text_box.insert("1.0", f"{existing} {tag}".strip())

    def show_cheatsheet(self) -> None:
        top = self.winfo_toplevel()
        win = tk.Toplevel(top)
        win.title("OmniVoice tag cheatsheet")
        win.geometry("680x420")
        win.transient(top)

        cols = ("tag", "where", "what it does")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=18)
        tree.heading("tag", text="Tag")
        tree.heading("where", text="Position")
        tree.heading("what it does", text="What it does")
        tree.column("tag", width=180, anchor="w")
        tree.column("where", width=70, anchor="center")
        tree.column("what it does", width=400, anchor="w")
        for tag, pos, desc in TAG_DOCS:
            tree.insert("", "end", values=(tag, pos, desc))
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        note = (
            "Higher guidance (3.0+) makes tag effects more pronounced — but if your text contains "
            "non-whitelisted brackets, the model may pronounce them literally. Stay at 2.0 for safety, "
            "bump to 3.0 when you want stronger tag delivery."
        )
        ttk.Label(win, text=note, wraplength=640, foreground="gray").pack(
            padx=8, pady=(0, 8)
        )
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def check_health(self) -> None:
        threading.Thread(target=self._health_worker, daemon=True).start()

    def _health_worker(self) -> None:
        try:
            r = httpx.get(f"{self.server_var.get().rstrip('/')}/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                cuda = data.get("cuda")
                self.root.after(
                    0,
                    lambda: (
                        self.health_lbl.config(text="OK", foreground="green"),
                        self.status_var.set(f"Server up. CUDA={cuda}. Ready."),
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda: (
                        self.health_lbl.config(
                            text=f"HTTP {r.status_code}", foreground="red"
                        ),
                        self.status_var.set(f"Server returned {r.status_code}"),
                    ),
                )
        except Exception as e:
            err = str(e)
            self.root.after(
                0,
                lambda: (
                    self.health_lbl.config(text="DOWN", foreground="red"),
                    self.status_var.set(f"Server unreachable: {err}"),
                ),
            )

    def generate(self) -> None:
        text = self.text_box.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty text", "Type something first.")
            return
        ref = self.ref_var.get().strip()
        if ref and not Path(ref).exists():
            messagebox.showerror("Bad ref", f"Ref audio not found:\n{ref}")
            return

        instruct = self.instruct_var.get().strip()
        if instruct in ("", "(none)"):
            instruct = None
        gscale = round(float(self.guidance_var.get()), 2)
        asmr = self.asmr_var.get()
        self._asmr = None if asmr in ("", "(none)") else asmr

        self.gen_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.jp_var.set("")
        do_translate = bool(self.translate_var.get())
        self.status_var.set(
            "Translating to Japanese..."
            if do_translate
            else f"Generating... (instruct={instruct!r}, gscale={gscale})"
        )
        threading.Thread(
            target=self._gen_worker,
            args=(text, ref, instruct, gscale, do_translate),
            daemon=True,
        ).start()

    def _gen_worker(
        self,
        text: str,
        ref: str,
        instruct: str | None,
        gscale: float,
        do_translate: bool,
    ) -> None:
        try:
            t0 = time.time()
            if do_translate:
                text = translate.to_japanese(text)
                jp = text
                self.root.after(
                    0,
                    lambda: (
                        self.jp_var.set(f"JP: {jp}"),
                        self.status_var.set("Translated. Generating speech..."),
                    ),
                )
            payload: dict = {"text": text, "guidance_scale": gscale}
            if ref:
                payload["ref_audio"] = ref
            if instruct:
                payload["instruct"] = instruct
            if getattr(self, "_asmr", None):
                payload["asmr"] = self._asmr
            base = self.server_var.get().rstrip("/")
            r = httpx.post(f"{base}/synthesize", json=payload, timeout=600)
            r.raise_for_status()
            data = r.json()
            wav_url = f"{base}{data['audio_url']}"
            wav_bytes = httpx.get(wav_url, timeout=60).content
            local = OUT_DIR / f"gen_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            local.write_bytes(wav_bytes)
            elapsed = time.time() - t0
            self.last_wav = str(local)
            self.root.after(0, lambda: self._on_done(data, elapsed, local.name))
        except httpx.HTTPStatusError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text[:200]
            self.root.after(
                0, lambda: self._on_error(f"HTTP {e.response.status_code}: {detail}")
            )
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self._on_error(err))

    def _on_done(self, data: dict, elapsed: float, fname: str) -> None:
        self.gen_btn.config(state="normal")
        self.play_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.status_var.set(
            f"Done -> {fname}  |  audio {data['duration']}s, server RTF {data['rtf']}, total {elapsed:.1f}s"
        )
        self.play()

    def _on_error(self, err: str) -> None:
        self.gen_btn.config(state="normal")
        self.status_var.set(f"Error: {err}")
        messagebox.showerror("Generation failed", err)

    def play(self) -> None:
        if not self.last_wav:
            return
        winsound.PlaySound(self.last_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def stop(self) -> None:
        winsound.PlaySound(None, winsound.SND_PURGE)

    def save_as(self) -> None:
        if not self.last_wav:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile=Path(self.last_wav).name,
        )
        if path:
            Path(path).write_bytes(Path(self.last_wav).read_bytes())
            self.status_var.set(f"Saved copy to {path}")

    def open_folder(self) -> None:
        os.startfile(str(OUT_DIR))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("OmniVoice TTS")
    root.geometry("820x680")
    root.minsize(620, 520)
    GenerateTab(root).pack(fill="both", expand=True)
    root.mainloop()
