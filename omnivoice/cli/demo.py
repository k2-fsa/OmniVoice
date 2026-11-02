#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning, voice design and audiobook creation from book files.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.audiobook import audio_io, text_io, voice_store
from omnivoice.audiobook.audio_io import ExportError
from omnivoice.audiobook.text_io import ExtractionError
from omnivoice.audiobook.voice_store import VoiceError
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR model. Reference text auto-transcription"
        " will be unavailable.",
    )
    parser.add_argument(
        "--asr-model",
        default="openai/whisper-large-v3-turbo",
        help="ASR model path or HuggingFace repo id"
        " (default: openai/whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Where audiobooks are written (default: ./output).",
    )
    parser.add_argument(
        "--voices-dir",
        default=None,
        help="Where saved voices are kept (default: $OMNIVOICE_VOICES_DIR, or"
        " $XDG_CACHE_HOME/omnivoice/voices).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    output_dir: str | Path = "output",
    voices_dir: str | Path | None = None,
) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
    ):
        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed. Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label="Inference Steps",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    # =====================================================================
    # Audiobook helpers
    # =====================================================================
    output_dir = Path(output_dir).expanduser()
    voices_dir = (
        voice_store.set_voices_dir(voices_dir) if voices_dir else voice_store.VOICES_DIR
    )
    prompt_cache: Dict[str, Any] = {}

    def _lang_value(language):
        return language if (language and language != "Auto") else None

    def _resolve_voice(voice_mode, saved, ref_audio, ref_text, instruct):
        """Turn the audiobook voice selection into something generate() takes."""
        if voice_mode == "Saved voice":
            if not saved:
                raise VoiceError("Pick a saved voice, or choose another voice source.")
            voice = voice_store.get_voice(saved)
            if not voice:
                raise VoiceError(f"Saved voice '{saved}' no longer exists.")
            if voice["kind"] == "design":
                return {"kind": "design", "instruct": voice["instruct"]}

            if saved not in prompt_cache:
                prompt = voice_store.load_prompt(voice, model=model)
                if prompt is None:
                    raise VoiceError(
                        f"The reference clip for '{saved}' is missing. "
                        "Save the voice again."
                    )
                prompt_cache[saved] = prompt
            return {"kind": "clone", "prompt": prompt_cache[saved]}

        if voice_mode == "Reference audio":
            if not ref_audio:
                raise VoiceError("Upload or record a reference clip first.")
            return {
                "kind": "clone",
                "prompt": model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=(ref_text or "").strip() or None
                ),
            }

        if voice_mode == "Voice design":
            instruct = (instruct or "").strip()
            if not instruct:
                raise VoiceError(
                    "Describe the voice you want (e.g. 'male, british accent')."
                )
            return {"kind": "design", "instruct": instruct}

        return {"kind": "auto"}

    def _synth_segment(text, voice, language, num_step, speed):
        """Generate one segment of speech as a float32 waveform."""
        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or 32),
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        kw: Dict[str, Any] = dict(
            text=text,
            language=_lang_value(language),
            generation_config=gen_config,
        )
        if voice["kind"] == "clone":
            kw["voice_clone_prompt"] = voice["prompt"]
        elif voice["kind"] == "design":
            kw["instruct"] = voice["instruct"]
        if speed is not None and abs(float(speed) - 1.0) > 1e-3:
            kw["speed"] = float(speed)

        audio = model.generate(**kw)
        return np.asarray(audio[0], dtype=np.float32).reshape(-1)

    def _voice_choices(selected=None):
        names = voice_store.voice_names()
        value = selected if selected in names else (names[0] if names else None)
        return gr.update(choices=names, value=value)

    def _refresh_voices():
        return _voice_choices(), _voice_choices()

    def _load_saved_clip(saved):
        """Fill the reference audio + transcript boxes from a saved voice."""
        voice = voice_store.get_voice(saved) if saved else None
        if not voice:
            return gr.update(), gr.update()
        reference = voice_store.reference_path(voice)
        return (
            str(reference) if reference else None,
            voice.get("ref_text") or voice.get("instruct") or "",
        )

    def _save_voice_from_clip(name, ref_audio, ref_text, language):
        try:
            prompt = None
            if ref_audio:
                prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=(ref_text or "").strip() or None
                )
            voice = voice_store.save_clone_voice(
                name, ref_audio, ref_text, language, prompt
            )
            prompt_cache.pop(voice["name"], None)
            return (
                f"Saved voice **{voice['name']}** (reference clip).",
                _voice_choices(voice["name"]),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            return f"Could not save the voice: {exc}", _voice_choices()

    def _save_voice_from_design(name, instruct, language):
        try:
            voice = voice_store.save_design_voice(name, instruct, language)
            prompt_cache.pop(voice["name"], None)
            return (
                f"Saved voice design **{voice['name']}**.",
                _voice_choices(voice["name"]),
            )
        except VoiceError as exc:
            return f"Could not save the voice: {exc}", _voice_choices()

    def _delete_saved_voice(saved):
        try:
            if not saved:
                return "Pick a voice to delete.", _voice_choices()
            voice_store.delete_voice(saved)
            prompt_cache.pop(saved, None)
            return f"Deleted **{saved}**.", _voice_choices()
        except VoiceError as exc:
            return str(exc), _voice_choices()

    def _book_paths(file_paths):
        """Gradio hands back a list for multiple uploads, a single path for one."""
        if not file_paths:
            return []
        if isinstance(file_paths, (str, Path)):
            return [str(file_paths)]
        return [str(path) for path in file_paths if path]

    def _preview_books(file_paths, max_chars):
        paths = _book_paths(file_paths)
        if not paths:
            return "Drop one or more book files first."

        lines = []
        first_segments = ""
        total_words = 0
        for file_path in paths:
            name = Path(file_path).name
            try:
                text = text_io.extract_text(file_path)
            except ExtractionError as exc:
                lines.append(f"- **{name}** — {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                lines.append(
                    f"- **{name}** — could not read it: {type(exc).__name__}: {exc}"
                )
                continue

            segments = text_io.split_segments(text, max_chars)
            words = len(text.split())
            total_words += words
            lines.append(
                f"- **{name}** — {len(text):,} characters, ~{words:,} words, "
                f"{len(segments):,} segments, ~{words / 150:.0f} min of audio"
            )
            if len(paths) == 1:
                first_segments = "\n".join(segments[:3])

        heading = "**1 book**" if len(paths) == 1 else f"**{len(paths)} books**"
        report = (
            f"{heading} at {int(max_chars)} chars per segment — "
            f"~{total_words:,} words, ~{total_words / 150:.0f} min of audio in "
            "total.\n\n" + "\n".join(lines)
        )
        if first_segments:
            snippet = first_segments[:600].replace(chr(10), " ")
            report += f"\n\nFirst segments:\n\n> {snippet}"
        return report

    def _render_book(
        file_path,
        voice,
        language,
        max_chars,
        num_step,
        speed,
        pause,
        fmt,
        limit,
        tick=None,
    ):
        """Write one book's audio to disk and report how it went.

        Returns ``(path, summary, ok)``. Failures come back as a summary string
        rather than an exception, so one bad book cannot stop a whole batch.
        """
        try:
            text = text_io.extract_text(file_path)
        except ExtractionError as exc:
            return None, str(exc), False
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            logging.error(traceback.format_exc())
            return None, f"could not read it: {type(exc).__name__}: {exc}", False

        segments = text_io.split_segments(text, max_chars)
        if not segments:
            return None, "no speech segments were found in it", False

        limit = int(limit or 0)
        partial = 0 < limit < len(segments)
        if partial:
            segments = segments[:limit]

        stem = Path(file_path).stem
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", stem).strip() or "audiobook"
        if partial:
            stem = f"{stem} (preview)"

        out_path = audio_io.unique_path(output_dir, stem, fmt)
        try:
            # Segments go straight to disk, so a long book never sits in memory.
            sink = audio_io.AudioSink(out_path, model.sampling_rate, fmt, pause)
        except ExportError as exc:
            return None, str(exc), False

        failures = []
        try:
            for index, segment in enumerate(segments):
                if tick:
                    tick(index, len(segments), f"segment {index + 1}/{len(segments)}")
                for attempt in (1, 2):
                    try:
                        sink.write(
                            _synth_segment(segment, voice, language, num_step, speed)
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - retry once
                        logging.warning(
                            f"Segment {index + 1} attempt {attempt} failed: {exc}"
                        )
                        if attempt == 2:
                            failures.append(index + 1)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            logging.error(traceback.format_exc())
            sink.discard()
            return None, f"generation stopped: {type(exc).__name__}: {exc}", False

        if not sink.segments:
            sink.discard()
            return None, "generation failed for every segment — see the log", False

        if tick:
            tick(len(segments), len(segments), "finishing the audio file…")
        try:
            sink.finish()
        except ExportError as exc:
            sink.discard()
            return None, str(exc), False
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            logging.error(traceback.format_exc())
            sink.discard()
            return None, f"export failed: {type(exc).__name__}: {exc}", False

        summary = (
            f"{sink.segments:,} of {len(segments):,} segments · "
            f"~{sink.seconds / 60:.1f} min of audio · {fmt.upper()}"
        )
        if failures:
            summary += f" · skipped {len(failures)} segment(s): {failures[:20]}"
        return out_path, summary, True

    def _generate_audiobooks(
        file_paths,
        voice_mode,
        saved,
        ref_audio,
        ref_text,
        instruct,
        language,
        max_chars,
        num_step,
        speed,
        pause,
        fmt,
        limit,
        progress=gr.Progress(),
    ):
        """Turn every dropped book into its own audio file, one after another.

        The GPU works on one segment at a time, so the books run in sequence
        rather than in parallel, and a book that fails does not stop the rest.
        """
        paths = _book_paths(file_paths)
        if not paths:
            return None, None, "Drop one or more book files first."

        try:
            voice = _resolve_voice(voice_mode, saved, ref_audio, ref_text, instruct)
        except VoiceError as exc:
            return None, None, f"Error: {exc}"

        logging.info(
            f"Audiobook batch: {len(paths)} book(s), voice={voice['kind']}, "
            f"format={fmt}"
        )

        produced = []
        rows = []
        for index, file_path in enumerate(paths):
            name = Path(file_path).name
            label = f"Book {index + 1}/{len(paths)}: {name}"

            def tick(done, total, desc, index=index, label=label):
                # One bar across the batch, not one that restarts for every book.
                share = (done / total) if total else 0.0
                progress((index + share) / len(paths), desc=f"{label} — {desc}")

            out_path, summary, ok = _render_book(
                file_path,
                voice,
                language,
                max_chars,
                num_step,
                speed,
                pause,
                fmt,
                limit,
                tick,
            )
            logging.info(f"{label}: {summary}")
            if ok:
                produced.append(out_path)
            rows.append((name, out_path.name if out_path else "—", summary))

        files = [str(path) for path in produced]
        first = files[0] if files else None

        if len(paths) == 1:
            name, audio_name, summary = rows[0]
            if produced:
                status = (
                    f"### Done — `{audio_name}`\n{summary}\n\nSaved to `{produced[0]}`"
                )
            else:
                status = f"**{name}**: {summary}"
            return first, files, status

        status = f"### Done — {len(produced)} of {len(paths)} books"
        if len(produced) < len(paths):
            status += f" ({len(paths) - len(produced)} failed)"
        status += "\n\n| Book | Audio file | Detail |\n| --- | --- | --- |\n"
        for name, audio_name, summary in rows:
            status += f"| {name} | {audio_name} | {summary} |\n"
        status += f"\nSaved to `{output_dir}`"
        return first, files, status

    def _open_output_folder():
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(output_dir)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
            return f"Opened `{output_dir}`"
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            return f"Could not open the folder ({exc}). It is at `{output_dir}`"

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes
- **Audiobook** — Turn a whole book (TXT, MD, PDF, EPUB, DOC, DOCX, HTML) into a
  single audio file with a reusable saved voice

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference Audio / 参考音频",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label=("Reference Text (optional) / 参考音频文本（可选）"),
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional) / 语种 (可选)")
                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status / 状态", lines=2)

                def _clone_fn(
                    text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        instruct,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_instruct,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesize / 待合成文本",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate / 生成", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / 合成结果",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status / 状态", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            # Dialects have no English equivalent
                            if "Dialect" in v.split(" / ")[0]:
                                parts.append(zh.strip())
                            else:
                                parts.append(en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

            # ==============================================================
            # Audiobook
            # ==============================================================
            with gr.TabItem("Audiobook / 有声书"):
                gr.Markdown(
                    "Drop one or more book files and pick a voice. Supported: "
                    "**TXT, MD, PDF, EPUB, DOC, DOCX, HTML**. Each book becomes its "
                    f"own file, named after the book, in `{output_dir}`. Books in a "
                    "batch are read one after another. PDF needs `pypdf`; M4A export "
                    "needs ffmpeg."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        ab_files = gr.File(
                            label="Book files (drop one or several here) / 书籍文件",
                            file_count="multiple",
                            file_types=sorted(text_io.SUPPORTED_EXTS),
                            type="filepath",
                        )
                        ab_preview_btn = gr.Button("Check files / 检查文件", size="sm")
                        ab_preview = gr.Markdown()

                        with gr.Accordion("Audiobook Settings / 有声书设置", open=True):
                            ab_lang = _lang_dropdown()
                            ab_chars = gr.Slider(
                                120,
                                600,
                                value=300,
                                step=20,
                                label="Segment length (characters) / 分段长度",
                                info="Shorter segments are safer for long books; "
                                "longer ones sound more natural.",
                            )
                            ab_pause = gr.Slider(
                                0.0,
                                1.5,
                                value=0.3,
                                step=0.05,
                                label="Pause between segments (seconds) / 段间停顿",
                            )
                            ab_fmt = gr.Radio(
                                list(audio_io.FORMATS),
                                value="mp3",
                                label="Output format / 输出格式",
                            )
                            ab_steps = gr.Slider(
                                8,
                                64,
                                value=32,
                                step=1,
                                label="Inference Steps / 推理步数",
                            )
                            ab_speed = gr.Slider(
                                0.5,
                                1.5,
                                value=1.0,
                                step=0.05,
                                label="Speed / 语速",
                            )
                            ab_limit = gr.Number(
                                value=0,
                                precision=0,
                                label="Limit segments (0 = full book) / 限制分段",
                                info="Use a small number to test the voice first.",
                            )
                        ab_btn = gr.Button(
                            "Create audiobooks / 生成有声书", variant="primary"
                        )
                        ab_open = gr.Button(
                            "Open output folder / 打开输出目录", size="sm"
                        )
                        ab_folder_status = gr.Markdown()

                    with gr.Column(scale=1):
                        ab_mode = gr.Radio(
                            choices=[
                                "Saved voice",
                                "Reference audio",
                                "Voice design",
                                "Automatic",
                            ],
                            value="Saved voice",
                            label="Voice source / 声音来源",
                            info="Saved voices are reusable and give the most "
                            "consistent narration.",
                        )
                        ab_saved = gr.Dropdown(
                            label="Saved voice / 已保存声音",
                            choices=voice_store.voice_names(),
                            value=(voice_store.voice_names() or [None])[0],
                            interactive=True,
                        )
                        ab_refresh = gr.Button(
                            "Refresh saved voices / 刷新声音列表", size="sm"
                        )

                        with gr.Group(visible=False) as ab_ref_group:
                            ab_ref_audio = gr.Audio(
                                label="Reference Audio / 参考音频",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            ab_ref_text = gr.Textbox(
                                label="Reference Text (optional) / "
                                "参考音频文本（可选）",
                                lines=2,
                                placeholder="Leave empty to auto-transcribe via ASR.",
                            )

                        with gr.Group(visible=False) as ab_design_group:
                            gr.Markdown(
                                "Comma-separated attributes, freely combined: gender "
                                "(male/female), age (child … elderly), pitch (very low "
                                "… very high), style (whisper), English accent "
                                "(american, british, …), or a Chinese dialect "
                                "(四川话, 陕西话 …). Example: "
                                "`female, low pitch, british accent`"
                            )
                            ab_instruct = gr.Textbox(
                                label="Voice description / 声音描述",
                                lines=2,
                                placeholder="female, low pitch, british accent",
                            )

                        with gr.Accordion(
                            "Save the reference as a voice / 保存声音", open=False
                        ):
                            ab_voice_name = gr.Textbox(
                                label="Voice name / 声音名称", lines=1
                            )
                            with gr.Row():
                                ab_save_clone = gr.Button(
                                    "Save reference voice / 保存参考声音", size="sm"
                                )
                                ab_save_design = gr.Button(
                                    "Save voice design / 保存声音设计", size="sm"
                                )

                        with gr.Accordion(
                            "Delete a saved voice / 删除已保存声音", open=False
                        ):
                            ab_delete_target = gr.Dropdown(
                                label="Saved voice / 已保存声音",
                                choices=voice_store.voice_names(),
                                interactive=True,
                            )
                            ab_delete = gr.Button(
                                "Delete voice / 删除声音", size="sm", variant="stop"
                            )

                        ab_voice_status = gr.Markdown()

                ab_audio = gr.Audio(
                    label="Audiobook (first file) / 有声书",
                    type="filepath",
                    interactive=False,
                )
                ab_download = gr.File(
                    label="Download audiobooks / 下载有声书",
                    file_count="multiple",
                    interactive=False,
                )
                ab_status = gr.Markdown()

                def _show_ab_groups(choice):
                    return (
                        gr.update(visible=choice == "Reference audio"),
                        gr.update(visible=choice == "Voice design"),
                        gr.update(visible=choice == "Saved voice"),
                    )

                ab_mode.change(
                    _show_ab_groups,
                    inputs=ab_mode,
                    outputs=[ab_ref_group, ab_design_group, ab_saved],
                )
                ab_refresh.click(_refresh_voices, outputs=[ab_saved, ab_delete_target])
                ab_saved.change(
                    _load_saved_clip,
                    inputs=ab_saved,
                    outputs=[ab_ref_audio, ab_ref_text],
                )
                ab_save_clone.click(
                    _save_voice_from_clip,
                    inputs=[ab_voice_name, ab_ref_audio, ab_ref_text, ab_lang],
                    outputs=[ab_voice_status, ab_saved],
                )
                ab_save_design.click(
                    _save_voice_from_design,
                    inputs=[ab_voice_name, ab_instruct, ab_lang],
                    outputs=[ab_voice_status, ab_saved],
                )
                ab_delete.click(
                    _delete_saved_voice,
                    inputs=ab_delete_target,
                    outputs=[ab_voice_status, ab_saved],
                )
                ab_preview_btn.click(
                    _preview_books,
                    inputs=[ab_files, ab_chars],
                    outputs=ab_preview,
                )
                ab_btn.click(
                    _generate_audiobooks,
                    inputs=[
                        ab_files,
                        ab_mode,
                        ab_saved,
                        ab_ref_audio,
                        ab_ref_text,
                        ab_instruct,
                        ab_lang,
                        ab_chars,
                        ab_steps,
                        ab_speed,
                        ab_pause,
                        ab_fmt,
                        ab_limit,
                    ],
                    outputs=[ab_audio, ab_download, ab_status],
                )
                ab_open.click(_open_output_folder, outputs=ab_folder_status)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    print("Model loaded.")

    demo = build_demo(
        model,
        checkpoint,
        output_dir=args.output_dir,
        voices_dir=args.voices_dir,
    )

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
