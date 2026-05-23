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
Gradio demo for OmniVoice — with streaming generation and voice library.

Three tabs:
  1. Voice Clone   — clone from uploaded audio OR saved voice; streaming output.
  2. Voice Design  — create custom voices via speaker attributes.
  3. Voice Library — pre-clone voices, name them, manage (list / delete).

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.utils.voice_library import VoiceLibrary

# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

# Sentinel shown in the saved-voice dropdown meaning "use uploaded audio"
_UPLOAD_SENTINEL = "── Upload audio ──"

# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
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
        "--library-dir",
        default=None,
        help="Directory for the voice library. Default: ~/.omnivoice/voices/",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    library_dir: Optional[str] = None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate
    voice_lib = VoiceLibrary(library_dir)

    # ------------------------------------------------------------------
    # Helper: saved-voice dropdown choices
    # ------------------------------------------------------------------
    def _voice_choices():
        return [_UPLOAD_SENTINEL] + voice_lib.names()

    # ------------------------------------------------------------------
    # Helper: generation settings accordion (shared by Clone & Design tabs)
    # ------------------------------------------------------------------
    def _gen_settings():
        with gr.Accordion("Generation Settings / Настройки генерации (optional)", open=False):
            sp = gr.Slider(
                0.5, 1.5, value=1.0, step=0.05,
                label="Speed / Скорость",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration / Длительность (seconds)",
                info="Leave empty to use speed. Set a fixed duration to override speed.",
            )
            ns = gr.Slider(
                4, 64, value=32, step=1,
                label="Inference Steps / Шаги вывода",
                info="Default: 32. Lower = faster, higher = better quality.",
            )
            dn = gr.Checkbox(
                label="Denoise / Шумоподавление", value=True,
                info="Default: enabled.",
            )
            gs = gr.Slider(
                0.0, 4.0, value=2.0, step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt / Предобработка референса", value=True,
                info="Silence removal and trimming on reference audio.",
            )
            po = gr.Checkbox(
                label="Postprocess Output / Постобработка вывода", value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    # ------------------------------------------------------------------
    # Voice-design instruct builder
    # ------------------------------------------------------------------
    def _build_instruct(groups):
        selected = [g for g in groups if g and g != "Auto"]
        if not selected:
            return None
        parts = []
        for v in selected:
            if " / " in v:
                en, zh = v.split(" / ", 1)
                parts.append(zh.strip() if "Dialect" in en else en.strip())
            else:
                parts.append(v)
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Core generator: clone tab (streaming)
    # ------------------------------------------------------------------
    def _clone_streaming(
        text, lang, saved_voice, ref_aud, ref_text,
        instruct, ns, gs, dn, sp, du, pp, po,
    ) -> Iterator[Tuple]:
        if not text or not text.strip():
            yield None, "⚠️ Please enter the text to synthesise."
            return

        # --- Resolve voice source ---
        voice_prompt = None
        if saved_voice and saved_voice != _UPLOAD_SENTINEL:
            try:
                voice_prompt = voice_lib.load(saved_voice)
            except KeyError as e:
                yield None, f"⚠️ {e}"
                return
            except Exception as e:
                yield None, f"Error loading saved voice: {type(e).__name__}: {e}"
                return
        elif ref_aud:
            try:
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_aud,
                    ref_text=ref_text.strip() if ref_text else None,
                )
            except Exception as e:
                yield None, f"Error processing reference audio: {type(e).__name__}: {e}"
                return
        else:
            yield None, "⚠️ Please upload a reference audio or select a saved voice."
            return

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(ns or 32),
            guidance_scale=float(gs) if gs is not None else 2.0,
            denoise=bool(dn) if dn is not None else True,
            preprocess_prompt=bool(pp),
            postprocess_output=bool(po),
        )

        lang_resolved = lang if (lang and lang != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(),
            language=lang_resolved,
            voice_clone_prompt=voice_prompt,
            generation_config=gen_config,
        )
        if sp is not None and float(sp) != 1.0:
            kw["speed"] = float(sp)
        if du is not None and float(du) > 0:
            kw["duration"] = float(du)
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        yield None, "⏳ Starting generation…"
        try:
            for audio, status in model.generate_streaming(**kw):
                waveform = (audio * 32767).astype(np.int16)
                yield (sampling_rate, waveform), status
        except Exception as e:
            yield None, f"Error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Core function: design tab (non-streaming, usually short)
    # ------------------------------------------------------------------
    def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
        if not text or not text.strip():
            return None, "⚠️ Please enter the text to synthesise."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(ns or 32),
            guidance_scale=float(gs) if gs is not None else 2.0,
            denoise=bool(dn) if dn is not None else True,
            preprocess_prompt=bool(pp),
            postprocess_output=bool(po),
        )
        lang_resolved = lang if (lang and lang != "Auto") else None
        instruct = _build_instruct(list(groups))

        kw: Dict[str, Any] = dict(
            text=text.strip(),
            language=lang_resolved,
            generation_config=gen_config,
        )
        if sp is not None and float(sp) != 1.0:
            kw["speed"] = float(sp)
        if du is not None and float(du) > 0:
            kw["duration"] = float(du)
        if instruct:
            kw["instruct"] = instruct

        try:
            audio_list = model.generate(**kw)
        except Exception as e:
            return None, f"Error: {type(e).__name__}: {e}"

        audio = audio_list[0]
        waveform = (audio * 32767).astype(np.int16)
        return (sampling_rate, waveform), "Done."

    # ------------------------------------------------------------------
    # Voice library helpers
    # ------------------------------------------------------------------
    def _lib_save(name, ref_aud, ref_text, preprocess):
        """Returns (status, table, vc_saved_update, lib_del_dd_update)."""
        if not name or not name.strip():
            return (
                "⚠️ Please enter a name for the voice.",
                _voice_table(),
                gr.update(),
                gr.update(),
            )
        if not ref_aud:
            return (
                "⚠️ Please upload a reference audio.",
                _voice_table(),
                gr.update(),
                gr.update(),
            )
        name = name.strip()
        try:
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_aud,
                ref_text=ref_text.strip() if ref_text else None,
                preprocess_prompt=bool(preprocess),
            )
            voice_lib.save(name, prompt)
            new_choices = _voice_choices()
            return (
                f"✅ Voice '{name}' saved successfully!",
                _voice_table(),
                gr.update(choices=new_choices),
                gr.update(choices=voice_lib.names()),
            )
        except Exception as e:
            return (
                f"Error: {type(e).__name__}: {e}",
                _voice_table(),
                gr.update(),
                gr.update(),
            )

    def _lib_delete(name):
        if not name or name == _UPLOAD_SENTINEL:
            return "⚠️ Please select a voice to delete.", _voice_table(), gr.update()
        ok = voice_lib.delete(name)
        msg = f"✅ Voice '{name}' deleted." if ok else f"⚠️ Voice '{name}' not found."
        new_choices = _voice_choices()
        return msg, _voice_table(), gr.update(choices=new_choices, value=_UPLOAD_SENTINEL)

    def _voice_table():
        voices = voice_lib.list_voices()
        rows = []
        for v in voices:
            preview = v["ref_text"]
            if len(preview) > 70:
                preview = preview[:70] + "…"
            rows.append([v["name"], preview])
        return rows

    def _lib_refresh():
        """Refresh both the library table and the clone-tab saved-voice dropdown."""
        new_choices = _voice_choices()
        table = _voice_table()
        return (
            gr.update(choices=new_choices),   # clone-tab saved-voice dd
            gr.update(choices=new_choices[1:]),  # lib-tab delete dd
            table,
        )

    # ------------------------------------------------------------------
    # CSS / Theme
    # ------------------------------------------------------------------
    theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
    css = """
    .gradio-container { max-width: 100% !important; font-size: 16px !important; }
    .gradio-container h1 { font-size: 1.5em !important; }
    .gradio-container .prose { font-size: 1.1em !important; }
    .compact-audio audio { height: 60px !important; }
    .compact-audio .waveform { min-height: 80px !important; }
    .lib-table { font-size: 0.9em; }
    .status-box textarea { font-size: 0.95em; color: #444; }
    .voice-badge {
        display: inline-block; background: #e8f4ff; border-radius: 6px;
        padding: 2px 10px; margin: 2px; font-size: 0.85em; color: #2060c0;
    }
    """

    def _lang_dropdown(label="Language (optional) / 语种 (可选)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # ==================================================================
    # Build Gradio blocks
    # ==================================================================
    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:

        gr.Markdown(
            """
# 🎙️ OmniVoice Demo

State-of-the-art text-to-speech for **600+ languages** — voice cloning, voice design,
and a **persistent voice library** for instant reuse of saved voice profiles.

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice) by Xiaomi AI Lab Next-gen Kaldi team.
"""
        )

        with gr.Tabs():

            # ==============================================================
            # TAB 1 — Voice Clone (with streaming + saved-voice selector)
            # ==============================================================
            with gr.TabItem("🎤 Voice Clone"):
                with gr.Row():
                    # ---- Left column: inputs ----
                    with gr.Column(scale=1):

                        vc_text = gr.Textbox(
                            label="Text to Synthesise / Текст для синтеза",
                            lines=4,
                            placeholder="Enter the text you want to synthesise…",
                        )

                        # --- Saved voice selector ---
                        vc_saved = gr.Dropdown(
                            label="Saved Voice / Сохранённый голос",
                            choices=_voice_choices(),
                            value=_UPLOAD_SENTINEL,
                            interactive=True,
                            info=(
                                "Select a pre-cloned voice from the library, "
                                "or keep '── Upload audio ──' to use the fields below."
                            ),
                        )

                        # --- Upload-audio group (hidden when a saved voice is chosen) ---
                        with gr.Group() as vc_upload_group:
                            vc_ref_audio = gr.Audio(
                                label="Reference Audio / Референсное аудио",
                                type="filepath",
                                elem_classes="compact-audio",
                            )
                            gr.Markdown(
                                "<span style='font-size:0.85em;color:#888;'>"
                                "Recommended: 3–10 seconds of clear speech."
                                "</span>"
                            )
                            vc_ref_text = gr.Textbox(
                                label="Reference Text (optional) / Текст референса (необяз.)",
                                lines=2,
                                placeholder=(
                                    "Transcript of the reference audio. "
                                    "Leave empty to auto-transcribe via ASR."
                                ),
                            )

                        vc_lang = _lang_dropdown()

                        with gr.Accordion("Instruct (optional)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)

                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()

                        vc_btn = gr.Button("▶ Generate / Сгенерировать", variant="primary")

                        # --- Quick "save this audio as a voice" panel ---
                        with gr.Accordion(
                            "💾 Clone & Save to Library / Клонировать и сохранить", open=False
                        ):
                            gr.Markdown(
                                "Clone the **uploaded** reference audio and save it under "
                                "a name for instant reuse later."
                            )
                            vc_save_name = gr.Textbox(
                                label="Voice Name / Имя голоса",
                                placeholder="e.g. Alice, Speaker1…",
                            )
                            vc_save_btn = gr.Button(
                                "📌 Save Voice / Сохранить голос", variant="secondary"
                            )
                            vc_save_status = gr.Textbox(
                                label="Save Status / Статус сохранения",
                                interactive=False,
                                elem_classes="status-box",
                            )

                    # ---- Right column: outputs ----
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output Audio / Результат",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(
                            label="Status / Статус",
                            lines=2,
                            interactive=False,
                            elem_classes="status-box",
                        )

                # Toggle upload group visibility
                def _toggle_upload(saved):
                    show = saved == _UPLOAD_SENTINEL
                    return gr.update(visible=show)

                vc_saved.change(_toggle_upload, inputs=[vc_saved], outputs=[vc_upload_group])

                # Generate (streaming)
                vc_btn.click(
                    _clone_streaming,
                    inputs=[
                        vc_text, vc_lang, vc_saved, vc_ref_audio, vc_ref_text,
                        vc_instruct, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

                # Quick-save from Voice Clone tab (uses the same _lib_save returning 4 values)
                def _quick_save(name, ref_aud, ref_text, pp_flag):
                    status, _table, vc_saved_upd, _del_upd = _lib_save(
                        name, ref_aud, ref_text, pp_flag
                    )
                    return status, vc_saved_upd

                vc_save_btn.click(
                    _quick_save,
                    inputs=[vc_save_name, vc_ref_audio, vc_ref_text, vc_pp],
                    outputs=[vc_save_status, vc_saved],
                )

            # ==============================================================
            # TAB 2 — Voice Design (unchanged logic, no streaming needed)
            # ==============================================================
            with gr.TabItem("🎨 Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to Synthesise / Текст для синтеза",
                            lines=4,
                            placeholder="Enter the text you want to synthesise…",
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

                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                        vd_btn = gr.Button("▶ Generate / Сгенерировать", variant="primary")

                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output Audio / Результат",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(
                            label="Status / Статус",
                            lines=2,
                            interactive=False,
                            elem_classes="status-box",
                        )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text, vd_lang, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po,
                    ] + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

            # ==============================================================
            # TAB 3 — Voice Library
            # ==============================================================
            with gr.TabItem("📚 Voice Library / Библиотека голосов"):

                gr.Markdown(
                    "Pre-clone any reference audio, give it a name, and reuse it "
                    "instantly from the **Voice Clone** tab — no re-uploading needed."
                )

                with gr.Row():
                    # ---- Left: add a new voice ----
                    with gr.Column(scale=1):
                        gr.Markdown("### ➕ Add / Clone New Voice")
                        lib_name = gr.Textbox(
                            label="Voice Name / Имя голоса",
                            placeholder="e.g. Alice, CEO, Narrator…",
                        )
                        lib_ref_audio = gr.Audio(
                            label="Reference Audio / Референсное аудио",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "3–10 seconds recommended for best cloning quality."
                            "</span>"
                        )
                        lib_ref_text = gr.Textbox(
                            label="Reference Text (optional) / Текст референса (необяз.)",
                            lines=2,
                            placeholder=(
                                "Transcript of the audio. "
                                "Leave empty to auto-transcribe."
                            ),
                        )
                        lib_preprocess = gr.Checkbox(
                            label="Preprocess Audio / Предобработать аудио",
                            value=True,
                            info="Trim silences and normalise volume before cloning.",
                        )
                        lib_save_btn = gr.Button(
                            "🧬 Clone & Save / Клонировать и сохранить",
                            variant="primary",
                        )
                        lib_save_status = gr.Textbox(
                            label="Status / Статус",
                            interactive=False,
                            lines=2,
                            elem_classes="status-box",
                        )

                    # ---- Right: manage saved voices ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 🗂️ Saved Voices / Сохранённые голоса")

                        lib_refresh_btn = gr.Button("🔄 Refresh / Обновить")

                        lib_table = gr.Dataframe(
                            headers=["Name / Имя", "Reference Text Preview"],
                            value=_voice_table(),
                            label="",
                            interactive=False,
                            elem_classes="lib-table",
                            wrap=True,
                        )

                        gr.Markdown("---")
                        gr.Markdown("#### 🗑️ Delete a Voice / Удалить голос")

                        lib_del_dd = gr.Dropdown(
                            label="Select Voice to Delete / Выбрать голос для удаления",
                            choices=voice_lib.names(),
                            value=None,
                            interactive=True,
                        )
                        lib_del_btn = gr.Button(
                            "🗑️ Delete / Удалить", variant="stop"
                        )
                        lib_del_status = gr.Textbox(
                            label="Delete Status / Статус удаления",
                            interactive=False,
                            elem_classes="status-box",
                        )

                # Wiring: save (returns status + table + updated dropdowns in both tabs)
                lib_save_btn.click(
                    _lib_save,
                    inputs=[lib_name, lib_ref_audio, lib_ref_text, lib_preprocess],
                    outputs=[lib_save_status, lib_table, vc_saved, lib_del_dd],
                )

                # Wiring: delete (returns status + table + updated delete dropdown)
                lib_del_btn.click(
                    _lib_delete,
                    inputs=[lib_del_dd],
                    outputs=[lib_del_status, lib_table, lib_del_dd],
                ).then(
                    # Also refresh the clone-tab saved-voice dropdown
                    lambda: gr.update(choices=_voice_choices(), value=_UPLOAD_SENTINEL),
                    inputs=[],
                    outputs=[vc_saved],
                )

                # Wiring: refresh all dropdowns + table
                lib_refresh_btn.click(
                    _lib_refresh,
                    inputs=[],
                    outputs=[vc_saved, lib_del_dd, lib_table],
                )

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

    logging.info("Loading model from %s on device=%s …", checkpoint, device)
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
        asr_model_name=args.asr_model,
    )
    logging.info("Model loaded.")

    demo = build_demo(
        model,
        checkpoint,
        library_dir=args.library_dir,
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
