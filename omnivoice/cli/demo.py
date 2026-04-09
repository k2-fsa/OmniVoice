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

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
from typing import Any, Dict

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.i18n import init_i18n
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"




# ---------------------------------------------------------------------------
# Voice Design instruction templates
# ---------------------------------------------------------------------------
# Each option is displayed as "English / 中文".
# The model expects English for accents and Chinese for dialects.
_CATEGORIES = {
    _("Gender"): [(_("Male"), "male"), (_("Female"), "female")],
    _("Age"): [
        (_("Child"), "child"),
        (_("Teenager"), "teenager"),
        (_("Young Adult"), "young adult"),
        (_("Middle-aged"), "middle-aged"),
        (_("Elderly"), "elderly"),
    ],
    _("Pitch"): [
        (_("Very Low Pitch"), "very low pitch"),
        (_("Low Pitch"), "low pitch"),
        (_("Moderate Pitch"), "moderate pitch"),
        (_("High Pitch"), "high pitch"),
        (_("Very High Pitch"), "very high pitch"),
    ],
    _("Style"): [(_("Whisper"), "whisper")],
    _("English Accent"): [
        (_("American Accent"), "american accent"),
        (_("Australian Accent"), "australian accent"),
        (_("British Accent"), "british accent"),
        (_("Chinese Accent"), "chinese accent"),
        (_("Canadian Accent"), "canadian accent"),
        (_("Indian Accent"), "indian accent"),
        (_("Korean Accent"), "korean accent"),
        (_("Portuguese Accent"), "portuguese accent"),
        (_("Russian Accent"), "russian accent"),
        (_("Japanese Accent"), "japanese accent"),
    ],
    _("Chinese Dialect"): [
        (_("Henan Dialect"), "河南话"),
        (_("Shaanxi Dialect"), "陕西话"),
        (_("Sichuan Dialect"), "四川话"),
        (_("Guizhou Dialect"), "贵州话"),
        (_("Yunnan Dialect"), "云南话"),
        (_("Guilin Dialect"), "桂林话"),
        (_("Jinan Dialect"), "济南话"),
        (_("Shijiazhuang Dialect"), "石家庄话"),
        (_("Gansu Dialect"), "甘肃话"),
        (_("Ningxia Dialect"), "宁夏话"),
        (_("Qingdao Dialect"), "青岛话"),
        (_("Northeast Dialect"), "东北话"),
    ],
}

_ATTR_INFO = {
    _("English Accent"): _("Only effective for English speech."),
    _("Chinese Dialect"): _("Only effective for Chinese speech."),
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
        help=_("Skip loading Whisper ASR model. Reference text auto-transcription will be unavailable."),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Interface language (en, pt_BR, zh).",
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
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
            return None, _("Please enter the text to synthesize.")

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

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, _("Error: {error_type}: {error_msg}").format(
                error_type=type(e).__name__, error_msg=e
            )

        waveform = audio[0].squeeze(0).numpy()  # (T,)
        waveform = (waveform * 32767).astype(np.int16)
        return (sampling_rate, waveform), _("Done.")

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
    def _lang_dropdown(label=None, value="Auto"):
        if label is None:
            label = _("Language (optional)")
        return gr.Dropdown(
            label=label,
            choices=[(_("Auto"), "Auto")] + sorted(lang_display_name(n) for n in LANG_NAMES),
            value=value,
            allow_custom_value=False,
            interactive=True,
            info=_("Keep as Auto to auto-detect the language."),
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion(_("Generation Settings (optional)"), open=False):
            sp = gr.Slider(
                0.5,
                1.5,
                value=1.0,
                step=0.05,
                label=_("Speed"),
                info=_("1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set."),
            )
            du = gr.Number(
                value=None,
                label=_("Duration (seconds)"),
                info=_("Leave empty to use speed. Set a fixed duration to override speed."),
            )
            ns = gr.Slider(
                4,
                64,
                value=32,
                step=1,
                label=_("Inference Steps"),
                info=_("Default: 32. Lower = faster, higher = better quality."),
            )
            dn = gr.Checkbox(
                label=_("Denoise"),
                value=True,
                info=_("Default: enabled. Uncheck to disable denoising."),
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label=_("Guidance Scale (CFG)"),
                info=_("Default: 2.0."),
            )
            pp = gr.Checkbox(
                label=_("Preprocess Prompt"),
                value=True,
                info=_(
                    "apply silence removal and trimming to the reference "
                    "audio, add punctuation in the end of reference text (if not already)"
                ),
            )
            po = gr.Checkbox(
                label=_("Postprocess Output"),
                value=True,
                info=_("Remove long silences from generated audio."),
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title=_("OmniVoice Demo")) as demo:
        gr.Markdown(_("MD_DEMO_HEADER"))

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem(_("Voice Clone")):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label=_("Text to Synthesize"),
                            lines=4,
                            placeholder=_("Enter the text you want to synthesize..."),
                        )
                        vc_ref_audio = gr.Audio(
                            label=_("Reference Audio"),
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            _("<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "</span>")
                        )
                        vc_ref_text = gr.Textbox(
                            label=_("Reference Text (optional)"),
                            lines=2,
                            placeholder=_("Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models."),
                        )
                        vc_lang = _lang_dropdown()
                        with gr.Accordion(_("Instruct (optional)"), open=False):
                            vc_instruct = gr.Textbox(label=_("Instruct"), lines=2)
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button(_("Generate"), variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label=_("Output Audio"),
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label=_("Status"), lines=2)

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
            with gr.TabItem(_("Voice Design")):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label=_("Text to Synthesize"),
                            lines=4,
                            placeholder=_("Enter the text you want to synthesize..."),
                        )
                        vd_lang = _lang_dropdown()

                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[(_("Auto"), "Auto")] + _choices,
                                    value="Auto",
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
                        vd_btn = gr.Button(_("Generate"), variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label=_("Output Audio"),
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label=_("Status"), lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    The values in groups are now the canonical English/Chinese strings
                    because we use (Label, Value) tuples in the Dropdowns.
                    """
                    auto_label = "Auto"  # Use the canonical "Auto" value
                    selected = [g for g in groups if g and g != auto_label]
                    if not selected:
                        return None
                    return ", ".join(selected)

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

    # Initialize i18n
    init_i18n(args.lang)

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(_("Loading model from {checkpoint}, device={device} ...").format(
        checkpoint=checkpoint, device=device
    ))
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=not args.no_asr,
    )
    print(_("Model loaded."))

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
