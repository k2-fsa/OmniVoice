#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Han Zhu)
#
# Licensed under the Apache License, Version 2.0

"""MLX inference backend for OmniVoice.

This module keeps the public inference behavior of :class:`omnivoice.OmniVoice`
but runs the diffusion language model and OmniVoice audio projections with MLX.
The Higgs audio tokenizer is still loaded through Transformers/PyTorch because
that tokenizer does not have an MLX implementation in this repository.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
import torchaudio

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - exercised on non-Apple CI
    raise ImportError(
        "The MLX backend requires the optional 'mlx' package. Install with "
        "`pip install -e '.[mlx]'` on Apple Silicon."
    ) from exc

from safetensors import safe_open
from transformers import AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel

from omnivoice.models.omnivoice import (
    GenerationTask,
    OmniVoiceGenerationConfig,
    VoiceClonePrompt,
    _combine_text,
    _get_time_steps,
    _resolve_instruct,
    _resolve_language,
    _resolve_model_path,
    _tokenize_with_nonverbal_tags,
)
from omnivoice.utils.audio import (
    cross_fade_chunks,
    fade_and_pad_audio,
    load_audio,
    remove_silence,
    trim_long_audio,
)
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation
from omnivoice.utils.voice_design import _ZH_RE

logger = logging.getLogger(__name__)


@dataclass
class MLXQwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLXQwen3Config":
        rope = data.get("rope_parameters") or {}
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
            head_dim=int(data.get("head_dim", data["hidden_size"] // data["num_attention_heads"])),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", data.get("rope_theta", 1_000_000.0))),
            attention_bias=bool(data.get("attention_bias", False)),
        )


@dataclass
class MLXOmniVoiceConfig:
    audio_vocab_size: int
    audio_mask_id: int
    num_audio_codebook: int
    audio_codebook_weights: list[float]
    llm_config: MLXQwen3Config

    @classmethod
    def from_pretrained(cls, model_path: str) -> "MLXOmniVoiceConfig":
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            audio_vocab_size=int(data.get("audio_vocab_size", 1025)),
            audio_mask_id=int(data.get("audio_mask_id", 1024)),
            num_audio_codebook=int(data.get("num_audio_codebook", 8)),
            audio_codebook_weights=[
                float(x)
                for x in data.get("audio_codebook_weights", [8, 8, 6, 6, 4, 4, 2, 2])
            ],
            llm_config=MLXQwen3Config.from_dict(data["llm_config"]),
        )


class MLXLinear:
    def __init__(self, weight: mx.array, bias: Optional[mx.array] = None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class MLXRMSNorm:
    def __init__(self, weight: mx.array, eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        output = x * mx.rsqrt(mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + self.eps)
        return output.astype(x.dtype) * self.weight


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(x: mx.array, rope_theta: float) -> mx.array:
    seq_len = x.shape[-2]
    head_dim = x.shape[-1]
    positions = mx.arange(seq_len, dtype=mx.float32)
    dims = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv_freq = mx.power(mx.array(rope_theta, dtype=mx.float32), -dims / head_dim)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    cos = mx.cos(emb)[None, None, :, :]
    sin = mx.sin(emb)[None, None, :, :]
    return (x * cos + _rotate_half(x) * sin).astype(x.dtype)


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    return mx.repeat(x, repeats, axis=1)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class MLXQwen3Attention:
    def __init__(self, cfg: MLXQwen3Config, weights: dict[str, mx.array], prefix: str):
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = _linear_from_weights(weights, f"{prefix}.q_proj")
        self.k_proj = _linear_from_weights(weights, f"{prefix}.k_proj")
        self.v_proj = _linear_from_weights(weights, f"{prefix}.v_proj")
        self.o_proj = _linear_from_weights(weights, f"{prefix}.o_proj")
        self.q_norm = _optional_rms_norm(weights, f"{prefix}.q_norm", cfg.rms_norm_eps)
        self.k_norm = _optional_rms_norm(weights, f"{prefix}.k_norm", cfg.rms_norm_eps)

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array]) -> mx.array:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q = _apply_rope(q, self.cfg.rope_theta)
        k = _apply_rope(k, self.cfg.rope_theta)

        repeats = self.num_heads // self.num_kv_heads
        k = _repeat_kv(k, repeats)
        v = _repeat_kv(v, repeats)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if attention_mask is not None:
            scores = mx.where(attention_mask, scores, mx.array(-1e9, dtype=scores.dtype))
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        values = probs @ v
        values = values.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(values)


class MLXQwen3MLP:
    def __init__(self, weights: dict[str, mx.array], prefix: str):
        self.gate_proj = _linear_from_weights(weights, f"{prefix}.gate_proj")
        self.up_proj = _linear_from_weights(weights, f"{prefix}.up_proj")
        self.down_proj = _linear_from_weights(weights, f"{prefix}.down_proj")

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class MLXQwen3Layer:
    def __init__(self, cfg: MLXQwen3Config, weights: dict[str, mx.array], layer_idx: int):
        prefix = f"llm.layers.{layer_idx}"
        self.input_layernorm = MLXRMSNorm(weights[f"{prefix}.input_layernorm.weight"], cfg.rms_norm_eps)
        self.self_attn = MLXQwen3Attention(cfg, weights, f"{prefix}.self_attn")
        self.post_attention_layernorm = MLXRMSNorm(
            weights[f"{prefix}.post_attention_layernorm.weight"],
            cfg.rms_norm_eps,
        )
        self.mlp = MLXQwen3MLP(weights, f"{prefix}.mlp")

    def __call__(self, x: mx.array, attention_mask: Optional[mx.array]) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MLXQwen3Model:
    def __init__(self, cfg: MLXQwen3Config, weights: dict[str, mx.array]):
        self.embed_tokens = weights["llm.embed_tokens.weight"]
        self.layers = [MLXQwen3Layer(cfg, weights, i) for i in range(cfg.num_hidden_layers)]
        self.norm = MLXRMSNorm(weights["llm.norm.weight"], cfg.rms_norm_eps)

    def embed(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens[input_ids]

    def __call__(self, inputs_embeds: mx.array, attention_mask: Optional[mx.array]) -> mx.array:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.norm(hidden_states)


class OmniVoiceMLX:
    """OmniVoice inference model backed by MLX for Apple Silicon."""

    def __init__(self, config: MLXOmniVoiceConfig, weights: dict[str, mx.array]):
        self.config = config
        self.llm = MLXQwen3Model(config.llm_config, weights)
        self.audio_embeddings = weights["audio_embeddings.weight"]
        self.audio_head_weight = weights["audio_heads.weight"]
        self.codebook_layer_offsets = (
            np.arange(config.num_audio_codebook, dtype=np.int64) * config.audio_vocab_size
        )
        total_weight = sum(config.audio_codebook_weights)
        self.normalized_audio_codebook_weights = [w / total_weight for w in config.audio_codebook_weights]

        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.feature_extractor = None
        self.duration_estimator = None
        self.sampling_rate = None
        self._asr_pipe = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        dtype: Union[str, Any] = "float16",
        load_asr: bool = False,
        asr_model_name: str = "openai/whisper-large-v3-turbo",
        audio_tokenizer_device: str = "cpu",
        **kwargs,
    ) -> "OmniVoiceMLX":
        if args:
            logger.debug("Ignoring positional arguments for MLX backend: %s", args)
        kwargs.pop("device_map", None)
        kwargs.pop("train", None)

        mx_dtype = _resolve_mx_dtype(dtype)
        resolved_path = _resolve_model_path(pretrained_model_name_or_path)
        config = MLXOmniVoiceConfig.from_pretrained(resolved_path)
        weights = _load_safetensors(resolved_path, mx_dtype)
        model = cls(config, weights)

        model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)

        audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")
        if not os.path.isdir(audio_tokenizer_path):
            audio_tokenizer_path = _resolve_model_path("eustlb/higgs-audio-v2-tokenizer")

        model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_path,
            device_map=audio_tokenizer_device,
        )
        model.feature_extractor = AutoFeatureExtractor.from_pretrained(audio_tokenizer_path)
        model.sampling_rate = model.feature_extractor.sampling_rate
        model.duration_estimator = RuleDurationEstimator()

        if load_asr:
            model.load_asr_model(model_name=asr_model_name)

        mx.eval(model.audio_embeddings, model.audio_head_weight)
        return model

    def supported_language_ids(self) -> set[str]:
        return LANG_IDS

    def supported_language_names(self) -> set[str]:
        return LANG_NAMES

    def load_asr_model(self, model_name: str = "openai/whisper-large-v3-turbo"):
        from transformers import pipeline as hf_pipeline

        logger.info("Loading ASR model %s ...", model_name)
        self._asr_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=_resolve_model_path(model_name),
            dtype=torch.float32,
            device_map="cpu",
        )

    def transcribe(self, audio: Union[str, tuple]) -> str:
        if self._asr_pipe is None:
            raise RuntimeError("ASR model is not loaded. Call model.load_asr_model() first.")
        if isinstance(audio, str):
            return self._asr_pipe(audio)["text"].strip()
        waveform, sr = audio
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        audio_input = {"array": np.squeeze(waveform), "sampling_rate": sr}
        return self._asr_pipe(audio_input)["text"].strip()

    def generate(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[VoiceClonePrompt, list[VoiceClonePrompt], None] = None,
        instruct: Union[str, list[str], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
        speed: Union[float, list[Optional[float]], None] = None,
        generation_config: Optional[OmniVoiceGenerationConfig] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        if self.audio_tokenizer is None or self.text_tokenizer is None:
            raise RuntimeError("Model is not loaded with audio/text tokenizers.")

        gen_config = generation_config or OmniVoiceGenerationConfig.from_dict(kwargs)
        full_task = self._preprocess_all(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
            speed=speed,
            duration=duration,
        )

        short_idx, long_idx = full_task.get_indices(gen_config, self.audio_tokenizer.config.frame_rate)
        results: list[Any] = [None] * full_task.batch_size

        if short_idx:
            short_task = full_task.slice_task(short_idx)
            assert short_task is not None
            for idx, res in zip(short_idx, self._generate_iterative(short_task, gen_config)):
                results[idx] = res

        if long_idx:
            long_task = full_task.slice_task(long_idx)
            assert long_task is not None
            for idx, res in zip(long_idx, self._generate_chunked(long_task, gen_config)):
                results[idx] = res

        generated_audios = []
        for i, result in enumerate(results):
            assert result is not None, f"Result {i} was not generated"
            generated_audios.append(self._decode_and_post_process(result, full_task.ref_rms[i], gen_config))
        return generated_audios

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
    ) -> VoiceClonePrompt:
        if self.audio_tokenizer is None:
            raise RuntimeError("Audio tokenizer is not loaded.")

        if isinstance(ref_audio, str):
            ref_wav = load_audio(ref_audio, self.sampling_rate)
        else:
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0, keepdims=True)
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform),
                    orig_freq=sr,
                    new_freq=self.sampling_rate,
                ).numpy()
            ref_wav = waveform

        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        if preprocess_prompt:
            if ref_text is None:
                ref_wav = trim_long_audio(ref_wav, self.sampling_rate, trim_threshold=20.0)
            ref_wav = remove_silence(
                ref_wav,
                self.sampling_rate,
                mid_sil=200,
                lead_sil=100,
                trail_sil=200,
            )
            if ref_wav.shape[-1] == 0:
                raise ValueError(
                    "Reference audio is empty after silence removal. "
                    "Try setting preprocess_prompt=False."
                )

        ref_duration = ref_wav.shape[-1] / self.sampling_rate
        if ref_duration > 20.0:
            logger.warning(
                "Reference audio is %.1fs long (>20s). This may slow generation and degrade cloning quality.",
                ref_duration,
            )

        if ref_text is None:
            if self._asr_pipe is None:
                self.load_asr_model()
            ref_text = self.transcribe((ref_wav, self.sampling_rate))

        chunk_size = self.audio_tokenizer.config.hop_length
        clip_size = int(ref_wav.shape[-1] % chunk_size)
        ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        ref_audio_tokens = self.audio_tokenizer.encode(ref_wav_tensor.unsqueeze(0)).audio_codes.squeeze(0)

        if preprocess_prompt:
            ref_text = add_punctuation(ref_text)

        return VoiceClonePrompt(ref_audio_tokens=ref_audio_tokens, ref_text=ref_text, ref_rms=ref_rms)

    def _preprocess_all(
        self,
        text: Union[str, list[str]],
        language: Union[str, list[str], None] = None,
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt: Union[VoiceClonePrompt, list[VoiceClonePrompt], None] = None,
        instruct: Union[str, list[str], None] = None,
        preprocess_prompt: bool = True,
        speed: Union[float, list[Optional[float]], None] = None,
        duration: Union[float, list[Optional[float]], None] = None,
    ) -> GenerationTask:
        text_list = [text] if isinstance(text, str) else text
        batch_size = len(text_list)

        language_list = [_resolve_language(lang) for lang in self._ensure_list(language, batch_size)]
        instruct_list = self._ensure_list(instruct, batch_size)
        for i, item in enumerate(instruct_list):
            if item is not None:
                instruct_list[i] = _resolve_instruct(item, use_zh=bool(text_list[i] and _ZH_RE.search(text_list[i])))

        if voice_clone_prompt is not None and (ref_text is not None or ref_audio is not None):
            logger.warning("Both voice_clone_prompt and ref_text/ref_audio are provided; ref_text/ref_audio ignored.")

        if voice_clone_prompt is None and ref_audio is not None:
            ref_text_list = self._ensure_list(ref_text, batch_size, auto_repeat=False)
            ref_audio_list = self._ensure_list(ref_audio, batch_size, auto_repeat=False)
            voice_clone_prompt = [
                self.create_voice_clone_prompt(
                    ref_audio=ref_audio_list[i],
                    ref_text=ref_text_list[i],
                    preprocess_prompt=preprocess_prompt,
                )
                for i in range(len(ref_text_list))
            ]

        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)
        if voice_clone_prompt_list[0] is not None:
            ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
            ref_audio_tokens_list = [vc.ref_audio_tokens for vc in voice_clone_prompt_list]
            ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]
        else:
            ref_text_list = [None] * batch_size
            ref_audio_tokens_list = [None] * batch_size
            ref_rms_list = [None] * batch_size

        user_speed = None
        if speed is not None:
            user_speed = [float(speed)] * batch_size if isinstance(speed, (int, float)) else list(speed)

        durations = None
        if duration is not None:
            durations = [float(duration)] * batch_size if isinstance(duration, (int, float)) else list(duration)

        num_target_tokens_list = []
        for i in range(batch_size):
            has_dur = durations is not None and durations[i] is not None
            item_speed = 1.0 if has_dur else (user_speed[i] if user_speed else 1.0)
            num_target_tokens_list.append(
                self._estimate_target_tokens(
                    text_list[i],
                    ref_text_list[i],
                    ref_audio_tokens_list[i].size(-1) if ref_audio_tokens_list[i] is not None else None,
                    speed=item_speed,
                )
            )

        speed_list: Optional[List[float]] = None
        if durations is not None:
            frame_rate = self.audio_tokenizer.config.frame_rate
            speed_list = []
            for i in range(batch_size):
                if durations[i] is not None:
                    target_tokens = max(1, int(durations[i] * frame_rate))
                    est = num_target_tokens_list[i]
                    speed_list.append(est / target_tokens if target_tokens > 0 else 1.0)
                    num_target_tokens_list[i] = target_tokens
                else:
                    s = user_speed[i] if user_speed else None
                    speed_list.append(s if s is not None else 1.0)
        elif user_speed is not None:
            speed_list = [s if s is not None else 1.0 for s in user_speed]

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=num_target_tokens_list,
            langs=language_list,
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_tokens_list,
            ref_rms=ref_rms_list,
            speed=speed_list,
        )

    def _ensure_list(self, x: Union[Any, List[Any]], batch_size: int, auto_repeat: bool = True) -> List[Any]:
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in (1, batch_size):
            raise ValueError(f"should be either the number of the text or 1, but got {len(x_list)}")
        if auto_repeat and len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        if num_ref_audio_tokens is None or ref_text is None or len(ref_text) == 0:
            ref_text = "Nice to meet you."
            num_ref_audio_tokens = 25
        est = self.duration_estimator.estimate_duration(text, ref_text, num_ref_audio_tokens)
        if speed > 0 and speed != 1.0:
            est = est / speed
        return max(1, int(est))

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
    ) -> dict[str, np.ndarray]:
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        style_text += f"<|lang_start|>{lang if lang else 'None'}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct if instruct else 'None'}<|instruct_end|>"

        style_ids = self.text_tokenizer(style_text, return_tensors="np").input_ids.astype(np.int64)
        style_tokens = np.repeat(style_ids, self.config.num_audio_codebook, axis=0)[None, :, :]

        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_ids = _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer).numpy().astype(np.int64)
        text_tokens = np.repeat(text_ids, self.config.num_audio_codebook, axis=0)[None, :, :]

        target_audio_tokens = np.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=np.int64,
        )

        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(_audio_tokens_to_numpy(ref_audio_tokens)[None, :, :])
        parts.append(target_audio_tokens)
        cond_input_ids = np.concatenate(parts, axis=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)
        cond_audio_mask = np.zeros((1, cond_total_length), dtype=bool)
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return {"input_ids": cond_input_ids, "audio_mask": cond_audio_mask}

    def _prepare_embed_inputs(self, input_ids: mx.array, audio_mask: mx.array) -> mx.array:
        text_embeds = self.llm.embed(input_ids[:, 0, :])
        offsets = mx.array(self.codebook_layer_offsets.reshape(1, -1, 1), dtype=input_ids.dtype)
        shifted_ids = (input_ids * audio_mask[:, None, :].astype(input_ids.dtype)) + offsets
        audio_embeds = mx.sum(self.audio_embeddings[shifted_ids], axis=1)
        return mx.where(audio_mask[:, :, None], audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: mx.array,
        audio_mask: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        hidden_states = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = hidden_states @ self.audio_head_weight.T
        audio_logits = logits_flat.reshape(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).transpose(0, 2, 1, 3)
        return audio_logits

    def _generate_iterative(
        self,
        task: GenerationTask,
        gen_config: OmniVoiceGenerationConfig,
    ) -> List[np.ndarray]:
        batch_size = task.batch_size
        inputs_list = [
            self._prepare_inference_inputs(
                task.texts[i],
                task.target_lens[i],
                task.ref_texts[i],
                task.ref_audio_tokens[i],
                task.langs[i],
                task.instructs[i],
                gen_config.denoise,
            )
            for i in range(batch_size)
        ]

        c_lens = [inp["input_ids"].shape[2] for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id
        batch_input_ids = np.full(
            (2 * batch_size, self.config.num_audio_codebook, max_c_len),
            pad_id,
            dtype=np.int64,
        )
        batch_audio_mask = np.zeros((2 * batch_size, max_c_len), dtype=bool)
        batch_attention_mask = np.zeros((2 * batch_size, 1, max_c_len, max_c_len), dtype=bool)

        for i, inp in enumerate(inputs_list):
            c_len = c_lens[i]
            u_len = task.target_lens[i]
            batch_input_ids[i, :, :c_len] = inp["input_ids"]
            batch_audio_mask[i, :c_len] = inp["audio_mask"]
            batch_attention_mask[i, :, :c_len, :c_len] = True

            batch_input_ids[batch_size + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[batch_size + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[batch_size + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = np.arange(u_len, max_c_len)
                batch_attention_mask[batch_size + i, :, pad_diag, pad_diag] = True

        tokens = np.full(
            (batch_size, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=np.int64,
        )

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem)
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_penalty = (
            np.arange(self.config.num_audio_codebook, dtype=np.float32).reshape(1, -1, 1)
            * gen_config.layer_penalty_factor
        )

        for step in range(gen_config.num_step):
            batch_logits = self.forward(
                input_ids=mx.array(batch_input_ids),
                audio_mask=mx.array(batch_audio_mask),
                attention_mask=mx.array(batch_attention_mask),
            ).astype(mx.float32)
            mx.eval(batch_logits)

            for i in range(batch_size):
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len = c_lens[i]
                t_len = task.target_lens[i]
                c_logits = np.asarray(batch_logits[i : i + 1, :, c_len - t_len : c_len, :], dtype=np.float32)
                u_logits = np.asarray(batch_logits[batch_size + i : batch_size + i + 1, :, :t_len, :], dtype=np.float32)
                pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits, gen_config)

                scores = scores - layer_penalty
                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample_np(scores, gen_config.position_temperature)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores[sample_tokens != self.config.audio_mask_id] = -np.inf
                flat_scores = scores.reshape(-1)
                if k >= flat_scores.size:
                    topk_idx = np.arange(flat_scores.size)
                else:
                    topk_idx = np.argpartition(flat_scores, -k)[-k:]

                flat_tokens = sample_tokens.reshape(-1)
                flat_pred = pred_tokens.reshape(-1)
                flat_tokens[topk_idx] = flat_pred[topk_idx]

                sample_tokens = flat_tokens.reshape(sample_tokens.shape)
                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[batch_size + i : batch_size + i + 1, :, :t_len] = sample_tokens

        return [tokens[i, :, : task.target_lens[i]].copy() for i in range(batch_size)]

    def _predict_tokens_with_scoring(
        self,
        c_logits: np.ndarray,
        u_logits: np.ndarray,
        gen_config: OmniVoiceGenerationConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        if gen_config.guidance_scale != 0:
            c_log_probs = _log_softmax_np(c_logits, axis=-1)
            u_log_probs = _log_softmax_np(u_logits, axis=-1)
            log_probs = _log_softmax_np(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                axis=-1,
            )
        else:
            log_probs = _log_softmax_np(c_logits, axis=-1)

        log_probs[..., self.config.audio_mask_id] = -np.inf

        if gen_config.class_temperature > 0.0:
            filtered = _filter_top_k_np(log_probs, ratio=0.1)
            pred_tokens = np.argmax(_gumbel_sample_np(filtered, gen_config.class_temperature), axis=-1)
        else:
            pred_tokens = np.argmax(log_probs, axis=-1)

        confidence_scores = np.max(log_probs, axis=-1)
        return pred_tokens.astype(np.int64), confidence_scores.astype(np.float32)

    def _generate_chunked(
        self,
        task: GenerationTask,
        gen_config: OmniVoiceGenerationConfig,
    ) -> List[List[np.ndarray]]:
        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration
                * self.audio_tokenizer.config.frame_rate
                / avg_tokens_per_char
            )
            all_chunks.append(chunk_text_punctuation(task.texts[i], chunk_len=text_chunk_len, min_chunk_len=3))

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have ref_audio."
        )

        max_num_chunks = max(len(c) for c in all_chunks)
        chunk_results: list[list[np.ndarray]] = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    _audio_token_len(ref_audios[j]) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if indices:
                    _run_batch(
                        indices,
                        texts=[all_chunks[i][ci] for i in indices],
                        ref_audios=[task.ref_audio_tokens[i] for i in indices],
                        ref_texts=[task.ref_texts[i] for i in indices],
                    )
        else:
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}
            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if indices:
                    _run_batch(
                        indices,
                        texts=[all_chunks[i][ci] for i in indices],
                        ref_audios=[first_chunk_map[i] for i in indices],
                        ref_texts=[all_chunks[i][0] for i in indices],
                    )

        return chunk_results

    def _decode_and_post_process(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        rms: Union[float, None],
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        tokenizer_device = self.audio_tokenizer.device
        with torch.inference_mode():
            if isinstance(tokens, list):
                chunk_audios = [
                    self.audio_tokenizer.decode(_to_torch_tokens(t, tokenizer_device).unsqueeze(0))
                    .audio_values[0]
                    .detach()
                    .cpu()
                    .numpy()
                    for t in tokens
                ]
                audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
            else:
                audio_waveform = (
                    self.audio_tokenizer.decode(_to_torch_tokens(tokens, tokenizer_device).unsqueeze(0))
                    .audio_values[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
        audio_waveform = self._post_process_audio(audio_waveform, gen_config.postprocess_output, rms)
        return audio_waveform.squeeze(0)

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        postprocess_output: bool,
        ref_rms: Union[float, None],
    ) -> np.ndarray:
        if postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        if ref_rms is not None and ref_rms < 0.1:
            generated_audio = generated_audio * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(generated_audio).max()
            if peak > 1e-6:
                generated_audio = generated_audio / peak * 0.5

        return fade_and_pad_audio(generated_audio, sample_rate=self.sampling_rate)


def _linear_from_weights(weights: dict[str, mx.array], prefix: str) -> MLXLinear:
    return MLXLinear(weights[f"{prefix}.weight"], weights.get(f"{prefix}.bias"))


def _optional_rms_norm(weights: dict[str, mx.array], prefix: str, eps: float) -> Optional[MLXRMSNorm]:
    key = f"{prefix}.weight"
    if key not in weights:
        return None
    return MLXRMSNorm(weights[key], eps)


def _iter_safetensor_files(model_path: str) -> Iterable[str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        for name in sorted(set(index["weight_map"].values())):
            yield os.path.join(model_path, name)
        return
    yield os.path.join(model_path, "model.safetensors")


def _load_safetensors(model_path: str, dtype: mx.Dtype) -> dict[str, mx.array]:
    quant_config = _load_quantization_config(model_path)
    if quant_config:
        weights = _load_quantized_safetensors(model_path, dtype, quant_config)
        _check_required_weights(weights)
        return weights

    weights: dict[str, mx.array] = {}
    for path in _iter_safetensor_files(model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing safetensors file: {path}")
        for key, tensor in mx.load(path).items():
            if tensor.dtype in (mx.float16, mx.float32, mx.bfloat16):
                weights[key] = tensor.astype(dtype)
            else:
                weights[key] = tensor
    _check_required_weights(weights)
    return weights


def _check_required_weights(weights: dict[str, mx.array]) -> None:
    required = [
        "llm.embed_tokens.weight",
        "llm.norm.weight",
        "audio_embeddings.weight",
        "audio_heads.weight",
    ]
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError(f"Checkpoint is missing required OmniVoice weights: {missing}")


def _load_quantization_config(model_path: str) -> Optional[dict[str, Any]]:
    manifest_path = Path(model_path) / "mlx_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    quant = manifest.get("quantization")
    if not quant or quant.get("format") != "omnivoice-rowwise":
        return None
    return quant


def _load_quantized_safetensors(
    model_path: str,
    dtype: mx.Dtype,
    quant_config: dict[str, Any],
) -> dict[str, mx.array]:
    quantized = quant_config.get("tensors", {})
    weights: dict[str, mx.array] = {}

    for path in _iter_safetensor_files(model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing safetensors file: {path}")
        with safe_open(path, framework="np") as f:
            keys = set(f.keys())
            for key in sorted(keys):
                if key.endswith(".scales"):
                    continue
                if key in quantized:
                    weights[key] = mx.array(
                        _dequantize_tensor_np(f, key, quantized[key]),
                        dtype=dtype,
                    )
                    continue
                if key.rsplit(".", 1)[0] in quantized and key.endswith(".scales"):
                    continue
                tensor = f.get_tensor(key)
                if np.issubdtype(tensor.dtype, np.floating):
                    weights[key] = mx.array(tensor, dtype=dtype)
                else:
                    weights[key] = mx.array(tensor)
    return weights


def _dequantize_tensor_np(handle, key: str, info: dict[str, Any]) -> np.ndarray:
    bits = int(info["bits"])
    shape = tuple(int(x) for x in info["shape"])
    group_size = int(info["group_size"])
    cols = shape[-1]
    flat_rows = int(np.prod(shape[:-1]))
    groups = math.ceil(cols / group_size)
    scales = handle.get_tensor(f"{key}.scales").astype(np.float32).reshape(flat_rows, groups)

    if bits == 8:
        q = handle.get_tensor(key).astype(np.float32).reshape(flat_rows, groups, group_size)
    elif bits == 4:
        packed = handle.get_tensor(key).reshape(flat_rows, groups, group_size // 2)
        low = (packed & 0x0F).astype(np.int16) - 8
        high = ((packed >> 4) & 0x0F).astype(np.int16) - 8
        q = np.empty((flat_rows, groups, group_size), dtype=np.float32)
        q[:, :, 0::2] = low
        q[:, :, 1::2] = high
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")

    dequant = (q * scales[:, :, None]).reshape(flat_rows, groups * group_size)
    return dequant[:, :cols].reshape(shape).astype(np.float16)


def _resolve_mx_dtype(dtype: Union[str, Any]) -> mx.Dtype:
    value = str(dtype).lower()
    if "bfloat16" in value or "bf16" in value:
        return mx.bfloat16
    if "float32" in value or "fp32" in value:
        return mx.float32
    if "float16" in value or "fp16" in value or value == "half":
        return mx.float16
    raise ValueError(f"Unsupported MLX dtype: {dtype}")


def _to_torch_tokens(tokens: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(tokens, dtype=np.int64)).to(device)


def _audio_token_len(tokens: Any) -> int:
    if isinstance(tokens, np.ndarray):
        return int(tokens.shape[-1])
    return int(tokens.size(-1))


def _audio_tokens_to_numpy(tokens: Any) -> np.ndarray:
    if isinstance(tokens, np.ndarray):
        return np.asarray(tokens, dtype=np.int64)
    return tokens.cpu().numpy().astype(np.int64)


def _log_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _filter_top_k_np(logits: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    k = math.ceil(ratio * logits.shape[-1])
    idx = np.argpartition(logits, -k, axis=-1)[..., -k:]
    filtered = np.full_like(logits, -np.inf)
    np.put_along_axis(filtered, idx, np.take_along_axis(logits, idx, axis=-1), axis=-1)
    return filtered


def _gumbel_sample_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    uniform = np.random.random(size=scaled.shape).astype(np.float32)
    gumbel = -np.log(-np.log(uniform + 1e-10) + 1e-10)
    return scaled + gumbel
