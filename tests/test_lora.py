#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../LICENSE for clarification regarding multiple authors
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

"""Tests for LoRA fine-tuning support (config, model wrapping, checkpoint
save/resume, and adapter merging).

Requires a local OmniVoice model directory, set via the
``OMNIVOICE_TEST_MODEL_PATH`` environment variable. Tests are skipped if it
is not set or does not exist, since they load the actual pretrained model.
"""

import json
import os

import pytest
import torch

MODEL_PATH = os.environ.get("OMNIVOICE_TEST_MODEL_PATH")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason="Set OMNIVOICE_TEST_MODEL_PATH to a local OmniVoice model directory to run these tests.",
)


def _lora_config(output_dir, **overrides):
    from omnivoice.training.config import TrainingConfig

    kwargs = {
        "llm_name_or_path": MODEL_PATH,
        "init_from_checkpoint": MODEL_PATH,
        "attn_implementation": "sdpa",
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "output_dir": output_dir,
        "steps": 2,
    }
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


def test_training_config_lora_defaults():
    from omnivoice.training.config import TrainingConfig

    config = TrainingConfig()
    assert config.use_lora is False
    assert config.lora_r == 16
    assert "q_proj" in config.lora_target_modules
    assert "audio_embeddings" in config.lora_modules_to_save
    assert "audio_heads" in config.lora_modules_to_save


def test_training_config_lora_json_roundtrip(tmp_path):
    from omnivoice.training.config import TrainingConfig

    config = _lora_config(str(tmp_path))
    json_path = tmp_path / "train_config.json"
    config.save_to_json(str(json_path))

    with open(json_path) as f:
        data = json.load(f)
    assert data["use_lora"] is True
    assert data["lora_r"] == 4

    reloaded = TrainingConfig.from_json(str(json_path))
    assert reloaded.use_lora is True
    assert reloaded.lora_r == 4


def test_build_model_with_lora_freezes_base_and_keeps_io_heads_trainable(tmp_path):
    from omnivoice.training.builder import build_model_and_tokenizer

    config = _lora_config(str(tmp_path))
    model, _tokenizer = build_model_and_tokenizer(config)

    from peft import PeftModel

    assert isinstance(model, PeftModel)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any("lora_A" in n for n in trainable)
    assert any("lora_B" in n for n in trainable)
    # audio_embeddings/audio_heads are modules_to_save -> fully trainable
    assert any("audio_embeddings" in n for n in trainable)
    assert any("audio_heads" in n for n in trainable)
    # base LLM linear weights themselves (not the lora_A/B deltas) stay frozen
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert any(n.endswith("self_attn.q_proj.base_layer.weight") for n in frozen)


def test_build_model_without_lora_is_plain_omnivoice(tmp_path):
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.training.builder import build_model_and_tokenizer

    config = _lora_config(str(tmp_path), use_lora=False)
    model, _tokenizer = build_model_and_tokenizer(config)

    assert type(model) is OmniVoice
    assert all(p.requires_grad for p in model.parameters())


def test_lora_forward_and_backward(tmp_path):
    from omnivoice.training.builder import build_model_and_tokenizer

    config = _lora_config(str(tmp_path))
    model, _tokenizer = build_model_and_tokenizer(config)

    batch, num_codebooks, seq_len, cond_len = 1, 8, 20, 10
    input_ids = torch.randint(0, 100, (batch, num_codebooks, seq_len))
    audio_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    audio_mask[:, cond_len:] = True
    labels = torch.full((batch, num_codebooks, seq_len), -100, dtype=torch.long)
    labels[:, :, cond_len:] = torch.randint(
        0, 100, (batch, num_codebooks, seq_len - cond_len)
    )

    out = model(input_ids=input_ids, audio_mask=audio_mask, labels=labels)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()

    grad_params = [p for p in model.parameters() if p.grad is not None]
    assert len(grad_params) > 0


def test_lora_checkpoint_save_and_resume(tmp_path):
    from accelerate import Accelerator

    from omnivoice.training.builder import build_model_and_tokenizer
    from omnivoice.training.checkpoint import load_checkpoint, save_checkpoint

    output_dir = str(tmp_path / "exp")
    config = _lora_config(output_dir)

    model, tokenizer = build_model_and_tokenizer(config)
    accelerator = Accelerator(project_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    save_checkpoint(accelerator, model, tokenizer, output_dir, step=1, keep_last_n=-1)

    checkpoint_dir = os.path.join(output_dir, "checkpoint-1")
    saved_files = set(os.listdir(checkpoint_dir))
    # Small PEFT adapter files, used for inference/merging.
    assert {"adapter_config.json", "adapter_model.safetensors"} <= saved_files
    # Full accelerator training state, used for resuming.
    assert {"optimizer.bin"} <= saved_files

    # Resuming requires rebuilding the same LoRA architecture first, then
    # restoring accelerator state on top of it (mirrors full fine-tuning).
    model2, _tokenizer2 = build_model_and_tokenizer(config)
    accelerator2 = Accelerator(project_dir=output_dir)
    optimizer2 = torch.optim.AdamW(
        [p for p in model2.parameters() if p.requires_grad], lr=1e-4
    )
    model2, optimizer2 = accelerator2.prepare(model2, optimizer2)
    step = load_checkpoint(accelerator2, checkpoint_dir)
    assert step == 1


def test_merge_lora_produces_deployable_model(tmp_path):
    from accelerate import Accelerator

    from omnivoice.cli.merge_lora import main as merge_main
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.training.builder import build_model_and_tokenizer
    from omnivoice.training.checkpoint import save_checkpoint

    output_dir = str(tmp_path / "exp")
    config = _lora_config(output_dir)
    model, tokenizer = build_model_and_tokenizer(config)
    accelerator = Accelerator(project_dir=output_dir)
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    save_checkpoint(accelerator, model, tokenizer, output_dir, step=1, keep_last_n=-1)
    checkpoint_dir = os.path.join(output_dir, "checkpoint-1")

    merged_dir = str(tmp_path / "merged")
    argv = [
        "omnivoice-merge-lora",
        "--base_model",
        MODEL_PATH,
        "--lora_adapter",
        checkpoint_dir,
        "--output_dir",
        merged_dir,
    ]
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        merge_main()
    finally:
        sys.argv = old_argv

    merged_files = set(os.listdir(merged_dir))
    assert {"config.json", "model.safetensors", "audio_tokenizer"} <= merged_files

    # Merged model loads and runs like any plain OmniVoice checkpoint.
    reloaded = OmniVoice.from_pretrained(
        merged_dir, device_map="cpu", dtype=torch.float32
    )
    assert type(reloaded) is OmniVoice
