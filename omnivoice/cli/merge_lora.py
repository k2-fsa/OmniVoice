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

"""Merge a LoRA adapter into its base model for normal OmniVoice deployment.

Loads the base model, merges the LoRA adapter weights into it, and saves the
result as a plain HuggingFace-format OmniVoice checkpoint (config.json +
model.safetensors + tokenizer + audio_tokenizer) that can be loaded directly
with ``OmniVoice.from_pretrained()`` — no PEFT dependency needed at inference
time.

Usage:
    omnivoice-merge-lora \\
        --base_model /path/to/OmniVoice \\
        --lora_adapter exp/omnivoice_lora/checkpoint-5000 \\
        --output_dir exp/omnivoice_lora/merged
"""

import argparse
import logging
import os
import shutil

import torch
from transformers import AutoTokenizer

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path
from omnivoice.utils.lora import load_lora_adapter

logger = logging.getLogger(__name__)

# Files that are part of a deployable OmniVoice directory but are not part of
# the model's state dict, so save_pretrained() does not (re)write them.
_EXTRA_DEPLOY_PATHS = ["audio_tokenizer", "chat_template.jinja"]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base OmniVoice model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model checkpoint path (local dir or HuggingFace repo id) "
        "the LoRA adapter was trained from (i.e. its init_from_checkpoint).",
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory (e.g. a LoRA training "
        "checkpoint) containing adapter_config.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the merged, deployable OmniVoice model to.",
    )
    return parser


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    args = get_parser().parse_args()

    logger.info(f"Loading base model from {args.base_model} ...")
    model = OmniVoice.from_pretrained(args.base_model, train=True, dtype=torch.float32)

    merged = load_lora_adapter(model, args.lora_adapter, merge=True)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving merged model to {args.output_dir} ...")
    merged.save_pretrained(args.output_dir)

    # Prefer the tokenizer saved alongside the adapter (training checkpoints
    # save it there); it is identical to the base model's unless new special
    # tokens were added, in which case it is the correct one to keep.
    tokenizer_source = (
        args.lora_adapter
        if os.path.exists(os.path.join(args.lora_adapter, "tokenizer_config.json"))
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.save_pretrained(args.output_dir)

    # Copy over deployment files that aren't part of the model state dict
    # (audio tokenizer, chat template) so the output directory is a
    # self-contained, directly loadable OmniVoice checkpoint.
    resolved_base = _resolve_model_path(args.base_model)
    for name in _EXTRA_DEPLOY_PATHS:
        src = os.path.join(resolved_base, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    logger.info(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
