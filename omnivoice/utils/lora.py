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

"""LoRA adapter loading shared by inference and merge CLIs.

Used by ``omnivoice.cli.infer`` (``--lora_adapter``) and
``omnivoice.cli.merge_lora``.
"""

import logging
from typing import Union

from omnivoice.models.omnivoice import OmniVoice

logger = logging.getLogger(__name__)


def load_lora_adapter(
    model: OmniVoice, adapter_path: str, merge: bool = True
) -> Union[OmniVoice, "peft.PeftModel"]:  # noqa: F821
    """Attach a trained LoRA adapter to a loaded ``OmniVoice`` model.

    Args:
        model: An ``OmniVoice`` model, as returned by
            :meth:`OmniVoice.from_pretrained`.
        adapter_path: Path to a LoRA adapter directory containing
            ``adapter_config.json`` and ``adapter_model.safetensors``, e.g.
            a LoRA training checkpoint directory.
        merge: If ``True`` (default), merge the adapter weights into
            ``model`` in-memory and return the plain ``OmniVoice`` model, so
            its custom methods (``generate``, ``create_voice_clone_prompt``,
            etc.) keep working unmodified. If ``False``, return the
            ``peft.PeftModel`` wrapper with the adapter kept separate.

    Returns:
        The merged ``OmniVoice`` model (``merge=True``) or the wrapping
        ``PeftModel`` (``merge=False``).
    """
    from peft import PeftModel

    logger.info("Loading LoRA adapter from %s", adapter_path)
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    if merge:
        return peft_model.merge_and_unload()
    return peft_model
