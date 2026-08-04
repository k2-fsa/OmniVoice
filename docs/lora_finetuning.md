# LoRA Finetuning

In addition to full finetuning (see [training.md](training.md)), OmniVoice supports parameter-efficient finetuning via [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation), using [PEFT](https://github.com/huggingface/peft). LoRA freezes the pretrained LLM backbone and trains small low-rank adapter matrices instead — typically **3-5% of parameters trainable** — which means:

- Much lower optimizer memory (Adam keeps 2 extra tensors per trainable parameter).
- Much smaller adapter checkpoints to ship (tens of MB vs. the full model size).
- The original pretrained weights stay untouched on disk; the adapter can be dropped to recover the original model exactly.

LoRA is **opt-in** via config — with `use_lora: false` (the default), training behaves exactly as documented in [training.md](training.md).

## Installation

```bash
pip install "omnivoice[lora]"
# or, from a source checkout:
pip install peft
```

## What Gets Adapted

| Config field | Applies to | Why |
|---|---|---|
| `lora_target_modules` | LLM attention/MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` by default) | This is where the pretrained language/text-conditioning knowledge lives — the standard set of layers LoRA adapts for transformer LLMs. |
| `lora_modules_to_save` | `audio_embeddings`, `audio_heads` by default | These are OmniVoice-specific audio token I/O layers, not part of the pretrained LLM. There's no pretrained knowledge in them to preserve via low-rank adapters, so they're kept **fully trainable** instead and saved alongside the adapter. |

Everything else (the LLM's other weights, layernorms, etc.) stays frozen, exactly as loaded from the base checkpoint.

## Configuration

Start from [examples/config/train_config_finetune_lora.json](../examples/config/train_config_finetune_lora.json) — identical to [train_config_finetune_sdpa.json](../examples/config/train_config_finetune_sdpa.json) plus the LoRA fields:

```json
{
    "init_from_checkpoint": "k2-fsa/OmniVoice",

    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_modules_to_save": ["audio_embeddings", "audio_heads"],

    "learning_rate": 1e-4,
    "attn_implementation": "sdpa"
}
```

| Field | Description | Default |
|---|---|---|
| `use_lora` | Enable LoRA finetuning instead of full finetuning | `false` |
| `lora_r` | LoRA rank — higher = more capacity, more trainable params | `16` |
| `lora_alpha` | LoRA scaling factor (effective scale is `lora_alpha / lora_r`) | `32` |
| `lora_dropout` | Dropout applied inside the LoRA adapter path | `0.05` |
| `lora_bias` | Which biases to train: `"none"`, `"all"`, or `"lora_only"` | `"none"` |
| `lora_target_modules` | LLM submodule names to attach LoRA adapters to | attention + MLP projections |
| `lora_modules_to_save` | Non-LLM modules trained in full instead of via LoRA | `audio_embeddings`, `audio_heads` |

A higher `learning_rate` than full finetuning (`1e-4` vs. `5e-5`) is typically used since the adapter matrices are randomly initialized and trained from scratch, rather than finetuning already-converged pretrained weights — a higher LR converges faster without the instability risk full finetuning has at that LR.

## Launching Training

```bash
accelerate launch \
    --gpu_ids "0,1" \
    --num_processes 2 \
    -m omnivoice.cli.train \
    --train_config examples/config/train_config_finetune_lora.json \
    --data_config examples/config/data_config_finetune.json \
    --output_dir exp/omnivoice_finetune_lora
```

Or run the full example pipeline (tokenization + training):

```bash
bash examples/run_finetune_lora.sh
```

On startup you'll see a line like:

```text
trainable params: 19,316,736 || all params: 631,894,016 || trainable%: 3.0570
```

confirming only the adapters plus `audio_embeddings`/`audio_heads` are being trained.

## How Many Checkpoints Get Saved

Same three fields as full finetuning govern this:

| Field | Effect |
|---|---|
| `steps` | Total training steps |
| `save_steps` | A checkpoint is saved every `save_steps` steps |
| `keep_last_n_checkpoints` | `-1` keeps every checkpoint; a positive `N` rotates old ones out, keeping only the most recent `N` |

Number of checkpoints saved = `steps / save_steps` (when `steps` is a multiple of `save_steps`), capped at `keep_last_n_checkpoints` if positive.

## Resuming Training

Identical mechanism to full finetuning — set `resume_from_checkpoint`:

```json
{
    "resume_from_checkpoint": "exp/omnivoice_finetune_lora/checkpoint-500"
}
```

Each run rebuilds the same LoRA-wrapped model architecture from `init_from_checkpoint` plus the `lora_*` config fields, then restores optimizer/scheduler/RNG state and adapter weights from the checkpoint on top of it. Keep `init_from_checkpoint` and the `lora_*` fields unchanged across a resume — a mismatch will surface as a shape/key error when `accelerate` restores state.

## Adapter Inference

Apply a trained adapter on top of the base model at inference time — merged into the model in-memory (not written to disk), so generation uses the regular `OmniVoice.generate()` path unmodified:

```bash
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --lora_adapter exp/omnivoice_finetune_lora/checkpoint-500 \
    --text "Hello, this is a text for text-to-speech." \
    --ref_audio ref.wav --ref_text "Reference transcript." \
    --output out.wav
```

`--lora_adapter` accepts any directory containing `adapter_config.json` + `adapter_model.safetensors` — i.e. any LoRA training checkpoint directory.

## Adapter Merging

To deploy with **no PEFT dependency at inference time**, merge the adapter into the base model once and save the result as a normal, standalone OmniVoice checkpoint:

```bash
omnivoice-merge-lora \
    --base_model k2-fsa/OmniVoice \
    --lora_adapter exp/omnivoice_finetune_lora/checkpoint-500 \
    --output_dir exp/omnivoice_finetune_lora/merged
```

This loads the base model, merges the LoRA deltas into its weights, and writes a complete, self-contained OmniVoice directory (model weights, tokenizer, and audio tokenizer).

## Using the Merged Model

The merged directory is a normal OmniVoice checkpoint — use it exactly like any other:

```bash
omnivoice-infer \
    --model exp/omnivoice_finetune_lora/merged \
    --text "Hello, this is a text for text-to-speech." \
    --instruct "male, British accent" \
    --output out.wav
```

```python
from omnivoice.models.omnivoice import OmniVoice

model = OmniVoice.from_pretrained("exp/omnivoice_finetune_lora/merged", device_map="cuda:0")
audios = model.generate(text="Hello world", language="en")
```

The merged model is saved in whatever dtype the base model was loaded in when merging (`float32` by default, since `omnivoice-merge-lora` loads the base model with `dtype=torch.float32`). Pass `--dtype fp32` / `dtype=torch.float32` at inference to match exactly, or `fp16`/`bf16` for less VRAM and faster inference — `from_pretrained` casts down automatically, and quality differences are usually negligible.

## Checkpoint Structure

A LoRA training checkpoint (`exp/.../checkpoint-<step>/`) contains two independent sets of files:

```text
checkpoint-500/
├── adapter_config.json        # PEFT LoRA config (small)
├── adapter_model.safetensors  # LoRA weights + audio_embeddings/audio_heads (small)
├── tokenizer.json             # text tokenizer, saved for adapter/merge use
├── tokenizer_config.json
├── model.safetensors          # accelerate's full training state (base + adapter weights)
├── optimizer.bin              # optimizer state, for resuming
├── scheduler.bin              # LR scheduler state, for resuming
└── random_states_0.pkl        # RNG state, for resuming
```

- **For adapter inference / merging**, only `adapter_config.json` + `adapter_model.safetensors` (+ tokenizer) are used.
- **For resuming training**, `accelerate`'s own state files (`model.safetensors`, `optimizer.bin`, `scheduler.bin`, `random_states_*.pkl`) are used — these include the full (frozen + trainable) model state, since `accelerate`'s checkpointing does not distinguish frozen from trainable parameters. This makes `checkpoint-*/` directories larger than the adapter alone, but keeps resuming exactly as reliable as full finetuning's. If you only need the adapter for deployment, you can delete everything except `adapter_config.json`/`adapter_model.safetensors`/the tokenizer files once training is done.

## Troubleshooting

**`ModuleNotFoundError: No module named 'peft'`**
Run `pip install peft` (or `pip install "omnivoice[lora]"`).

**Resume fails with a shape/key mismatch error**
`resume_from_checkpoint` must be resumed with the *same* `init_from_checkpoint`, `lora_r`, `lora_alpha`, `lora_target_modules`, and `lora_modules_to_save` as the run that produced the checkpoint — these determine the model architecture `accelerate`'s state restore expects to find.

**`omnivoice-merge-lora` / `--lora_adapter` can't find the adapter**
Point it at a checkpoint *directory*, not a specific file — it must contain `adapter_config.json`.

**Training loss looks the same as a frozen model / doesn't move**
Check the `trainable params: ...` line printed at startup. If it's `0` or doesn't include LoRA layers, verify `use_lora: true` is set in the config passed to `--train_config`, and that `lora_target_modules` names match submodules in your `llm_name_or_path`'s architecture.

**Checkpoint saves or merging are unexpectedly slow**
Each checkpoint writes a full base-model-sized file (accelerate's training state — see [Checkpoint Structure](#checkpoint-structure)), and merging both reads and writes one. If disk space is tight or I/O is contended, these can take much longer than expected; check available disk space first.
