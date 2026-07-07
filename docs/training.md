# Training

## Training Config

All training is controlled by a JSON training config file and a JSON data config file.

See [examples/config/](../examples/config/) for ready-to-use configs.

Training config file on Emilia is: [examples/config/train_config_emilia.json](../examples/config/train_config_emilia.json)

Data config file for Emilia is: [examples/config/data_config_emilia.json](../examples/config/data_config_emilia.json)


Key fields in training config file:

| Field | Description | Default |
|---|---|---|
| `llm_name_or_path` | local LLM path or huggingface id | Qwen/Qwen3-0.6B |
| `steps` | Total training steps | 300,000 |
| `learning_rate` | Peak learning rate | 1e-4 |
| `batch_tokens` | Tokens per batch on each GPU | 8192 |
| `attn_implementation` | Attention backend: `"flex_attention"` or `"sdpa"` | `"flex_attention"` |

`output_dir` and `data_config` are passed via command line (see below).

## Attention Implementation

By default, training uses `flex_attention`, which requires PyTorch ≥ 2.5 and a compatible GPU (e.g. NVIDIA Ampere or newer). If your environment does not support `flex_attention`, set `attn_implementation` to `"sdpa"` in your training config. See [examples/config/train_config_finetune_sdpa.json](../examples/config/train_config_finetune_sdpa.json) for a ready-to-use SDPA config:

```json
{
    "attn_implementation": "sdpa",
    "max_sample_tokens": 2000,
    "min_sample_tokens": 50,
    "max_batch_size": 64
}
```

`"sdpa"` uses PyTorch's built-in scaled dot-product attention and works on a wider range of hardware.

The following fields only apply when `attn_implementation != "flex_attention"`:

| Field | Description | Default |
|---|---|---|
| `max_sample_tokens` | Maximum token length per sample; longer samples are dropped | 2000 |
| `min_sample_tokens` | Minimum token length per sample; shorter samples are dropped | 50 |
| `max_batch_size` | Cap on the number of samples per batch | 64 |

`batch_tokens` remains the primary control for memory usage — it sets the total token budget per batch. `max_batch_size` is a safety guard to prevent a batch of many short samples from creating an unusually large batch dimension.

### Batching strategy

The two backends use **different batching strategies**, which are selected automatically:

| Backend | Batching strategy | Batch shape | Notes |
|---|---|---|---|
| `flex_attention` | Sequence packing | `[1, C, batch_tokens]` | Multiple samples concatenated into one long sequence; document boundaries tracked via `document_ids` |
| `sdpa` | Length-grouped padding | `[B, C, max_len]` | Samples with similar token lengths are grouped into the same batch and padded to the local maximum length |

**Why different strategies?**

- With `flex_attention`, sequence packing is memory-efficient because a compact `BlockMask` (not a dense matrix) describes which tokens can attend to each other across document boundaries.
- With `sdpa`, length-grouped padding is used instead: samples of similar token lengths are batched together and padded to the local maximum, so a lightweight `[B, 1, max_len, max_len]` boolean attention mask suffices with low overhead and minimal wasted padding.

## Launching Training

```bash
accelerate launch \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    --num_processes 8 \
    -m omnivoice.cli.train \
    --train_config config/train_config_emilia.json \
    --data_config config/data_config_emilia.json \
    --output_dir exp/omnivoice_emilia
```

## Resuming Training

Set `resume_from_checkpoint` in your training config to resume from an existing checkpoint:

```json
{
    "resume_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## Initializing from a Pretrained Model

To start training from a pretrained OmniVoice checkpoint (for fine-tuning):

```json
{
    "init_from_checkpoint": "exp/omnivoice/checkpoint-100000"
}
```

## Monitoring

Training logs to TensorBoard:
```bash
tensorboard --logdir exp/omnivoice_emilia/tensorboard
```
