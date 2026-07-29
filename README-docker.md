# OmniVoice Docker deployment

OmniVoice uses one parameterized `Dockerfile` for both runtime variants:

- CPU builds copy `uv.cpu.lock`, which resolves CPU-only PyTorch wheels.
- NVIDIA builds copy `uv.lock`, which resolves the CUDA 12.8 PyTorch wheels.

The operating-system packages, application installation, non-root runtime, and
startup command are otherwise identical. `compose.yaml` contains shared
settings; `compose.cpu.yaml` and `compose.gpu.yaml` select the dependency lock,
image tag, and inference device.

## Prerequisites

- Docker Engine or Docker Desktop with Compose v2.
- Network access during the image build and first model download.
- Enough disk space for the image/cache and enough RAM to load OmniVoice.
- For GPU deployment: a supported NVIDIA GPU, driver, and NVIDIA Container
  Toolkit (or equivalent Docker Desktop/WSL 2 integration).

CPU inference is memory-intensive and substantially slower than GPU inference.
Model download and startup can take many minutes.

## Quick start

CPU:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up --build -d
docker compose -f compose.yaml -f compose.cpu.yaml ps
docker compose -f compose.yaml -f compose.cpu.yaml logs -f omnivoice
```

NVIDIA GPU:

```bash
nvidia-smi
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml ps
```

Open <http://localhost:8001>. The container starts Gradio non-interactively and
does not preload Whisper ASR. If reference text is omitted during voice
cloning, Whisper can still load lazily on that request.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Hugging Face model ID or local container path |
| `OMNIVOICE_DEVICE` | `cpu`/`cuda` in overrides | Main inference device |
| `OMNIVOICE_HOST` | `0.0.0.0` | Gradio listen address inside the container |
| `OMNIVOICE_PORT` | `8001` | Published and container port |
| `OMNIVOICE_PUBLISH_HOST` | `0.0.0.0` | Host interface on which Docker publishes the port |
| `OMNIVOICE_ROOT_PATH` | empty | Optional reverse-proxy root path |
| `OMNIVOICE_HEALTH_START_PERIOD` | `20m` | Grace period for download/model loading |
| `OMNIVOICE_NORMALIZE_TEXT` | `false` | Initial state of the Vietnamese normalization checkbox |
| `OMNIVOICE_BAMIBERT_HOST_PATH` | `./artifacts/models/bamibert_augmented_best` | Host BamiBERT directory |
| `OMNIVOICE_BAMIBERT_DEVICE` | `cpu` | Device for the normalization detector |

Example:

```bash
OMNIVOICE_PORT=8010 \
OMNIVOICE_PUBLISH_HOST=127.0.0.1 \
OMNIVOICE_NORMALIZE_TEXT=true \
docker compose -f compose.yaml -f compose.cpu.yaml up --build -d
```

The `OMNIVOICE_PORT` value configures both Gradio and the container port, so a
single setting is sufficient.

## Vietnamese normalization

Model weights are deliberately not baked into the image. Put the latest
BamiBERT model at the default host path:

```text
artifacts/models/bamibert_augmented_best/
```

or set `OMNIVOICE_BAMIBERT_HOST_PATH` to another readable directory. Compose
mounts it read-only at `/models/bamibert` for both CPU and GPU deployments.
The detector defaults to CPU to avoid consuming TTS GPU memory; override
`OMNIVOICE_BAMIBERT_DEVICE` when appropriate.

Enable normalization either with `OMNIVOICE_NORMALIZE_TEXT=true` or with the
checkbox in both Gradio generation tabs. Only target text is semantically
normalized. Unicode-safe cleanup remains active for target and reference text.
Detector failures preserve the complete target rather than handing partially
normalized text to inference.

## Cache and restart behavior

`omnivoice_huggingface-cache` stores model snapshots and
`omnivoice_outputs` stores Gradio outputs. They survive container replacement
and normal Compose shutdown, so restarting does not download cached snapshots
again:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml restart
```

The service uses `restart: unless-stopped`, an init process, a 30-second stop
grace period, and the exec-form application command. On startup, a small
entrypoint repairs ownership once for volumes created by older root-running
images, then immediately drops to UID/GID `10001`. The internal cache and
output paths are fixed so this narrowly scoped migration cannot traverse an
arbitrary configured path. Do not use `down -v` unless you intentionally want
to delete the model and output volumes.

BuildKit cache mounts preserve downloaded Python packages between builds.
Dependency layers only invalidate when `pyproject.toml`, the selected lockfile,
or the pinned `uv` version changes; application-source changes reuse those
layers.

## Health check

The service becomes healthy only after model loading completes and Gradio
answers on its configured port. The default health start period is 20 minutes:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml ps
docker inspect --format '{{json .State.Health}}' omnivoice-ui
```

Increase `OMNIVOICE_HEALTH_START_PERIOD` on slow CPU or network-constrained
hosts. A container in `starting` state during the initial model load is
expected; inspect logs before treating it as failed.

## Validation

Validate the merged Compose configurations:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml config
docker compose -f compose.yaml -f compose.gpu.yaml config
```

Confirm the CPU image did not install a CUDA build:

```bash
docker run --rm omnivoice-ui:cpu python -c \
  "import torch, torchaudio, gradio, omnivoice; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

For the CPU image, `torch.version.cuda` must be `None` and
`torch.cuda.is_available()` must be `False`.

GPU Compose configuration can be validated without an adapter, but runtime
validation requires an NVIDIA-enabled Docker host.

## Operations and troubleshooting

Use the same Compose file pair for every operation:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml logs --tail=200 omnivoice
docker compose -f compose.yaml -f compose.cpu.yaml stop
docker compose -f compose.yaml -f compose.cpu.yaml down
```

- **Health remains `starting`:** follow the logs and check RAM, disk space, and
  network access. Increase the health start period if model loading is healthy
  but slow.
- **Container exits with code 137:** the host likely killed it for running out
  of memory.
- **Port already allocated:** change `OMNIVOICE_PORT`.
- **BamiBERT cannot load:** verify the host directory contains the complete
  model and is readable by container UID `10001`.
- **Model download repeats:** verify the `omnivoice_huggingface-cache` volume is
  mounted and `HF_HOME` matches its container target.
- **NVIDIA device unavailable:** check `nvidia-smi`, the NVIDIA Container
  Toolkit, Docker daemon configuration, and host driver compatibility.
- **Permission errors:** keep the named volumes. The startup migration handles
  volumes from older root-running images; bind-mounting replacement cache or
  output paths is outside the supported Compose configuration.

Generated audio, model weights, checkpoints, Hugging Face caches, datasets, and
secrets are excluded from the image build context and must not be committed.
