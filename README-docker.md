# OmniVoice Docker deployment

OmniVoice has separate reproducible CPU and CUDA image workflows:

- `compose.yaml` contains the common port, cache volume, health check, and CUDA
  image defaults.
- `compose.cpu.yaml` selects `Dockerfile.cpu`, the `omnivoice-ui:cpu` image,
  and `--device cpu`. It does not request a GPU.
- `compose.gpu.yaml` retains the CUDA image from `Dockerfile` and adds an
  NVIDIA GPU reservation.

`uv.cpu.lock` selects CPU-only PyTorch packages. The CPU image must not contain
CUDA or NVIDIA runtime packages. The regular `uv.lock` intentionally selects
the CUDA 12.8 PyTorch build, so the GPU image is expected to contain large
CUDA/NVIDIA dependencies.

## Prerequisites

- Docker Engine or Docker Desktop with Docker Compose v2 (`docker compose`).
- Network access while building and during the first model download.
- Enough disk space for the selected image and the model cache.
- On Windows, Docker Desktop running with WSL 2 integration enabled.

An NVIDIA deployment also requires a supported NVIDIA GPU, a sufficiently
recent NVIDIA driver, and NVIDIA Container Toolkit or compatible Docker
Desktop/WSL 2 GPU support.

## Validate Compose configuration

```bash
# Common configuration
docker compose -f compose.yaml config

# CPU configuration
docker compose -f compose.yaml -f compose.cpu.yaml config

# NVIDIA configuration
docker compose -f compose.yaml -f compose.gpu.yaml config
```

## CPU-only image

Build only the CPU image:

```bash
docker compose --progress=plain -f compose.yaml -f compose.cpu.yaml build
```

Run packaging smoke tests without loading or downloading the OmniVoice model:

```bash
docker image inspect omnivoice-ui:cpu --format "{{.Created}} {{.Size}}"

docker run --rm --entrypoint omnivoice-demo omnivoice-ui:cpu --help

docker run --rm --entrypoint python omnivoice-ui:cpu -c \
  "import torch, torchaudio, gradio, omnivoice; print('Imports OK'); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

For the CPU image, `torch.version.cuda` must be `None` and
`torch.cuda.is_available()` must be `False`.

Start the CPU UI only when the machine has enough memory and disk space:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

Open <http://localhost:8001> after the health check passes. First startup may
take a long time because it downloads and loads the model. CPU inference can be
much slower than GPU inference, and packaging success does not prove acceptable
model latency or that the full model fits in available RAM.

Manage the CPU deployment:

```bash
# Status
docker compose -f compose.yaml -f compose.cpu.yaml ps

# Follow logs
docker compose -f compose.yaml -f compose.cpu.yaml logs -f omnivoice

# Stop services without removing them
docker compose -f compose.yaml -f compose.cpu.yaml stop

# Restart the service
docker compose -f compose.yaml -f compose.cpu.yaml restart omnivoice

# Remove the service container and network, but preserve the named cache volume
docker compose -f compose.yaml -f compose.cpu.yaml down
```

## CUDA/NVIDIA image

Build the CUDA-capable image without starting it:

```bash
docker compose --progress=plain -f compose.yaml build
```

Verify CUDA packaging without loading the model:

```bash
docker run --rm --entrypoint python omnivoice-ui:468e927 -c \
  "import torch, gradio, omnivoice; print('Imports OK'); print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

Packaging validation on a machine without an NVIDIA GPU does not prove GPU
inference works. Validate the NVIDIA driver and container runtime on the target
host before deployment:

```bash
nvidia-smi
docker info
```

Build and launch on the NVIDIA host:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

Manage the NVIDIA deployment:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml ps
docker compose -f compose.yaml -f compose.gpu.yaml logs -f omnivoice
docker compose -f compose.yaml -f compose.gpu.yaml stop
docker compose -f compose.yaml -f compose.gpu.yaml restart omnivoice
docker compose -f compose.yaml -f compose.gpu.yaml down
```

## Persistent model cache

Both deployments mount the existing named volume
`omnivoice_huggingface-cache` at `/cache/huggingface`. Model files survive
container stops, restarts, rebuilds, and normal `docker compose down`, avoiding
large repeated downloads.

Do not run `docker compose down -v` unless deletion of the model cache is
explicitly intended. The `-v` option removes Compose-managed volumes, including
the persistent Hugging Face cache.
