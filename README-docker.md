# OmniVoice Docker deployment

OmniVoice provides separate reproducible CPU and NVIDIA/CUDA image workflows.
`compose.yaml` contains shared settings; `compose.cpu.yaml` selects the CPU-only
image and device; `compose.gpu.yaml` requests an NVIDIA GPU and selects CUDA.

The CPU image uses `uv.cpu.lock`, which selects CPU-only PyTorch packages. The
GPU image uses `uv.lock`, which intentionally selects the CUDA 12.8 PyTorch
build. Neither workflow changes model architecture, generation settings, or
audio processing.

## Prerequisites

- Docker Engine or Docker Desktop with Compose v2 (`docker compose`).
- Network access while building and during the first model download.
- Enough disk space for the image and model cache, and enough RAM to load the
  model. CPU inference is memory-intensive and may take several minutes.
- On Windows, Docker Desktop with WSL 2 integration enabled.

NVIDIA deployment also requires a supported GPU, a compatible NVIDIA driver,
and NVIDIA Container Toolkit (or compatible Docker Desktop/WSL 2 GPU support).

## Configuration

The deployment has safe defaults matching the tested CPU setup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIVOICE_DEVICE` | auto; `cpu`/`cuda` in overrides | Inference device |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Hugging Face model ID or local container path |
| `OMNIVOICE_HOST` | `0.0.0.0` | Gradio listen address |
| `OMNIVOICE_PORT` | `8001` | Host and container port |
| `OMNIVOICE_OUTPUT_DIR` | `/app/outputs` | Gradio temporary/generated files |
| `HF_HOME` | `/cache/huggingface` | Hugging Face cache location |
| `OMNIVOICE_ROOT_PATH` | empty | Optional reverse-proxy root path |

Set variables in the shell before `docker compose`, for example:

```bash
OMNIVOICE_PORT=8010 OMNIVOICE_MODEL=/models/OmniVoice \
  docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

A local model path must also be mounted into the container. The service keeps
the Hugging Face cache in the `omnivoice_huggingface-cache` volume and outputs
in `omnivoice_outputs`. `/app/outputs` is created writable in both images.

## Validate configuration

```bash
docker compose -f compose.yaml config
docker compose -f compose.yaml -f compose.cpu.yaml config
docker compose -f compose.yaml -f compose.gpu.yaml config
```

## CPU deployment

Build and start the CPU-only image:

```bash
docker compose --progress=plain -f compose.yaml -f compose.cpu.yaml build
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

Open <http://localhost:8001>. On first start, OmniVoice downloads the model to
the persistent cache and then loads it into RAM. The health check allows up to
15 minutes for this initialization. CPU speech generation was tested
end-to-end, but inference is extremely slow and may take several minutes per
request depending on the host.

Confirm the CPU package selection without downloading the model:

```bash
docker run --rm --entrypoint python omnivoice-ui:cpu -c \
  "import torch, torchaudio, gradio, omnivoice; print('Imports OK'); print('PyTorch:', torch.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

For the CPU image, `torch.version.cuda` must be `None` and
`torch.cuda.is_available()` must be `False`.

## NVIDIA/CUDA deployment

On an NVIDIA host, build and start the CUDA configuration:

```bash
nvidia-smi
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

The CUDA configuration and package path are preserved and Compose-valid, but
CUDA execution was not runtime-tested locally because the development machine
has no NVIDIA GPU. Do not treat a successful image build as proof that the
target host's driver/runtime combination works.

## Operations

Use the same `-f` arguments for the deployment you started. These examples use
CPU:

```bash
# Status and health
docker compose -f compose.yaml -f compose.cpu.yaml ps

# Follow application and model-loading logs
docker compose -f compose.yaml -f compose.cpu.yaml logs -f omnivoice

# Live container CPU and memory usage
docker stats omnivoice-ui

# Stop containers while preserving them
docker compose -f compose.yaml -f compose.cpu.yaml stop

# Remove containers and the network while preserving named volumes
docker compose -f compose.yaml -f compose.cpu.yaml down
```

Normal `down` preserves model and output volumes. Do not use `down -v` unless
you intentionally want to delete both named volumes and redownload the model.

## Troubleshooting

- **Health remains `starting`:** follow the logs. Initial download and model
  loading can take a long time; confirm the host has enough RAM and disk space.
- **Port already allocated:** set another port, such as
  `OMNIVOICE_PORT=8010`, and browse to that port.
- **Model download fails:** verify outbound network/DNS access and available
  cache volume space. A configured Hugging Face token or mirror can be passed
  through the environment when required by the model source.
- **Container exits or is killed:** inspect `docker compose ... logs` and host
  memory usage. An out-of-memory kill commonly has exit code 137.
- **CPU image contains CUDA:** rebuild with both Compose files and no stale tag:
  `docker compose -f compose.yaml -f compose.cpu.yaml build --no-cache`.
- **NVIDIA device unavailable:** run `nvidia-smi` on the host and verify Docker
  NVIDIA runtime support before starting the GPU override.
- **Permission errors under outputs/cache:** inspect the named-volume mounts and
  avoid replacing container paths with host directories lacking write access.

Generated audio, outputs, model checkpoints, and Hugging Face caches are
excluded from the Docker build context and must not be committed.
