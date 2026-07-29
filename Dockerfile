# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG UV_VERSION=0.10.4
ARG OMNIVOICE_LOCKFILE=uv.lock

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    OMNIVOICE_OUTPUT_DIR=/app/outputs \
    OMNIVOICE_HOST=0.0.0.0 \
    OMNIVOICE_PORT=8001 \
    OMNIVOICE_MODEL=k2-fsa/OmniVoice \
    OMNIVOICE_BAMIBERT_MODEL=/models/bamibert \
    OMNIVOICE_BAMIBERT_DEVICE=cpu \
    OMNIVOICE_NORMALIZE_TEXT=false \
    GRADIO_TEMP_DIR=/app/outputs \
    GRADIO_ANALYTICS_ENABLED=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN groupadd --system --gid 10001 omnivoice \
    && useradd --system --uid 10001 --gid omnivoice --home-dir /app omnivoice \
    && install -d -o omnivoice -g omnivoice \
        /app/outputs /cache/huggingface /models/bamibert

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml README.md LICENSE ./
COPY ${OMNIVOICE_LOCKFILE} ./uv.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=omnivoice:omnivoice omnivoice ./omnivoice

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && .venv/bin/python -c "import gradio, omnivoice, torch, torchaudio"

COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

STOPSIGNAL SIGTERM

ENTRYPOINT ["docker-entrypoint"]
CMD ["omnivoice-demo", "--no-asr"]
