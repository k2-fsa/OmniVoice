# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ARG UV_VERSION=0.10.4

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    OMNIVOICE_OUTPUT_DIR=/app/outputs \
    GRADIO_TEMP_DIR=/app/outputs \
    GRADIO_ANALYTICS_ENABLED=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/outputs /cache/huggingface \
    && chmod 0777 /app/outputs /cache/huggingface

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md LICENSE ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY omnivoice ./omnivoice

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8001

CMD ["omnivoice-demo", "--no-asr"]
