from __future__ import annotations

import os
from pathlib import Path


OFFLINE_ENV_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


def configure_offline_defaults() -> None:
    for key, value in OFFLINE_ENV_DEFAULTS.items():
        os.environ[key] = value


def network_access_allowed() -> bool:
    return False


def resolve_local_or_allowed(name_or_path: str, *, resource: str) -> str:
    path = Path(name_or_path).expanduser()
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(name_or_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Modo offline ativo: {resource} '{name_or_path}' nao existe localmente "
            "nem foi encontrado no cache local do Hugging Face. Informe um caminho "
            "local ja baixado. Downloads externos estao bloqueados."
        ) from exc



def ensure_path_inside(root: os.PathLike[str] | str, candidate: os.PathLike[str] | str) -> Path:
    root_path = Path(root).resolve()
    candidate_path = Path(candidate).resolve()
    try:
        candidate_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(
            f"Caminho fora do cache bloqueado pelo modo offline: {candidate_path}"
        ) from exc
    return candidate_path
