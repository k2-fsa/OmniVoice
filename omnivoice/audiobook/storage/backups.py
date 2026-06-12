from __future__ import annotations

import hashlib
import json
import posixpath
import zipfile
from pathlib import Path
from typing import Iterable

from omnivoice.audiobook.storage.repository import WorkspaceRepository


FORBIDDEN_BACKUP_PATTERNS = [
    "authorization",
    "api_key",
    "apikey",
    "bearer ",
    "openrouter_api_key",
    "".join(["sk", "-or", "-"]),
]


class BackupError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_arcname(path: Path, root: Path) -> str:
    relative = path.resolve().relative_to(root.resolve())
    name = relative.as_posix()
    if name.startswith("../") or "/../" in name or name == "..":
        raise BackupError(f"Unsafe backup path: {path}")
    return name


def _scan_text_for_secrets(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    lowered = content.lower()
    for pattern in FORBIDDEN_BACKUP_PATTERNS:
        if pattern in lowered:
            raise BackupError(f"Refusing to include potential secret-bearing file: {path}")


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def export_project_backup(
    repository: WorkspaceRepository,
    *,
    project_id: int,
    project_root: Path,
    output_zip: Path,
    include_manuscript: bool = False,
) -> Path:
    snapshot = repository.project_snapshot(project_id)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format": "omnivoice-project-backup-v1",
        "project": snapshot["project"],
        "include_manuscript": include_manuscript,
        "database_snapshot": snapshot,
        "files": [],
        "excluded": ["secrets", "authorization_headers", "raw_provider_payloads"],
    }

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in _iter_files(project_root):
            if output_zip.resolve() == path.resolve():
                continue
            if not include_manuscript and path.suffix.lower() == ".docx":
                continue
            _scan_text_for_secrets(path)
            arcname = _safe_arcname(path, project_root)
            archive.write(path, f"project/{arcname}")
            manifest["files"].append({"path": f"project/{arcname}", "sha256": sha256_file(path)})
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        archive.writestr("manifest.json", manifest_bytes)
    return output_zip


def _validate_archive_member(member: zipfile.ZipInfo) -> None:
    name = member.filename.replace("\\", "/")
    normalized = posixpath.normpath(name)
    if normalized.startswith("../") or normalized == ".." or posixpath.isabs(normalized):
        raise BackupError(f"Unsafe archive path: {member.filename}")
    if member.external_attr >> 16 & 0o170000 == 0o120000:
        raise BackupError(f"Refusing to restore symlink: {member.filename}")


def import_project_backup(archive_path: Path, destination: Path, *, overwrite: bool = False) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            _validate_archive_member(member)
        try:
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise BackupError("Backup archive is missing manifest.json") from exc
        for file_info in manifest.get("files", []):
            if not isinstance(file_info, dict):
                raise BackupError("Backup manifest has invalid file entry")
            name = str(file_info.get("path") or "")
            expected = str(file_info.get("sha256") or "")
            _validate_archive_member(zipfile.ZipInfo(name))
            target = destination / name
            if target.exists() and not overwrite:
                raise BackupError(f"Refusing to overwrite existing restore path: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(name)
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                raise BackupError(f"Backup hash mismatch for {name}")
            target.write_bytes(data)
    return manifest
