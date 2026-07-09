from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ProjectRecord:
    id: int
    slug: str
    title: str
    author: str
    genre: str
    language: str
    status: str
    root_path: Optional[str]


@dataclass
class SecretMetadataRecord:
    provider: str
    fingerprint: str
    configured: bool
    storage_mode: str
    last_test_status: Optional[str] = None


@dataclass
class TokenUsageRecord:
    project_id: int
    estimated_input: int
    estimated_output: int
    actual_input: Optional[int] = None
    actual_output: Optional[int] = None
    source: str = "estimate"
    provider_run_id: Optional[int] = None


@dataclass
class AudioAssetRecord:
    project_id: int
    role: str
    path: str
    chapter_id: Optional[str] = None
    segment_id: Optional[str] = None
    sha256: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    status: str = "available"


class WorkspaceRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def create_project(
        self,
        *,
        slug: str,
        title: str,
        author: str = "",
        genre: str = "technical",
        language: str = "pt-BR",
        root_path: Optional[str] = None,
    ) -> ProjectRecord:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO projects(slug, title, author, genre, language, root_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (slug, title, author, genre, language, root_path),
            )
        return self.get_project(int(cursor.lastrowid))

    def get_project(self, project_id: int) -> ProjectRecord:
        row = self.connection.execute(
            "SELECT * FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown project id: {project_id}")
        return ProjectRecord(
            id=int(row["id"]),
            slug=str(row["slug"]),
            title=str(row["title"]),
            author=str(row["author"]),
            genre=str(row["genre"]),
            language=str(row["language"]),
            status=str(row["status"]),
            root_path=row["root_path"],
        )

    def list_projects(self) -> list[ProjectRecord]:
        rows = self.connection.execute("SELECT * FROM projects ORDER BY updated_at DESC, id DESC").fetchall()
        return [self.get_project(int(row["id"])) for row in rows]

    def upsert_secret_metadata(self, record: SecretMetadataRecord) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO secret_metadata(provider, fingerprint, configured, storage_mode, last_test_status)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    configured = excluded.configured,
                    storage_mode = excluded.storage_mode,
                    last_test_status = excluded.last_test_status,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record.provider,
                    record.fingerprint,
                    1 if record.configured else 0,
                    record.storage_mode,
                    record.last_test_status,
                ),
            )

    def get_secret_metadata(self, provider: str) -> Optional[SecretMetadataRecord]:
        row = self.connection.execute(
            "SELECT * FROM secret_metadata WHERE provider = ?",
            (provider,),
        ).fetchone()
        if row is None:
            return None
        return SecretMetadataRecord(
            provider=str(row["provider"]),
            fingerprint=str(row["fingerprint"]),
            configured=bool(row["configured"]),
            storage_mode=str(row["storage_mode"]),
            last_test_status=row["last_test_status"],
        )

    def remove_secret_metadata(self, provider: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM secret_metadata WHERE provider = ?", (provider,))

    def add_token_usage(self, record: TokenUsageRecord) -> int:
        actual_total = (record.actual_input or 0) + (record.actual_output or 0)
        estimated_total = record.estimated_input + record.estimated_output
        total = actual_total if record.source == "actual" else estimated_total
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO token_usage(
                    provider_run_id, project_id, estimated_input, estimated_output,
                    actual_input, actual_output, total, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.provider_run_id,
                    record.project_id,
                    record.estimated_input,
                    record.estimated_output,
                    record.actual_input,
                    record.actual_output,
                    total,
                    record.source,
                ),
            )
        return int(cursor.lastrowid)

    def add_cost_estimate(
        self,
        *,
        project_id: int,
        input_cost: float,
        output_cost: float,
        currency: str = "USD",
        pricing_source: str = "user",
        is_actual: bool = False,
        provider_run_id: Optional[int] = None,
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO cost_estimates(
                    provider_run_id, project_id, currency, input_cost, output_cost,
                    total_cost, pricing_source, is_actual
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    provider_run_id,
                    project_id,
                    currency,
                    input_cost,
                    output_cost,
                    input_cost + output_cost,
                    pricing_source,
                    1 if is_actual else 0,
                ),
            )
        return int(cursor.lastrowid)

    def add_audio_asset(self, record: AudioAssetRecord) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO audio_assets(
                    project_id, chapter_id, segment_id, role, path, sha256,
                    duration_seconds, sample_rate_hz, channels, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.project_id,
                    record.chapter_id,
                    record.segment_id,
                    record.role,
                    record.path,
                    record.sha256,
                    record.duration_seconds,
                    record.sample_rate_hz,
                    record.channels,
                    record.status,
                ),
            )
        return int(cursor.lastrowid)

    def write_checkpoint(self, project_id: int, checkpoint_path: Path, state: dict[str, Any]) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO checkpoints(project_id, checkpoint_path, state_json) VALUES (?, ?, ?)",
                (project_id, str(checkpoint_path), json.dumps(state, ensure_ascii=False, sort_keys=True)),
            )
        return int(cursor.lastrowid)

    def project_snapshot(self, project_id: int) -> dict[str, Any]:
        project = self.get_project(project_id)
        tables = [
            "source_documents",
            "chunks",
            "audiobook_plans",
            "provider_runs",
            "token_usage",
            "cost_estimates",
            "audio_assets",
            "qc_reports",
            "checkpoints",
            "backups",
        ]
        snapshot: dict[str, Any] = {"project": project.__dict__}
        for table in tables:
            rows = self.connection.execute(
                f"SELECT * FROM {table} WHERE project_id = ? ORDER BY id",
                (project_id,),
            ).fetchall()
            snapshot[table] = [dict(row) for row in rows]
        return snapshot
