"""Local SQLite project vault for audiobook workflows."""

from omnivoice.audiobook.storage.backups import export_project_backup, import_project_backup
from omnivoice.audiobook.storage.paths import ProjectPaths
from omnivoice.audiobook.storage.repository import (
    AudioAssetRecord,
    ProjectRecord,
    SecretMetadataRecord,
    TokenUsageRecord,
    WorkspaceRepository,
)
from omnivoice.audiobook.storage.schema import connect_workspace_db, initialize_workspace_db
from omnivoice.audiobook.storage.secrets import EnvironmentSecretStore, InMemorySecretStore

__all__ = [
    "AudioAssetRecord",
    "EnvironmentSecretStore",
    "InMemorySecretStore",
    "ProjectPaths",
    "ProjectRecord",
    "SecretMetadataRecord",
    "TokenUsageRecord",
    "WorkspaceRepository",
    "connect_workspace_db",
    "export_project_backup",
    "import_project_backup",
    "initialize_workspace_db",
]
