import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from omnivoice.audiobook.costing import estimate_chunk_usage, estimate_cost
from omnivoice.audiobook.storage.backups import BackupError, export_project_backup, import_project_backup
from omnivoice.audiobook.storage.repository import SecretMetadataRecord, WorkspaceRepository
from omnivoice.audiobook.storage.schema import initialize_workspace_db
from omnivoice.audiobook.storage.secrets import InMemorySecretStore
from omnivoice.audiobook.workspace_ui import ApiWorkspaceController


class AudiobookWorkspaceStorageTest(unittest.TestCase):
    def test_project_and_secret_metadata_do_not_store_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "workspace.sqlite"
            connection = initialize_workspace_db(db)
            repository = WorkspaceRepository(connection)

            project = repository.create_project(slug="book", title="Book")
            secret = InMemorySecretStore().save("openrouter", "sk-test-secret-value")
            repository.upsert_secret_metadata(
                SecretMetadataRecord(
                    provider="openrouter",
                    fingerprint=secret.fingerprint,
                    configured=True,
                    storage_mode=secret.storage_mode,
                )
            )

            self.assertEqual(project.slug, "book")
            db_text = db.read_bytes().decode("utf-8", errors="ignore")
            self.assertNotIn("sk-test-secret-value", db_text)
            metadata = repository.get_secret_metadata("openrouter")
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.storage_mode, "session")
            connection.close()

    def test_controller_clears_key_output_after_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = ApiWorkspaceController(Path(tmp) / "workspace.sqlite")

            status, key_value = controller.save_api_key("openrouter", "sk-test-secret-value")

            self.assertEqual(key_value, "")
            self.assertIn("Configured provider: openrouter", status)
            self.assertNotIn("sk-test-secret-value", status)
            controller.close()

    def test_cost_estimate_is_deterministic(self):
        usage = estimate_chunk_usage("one two three", expected_output_tokens=10)
        cost = estimate_cost(usage, input_per_million=1.0, output_per_million=2.0)

        self.assertEqual(usage.input_tokens, 5)
        self.assertEqual(usage.output_tokens, 10)
        self.assertAlmostEqual(cost.total_cost, 0.000025)

    def test_backup_excludes_docx_by_default_and_rejects_secret_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            (root / "plan.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            (root / "book.docx").write_bytes(b"private-docx")

            db = Path(tmp) / "workspace.sqlite"
            repository = WorkspaceRepository(initialize_workspace_db(db))
            project = repository.create_project(slug="book", title="Book", root_path=str(root))
            archive = Path(tmp) / "backup.zip"

            export_project_backup(repository, project_id=project.id, project_root=root, output_zip=archive)

            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
            self.assertIn("manifest.json", names)
            self.assertIn("project/plan.json", names)
            self.assertNotIn("project/book.docx", names)

            token = "Authorization: " + "Bearer " + "sk" + "-or-secret"
            (root / "payload.txt").write_text(token, encoding="utf-8")
            with self.assertRaises(BackupError):
                export_project_backup(repository, project_id=project.id, project_root=root, output_zip=Path(tmp) / "bad.zip")
            repository.connection.close()

    def test_restore_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../evil.txt", "bad")
                zipped.writestr("manifest.json", "{}")

            with self.assertRaises(BackupError):
                import_project_backup(archive, Path(tmp) / "restore")

    def test_schema_foreign_keys_are_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            connection = initialize_workspace_db(Path(tmp) / "workspace.sqlite")
            with self.assertRaises(sqlite3.IntegrityError):
                with connection:
                    connection.execute(
                        "INSERT INTO audio_assets(project_id, role, path) VALUES (?, ?, ?)",
                        (999, "raw", "missing.wav"),
                    )
            connection.close()


if __name__ == "__main__":
    unittest.main()
