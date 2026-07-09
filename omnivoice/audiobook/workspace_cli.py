from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice.audiobook.costing import estimate_chunk_usage, estimate_cost
from omnivoice.audiobook.storage.backups import export_project_backup, import_project_backup
from omnivoice.audiobook.storage.repository import ProjectRecord, SecretMetadataRecord, WorkspaceRepository
from omnivoice.audiobook.storage.schema import initialize_workspace_db
from omnivoice.audiobook.storage.secrets import InMemorySecretStore


SESSION_SECRET_STORE = InMemorySecretStore()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local audiobook workspace projects.")
    parser.add_argument("--db", required=True, help="SQLite workspace database path.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("init", help="Initialize the workspace database.")

    create = subcommands.add_parser("create-project", help="Create a project row.")
    create.add_argument("--slug", required=True)
    create.add_argument("--title", required=True)
    create.add_argument("--author", default="")
    create.add_argument("--genre", choices=["technical", "fiction"], default="technical")
    create.add_argument("--language", default="pt-BR")
    create.add_argument("--root-path")

    list_projects = subcommands.add_parser("list-projects", help="List projects.")
    list_projects.set_defaults(command="list-projects")

    key = subcommands.add_parser("save-session-key", help="Save a provider key for this process only.")
    key.add_argument("--provider", default="openrouter")
    key.add_argument("--api-key", required=True)

    estimate = subcommands.add_parser("estimate-cost", help="Estimate tokens and provider cost.")
    estimate.add_argument("--text", default="")
    estimate.add_argument("--text-file")
    estimate.add_argument("--expected-output-tokens", type=int, default=512)
    estimate.add_argument("--input-per-million", type=float, required=True)
    estimate.add_argument("--output-per-million", type=float, required=True)

    backup = subcommands.add_parser("export-backup", help="Export a safe project backup.")
    backup.add_argument("--project-id", type=int, required=True)
    backup.add_argument("--project-root", required=True)
    backup.add_argument("--output", required=True)
    backup.add_argument("--include-manuscript", action="store_true")

    restore = subcommands.add_parser("import-backup", help="Restore a project backup into a directory.")
    restore.add_argument("--archive", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument("--overwrite", action="store_true")
    return parser


def _project_to_dict(project: ProjectRecord) -> dict[str, object]:
    return project.__dict__


def main() -> None:
    args = _parser().parse_args()
    connection = initialize_workspace_db(args.db)
    repository = WorkspaceRepository(connection)

    if args.command == "init":
        print(json.dumps({"database": str(Path(args.db)), "initialized": True}, indent=2))
        return
    if args.command == "create-project":
        project = repository.create_project(
            slug=args.slug,
            title=args.title,
            author=args.author,
            genre=args.genre,
            language=args.language,
            root_path=args.root_path,
        )
        print(json.dumps(_project_to_dict(project), ensure_ascii=False, indent=2))
        return
    if args.command == "list-projects":
        print(json.dumps([_project_to_dict(project) for project in repository.list_projects()], ensure_ascii=False, indent=2))
        return
    if args.command == "save-session-key":
        result = SESSION_SECRET_STORE.save(args.provider, args.api_key)
        repository.upsert_secret_metadata(
            record=SecretMetadataRecord(
                provider=args.provider,
                fingerprint=result.fingerprint,
                configured=True,
                storage_mode=result.storage_mode,
            )
        )
        print(json.dumps({"provider": result.provider, "fingerprint": result.fingerprint, "storage_mode": result.storage_mode}, indent=2))
        return
    if args.command == "estimate-cost":
        text = args.text
        if args.text_file:
            text = Path(args.text_file).read_text(encoding="utf-8")
        usage = estimate_chunk_usage(text, expected_output_tokens=args.expected_output_tokens)
        cost = estimate_cost(
            usage,
            input_per_million=args.input_per_million,
            output_per_million=args.output_per_million,
        )
        print(json.dumps({"usage": usage.__dict__, "cost": cost.__dict__}, ensure_ascii=False, indent=2))
        return
    if args.command == "export-backup":
        path = export_project_backup(
            repository,
            project_id=args.project_id,
            project_root=Path(args.project_root),
            output_zip=Path(args.output),
            include_manuscript=args.include_manuscript,
        )
        print(json.dumps({"backup": str(path), "secrets_included": False}, indent=2))
        return
    if args.command == "import-backup":
        manifest = import_project_backup(Path(args.archive), Path(args.destination), overwrite=args.overwrite)
        print(json.dumps({"restored": True, "manifest": manifest}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
