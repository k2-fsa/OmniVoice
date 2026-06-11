import importlib
import sys
from pathlib import Path


def test_offline_modules_do_not_import_openrouter_or_http_clients():
    root = Path(__file__).resolve().parents[1]
    offline_files = [
        root / "omnivoice" / "audiobook" / "__init__.py",
        root / "omnivoice" / "audiobook" / "planner.py",
        root / "omnivoice" / "audiobook" / "cli.py",
        root / "omnivoice" / "audiobook" / "offline_audit.py",
        root / "omnivoice" / "audiobook" / "workflow_cli.py",
        root / "omnivoice" / "audiobook" / "mastering_cli.py",
        root / "omnivoice" / "audiobook" / "qc_cli.py",
    ]
    banned = [
        "from omnivoice.audiobook.openrouter import",
        "import urllib.request",
        "import requests",
        "import httpx",
        "import aiohttp",
        "import socket",
    ]

    for path in offline_files:
        content = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in content, f"{path} must not reference provider/network token {token}"


def test_offline_entrypoints_do_not_load_provider_module_at_runtime():
    for name in list(sys.modules):
        if name == "omnivoice.audiobook" or name.startswith("omnivoice.audiobook."):
            sys.modules.pop(name)

    for module_name in [
        "omnivoice.audiobook",
        "omnivoice.audiobook.cli",
        "omnivoice.audiobook.workflow_cli",
        "omnivoice.audiobook.mastering_cli",
        "omnivoice.audiobook.qc_cli",
        "omnivoice.audiobook.offline_audit",
    ]:
        importlib.import_module(module_name)

    assert "omnivoice.audiobook.openrouter" not in sys.modules


def test_script_entrypoints_keep_openrouter_separate():
    root = Path(__file__).resolve().parents[1]
    content = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'omnivoice-docx-audiobook-plan = "omnivoice.audiobook.cli:main"' in content
    assert 'omnivoice-audiobook-workflow = "omnivoice.audiobook.workflow_cli:main"' in content
    assert 'omnivoice-audiobook-master = "omnivoice.audiobook.mastering_cli:main"' in content
    assert 'omnivoice-audiobook-qc = "omnivoice.audiobook.qc_cli:main"' in content
    assert 'omnivoice-openrouter-audiobook-chunk = "omnivoice.audiobook.openrouter_cli:main"' in content
