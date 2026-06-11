from pathlib import Path


def test_offline_modules_do_not_import_openrouter_or_http_clients():
    root = Path(__file__).resolve().parents[1]
    offline_files = [
        root / "omnivoice" / "audiobook" / "planner.py",
        root / "omnivoice" / "audiobook" / "cli.py",
        root / "omnivoice" / "audiobook" / "offline_audit.py",
    ]
    banned = [
        "from omnivoice.audiobook.openrouter",
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
