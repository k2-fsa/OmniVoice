"""OmniVoice Studio app identity + persistent user config.

Single source of truth shared by two consumers:
  - the studio GUI (gui/studio.pyw, gui/server_panel.py)
  - the frozen launcher exe (installer/launcher.py, bundled via PyInstaller)

Config lives outside the repo at %APPDATA%\\OmniVoiceStudio\\config.json so the
installed exe can find the OmniVoice folder before any repo code runs. No
tkinter imports here — must stay importable from the frozen launcher.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "OmniVoice Studio"
APP_VERSION = "1.1.0"  # keep in sync with installer/omnivoice-studio.iss

CONFIG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "OmniVoiceStudio"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULTS: dict = {
    "omnivoice_dir": "",  # folder holding .venv/, gui/, server/, ref-output/
    "server_port": 9192,
    "server_ref_audio": "",  # --ref-audio passed to server/app.py; "" = omit
    "server_autostart": False,  # start the TTS server when the studio opens
}


def load_config() -> dict:
    """Return DEFAULTS overlaid with whatever is on disk (missing/corrupt ok)."""
    cfg = dict(DEFAULTS)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update({k: data[k] for k in DEFAULTS if k in data})
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict) -> None:
    """Persist only the known keys, creating the config dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    known = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
    CONFIG_PATH.write_text(json.dumps(known, indent=2), encoding="utf-8")


def validate_omnivoice_dir(path: str | Path) -> list[str]:
    """Return a list of problems with the candidate OmniVoice folder.

    Empty list means the folder is usable: it has the venv interpreter the
    studio (and demucs/torch/server) run on, and the studio entry point.
    """
    problems: list[str] = []
    if not path:
        return ["No folder selected."]
    root = Path(path)
    if not root.is_dir():
        return [f"Not a folder: {root}"]
    if not (root / ".venv" / "Scripts" / "pythonw.exe").exists():
        problems.append(r"Missing .venv\Scripts\pythonw.exe (venv not set up here)")
    if not (root / "gui" / "studio.pyw").exists():
        problems.append(r"Missing gui\studio.pyw (not an OmniVoice folder?)")
    return problems


def default_ref_audio(root: str | Path) -> str:
    """The stock reference WAV shipped in the OmniVoice folder, if present."""
    ref = Path(root) / "ref" / "default_ref.wav"
    return str(ref) if ref.exists() else ""
