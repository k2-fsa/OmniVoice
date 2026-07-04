#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent named voice-clone library.

Saves ``VoiceClonePrompt`` objects to disk so you can:
- Clone a voice once and reuse it without re-uploading reference audio.
- Give meaningful names to saved clones.
- Load / delete / list clones programmatically or from the Gradio UI.

Storage layout (each voice = two files)::

    <library_dir>/
        my_voice.json    <- metadata: display name, ref_text, ref_rms
        my_voice.pt      <- serialised ref_audio_tokens tensor  (C, T)

Example::

    from omnivoice import OmniVoice
    from omnivoice.utils.voice_library import VoiceLibrary

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", ...)
    lib   = VoiceLibrary()                        # default: ~/.omnivoice/voices/

    # Clone once and persist
    prompt = model.create_voice_clone_prompt("speaker.wav")
    lib.save("Alice", prompt)

    # Reuse later — no audio file needed
    prompt = lib.load("Alice")
    audio  = model.generate("Hello world", voice_clone_prompt=prompt)
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from omnivoice.models.omnivoice import VoiceClonePrompt

# Default storage directory (~/.omnivoice/voices/ — cross-platform)
DEFAULT_LIBRARY_DIR: Path = Path.home() / ".omnivoice" / "voices"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_stem(name: str) -> str:
    """Convert a display name to a filesystem-safe file stem (max 80 chars)."""
    stem = re.sub(r"[^\w\-.]", "_", name.strip())
    stem = re.sub(r"_+", "_", stem).strip("_.")
    return (stem or "voice")[:80]


# ---------------------------------------------------------------------------
# VoiceLibrary
# ---------------------------------------------------------------------------


class VoiceLibrary:
    """Persistent library of named VoiceClonePrompt objects.

    Args:
        library_dir: Directory where voices are stored.
            Defaults to ``~/.omnivoice/voices/``.
    """

    def __init__(self, library_dir: Optional[str] = None):
        self.library_dir = Path(library_dir or DEFAULT_LIBRARY_DIR)
        self.library_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save(self, name: str, prompt: "VoiceClonePrompt") -> None:
        """Save *prompt* under the given display *name*.

        Overwrites silently if a voice with this name already exists.

        Args:
            name: Human-readable display name (e.g. ``"Alice"``).
            prompt: The VoiceClonePrompt to persist.

        Raises:
            ValueError: If *name* is empty.
        """
        name = name.strip()
        if not name:
            raise ValueError("Voice name must not be empty.")

        safe = _safe_stem(name)
        torch.save(prompt.ref_audio_tokens, self.library_dir / f"{safe}.pt")
        meta: Dict = {
            "name": name,
            "safe_name": safe,
            "ref_text": prompt.ref_text,
            "ref_rms": float(prompt.ref_rms),
        }
        (self.library_dir / f"{safe}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, name: str) -> "VoiceClonePrompt":
        """Load a saved voice clone by display name.

        Args:
            name: The display name used in :meth:`save`.

        Returns:
            VoiceClonePrompt ready to pass to ``model.generate()``.

        Raises:
            KeyError: If no voice with this name is found.
        """
        from omnivoice.models.omnivoice import VoiceClonePrompt  # avoid circular import

        meta, safe = self._find(name)
        tokens = torch.load(self.library_dir / f"{safe}.pt", weights_only=True)
        return VoiceClonePrompt(
            ref_audio_tokens=tokens,
            ref_text=meta["ref_text"],
            ref_rms=float(meta["ref_rms"]),
        )

    def delete(self, name: str) -> bool:
        """Delete a saved voice clone.

        Args:
            name: Display name of the voice to delete.

        Returns:
            ``True`` if found and deleted, ``False`` if not found.
        """
        try:
            meta, safe = self._find(name)
        except KeyError:
            return False
        (self.library_dir / f"{safe}.json").unlink(missing_ok=True)
        (self.library_dir / f"{safe}.pt").unlink(missing_ok=True)
        return True

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename a saved voice clone.

        Args:
            old_name: Current display name.
            new_name: New display name.

        Raises:
            KeyError: If *old_name* is not found.
        """
        prompt = self.load(old_name)
        self.delete(old_name)
        self.save(new_name, prompt)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_voices(self) -> List[Dict]:
        """Return sorted metadata dicts for all saved voices.

        Each dict contains: ``name``, ``safe_name``, ``ref_text``, ``ref_rms``.
        """
        voices: List[Dict] = []
        for path in sorted(self.library_dir.glob("*.json")):
            try:
                voices.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
        return voices

    def names(self) -> List[str]:
        """Return display names of all saved voices (alphabetically sorted)."""
        return [v["name"] for v in self.list_voices()]

    def exists(self, name: str) -> bool:
        """Return ``True`` if a voice with this display name exists."""
        try:
            self._find(name)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find(self, name: str) -> Tuple[Dict, str]:
        """Locate metadata by display name.  Returns ``(meta, safe_stem)``."""
        # Fast path: safe stem derived from name exists and name matches
        safe = _safe_stem(name)
        path = self.library_dir / f"{safe}.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("name") == name:
                return meta, safe

        # Slow path: linear scan (handles name collisions on safe stems)
        for meta in self.list_voices():
            if meta.get("name") == name:
                return meta, meta["safe_name"]

        raise KeyError(f"Voice '{name}' not found in library at {self.library_dir}.")

    def __repr__(self) -> str:
        ns = self.names()
        return f"VoiceLibrary(dir={self.library_dir!r}, voices={len(ns)}: {ns})"
