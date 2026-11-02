"""Audiobook creation helpers for OmniVoice.

``text_io`` extracts and segments book text, ``audio_io`` streams the generated
segments to an audio file, and ``voice_store`` keeps reusable voices on disk.
The Gradio demo's *Audiobook* tab is built on these.
"""

from omnivoice.audiobook import audio_io, text_io, voice_store

__all__ = ["audio_io", "text_io", "voice_store"]
