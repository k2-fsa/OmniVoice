import os
import json
import locale
from typing import Optional, Dict

# Global dictionary to hold translations for the current active language
_current_translations: Dict[str, str] = {}
_current_lang: str = "en"


def get_locales_dir() -> str:
    """Return the absolute path to the locales directory."""
    # Locales is expected to be inside the omnivoice package for portability
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "locales")


def init_i18n(lang: Optional[str] = None):
    """Initialize i18n from JSON files for the given language.

    If lang is None, it tries to detect the system language.
    Defaults to English if detection fails or language is not available.
    """
    global _current_translations, _current_lang

    locales_dir = get_locales_dir()

    if lang is None:
        try:
            # Try to get system default locale
            system_lang, _ = locale.getdefaultlocale()
            if system_lang:
                lang = system_lang
        except Exception:
            lang = "en"

    # Normalize lang (e.g. pt_BR, en_US)
    if lang:
        lang = lang.replace("-", "_")

    _current_lang = lang

    # Try mapping to sub-languages if exact not found
    # e.g. pt_BR -> pt
    lang_files = [f"{lang}.json"]
    if "_" in lang:
        lang_files.append(f"{lang.split('_')[0]}.json")

    # Always fallback to English for the internal dictionary keys
    # English often defines the available keys
    _current_translations = {}

    # Load from JSON
    success = False
    for filename in lang_files:
        path = os.path.join(locales_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    _current_translations.update(json.load(f))
                success = True
                break
            except Exception as e:
                print(f"Error loading translation file {path}: {e}")

    # Make _() available in builtins for easier access
    import builtins

    builtins._ = translate  # noqa
    return success


def translate(message: str) -> str:
    """Lookup a translation for the given message (key)."""
    # Simply return the value from current translations if it exists
    # otherwise return the message key itself (standard i18n behavior)
    return _current_translations.get(message, message)


# Default initialization (detect system)
init_i18n()
