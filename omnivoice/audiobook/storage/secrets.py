from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional


def key_fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    suffix = value[-4:] if len(value) >= 4 else "short"
    return f"sha256:{digest[:12]}:last4:{suffix}"


@dataclass
class SecretSaveResult:
    provider: str
    fingerprint: str
    storage_mode: str
    persistent: bool


class SecretStoreError(RuntimeError):
    pass


class InMemorySecretStore:
    storage_mode = "session"

    def __init__(self):
        self._values: dict[str, str] = {}

    def save(self, provider: str, api_key: str) -> SecretSaveResult:
        if not api_key.strip():
            raise SecretStoreError("API key must not be empty")
        self._values[provider] = api_key
        return SecretSaveResult(
            provider=provider,
            fingerprint=key_fingerprint(api_key),
            storage_mode=self.storage_mode,
            persistent=False,
        )

    def get(self, provider: str) -> Optional[str]:
        return self._values.get(provider)

    def remove(self, provider: str) -> None:
        self._values.pop(provider, None)


class EnvironmentSecretStore:
    storage_mode = "environment"

    def __init__(self, variable_by_provider: Optional[dict[str, str]] = None):
        self.variable_by_provider = variable_by_provider or {"openrouter": "OPENROUTER_API_KEY"}

    def save(self, provider: str, api_key: str) -> SecretSaveResult:
        raise SecretStoreError("Environment secret store is read-only; set the API key in the process environment")

    def get(self, provider: str) -> Optional[str]:
        variable = self.variable_by_provider.get(provider)
        return os.environ.get(variable or "")

    def metadata(self, provider: str) -> Optional[SecretSaveResult]:
        value = self.get(provider)
        if not value:
            return None
        return SecretSaveResult(
            provider=provider,
            fingerprint=key_fingerprint(value),
            storage_mode=self.storage_mode,
            persistent=True,
        )

    def remove(self, provider: str) -> None:
        raise SecretStoreError("Environment secret store cannot remove process environment values")
