from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from omnivoice._offline import OFFLINE_ENV_DEFAULTS, configure_offline_defaults, network_access_allowed


@dataclass
class OfflineAuditResult:
    passed: bool
    env: Dict[str, str]
    findings: List[str]

    def to_dict(self):
        return {
            "passed": self.passed,
            "env": self.env,
            "findings": self.findings,
        }


def audit_offline_runtime() -> OfflineAuditResult:
    configure_offline_defaults()
    findings: List[str] = []
    env_snapshot = {key: os.environ.get(key, "") for key in OFFLINE_ENV_DEFAULTS}

    for key, expected in OFFLINE_ENV_DEFAULTS.items():
        actual = os.environ.get(key)
        if actual != expected:
            findings.append(f"{key}={actual!r}, expected {expected!r}")

    if network_access_allowed():
        findings.append("network_access_allowed() returned True")

    return OfflineAuditResult(passed=not findings, env=env_snapshot, findings=findings)
