"""Hard runtime barrier enforcing zero data leakage from blind_holdout/."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

_BLIND_AUDIT_REASON: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "evoalign_blind_audit_reason", default=None
)

BLIND_DIR_NAME = "blind_holdout"
AUDIT_LOG_PATH = Path("evoalign_os/outputs/blind_access_audit.jsonl")


class DataLeakageViolation(PermissionError):
    """Raised when training/evolution code attempts to touch blind holdout data."""


class BlindAuditContext:
    """Explicit context manager required for one-shot final blind evaluation."""

    def __init__(self, reason: str, caller: str = "final_evaluator"):
        if not reason or len(reason.strip()) < 4:
            raise ValueError("BlindAuditContext requires a descriptive audit reason.")
        self.reason = reason.strip()
        self.caller = caller
        self._token = None

    def __enter__(self):
        self._token = _BLIND_AUDIT_REASON.set(f"{self.caller}:{self.reason}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _BLIND_AUDIT_REASON.reset(self._token)
        return False


def is_blind_path(path: str | os.PathLike) -> bool:
    norm = os.path.normpath(os.path.abspath(str(path))).replace("\\", "/")
    parts = [p for p in norm.split("/") if p]
    return BLIND_DIR_NAME in parts


def guard_read_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    if is_blind_path(p):
        reason = _BLIND_AUDIT_REASON.get()
        if not reason:
            raise DataLeakageViolation(
                f"Blocked unauthorized read of blind holdout path '{p}'. "
                "Training/development/RSI evolution must never inspect blind_holdout/."
            )
        AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        sha = None
        if p.exists() and p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha = h.hexdigest()
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "path": str(p),
            "sha256": sha,
        }
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return p
