"""Shared path setup helpers for test modules."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT_STR = str(_REPO_ROOT)


def ensure_repo_root_on_path() -> None:
    """Ensure repository root is importable for tests that import from python/."""
    if _REPO_ROOT_STR not in sys.path:
        sys.path.insert(0, _REPO_ROOT_STR)
