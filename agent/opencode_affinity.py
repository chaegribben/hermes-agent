"""OpenCode relay session-affinity header for the Rigel compatibility hotfix.

OpenCode Go/Zen require ``x-opencode-session`` to keep requests from one
Hermes conversation pinned to the same backend.  This July Hermes tree already
has a stable per-conversation ``AIAgent.session_id`` and a concurrency-safe
``HERMES_SESSION_ID`` ContextVar bridge for auxiliary calls, so the hotfix uses
those directly rather than backporting the newer affinity-scope machinery.
"""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

OPENCODE_SESSION_HEADER = "x-opencode-session"
_OPEN_CODE_PROVIDERS = frozenset({"opencode-go", "opencode-zen"})


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """Return True when the provider or URL targets the OpenCode relay."""
    if str(provider or "").strip().lower() in _OPEN_CODE_PROVIDERS:
        return True
    try:
        hostname = (urlparse(str(base_url or "")).hostname or "").lower().rstrip(".")
    except Exception:
        return False
    return hostname == "opencode.ai" or hostname.endswith(".opencode.ai")


def _ambient_session_id() -> str:
    """Read the active Hermes session without introducing process-global state."""
    try:
        from gateway.session_context import get_session_env
        return str(get_session_env("HERMES_SESSION_ID", "") or "").strip()
    except Exception:
        return ""


def opencode_session_headers(
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return the required OpenCode affinity header for one Hermes session."""
    if not is_opencode_target(provider, base_url):
        return {}
    key = str(session_id or "").strip() or _ambient_session_id()
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(
    kwargs: dict[str, Any],
    provider: Optional[str],
    base_url: Optional[str],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Merge OpenCode affinity into ``extra_headers`` without overriding callers."""
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs
