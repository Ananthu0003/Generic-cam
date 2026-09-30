"""FastAPI dependency providers for VexCAM API.

Use ``get_session`` in route function signatures instead of calling helpers
directly.  This decouples session retrieval from state.py and makes it easy
to override in tests.

Example::

    @router.get("/setups")
    def get_setups(sess = Depends(get_session)):
        ...
"""

import time
from fastapi import Request

from .config import SESSION_TTL_SECONDS

# ---------------------------------------------------------------------------
# Module-level session store (shared across the process)
# ---------------------------------------------------------------------------

_sessions: dict = {}
_default_session = None  # lazily initialised to avoid circular imports


def _get_default_session():
    global _default_session
    if _default_session is None:
        from .state import SessionState
        _default_session = SessionState()
    return _default_session


def _evict_expired() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid
        for sid, (_, ts) in _sessions.items()
        if now - ts > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


def get_session(request: Request):
    """Return (or lazily create) the ``SessionState`` for this request.

    Session identity is determined by the ``X-Session-ID`` header.
    If the header is absent the process-level default session is returned.
    """
    from .state import SessionState  # lazy import to avoid circular dep

    _evict_expired()
    sid = request.headers.get("X-Session-ID", "")

    if not sid:
        return _get_default_session()

    if sid not in _sessions:
        _sessions[sid] = (SessionState(), time.time())

    state, _ = _sessions[sid]
    _sessions[sid] = (state, time.time())  # refresh TTL
    return state


# Alias kept for backwards compatibility
_get_session = get_session
