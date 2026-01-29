"""
Base utilities for tau-bench tools adapted to agent_framework.

Provides:
- Active database management
- Tool call logging (coroutine-isolated via ContextVar)
- Common utilities
"""

from contextvars import ContextVar
from typing import Dict, Any, List, Optional
from src.multiagent.database.store import RetailDatabase


# =============================================================================
# ACTIVE DATABASE MANAGEMENT
# =============================================================================

# ContextVar for coroutine-isolated database (each async workflow gets its own)
_active_database: ContextVar[Optional[RetailDatabase]] = ContextVar(
    'active_database', default=None
)


def set_active_database(db: RetailDatabase) -> None:
    """Set the active database for the current coroutine."""
    _active_database.set(db)


def get_active_database() -> RetailDatabase:
    """Get the active database for the current coroutine."""
    db = _active_database.get()
    if db is None:
        raise RuntimeError("No active database. Call set_active_database() first.")
    return db


def clear_active_database() -> None:
    """Clear the active database reference for the current coroutine."""
    _active_database.set(None)


# =============================================================================
# TOOL CALL LOGGING (ContextVar for coroutine isolation)
# =============================================================================

# ContextVar ensures each concurrent workflow has its own isolated log
# This prevents race conditions when running with MAX_CONCURRENT > 1
_tool_call_log: ContextVar[List[Dict[str, Any]]] = ContextVar('tool_call_log')


def log_tool_call(name: str, args: Dict[str, Any], result: Any) -> None:
    """Log a tool call (isolated per coroutine)."""
    try:
        log = _tool_call_log.get()
    except LookupError:
        # First call in this coroutine - initialize empty list
        log = []
        _tool_call_log.set(log)

    log.append({
        "tool": name,
        "arguments": args,
        "result": str(result)[:500] if result else None
    })


def reset_tool_log() -> None:
    """Reset the tool call log for the current coroutine."""
    _tool_call_log.set([])  # New list for this coroutine


def get_tool_log() -> List[Dict[str, Any]]:
    """Get a copy of the tool call log for the current coroutine."""
    try:
        return _tool_call_log.get().copy()
    except LookupError:
        return []  # No log initialized yet
