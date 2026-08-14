"""Compatibility facade for scheduler_failure.terminal_snapshot."""

try:
    from scheduler_failure.terminal_snapshot import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.terminal_snapshot import *  # noqa: F401,F403
