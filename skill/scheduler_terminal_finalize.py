"""Compatibility facade for scheduler_failure.terminal_finalize."""

try:
    from scheduler_failure.terminal_finalize import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.terminal_finalize import *  # noqa: F401,F403
