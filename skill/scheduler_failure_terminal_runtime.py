"""Compatibility facade for scheduler_failure.terminal_runtime."""

try:
    from scheduler_failure.terminal_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.terminal_runtime import *  # noqa: F401,F403
