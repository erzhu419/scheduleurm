"""Compatibility facade for scheduler_failure.crash_runtime."""

try:
    from scheduler_failure.crash_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.crash_runtime import *  # noqa: F401,F403
