"""Compatibility facade for scheduler_running.runtime."""

try:
    from scheduler_running.runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_running.runtime import *  # noqa: F401,F403
