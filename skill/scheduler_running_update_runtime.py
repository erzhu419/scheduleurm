"""Compatibility facade for scheduler_running.update_runtime."""

try:
    from scheduler_running.update_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_running.update_runtime import *  # noqa: F401,F403
