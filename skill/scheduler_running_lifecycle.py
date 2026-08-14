"""Compatibility facade for scheduler_running.lifecycle."""

try:
    from scheduler_running.lifecycle import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_running.lifecycle import *  # noqa: F401,F403
