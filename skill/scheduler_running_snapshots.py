"""Compatibility facade for scheduler_running.snapshots."""

try:
    from scheduler_running.snapshots import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_running.snapshots import *  # noqa: F401,F403
