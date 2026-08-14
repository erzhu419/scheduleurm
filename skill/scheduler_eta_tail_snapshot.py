"""Compatibility facade for scheduler_eta.tail_snapshot."""

try:
    from scheduler_eta.tail_snapshot import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_eta.tail_snapshot import *  # noqa: F401,F403
