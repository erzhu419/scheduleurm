"""Compatibility facade for scheduler_eta.state."""

try:
    from scheduler_eta.state import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_eta.state import *  # noqa: F401,F403
