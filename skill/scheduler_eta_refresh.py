"""Compatibility facade for scheduler_eta.refresh."""

try:
    from scheduler_eta.refresh import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_eta.refresh import *  # noqa: F401,F403
