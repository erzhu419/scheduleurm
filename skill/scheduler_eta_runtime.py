"""Compatibility facade for scheduler_eta.runtime."""

try:
    from scheduler_eta.runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_eta.runtime import *  # noqa: F401,F403
