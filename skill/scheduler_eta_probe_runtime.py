"""Compatibility facade for scheduler_eta.probe_runtime."""

try:
    from scheduler_eta.probe_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_eta.probe_runtime import *  # noqa: F401,F403
