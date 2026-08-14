"""Compatibility facade for scheduler_failure.analysis."""

try:
    from scheduler_failure.analysis import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.analysis import *  # noqa: F401,F403
