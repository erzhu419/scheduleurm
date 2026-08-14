"""Compatibility facade for scheduler_failure.runtime."""

try:
    from scheduler_failure.runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.runtime import *  # noqa: F401,F403
