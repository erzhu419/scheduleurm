"""Compatibility facade for scheduler_failure.crash_requeue."""

try:
    from scheduler_failure.crash_requeue import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.crash_requeue import *  # noqa: F401,F403
