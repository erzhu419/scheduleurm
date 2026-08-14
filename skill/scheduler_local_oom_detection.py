"""Compatibility facade for scheduler_failure.local_oom_detection."""

try:
    from scheduler_failure.local_oom_detection import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.local_oom_detection import *  # noqa: F401,F403
