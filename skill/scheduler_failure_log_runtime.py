"""Compatibility facade for scheduler_failure.log_runtime."""

try:
    from scheduler_failure.log_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.log_runtime import *  # noqa: F401,F403
