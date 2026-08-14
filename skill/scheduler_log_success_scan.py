"""Compatibility facade for scheduler_failure.log_success_scan."""

try:
    from scheduler_failure.log_success_scan import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.log_success_scan import *  # noqa: F401,F403
