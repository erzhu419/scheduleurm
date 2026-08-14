"""Compatibility facade for scheduler_failure.terminal_diagnosis."""

try:
    from scheduler_failure.terminal_diagnosis import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.terminal_diagnosis import *  # noqa: F401,F403
