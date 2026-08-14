"""Compatibility facade for scheduler_running.probe_state_runtime."""

try:
    from scheduler_running.probe_state_runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_running.probe_state_runtime import *  # noqa: F401,F403
