"""Compatibility facade for scheduler runtime wiring helpers."""

if __package__:
    from .scheduler_runtime.wiring import *  # noqa: F401,F403
else:
    from scheduler_runtime.wiring import *  # noqa: F401,F403
