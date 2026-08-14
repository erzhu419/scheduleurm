"""Compatibility facade for scheduler runtime identity helpers."""

if __package__:
    from .scheduler_runtime.identity import *  # noqa: F401,F403
else:
    from scheduler_runtime.identity import *  # noqa: F401,F403
