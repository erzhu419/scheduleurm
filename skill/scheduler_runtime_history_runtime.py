"""Compatibility facade for scheduler runtime history runtime helpers."""

if __package__:
    from .scheduler_runtime.history_runtime import *  # noqa: F401,F403
else:
    from scheduler_runtime.history_runtime import *  # noqa: F401,F403
