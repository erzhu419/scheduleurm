"""Compatibility facade for scheduler runtime bootstrap helpers."""

if __package__:
    from .scheduler_runtime.bootstrap import *  # noqa: F401,F403
else:
    from scheduler_runtime.bootstrap import *  # noqa: F401,F403
