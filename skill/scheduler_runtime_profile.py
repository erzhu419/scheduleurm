"""Compatibility facade for scheduler runtime profile helpers."""

if __package__:
    from .scheduler_runtime.profile import *  # noqa: F401,F403
else:
    from scheduler_runtime.profile import *  # noqa: F401,F403
