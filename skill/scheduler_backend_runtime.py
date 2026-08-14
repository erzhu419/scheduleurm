"""Compatibility facade for backend runtime helpers."""

if __package__:
    from .scheduler_backend.runtime import *  # noqa: F401,F403
else:
    from scheduler_backend.runtime import *  # noqa: F401,F403
