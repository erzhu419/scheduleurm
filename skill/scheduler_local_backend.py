"""Compatibility facade for local backend helpers."""

if __package__:
    from .scheduler_backend.local_backend import *  # noqa: F401,F403
else:
    from scheduler_backend.local_backend import *  # noqa: F401,F403
