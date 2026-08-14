"""Compatibility facade for local probe helpers."""

if __package__:
    from .scheduler_backend.local_probe import *  # noqa: F401,F403
else:
    from scheduler_backend.local_probe import *  # noqa: F401,F403
