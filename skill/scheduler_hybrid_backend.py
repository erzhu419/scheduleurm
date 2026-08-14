"""Compatibility facade for hybrid backend helpers."""

if __package__:
    from .scheduler_backend.hybrid import *  # noqa: F401,F403
else:
    from scheduler_backend.hybrid import *  # noqa: F401,F403
