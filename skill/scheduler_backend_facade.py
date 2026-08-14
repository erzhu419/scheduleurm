"""Compatibility facade for backend facade helpers."""

if __package__:
    from .scheduler_backend.facade import *  # noqa: F401,F403
else:
    from scheduler_backend.facade import *  # noqa: F401,F403
