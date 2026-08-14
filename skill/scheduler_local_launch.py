"""Compatibility facade for local launch helpers."""

if __package__:
    from .scheduler_backend.local_launch import *  # noqa: F401,F403
else:
    from scheduler_backend.local_launch import *  # noqa: F401,F403
