"""Compatibility facade for checkpoint submit helpers."""

if __package__:
    from .scheduler_submit.checkpoint import *  # noqa: F401,F403
else:
    from scheduler_submit.checkpoint import *  # noqa: F401,F403
