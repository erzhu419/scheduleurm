"""Compatibility facade for submit command helpers."""

if __package__:
    from .scheduler_submit.command import *  # noqa: F401,F403
else:
    from scheduler_submit.command import *  # noqa: F401,F403
