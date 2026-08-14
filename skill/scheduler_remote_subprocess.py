"""Compatibility facade for remote subprocess helpers."""

if __package__:
    from .scheduler_remote.subprocess import *  # noqa: F401,F403
else:
    from scheduler_remote.subprocess import *  # noqa: F401,F403
