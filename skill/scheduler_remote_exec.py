"""Compatibility facade for remote execution helpers."""

if __package__:
    from .scheduler_remote.exec import *  # noqa: F401,F403
else:
    from scheduler_remote.exec import *  # noqa: F401,F403
