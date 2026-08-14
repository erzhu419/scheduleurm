"""Compatibility facade for Windows log helpers."""

if __package__:
    from .scheduler_windows.logs import *  # noqa: F401,F403
else:
    from scheduler_windows.logs import *  # noqa: F401,F403
