"""Compatibility facade for Windows backend helpers."""

if __package__:
    from .scheduler_windows.backend import *  # noqa: F401,F403
else:
    from scheduler_windows.backend import *  # noqa: F401,F403
