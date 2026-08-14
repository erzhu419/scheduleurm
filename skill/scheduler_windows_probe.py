"""Compatibility facade for Windows probe helpers."""

if __package__:
    from .scheduler_windows.probe import *  # noqa: F401,F403
else:
    from scheduler_windows.probe import *  # noqa: F401,F403
