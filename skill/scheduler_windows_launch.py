"""Compatibility facade for Windows launch helpers."""

if __package__:
    from .scheduler_windows.launch import *  # noqa: F401,F403
else:
    from scheduler_windows.launch import *  # noqa: F401,F403
