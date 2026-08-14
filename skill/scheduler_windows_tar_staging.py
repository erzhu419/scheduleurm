"""Compatibility facade for Windows tar staging helpers."""

if __package__:
    from .scheduler_windows.tar_staging import *  # noqa: F401,F403
else:
    from scheduler_windows.tar_staging import *  # noqa: F401,F403
