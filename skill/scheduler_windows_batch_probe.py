"""Compatibility facade for Windows batch probe helpers."""

if __package__:
    from .scheduler_windows.batch_probe import *  # noqa: F401,F403
else:
    from scheduler_windows.batch_probe import *  # noqa: F401,F403
