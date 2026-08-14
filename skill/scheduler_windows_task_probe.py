"""Compatibility facade for Windows task probe helpers."""

if __package__:
    from .scheduler_windows.task_probe import *  # noqa: F401,F403
else:
    from scheduler_windows.task_probe import *  # noqa: F401,F403
