"""Compatibility facade for scheduler task id helpers."""

if __package__:
    from .scheduler_task.ids import *  # noqa: F401,F403
else:
    from scheduler_task.ids import *  # noqa: F401,F403
