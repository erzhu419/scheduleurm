"""Compatibility facade for scheduler task log helpers."""

if __package__:
    from .scheduler_task.log import *  # noqa: F401,F403
else:
    from scheduler_task.log import *  # noqa: F401,F403
