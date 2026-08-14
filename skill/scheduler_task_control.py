"""Compatibility facade for scheduler task control helpers."""

if __package__:
    from .scheduler_task.control import *  # noqa: F401,F403
else:
    from scheduler_task.control import *  # noqa: F401,F403
