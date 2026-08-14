"""Compatibility facade for scheduler task identity helpers."""

if __package__:
    from .scheduler_task.identity import *  # noqa: F401,F403
else:
    from scheduler_task.identity import *  # noqa: F401,F403
