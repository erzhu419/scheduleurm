"""Compatibility facade for dispatch task gate helpers."""

if __package__:
    from .scheduler_dispatch.task_gates import *  # noqa: F401,F403
else:
    from scheduler_dispatch.task_gates import *  # noqa: F401,F403
