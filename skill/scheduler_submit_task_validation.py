"""Compatibility facade for submit task validation helpers."""

if __package__:
    from .scheduler_submit.task_validation import *  # noqa: F401,F403
else:
    from scheduler_submit.task_validation import *  # noqa: F401,F403
