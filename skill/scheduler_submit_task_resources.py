"""Compatibility facade for submit task resource helpers."""

if __package__:
    from .scheduler_submit.task_resources import *  # noqa: F401,F403
else:
    from scheduler_submit.task_resources import *  # noqa: F401,F403
