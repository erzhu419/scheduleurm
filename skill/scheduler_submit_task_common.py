"""Compatibility facade for shared submit task helpers."""

if __package__:
    from .scheduler_submit.task_common import *  # noqa: F401,F403
else:
    from scheduler_submit.task_common import *  # noqa: F401,F403
