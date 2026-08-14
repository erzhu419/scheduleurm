"""Compatibility facade for submit task construction helpers."""

if __package__:
    from .scheduler_submit.task import *  # noqa: F401,F403
else:
    from scheduler_submit.task import *  # noqa: F401,F403
