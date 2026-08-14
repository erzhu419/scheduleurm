"""Compatibility facade for CPU batch submit helpers."""

if __package__:
    from .scheduler_submit.cpu_batch import *  # noqa: F401,F403
else:
    from scheduler_submit.cpu_batch import *  # noqa: F401,F403
