"""Compatibility facade for scheduler batch completion helpers."""

if __package__:
    from .scheduler_result.batch_completion import *  # noqa: F401,F403
else:
    from scheduler_result.batch_completion import *  # noqa: F401,F403
