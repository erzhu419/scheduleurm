"""Compatibility facade for submit policy helpers."""

if __package__:
    from .scheduler_submit.policy import *  # noqa: F401,F403
else:
    from scheduler_submit.policy import *  # noqa: F401,F403
