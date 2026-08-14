"""Compatibility facade for submit training policy helpers."""

if __package__:
    from .scheduler_submit.training_policy import *  # noqa: F401,F403
else:
    from scheduler_submit.training_policy import *  # noqa: F401,F403
