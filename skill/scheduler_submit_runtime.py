"""Compatibility facade for submit runtime helpers."""

if __package__:
    from .scheduler_submit.runtime import *  # noqa: F401,F403
else:
    from scheduler_submit.runtime import *  # noqa: F401,F403
