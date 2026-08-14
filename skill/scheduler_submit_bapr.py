"""Compatibility facade for BAPR submit helpers."""

if __package__:
    from .scheduler_submit.bapr import *  # noqa: F401,F403
else:
    from scheduler_submit.bapr import *  # noqa: F401,F403
