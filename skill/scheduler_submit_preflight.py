"""Compatibility facade for submit preflight helpers."""

if __package__:
    from .scheduler_submit.preflight import *  # noqa: F401,F403
else:
    from scheduler_submit.preflight import *  # noqa: F401,F403
