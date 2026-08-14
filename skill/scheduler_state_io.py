"""Compatibility facade for scheduler state IO helpers."""

if __package__:
    from .scheduler_state.io import *  # noqa: F401,F403
else:
    from scheduler_state.io import *  # noqa: F401,F403
