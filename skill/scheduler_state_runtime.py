"""Compatibility facade for scheduler state runtime helpers."""

if __package__:
    from .scheduler_state.runtime import *  # noqa: F401,F403
else:
    from scheduler_state.runtime import *  # noqa: F401,F403
