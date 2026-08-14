"""Compatibility facade for scheduler lock helpers."""

if __package__:
    from .scheduler_state.locks import *  # noqa: F401,F403
else:
    from scheduler_state.locks import *  # noqa: F401,F403
