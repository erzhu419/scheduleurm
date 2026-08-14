"""Compatibility facade for scheduler archive helpers."""

if __package__:
    from .scheduler_state.archive import *  # noqa: F401,F403
else:
    from scheduler_state.archive import *  # noqa: F401,F403
