"""Compatibility facade for scheduler state compaction helpers."""

if __package__:
    from .scheduler_state.compaction import *  # noqa: F401,F403
else:
    from scheduler_state.compaction import *  # noqa: F401,F403
