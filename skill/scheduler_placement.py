"""Compatibility facade for scheduler_placement_engine.core."""

if __package__:
    from .scheduler_placement_engine.core import *  # noqa: F401,F403
else:
    from scheduler_placement_engine.core import *  # noqa: F401,F403
