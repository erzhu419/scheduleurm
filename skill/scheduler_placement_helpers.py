"""Compatibility facade for scheduler_placement_engine.helpers."""

if __package__:
    from .scheduler_placement_engine.helpers import *  # noqa: F401,F403
else:
    from scheduler_placement_engine.helpers import *  # noqa: F401,F403
