"""Compatibility facade for scheduler_placement_engine.post_dispatch_thresholds."""

if __package__:
    from .scheduler_placement_engine.post_dispatch_thresholds import *  # noqa: F401,F403
else:
    from scheduler_placement_engine.post_dispatch_thresholds import *  # noqa: F401,F403
