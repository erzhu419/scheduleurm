"""Compatibility facade for scheduler_placement_engine.preemption."""

if __package__:
    from .scheduler_placement_engine.preemption import *  # noqa: F401,F403
else:
    from scheduler_placement_engine.preemption import *  # noqa: F401,F403
