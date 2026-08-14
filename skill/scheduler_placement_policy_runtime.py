"""Compatibility facade for scheduler_placement_engine.policy_runtime."""

if __package__:
    from .scheduler_placement_engine.policy_runtime import *  # noqa: F401,F403
else:
    from scheduler_placement_engine.policy_runtime import *  # noqa: F401,F403
