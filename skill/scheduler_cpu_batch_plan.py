"""Compatibility facade for scheduler CPU batch planning helpers."""

if __package__:
    from .scheduler_cpu.batch_plan import *  # noqa: F401,F403
else:
    from scheduler_cpu.batch_plan import *  # noqa: F401,F403
