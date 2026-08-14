"""Compatibility facade for scheduler CPU planning helpers."""

if __package__:
    from .scheduler_cpu.plan import *  # noqa: F401,F403
else:
    from scheduler_cpu.plan import *  # noqa: F401,F403
