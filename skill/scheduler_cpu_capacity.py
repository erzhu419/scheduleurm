"""Compatibility facade for scheduler CPU capacity helpers."""

if __package__:
    from .scheduler_cpu.capacity import *  # noqa: F401,F403
else:
    from scheduler_cpu.capacity import *  # noqa: F401,F403
