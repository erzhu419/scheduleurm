"""Compatibility facade for scheduler CPU runtime wrappers."""

if __package__:
    from .scheduler_cpu.runtime import *  # noqa: F401,F403
else:
    from scheduler_cpu.runtime import *  # noqa: F401,F403
