"""Compatibility facade for scheduler CPU submit runtime wrappers."""

if __package__:
    from .scheduler_cpu.submit_runtime import *  # noqa: F401,F403
else:
    from scheduler_cpu.submit_runtime import *  # noqa: F401,F403
