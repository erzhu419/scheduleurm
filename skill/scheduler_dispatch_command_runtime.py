"""Compatibility facade for dispatch command runtime helpers."""

if __package__:
    from .scheduler_dispatch.command_runtime import *  # noqa: F401,F403
else:
    from scheduler_dispatch.command_runtime import *  # noqa: F401,F403
