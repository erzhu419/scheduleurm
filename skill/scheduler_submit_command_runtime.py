"""Compatibility facade for submit command runtime helpers."""

if __package__:
    from .scheduler_submit.command_runtime import *  # noqa: F401,F403
else:
    from scheduler_submit.command_runtime import *  # noqa: F401,F403
