"""Compatibility facade for scheduler command dependency wiring."""

if __package__:
    from .scheduler_commands.wiring import *  # noqa: F401,F403
else:
    from scheduler_commands.wiring import *  # noqa: F401,F403
