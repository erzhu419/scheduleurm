"""Compatibility facade for scheduler CLI parsing."""

if __package__:
    from .scheduler_commands.cli import *  # noqa: F401,F403
else:
    from scheduler_commands.cli import *  # noqa: F401,F403
