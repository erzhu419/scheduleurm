"""Compatibility facade for scheduler wait-for command helpers."""

if __package__:
    from .scheduler_commands.wait_for import *  # noqa: F401,F403
else:
    from scheduler_commands.wait_for import *  # noqa: F401,F403
