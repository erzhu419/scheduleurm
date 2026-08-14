"""Compatibility facade for bulk submit helpers."""

if __package__:
    from .scheduler_commands.bulk_submit import *  # noqa: F401,F403
else:
    from scheduler_commands.bulk_submit import *  # noqa: F401,F403
