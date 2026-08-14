"""Compatibility facade for scheduler bulk result conflict helpers."""

if __package__:
    from .scheduler_result.bulk_conflicts import *  # noqa: F401,F403
else:
    from scheduler_result.bulk_conflicts import *  # noqa: F401,F403
