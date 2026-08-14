"""Compatibility facade for scheduler runtime history recording helpers."""

if __package__:
    from .scheduler_runtime.history_record import *  # noqa: F401,F403
else:
    from scheduler_runtime.history_record import *  # noqa: F401,F403
