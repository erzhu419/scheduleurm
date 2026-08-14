"""Compatibility facade for watcher notification helpers."""

if __package__:
    from .scheduler_watch.notifications import *  # noqa: F401,F403
else:
    from scheduler_watch.notifications import *  # noqa: F401,F403
