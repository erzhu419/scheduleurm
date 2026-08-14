"""Compatibility facade for watcher command helpers."""

if __package__:
    from .scheduler_watch.command import *  # noqa: F401,F403
else:
    from scheduler_watch.command import *  # noqa: F401,F403
