"""Compatibility facade for watcher transition helpers."""

if __package__:
    from .scheduler_watch.transitions import *  # noqa: F401,F403
else:
    from scheduler_watch.transitions import *  # noqa: F401,F403
