"""Compatibility facade for watcher state helpers."""

if __package__:
    from .scheduler_watch.watcher_state import *  # noqa: F401,F403
else:
    from scheduler_watch.watcher_state import *  # noqa: F401,F403
