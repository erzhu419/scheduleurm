"""Compatibility facade for watcher runtime helpers."""

if __package__:
    from .scheduler_watch.runtime import *  # noqa: F401,F403
else:
    from scheduler_watch.runtime import *  # noqa: F401,F403
