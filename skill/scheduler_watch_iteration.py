"""Compatibility facade for watcher iteration helpers."""

if __package__:
    from .scheduler_watch.iteration import *  # noqa: F401,F403
else:
    from scheduler_watch.iteration import *  # noqa: F401,F403
