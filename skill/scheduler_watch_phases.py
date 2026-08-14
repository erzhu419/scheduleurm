"""Compatibility facade for watcher phase recorder helpers."""

if __package__:
    from .scheduler_watch.phases import *  # noqa: F401,F403
else:
    from scheduler_watch.phases import *  # noqa: F401,F403
