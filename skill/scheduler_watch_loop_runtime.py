"""Compatibility facade for watcher loop runtime helpers."""

if __package__:
    from .scheduler_watch.loop_runtime import *  # noqa: F401,F403
else:
    from scheduler_watch.loop_runtime import *  # noqa: F401,F403
