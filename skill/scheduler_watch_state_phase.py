"""Compatibility facade for watcher state phase helpers."""

if __package__:
    from .scheduler_watch.state_phase import *  # noqa: F401,F403
else:
    from scheduler_watch.state_phase import *  # noqa: F401,F403
