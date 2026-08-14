"""Compatibility facade for dispatch/watch dependency wiring helpers."""

if __package__:
    from .scheduler_dispatch.watch_wiring import *  # noqa: F401,F403
else:
    from scheduler_dispatch.watch_wiring import *  # noqa: F401,F403
