"""Compatibility facade for dispatch intent helpers."""

if __package__:
    from .scheduler_dispatch.intent import *  # noqa: F401,F403
else:
    from scheduler_dispatch.intent import *  # noqa: F401,F403
