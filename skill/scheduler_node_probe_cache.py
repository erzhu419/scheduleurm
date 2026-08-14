"""Compatibility facade for scheduler node probe cache helpers."""

if __package__:
    from .scheduler_probe.cache import *  # noqa: F401,F403
else:
    from scheduler_probe.cache import *  # noqa: F401,F403
