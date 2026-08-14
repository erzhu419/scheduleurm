"""Compatibility facade for scheduler probe-all helpers."""

if __package__:
    from .scheduler_probe.all import *  # noqa: F401,F403
else:
    from scheduler_probe.all import *  # noqa: F401,F403
