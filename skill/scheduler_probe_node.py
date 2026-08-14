"""Compatibility facade for scheduler node probe helpers."""

if __package__:
    from .scheduler_probe.node import *  # noqa: F401,F403
else:
    from scheduler_probe.node import *  # noqa: F401,F403
