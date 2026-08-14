"""Compatibility facade for scheduler node reporting helpers."""

if __package__:
    from .scheduler_node.reporting import *  # noqa: F401,F403
else:
    from scheduler_node.reporting import *  # noqa: F401,F403
