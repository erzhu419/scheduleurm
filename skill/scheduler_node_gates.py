"""Compatibility facade for scheduler node gate helpers."""

if __package__:
    from .scheduler_node.gates import *  # noqa: F401,F403
else:
    from scheduler_node.gates import *  # noqa: F401,F403
