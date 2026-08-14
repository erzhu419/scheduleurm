"""Compatibility facade for scheduler node inventory helpers."""

if __package__:
    from .scheduler_node.inventory import *  # noqa: F401,F403
else:
    from scheduler_node.inventory import *  # noqa: F401,F403
