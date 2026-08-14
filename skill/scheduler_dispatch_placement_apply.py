"""Compatibility facade for dispatch placement helpers."""

if __package__:
    from .scheduler_dispatch.placement_apply import *  # noqa: F401,F403
else:
    from scheduler_dispatch.placement_apply import *  # noqa: F401,F403
