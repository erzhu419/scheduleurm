"""Compatibility facade for scheduler resource estimate helpers."""

if __package__:
    from .scheduler_resource.estimates import *  # noqa: F401,F403
else:
    from scheduler_resource.estimates import *  # noqa: F401,F403
