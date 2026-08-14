"""Compatibility facade for scheduler resource forensics helpers."""

if __package__:
    from .scheduler_resource.forensics import *  # noqa: F401,F403
else:
    from scheduler_resource.forensics import *  # noqa: F401,F403
