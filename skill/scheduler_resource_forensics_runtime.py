"""Compatibility facade for scheduler resource forensics runtime wrappers."""

if __package__:
    from .scheduler_resource.forensics_runtime import *  # noqa: F401,F403
else:
    from scheduler_resource.forensics_runtime import *  # noqa: F401,F403
