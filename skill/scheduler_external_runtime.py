"""Compatibility facade for scheduler_external.runtime."""

if __package__:
    from .scheduler_external.runtime import *  # noqa: F401,F403
else:
    from scheduler_external.runtime import *  # noqa: F401,F403
