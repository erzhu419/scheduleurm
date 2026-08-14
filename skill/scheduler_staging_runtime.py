"""Compatibility facade for scheduler_staging.runtime."""

if __package__:
    from .scheduler_staging.runtime import *  # noqa: F401,F403
else:
    from scheduler_staging.runtime import *  # noqa: F401,F403
