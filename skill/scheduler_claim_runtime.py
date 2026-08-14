"""Compatibility facade for scheduler claim runtime types."""

if __package__:
    from .scheduler_claim.runtime import *  # noqa: F401,F403
else:
    from scheduler_claim.runtime import *  # noqa: F401,F403
