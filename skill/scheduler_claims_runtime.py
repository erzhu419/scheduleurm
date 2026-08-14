"""Compatibility facade for scheduler claim runtime wrappers."""

if __package__:
    from .scheduler_claim.wrappers import *  # noqa: F401,F403
else:
    from scheduler_claim.wrappers import *  # noqa: F401,F403
