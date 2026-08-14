"""Compatibility facade for scheduler claim tending helpers."""

if __package__:
    from .scheduler_claim.tending import *  # noqa: F401,F403
else:
    from scheduler_claim.tending import *  # noqa: F401,F403
