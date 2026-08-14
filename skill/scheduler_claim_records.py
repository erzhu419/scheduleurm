"""Compatibility facade for scheduler claim record helpers."""

if __package__:
    from .scheduler_claim.records import *  # noqa: F401,F403
else:
    from scheduler_claim.records import *  # noqa: F401,F403
