"""Compatibility facade for scheduler claim input helpers."""

if __package__:
    from .scheduler_claim.inputs import *  # noqa: F401,F403
else:
    from scheduler_claim.inputs import *  # noqa: F401,F403
