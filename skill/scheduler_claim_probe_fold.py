"""Compatibility facade for scheduler claim probe folding helpers."""

if __package__:
    from .scheduler_claim.probe_fold import *  # noqa: F401,F403
else:
    from scheduler_claim.probe_fold import *  # noqa: F401,F403
