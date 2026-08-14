"""Compatibility facade for scheduler claim remote script payload."""

if __package__:
    from .scheduler_claim.remote_script import *  # noqa: F401,F403
else:
    from scheduler_claim.remote_script import *  # noqa: F401,F403
