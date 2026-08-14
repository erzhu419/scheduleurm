"""Compatibility facade for local orphan recovery helpers."""

if __package__:
    from .scheduler_backend.local_orphan_recovery import *  # noqa: F401,F403
else:
    from scheduler_backend.local_orphan_recovery import *  # noqa: F401,F403
