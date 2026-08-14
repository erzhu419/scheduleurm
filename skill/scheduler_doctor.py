"""Compatibility facade for scheduler doctor helpers."""

if __package__:
    from .scheduler_diagnostics.doctor import *  # noqa: F401,F403
else:
    from scheduler_diagnostics.doctor import *  # noqa: F401,F403
