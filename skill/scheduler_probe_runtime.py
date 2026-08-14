"""Compatibility facade for scheduler probe runtime wrappers."""

if __package__:
    from .scheduler_probe.runtime import *  # noqa: F401,F403
else:
    from scheduler_probe.runtime import *  # noqa: F401,F403
