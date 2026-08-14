"""Compatibility facade for scheduler local profiling helpers."""

if __package__:
    from .scheduler_probe.profile_local import *  # noqa: F401,F403
else:
    from scheduler_probe.profile_local import *  # noqa: F401,F403
