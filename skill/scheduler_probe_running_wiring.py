"""Compatibility facade for scheduler probe/running dependency wiring."""

if __package__:
    from .scheduler_probe.running_wiring import *  # noqa: F401,F403
else:
    from scheduler_probe.running_wiring import *  # noqa: F401,F403
