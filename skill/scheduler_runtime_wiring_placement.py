"""Compatibility facade for placement runtime wiring helpers."""

if __package__:
    from .scheduler_runtime.wiring_placement import *  # noqa: F401,F403
else:
    from scheduler_runtime.wiring_placement import *  # noqa: F401,F403
