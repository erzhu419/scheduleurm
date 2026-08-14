"""Compatibility facade for external/control runtime wiring helpers."""

if __package__:
    from .scheduler_runtime.wiring_external_control import *  # noqa: F401,F403
else:
    from scheduler_runtime.wiring_external_control import *  # noqa: F401,F403
