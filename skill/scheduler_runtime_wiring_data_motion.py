"""Compatibility facade for data-motion runtime wiring helpers."""

if __package__:
    from .scheduler_runtime.wiring_data_motion import *  # noqa: F401,F403
else:
    from scheduler_runtime.wiring_data_motion import *  # noqa: F401,F403
