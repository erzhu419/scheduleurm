"""Compatibility facade for aggregate data-motion runtime assembly."""

if __package__:
    from .scheduler_runtime.data_motion import *  # noqa: F401,F403
else:
    from scheduler_runtime.data_motion import *  # noqa: F401,F403
