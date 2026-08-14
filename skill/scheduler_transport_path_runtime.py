"""Compatibility facade for transport path runtime helpers."""

if __package__:
    from .scheduler_transport.path_runtime import *  # noqa: F401,F403
else:
    from scheduler_transport.path_runtime import *  # noqa: F401,F403
