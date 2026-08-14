"""Compatibility facade for transport Windows runtime helpers."""

if __package__:
    from .scheduler_transport.windows_runtime import *  # noqa: F401,F403
else:
    from scheduler_transport.windows_runtime import *  # noqa: F401,F403
