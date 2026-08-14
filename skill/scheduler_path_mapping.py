"""Compatibility facade for scheduler transport path mapping helpers."""

if __package__:
    from .scheduler_transport.path_mapping import *  # noqa: F401,F403
else:
    from scheduler_transport.path_mapping import *  # noqa: F401,F403
