"""Compatibility facade for dispatch accounting helpers."""

if __package__:
    from .scheduler_dispatch.core import *  # noqa: F401,F403
else:
    from scheduler_dispatch.core import *  # noqa: F401,F403
