"""Compatibility facade for Windows host extras helpers."""

if __package__:
    from .scheduler_windows.host_extras import *  # noqa: F401,F403
else:
    from scheduler_windows.host_extras import *  # noqa: F401,F403
