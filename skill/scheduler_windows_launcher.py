"""Compatibility facade for embedded Windows launcher scripts."""

if __package__:
    from .scheduler_windows.launcher import *  # noqa: F401,F403
else:
    from scheduler_windows.launcher import *  # noqa: F401,F403
