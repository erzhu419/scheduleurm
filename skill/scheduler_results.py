"""Compatibility facade for scheduler result discovery helpers."""

if __package__:
    from .scheduler_result.discovery import *  # noqa: F401,F403
else:
    from scheduler_result.discovery import *  # noqa: F401,F403
