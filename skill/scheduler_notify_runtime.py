"""Compatibility facade for scheduler_notification.runtime."""

if __package__:
    from .scheduler_notification.runtime import *  # noqa: F401,F403
else:
    from scheduler_notification.runtime import *  # noqa: F401,F403
