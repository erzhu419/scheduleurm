"""Compatibility facade for scheduler_notification.notify."""

if __package__:
    from .scheduler_notification.notify import *  # noqa: F401,F403
else:
    from scheduler_notification.notify import *  # noqa: F401,F403
