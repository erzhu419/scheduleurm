"""Compatibility facade for submit shell inspection helpers."""

if __package__:
    from .scheduler_submit.shell import *  # noqa: F401,F403
else:
    from scheduler_submit.shell import *  # noqa: F401,F403
