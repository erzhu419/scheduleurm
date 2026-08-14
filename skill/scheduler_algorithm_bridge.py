"""Compatibility facade for scheduler_algorithm.bridge."""

if __package__:
    from .scheduler_algorithm.bridge import *  # noqa: F401,F403
else:
    from scheduler_algorithm.bridge import *  # noqa: F401,F403
