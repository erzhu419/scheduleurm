"""Compatibility facade for scheduler resource history runtime wrappers."""

if __package__:
    from .scheduler_resource.history_runtime import *  # noqa: F401,F403
else:
    from scheduler_resource.history_runtime import *  # noqa: F401,F403
