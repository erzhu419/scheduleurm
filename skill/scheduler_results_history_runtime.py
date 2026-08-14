"""Compatibility facade for scheduler result/history runtime wrappers."""

if __package__:
    from .scheduler_result.history_runtime import *  # noqa: F401,F403
else:
    from scheduler_result.history_runtime import *  # noqa: F401,F403
