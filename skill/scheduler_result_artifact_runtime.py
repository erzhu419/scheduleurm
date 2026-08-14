"""Compatibility facade for scheduler result artifact runtime wrappers."""

if __package__:
    from .scheduler_result.artifact_runtime import *  # noqa: F401,F403
else:
    from scheduler_result.artifact_runtime import *  # noqa: F401,F403
