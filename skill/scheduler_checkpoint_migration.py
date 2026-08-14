"""Compatibility facade for scheduler_migration.checkpoint."""

try:
    from scheduler_migration.checkpoint import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.checkpoint import *  # noqa: F401,F403
