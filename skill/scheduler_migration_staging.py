"""Compatibility facade for scheduler_migration.staging."""

try:
    from scheduler_migration.staging import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.staging import *  # noqa: F401,F403
