"""Compatibility facade for scheduler_migration.runtime."""

try:
    from scheduler_migration.runtime import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.runtime import *  # noqa: F401,F403
