"""Compatibility facade for scheduler_migration.policy."""

try:
    from scheduler_migration.policy import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.policy import *  # noqa: F401,F403
