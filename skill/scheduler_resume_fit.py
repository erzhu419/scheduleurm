"""Compatibility facade for scheduler_migration.resume_fit."""

try:
    from scheduler_migration.resume_fit import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.resume_fit import *  # noqa: F401,F403
