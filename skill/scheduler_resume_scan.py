"""Compatibility facade for scheduler_migration.resume_scan."""

try:
    from scheduler_migration.resume_scan import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.resume_scan import *  # noqa: F401,F403
