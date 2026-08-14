"""Compatibility facade for scheduler_migration.dispatch_resume_placement."""

try:
    from scheduler_migration.dispatch_resume_placement import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_migration.dispatch_resume_placement import *  # noqa: F401,F403
