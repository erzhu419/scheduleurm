"""Compatibility facade for scheduler_staging.code_tar."""

if __package__:
    from .scheduler_staging.code_tar import *  # noqa: F401,F403
else:
    from scheduler_staging.code_tar import *  # noqa: F401,F403
