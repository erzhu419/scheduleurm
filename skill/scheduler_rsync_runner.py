"""Compatibility facade for scheduler_data_motion.rsync."""

if __package__:
    from .scheduler_data_motion.rsync import *  # noqa: F401,F403
else:
    from scheduler_data_motion.rsync import *  # noqa: F401,F403
