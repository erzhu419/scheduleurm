"""Compatibility facade for scheduler CPU batch submit specs."""

if __package__:
    from .scheduler_cpu.batch_submit_spec import *  # noqa: F401,F403
else:
    from scheduler_cpu.batch_submit_spec import *  # noqa: F401,F403
