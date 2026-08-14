"""Compatibility facade for scheduler task lineage helpers."""

if __package__:
    from .scheduler_task.lineage import *  # noqa: F401,F403
else:
    from scheduler_task.lineage import *  # noqa: F401,F403
