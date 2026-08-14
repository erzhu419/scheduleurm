"""Compatibility facade for scheduler resource accounting helpers."""

if __package__:
    from .scheduler_resource.accounting import *  # noqa: F401,F403
else:
    from scheduler_resource.accounting import *  # noqa: F401,F403
