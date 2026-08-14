"""Compatibility facade for scheduler sudo CPU probe helpers."""

if __package__:
    from .scheduler_probe.sudo_cpu import *  # noqa: F401,F403
else:
    from scheduler_probe.sudo_cpu import *  # noqa: F401,F403
