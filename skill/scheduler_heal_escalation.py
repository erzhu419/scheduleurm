"""Compatibility facade for scheduler heal escalation helpers."""

if __package__:
    from .scheduler_diagnostics.heal_escalation import *  # noqa: F401,F403
else:
    from scheduler_diagnostics.heal_escalation import *  # noqa: F401,F403
