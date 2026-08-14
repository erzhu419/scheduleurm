"""Compatibility facade for scheduler doctor runtime exports."""

if __package__:
    from .scheduler_diagnostics.doctor_runtime import *  # noqa: F401,F403
else:
    from scheduler_diagnostics.doctor_runtime import *  # noqa: F401,F403
