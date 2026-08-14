"""Compatibility facade for scheduler diagnostics dependency wiring."""

if __package__:
    from .scheduler_diagnostics.wiring import *  # noqa: F401,F403
else:
    from scheduler_diagnostics.wiring import *  # noqa: F401,F403
