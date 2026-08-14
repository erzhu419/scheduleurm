"""Compatibility facade for environment smoke-test helpers."""

if __package__:
    from .scheduler_environment.env_smoke import *  # noqa: F401,F403
else:
    from scheduler_environment.env_smoke import *  # noqa: F401,F403
