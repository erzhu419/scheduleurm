"""Compatibility facade for docker launch wrapping helpers."""

if __package__:
    from .scheduler_environment.docker_wrap import *  # noqa: F401,F403
else:
    from scheduler_environment.docker_wrap import *  # noqa: F401,F403
