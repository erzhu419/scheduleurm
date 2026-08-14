"""Compatibility facade for docker/conda preload helpers."""

if __package__:
    from .scheduler_environment.docker_preload import *  # noqa: F401,F403
else:
    from scheduler_environment.docker_preload import *  # noqa: F401,F403
