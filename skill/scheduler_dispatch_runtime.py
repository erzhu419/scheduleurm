"""Compatibility facade for dispatch runtime helpers."""

if __package__:
    from .scheduler_dispatch.runtime import *  # noqa: F401,F403
else:
    from scheduler_dispatch.runtime import *  # noqa: F401,F403
