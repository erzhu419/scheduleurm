"""Compatibility facade for scheduler_failure.final_model_success."""

try:
    from scheduler_failure.final_model_success import *  # noqa: F401,F403
except ModuleNotFoundError:
    from .scheduler_failure.final_model_success import *  # noqa: F401,F403
