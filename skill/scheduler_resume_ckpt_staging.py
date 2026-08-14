"""Compatibility facade for resume checkpoint staging helpers."""

try:
    from scheduler_staging.resume_ckpt import (
        ResumeCkptStagingDeps,
        stage_resume_ckpt_for_launch,
        subprocess,
    )
except ModuleNotFoundError:
    from .scheduler_staging.resume_ckpt import (
        ResumeCkptStagingDeps,
        stage_resume_ckpt_for_launch,
        subprocess,
    )

__all__ = [
    "ResumeCkptStagingDeps",
    "stage_resume_ckpt_for_launch",
    "subprocess",
]
