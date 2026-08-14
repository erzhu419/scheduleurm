"""Migration staging package."""

from .checkpoint import CheckpointMigrationDeps, checkpoint_migration_extra_allowed_nodes
from .candidate_staging import stage_migration_candidates_outside_lock
from .ckpt_staging import stage_checkpoint_if_needed
from .cwd_staging import stage_cwd
from .dispatch_resume_placement import (
    DispatchResumePlacementDeps,
    ResumePlacementResult,
    resolve_resume_aware_placement,
    validate_selected_resume_checkpoint,
)
from .env_staging import verify_python_env
from .policy import (
    MigrationPolicyDeps,
    compute_node_load_seconds,
    consider_migration,
    identify_migration_candidates,
)
from .resume_fit import (
    ResumeFitDeps,
    estimate_gpu_fit_wait_seconds,
    estimate_node_fit_wait_seconds,
    record_staged_resume_location,
    resume_checkpoint_stage_check,
    resume_ckpt_stage_estimate_seconds,
    resume_ckpt_stage_key,
    resume_location_for_target,
    resume_stage_source_for_target,
    running_tasks_for_node,
    task_vram_pressure_mb,
    task_wait_eta_seconds,
)
from .resume_scan import (
    ResumeScanDeps,
    allow_initial_resume_scan_error,
    find_resume_on_node,
    refresh_resume_locations_for_task,
    resume_candidate_is_safe,
    resume_node_names,
    resume_scan_key,
    scan_resume_locations,
    task_requires_resume_scan,
)
from .staging import stage_for_migration
from .types import MigrationCandidateStagingDeps, MigrationStagingDeps

__all__ = [
    "CheckpointMigrationDeps",
    "DispatchResumePlacementDeps",
    "MigrationCandidateStagingDeps",
    "MigrationPolicyDeps",
    "MigrationStagingDeps",
    "ResumeFitDeps",
    "ResumePlacementResult",
    "ResumeScanDeps",
    "allow_initial_resume_scan_error",
    "checkpoint_migration_extra_allowed_nodes",
    "compute_node_load_seconds",
    "consider_migration",
    "estimate_gpu_fit_wait_seconds",
    "estimate_node_fit_wait_seconds",
    "find_resume_on_node",
    "identify_migration_candidates",
    "record_staged_resume_location",
    "refresh_resume_locations_for_task",
    "resolve_resume_aware_placement",
    "resume_candidate_is_safe",
    "resume_checkpoint_stage_check",
    "resume_ckpt_stage_estimate_seconds",
    "resume_ckpt_stage_key",
    "resume_location_for_target",
    "resume_node_names",
    "resume_scan_key",
    "resume_stage_source_for_target",
    "running_tasks_for_node",
    "scan_resume_locations",
    "stage_checkpoint_if_needed",
    "stage_cwd",
    "stage_for_migration",
    "stage_migration_candidates_outside_lock",
    "task_requires_resume_scan",
    "task_vram_pressure_mb",
    "task_wait_eta_seconds",
    "validate_selected_resume_checkpoint",
    "verify_python_env",
]
