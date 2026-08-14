"""Compatibility facade for launch CWD staging helpers."""

try:
    from scheduler_staging.code_tar import CodeTarStagingDeps
    from scheduler_windows.tar_staging import WindowsTarStagingDeps
    from scheduler_staging.launch_cwd import (
        DEFAULT_RSYNC_EXCLUDES,
        LAUNCH_CWD_STAGING_AUDIT_TOKENS,
        LaunchCwdStagingDeps,
        launch_input_stage_key,
        launch_input_stage_state,
        stage_input_paths_for_launch,
        stage_code_tar_for_launch,
        stage_cwd_for_launch,
        stage_local_dir_to_windows,
        subprocess,
        tar_exclude_args,
    )
except ModuleNotFoundError:
    from .scheduler_staging.code_tar import CodeTarStagingDeps
    from .scheduler_windows_tar_staging import WindowsTarStagingDeps
    from .scheduler_staging.launch_cwd import (
        DEFAULT_RSYNC_EXCLUDES,
        LAUNCH_CWD_STAGING_AUDIT_TOKENS,
        LaunchCwdStagingDeps,
        launch_input_stage_key,
        launch_input_stage_state,
        stage_input_paths_for_launch,
        stage_code_tar_for_launch,
        stage_cwd_for_launch,
        stage_local_dir_to_windows,
        subprocess,
        tar_exclude_args,
    )

__all__ = [
    "CodeTarStagingDeps",
    "DEFAULT_RSYNC_EXCLUDES",
    "LAUNCH_CWD_STAGING_AUDIT_TOKENS",
    "LaunchCwdStagingDeps",
    "WindowsTarStagingDeps",
    "launch_input_stage_key",
    "launch_input_stage_state",
    "stage_code_tar_for_launch",
    "stage_cwd_for_launch",
    "stage_input_paths_for_launch",
    "stage_local_dir_to_windows",
    "subprocess",
    "tar_exclude_args",
]
