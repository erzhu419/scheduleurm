"""Compatibility facade for submit policy helpers.

The implementation is split by concern so the submit path can evolve without
turning this module back into a mixed policy/parser monolith.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from .bapr import (
        bapr_batch_projection,
        bapr_run_seed_metas_from_cmd,
        default_conflict_path_key,
        infer_bapr_result_dirs_from_cmd,
        infer_bapr_run_seed_ckpt,
        split_bapr_seed_batch_submit_args,
        submit_context_is_bapr,
    )
    from .checkpoint import (
        candidate_training_source_paths,
        checkpoint_contract_reason,
        cmd_has_resume_flag,
        infer_command_managed_resume,
        infer_checkpoint_from_submit,
        infer_direct_jax_train_ckpt,
        infer_wrapper_ckpt,
        resume_capability_reason,
    )
    from .shell import (
        abs_under,
        arg_value,
        expand_simple_seed_loop_inner,
        extract_script_cd_dir,
        safe_read_text,
        script_invocation_from_cmd,
        seed_value_substitute,
        shell_expand_simple,
        shell_split_statements,
        simple_shell_env_from_script,
        submit_policy_text,
    )
    from .training_policy import (
        cmd_explicitly_cpu,
        cmd_looks_like_training,
        cpu_training_policy_reason,
        task_looks_like_training,
    )
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .bapr import (
        bapr_batch_projection,
        bapr_run_seed_metas_from_cmd,
        default_conflict_path_key,
        infer_bapr_result_dirs_from_cmd,
        infer_bapr_run_seed_ckpt,
        split_bapr_seed_batch_submit_args,
        submit_context_is_bapr,
    )
    from .checkpoint import (
        candidate_training_source_paths,
        checkpoint_contract_reason,
        cmd_has_resume_flag,
        infer_command_managed_resume,
        infer_checkpoint_from_submit,
        infer_direct_jax_train_ckpt,
        infer_wrapper_ckpt,
        resume_capability_reason,
    )
    from .shell import (
        abs_under,
        arg_value,
        expand_simple_seed_loop_inner,
        extract_script_cd_dir,
        safe_read_text,
        script_invocation_from_cmd,
        seed_value_substitute,
        shell_expand_simple,
        shell_split_statements,
        simple_shell_env_from_script,
        submit_policy_text,
    )
    from .training_policy import (
        cmd_explicitly_cpu,
        cmd_looks_like_training,
        cpu_training_policy_reason,
        task_looks_like_training,
    )


__all__ = [
    "abs_under",
    "arg_value",
    "bapr_batch_projection",
    "bapr_run_seed_metas_from_cmd",
    "candidate_training_source_paths",
    "checkpoint_contract_reason",
    "cmd_explicitly_cpu",
    "cmd_has_resume_flag",
    "cmd_looks_like_training",
    "cpu_training_policy_reason",
    "default_conflict_path_key",
    "expand_simple_seed_loop_inner",
    "extract_script_cd_dir",
    "infer_bapr_result_dirs_from_cmd",
    "infer_bapr_run_seed_ckpt",
    "infer_command_managed_resume",
    "infer_checkpoint_from_submit",
    "infer_direct_jax_train_ckpt",
    "infer_wrapper_ckpt",
    "resume_capability_reason",
    "safe_read_text",
    "script_invocation_from_cmd",
    "seed_value_substitute",
    "shell_expand_simple",
    "shell_split_statements",
    "simple_shell_env_from_script",
    "split_bapr_seed_batch_submit_args",
    "submit_context_is_bapr",
    "submit_policy_text",
    "task_looks_like_training",
]
