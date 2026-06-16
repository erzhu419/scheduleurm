"""Measured service certificates for theorem-facing dispatch.

The scheduler owns process launch and legacy safety rules.  This module only
binds a task/candidate pair to an existing measured service-cache row.  If a
task cannot be bound to an exact non-boundary profile, callers get an explicit
uncertified result instead of an interpolated service rate.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from simulation.defaults import build_default_cache
from simulation.service_cache import ProfileRecord, ServiceRateCache

from ..features import as_int


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_TOKENS = {
    "completed",
    "history",
    "profile",
    "profiles",
    "local",
    "bench",
    "project",
    "fabric",
    "cpu",
    "gpu",
    "hybrid",
    "v1",
    "c",
    "le",
    "p",
}


@dataclass(frozen=True)
class ServiceBinding:
    workload_key: str
    profile: int
    lower_service: float
    aggregate_rate: float
    mean_rate: float
    certified: bool
    reason: str
    source: str
    command_fingerprint: str = ""
    resource_kind: str = ""
    node_bucket: str = ""
    capacity_boundary: bool = False

    def lower_service_vector(self) -> dict[str, float]:
        if not self.certified or self.lower_service <= 0.0:
            return {}
        return {self.workload_key: float(self.lower_service)}

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload_key": self.workload_key,
            "profile": self.profile,
            "lower_service": self.lower_service,
            "aggregate_rate": self.aggregate_rate,
            "mean_rate": self.mean_rate,
            "certified": self.certified,
            "reason": self.reason,
            "source": self.source,
            "command_fingerprint": self.command_fingerprint,
            "resource_kind": self.resource_kind,
            "node_bucket": self.node_bucket,
            "capacity_boundary": self.capacity_boundary,
        }


@lru_cache(maxsize=1)
def default_service_cache() -> ServiceRateCache:
    return build_default_cache()


def infer_workload_key(
    task: Mapping[str, Any],
    cache: ServiceRateCache | None = None,
    *,
    admission_mode: str | None = None,
) -> str:
    """Infer the measured workload key for a scheduler task.

    The first block covers the controlled benchmark families used in the paper.
    The second block scores production-history keys by token overlap so a new
    production task can be admitted only when it resembles a measured family.
    """

    cache = cache or default_service_cache()
    explicit = _explicit_workload_key(task, cache)
    if explicit:
        return explicit

    fingerprint = str(task.get("command_fingerprint") or "").strip()
    if fingerprint:
        for key in cache.available_workloads():
            for record in cache.profiles(key, include_boundaries=True):
                if fingerprint and fingerprint == str(record.command_fingerprint or ""):
                    return key

    text = _task_text(task)
    lower = text.lower()
    identity_lower = _task_identity_text(task).lower()
    cpu_suffix = _cpu_bucket_suffix(task)
    if "freqduet" in lower:
        if "run_freqduet_snapshot_counterfactual_matrix.py" in lower:
            if cpu_suffix == "c9_16":
                return "freqduet_snapshot_counterfactual_c9_16_completed_history"
            return "freqduet_snapshot_counterfactual_c9_16_completed_history"
        if "run_freqduet_ablation.py" in lower:
            key = f"freqduet_cpu_ablation_{cpu_suffix}"
            if key.endswith("_c_le2"):
                key = "freqduet_cpu_ablation_c_le2_completed_history"
            elif key in {
                "freqduet_cpu_ablation_c3_8",
                "freqduet_cpu_ablation_c33_64",
                "freqduet_cpu_ablation_c65p",
            }:
                key = f"{key}_completed_history"
            return key
        if "runner_v3.py" in lower:
            if cpu_suffix == "c_le2":
                return "freqduet_runner_v3_c_le2_completed_history"
            if cpu_suffix == "c3_8":
                return "freqduet_runner_v3_c3_8_completed_history"
            if cpu_suffix == "c9_16":
                return "freqduet_runner_v3_c9_16_residual_completed_history"
            if cpu_suffix == "c17_32":
                return "freqduet_runner_v3_c17_32_completed_history"
            if cpu_suffix == "c33_64":
                return "freqduet_runner_v3_c33_64_completed_history"
    if "bamor" in lower and "mujoco" in lower:
        return f"bamor_mujoco_{cpu_suffix}_completed_history"
    if any(tok in lower for tok in ("jax", "matmul", "gpu_heavy", "gpu-bound", "gpu_bound")):
        return "gpu_heavy_jax_matmul"
    if any(tok in lower for tok in ("torch_cnn_progress_benchmark", "cnn_torch_gpu", "torch_cnn_progress_stack")):
        return "gpu_cnn_torch_progress_stack"
    if any(tok in lower for tok in ("torch_llm_progress_benchmark", "llm_torch_transformer", "torch_decoder_stack")):
        return "gpu_llm_torch_decoder_stack"
    if any(tok in lower for tok in ("resac_ant_node007", "node007_tqdm", "node007_eta_matrix_rl")):
        return "hybrid_rl_resac_ant_node007_tqdm"
    if any(tok in lower for tok in ("resac", "bapr", "mujoco", "ant-v", "ant_")):
        return "hybrid_rl_resac_ant"
    if any(tok in lower for tok in ("sleep_20ms", "light_control", "control-plane", "control_plane")):
        return "light_control_local"
    if (
        str(task.get("project") or "").strip().lower() == "scheduleurm"
        or "scheduleurm/auto-adopted" in identity_lower
        or "auto-adopted: scheduleurm" in identity_lower
    ):
        return "scheduleurm_control_plane_completed_history"
    if "freq-hrl" in lower or "freqhrl" in lower or "freq_hrl" in lower:
        if any(tok in lower for tok in ("analysis", "appendix", "matrix")):
            return "transit_freqhrl_analysis_matrix_c_le2_completed_history"
        if any(tok in lower for tok in ("deploy", "probe", "smoke", "control", "import")):
            return "transit_freqhrl_import_smoke_c_le2_completed_history"
        if "native-promotion" in lower or "native_promotion" in lower:
            if any(tok in lower for tok in ("wait-credit", "wait_credit")):
                return "transit_native_promotion_c9_16_wait_credit_shell_completed_history"
            if "bounded-wait" in lower or "bounded_wait" in lower:
                return "transit_native_promotion_c9_16_bounded_wait_completed_history"
            if "single" in lower or "shard_1_32_64" in lower:
                return "transit_native_promotion_c33_64_single_seed_completed_history"
            if "more-seeds" in lower or "residual" in lower:
                return "transit_native_promotion_c17_32_residual_completed_history"
            return "transit_native_promotion_c3_8_persistent_stress_completed_history"
        if "demand-estimator" in lower or "demand_estimator" in lower:
            return "transit_demand_estimator_c17_32_completed_history"
        if "trading" in lower:
            if "public" in lower or "yahoo" in lower:
                return "transit_trading_public_csv_c3_8_completed_history"
            if "promotion" in lower or "recovery" in lower:
                return "transit_trading_promotion_recovery_c17_32_completed_history"
            if "policy" in lower:
                return "transit_trading_policy_c33_64_completed_history"
        if "real-demand" in lower or "real_demand" in lower:
            if "alighting" in lower:
                return "transit_native_real_demand_alighting_c3_8_completed_history"
            if "merge_native_real_demand_shards" in lower or "merge-drift" in lower:
                return "transit_native_real_demand_alighting_c3_8_completed_history"
            if "safe-wait" in lower or "waitaware" in lower:
                return "transit_native_real_demand_safe_wait_c9_16_profile_extension"
    project_lower = str(task.get("project") or "").strip().lower()
    signature_lower = str(task.get("signature") or "").strip().lower()
    if project_lower == "offline-sumo" or any(tok in lower for tok in ("train_rlpd.py", "train_wsrl.py")):
        return "offline_sumo_eval_c33_64_completed_history"
    if "per_ckpt_eval.py" in lower:
        return "offline_sumo_eval_c33_64_completed_history"
    if "h2o+" in lower or "h2oplus" in lower or "simple_sac" in lower:
        if any(tok in lower for tok in ("train_awac_bus.py", "train_td3bc_bus.py", "train_iql_bus.py")):
            if "device cuda" in lower or "--device=cuda" in lower:
                return "h2oplus_snapshot_gpu_completed_history"
            return "h2oplus_shell_eval_c_le2_completed_history"
        if any(tok in lower for tok in ("dispatch_r3_evals.py", "dispatch_r3_nosnap_evals.py", "run_p4_hold_calibration.sh")):
            return "h2oplus_shell_eval_c_le2_completed_history"
        if "h2o+_bus_main.py" in lower and any(tok in lower for tok in ("snapshot", "contrastive", "dynamics")):
            return "h2oplus_snapshot_gpu_completed_history"
        if "pure_online_sac" in lower and "run_multiseed_eval.sh" in lower:
            return "sumo_eval_simple_sac_c_le2"
        if "eval_with_metrics.py" in lower:
            return "h2oplus_shell_eval_c_le2_completed_history"
        if "eval_daganzo.py" in lower or "run_multiseed_eval.sh" in lower or "shell" in lower:
            return "h2oplus_shell_eval_c_le2_completed_history"
        if "h2o+_bus_main.py" in lower:
            return "offline_sumo_eval_c33_64_completed_history"
    if "resac" in project_lower or "re-sac" in project_lower or "resac_bus" in lower:
        return "resac_bus_seed_extension_cpu_eval_completed_history"
    if "train_compare_baselines.py" in lower:
        return "bamor_train_compare_c3_8_completed_history"
    if "run_bamor_diagnostic_shard.py" in lower:
        if cpu_suffix == "c9_16":
            return "bamor_diagnostic_shard_c9_16_completed_history"
        if cpu_suffix == "c3_8":
            return "bamor_diagnostic_shard_c3_8_completed_history"
        return "bamor_diagnostic_shard_c17_32_completed_history"
    if "paper-longtrain" in lower or "paper_longtrain" in lower:
        return "freqduet_paper_longtrain_c17_32_completed_history"
    if "promoted-ep100" in lower or "promoted_ep100" in lower:
        return "freqduet_promoted_ep100_c65p_completed_history"
    if "env-smoke" in lower or "deps ok" in lower or "hpc-relay" in lower or "envbootstrap" in project_lower:
        return "scheduleurm_hpc_relay_smoke_completed_history"
    if "runner_v3" in lower and "allfreq" in lower:
        return "freqduet_runner_v3_allfreq_alllayers_c9_16"
    if any(tok in lower for tok in ("cpu_heavy_local", "cpu-heavy", "cpu_heavy", "prime_sieve")):
        return "cpu_heavy_local_bench"

    if _strict_admission_mode(admission_mode):
        return ""

    text_tokens = set(_tokens(text))
    best_key = ""
    best_score = 0
    for key in cache.available_workloads():
        key_tokens = set(_tokens(key))
        if not key_tokens:
            continue
        score = len(text_tokens & key_tokens)
        if score > best_score:
            best_key = key
            best_score = score
    return best_key if best_score >= 2 else ""


def bind_service(
    task: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    cache: ServiceRateCache | None = None,
    workload_key: str | None = None,
    admission_mode: str | None = None,
) -> ServiceBinding:
    cache = cache or default_service_cache()
    selected_key = workload_key or infer_workload_key(
        task,
        cache,
        admission_mode=admission_mode,
    )
    profile = max(1, as_int(features.get("post_task_count"), 1))
    if not selected_key:
        return _uncertified("", profile, "workload_key_not_inferred")
    record = cache.get(selected_key, profile)
    if record is None:
        return _uncertified(selected_key, profile, "exact_profile_missing")
    return binding_from_record(record)


def binding_from_record(record: ProfileRecord) -> ServiceBinding:
    aggregate = _finite_nonnegative(record.aggregate_rate)
    mean = _finite_nonnegative(record.mean_rate)
    certified = (not record.capacity_boundary) and aggregate > 0.0
    reason = "" if certified else (
        "capacity_boundary" if record.capacity_boundary else "nonpositive_service_rate"
    )
    return ServiceBinding(
        workload_key=record.workload_key,
        profile=int(record.profile),
        lower_service=aggregate if certified else 0.0,
        aggregate_rate=aggregate,
        mean_rate=mean,
        certified=certified,
        reason=reason,
        source=str(record.source),
        command_fingerprint=str(record.command_fingerprint),
        resource_kind=str(record.resource_kind),
        node_bucket=str(record.node_bucket),
        capacity_boundary=bool(record.capacity_boundary),
    )


def available_profile_table(cache: ServiceRateCache | None = None) -> dict[str, list[int]]:
    cache = cache or default_service_cache()
    return {
        key: [int(record.profile) for record in cache.profiles(key)]
        for key in cache.available_workloads()
    }


def _uncertified(workload_key: str, profile: int, reason: str) -> ServiceBinding:
    return ServiceBinding(
        workload_key=str(workload_key or ""),
        profile=int(profile),
        lower_service=0.0,
        aggregate_rate=0.0,
        mean_rate=0.0,
        certified=False,
        reason=reason,
        source="",
    )


def _explicit_workload_key(task: Mapping[str, Any], cache: ServiceRateCache) -> str:
    for field in ("theorem_workload_key", "workload_key", "service_workload_key"):
        value = str(task.get(field) or "").strip()
        if value and value in set(cache.available_workloads()):
            return value
    return ""


def _strict_admission_mode(admission_mode: str | None = None) -> bool:
    mode = str(admission_mode or os.environ.get("SCHEDULEURM_THEOREM_ADMISSION_MODE") or "").strip().lower()
    return mode in {"strict", "manifest", "fingerprint"}


def _task_text(task: Mapping[str, Any]) -> str:
    parts = [
        task.get("project"),
        task.get("signature"),
        task.get("description"),
        task.get("cmd"),
        task.get("cwd"),
        task.get("git_repo"),
    ]
    return " ".join(str(part or "") for part in parts)


def _task_identity_text(task: Mapping[str, Any]) -> str:
    parts = [
        task.get("project"),
        task.get("signature"),
        task.get("description"),
    ]
    return " ".join(str(part or "") for part in parts)


def _cpu_bucket_suffix(task: Mapping[str, Any]) -> str:
    cpu = max(0, as_int(task.get("cpu_cores"), 0))
    if cpu <= 2:
        return "c_le2"
    if cpu <= 8:
        return "c3_8"
    if cpu <= 16:
        return "c9_16"
    if cpu <= 32:
        return "c17_32"
    if cpu <= 64:
        return "c33_64"
    return "c65p"


def _tokens(text: Any) -> list[str]:
    out = []
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        if len(raw) <= 1 or raw in _STOP_TOKENS:
            continue
        out.append(raw)
    return out


def _finite_nonnegative(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)
