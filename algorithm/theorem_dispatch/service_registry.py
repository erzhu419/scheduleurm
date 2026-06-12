"""Measured service certificates for theorem-facing dispatch.

The scheduler owns process launch and legacy safety rules.  This module only
binds a task/candidate pair to an existing measured service-cache row.  If a
task cannot be bound to an exact non-boundary profile, callers get an explicit
uncertified result instead of an interpolated service rate.
"""
from __future__ import annotations

import math
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


def infer_workload_key(task: Mapping[str, Any], cache: ServiceRateCache | None = None) -> str:
    """Infer the measured workload key for a scheduler task.

    The first block covers the controlled benchmark families used in the paper.
    The second block scores production-history keys by token overlap so a new
    production task can be admitted only when it resembles a measured family.
    """

    text = _task_text(task)
    lower = text.lower()
    identity_lower = _task_identity_text(task).lower()
    cpu_suffix = _cpu_bucket_suffix(task)
    if "freqduet" in lower:
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
    if ("freq-hrl" in lower or "freqhrl" in lower) and any(
        tok in lower for tok in ("deploy", "probe", "smoke", "control", "import")
    ):
        return "transit_freqhrl_import_smoke_c_le2_completed_history"
    if any(tok in lower for tok in ("cpu_heavy_local", "cpu-heavy", "cpu_heavy", "prime_sieve")):
        return "cpu_heavy_local_bench"

    cache = cache or default_service_cache()
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
) -> ServiceBinding:
    cache = cache or default_service_cache()
    selected_key = workload_key or infer_workload_key(task, cache)
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
