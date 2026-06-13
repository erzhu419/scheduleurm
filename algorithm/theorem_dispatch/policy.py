"""Robust MaxWeight placement policy backed by measured service rows."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ..features import as_float, as_int, gpu_candidate_features
from ..scoring import RobustScoreWeights, weights_from_env
from .queue_state import queue_vector_for_task
from .service_registry import ServiceBinding, bind_service, default_service_cache


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return as_int(raw, 0)


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return as_float(raw, 0.0)


def _tuple_score(score: Any) -> Tuple[Any, ...]:
    if isinstance(score, tuple):
        return score
    if isinstance(score, list):
        return tuple(score)
    return (score,)


@dataclass(frozen=True)
class TheoremPolicyConfig:
    name: str = "theorem_maxweight_v1"
    max_tasks_per_gpu: int | None = None
    max_post_vram_frac: float | None = None
    max_gpu_util_pct: int | None = None
    sweet_spot_tasks_per_gpu: int | None = None
    uncertified_mode: str = "legacy"
    queue_ttl_s: float = 1.0
    penalty_per_extra_profile: float = 0.0
    profile_penalty_reference: int = 1
    backlog_penalty_reference: float = 0.0
    backlog_penalty_min_fraction: float = 0.25
    admission_mode: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_tasks_per_gpu": self.max_tasks_per_gpu,
            "max_post_vram_frac": self.max_post_vram_frac,
            "max_gpu_util_pct": self.max_gpu_util_pct,
            "sweet_spot_tasks_per_gpu": self.sweet_spot_tasks_per_gpu,
            "uncertified_mode": self.uncertified_mode,
            "queue_ttl_s": self.queue_ttl_s,
            "penalty_per_extra_profile": self.penalty_per_extra_profile,
            "profile_penalty_reference": self.profile_penalty_reference,
            "backlog_penalty_reference": self.backlog_penalty_reference,
            "backlog_penalty_min_fraction": self.backlog_penalty_min_fraction,
            "admission_mode": self.admission_mode,
        }


class TheoremMaxWeightPlacementPolicy:
    """GPU placement policy that emits theorem-grade lower-service traces."""

    def __init__(self, config: TheoremPolicyConfig | None = None):
        self.config = config or theorem_policy_config()
        self.name = self.config.name
        self._cache = default_service_cache()
        self._gpu_audit_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._fallback_weights: RobustScoreWeights = weights_from_env(
            self.config.sweet_spot_tasks_per_gpu)

    def snapshot(self) -> Dict[str, Any]:
        out = self.config.snapshot()
        out["service_cache_workload_count"] = len(self._cache.available_workloads())
        out["fallback_score_weights"] = self._fallback_weights.snapshot()
        return out

    def gpu_fit_block_reason(
        self,
        task: Dict[str, Any],
        gpu: Dict[str, Any],
        node_info: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        count = as_int(gpu.get("running_task_count"), 0)
        max_tasks = self.config.max_tasks_per_gpu
        if max_tasks is not None and max_tasks > 0 and count >= max_tasks:
            return f"algorithm:{self.name}: task_count {count}/{max_tasks}"

        util_cap = self.config.max_gpu_util_pct
        used_mb = as_int(gpu.get("used_mb"), 0)
        empty_used_mb = as_int(context.get("gpu_empty_used_mb"), 100)
        if (
            util_cap is not None
            and util_cap >= 0
            and used_mb > empty_used_mb
            and as_int(gpu.get("util_pct"), 0) >= util_cap
        ):
            return f"algorithm:{self.name}: util {as_int(gpu.get('util_pct'), 0)}%>={util_cap}%"

        max_frac = self.config.max_post_vram_frac
        if max_frac is not None and max_frac > 0:
            total = max(1, as_int(gpu.get("total_mb"), 1))
            need = max(0, as_int(task.get("est_vram_mb"), 0))
            post_frac = float(used_mb + need) / float(total)
            if post_frac > max_frac:
                return (
                    f"algorithm:{self.name}: post_vram_frac "
                    f"{post_frac:.3f}>{max_frac:.3f}"
                )

        if self.config.uncertified_mode == "block":
            features = self._features(task, {}, gpu, context)
            binding = bind_service(
                task,
                features,
                cache=self._cache,
                admission_mode=self.config.admission_mode,
            )
            if not binding.certified:
                return f"algorithm:{self.name}: service_certificate_required:{binding.reason}"
        return ""

    def gpu_score(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        legacy_score: Tuple[Any, ...],
        context: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        legacy_tuple = _tuple_score(legacy_score)
        audit = self._audit(task, node_state, gpu, context, legacy_tuple)
        self._gpu_audit_cache[self._audit_key(task, node_state, gpu)] = audit
        if not audit.get("theorem_ready"):
            return legacy_tuple
        robust_score = float(audit.get("robust_maxweight_score") or 0.0)
        lower = float(audit.get("selected_class_lower_service") or 0.0)
        penalty = float(audit.get("penalty_units") or 0.0)
        profile = int((audit.get("service_binding") or {}).get("profile") or 0)
        return (-robust_score, penalty, -lower, profile) + legacy_tuple

    def selected_gpu_audit(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        cached = self._gpu_audit_cache.get(self._audit_key(task, node_state, gpu))
        if cached:
            return cached
        return self._audit(task, node_state, gpu, context, None)

    def global_batch_select(
        self,
        tasks: list[Dict[str, Any]],
        nodes: list[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        from .batch_policy import select_global_batch_placements

        result = select_global_batch_placements(
            tasks,
            nodes,
            context=context,
            policy=self,
            max_batch_size=int(context.get("global_batch_size") or 4),
            max_configurations=int(
                context.get("global_batch_max_configurations")
                or context.get("global_max_configurations")
                or 10000
            ),
        )
        return result.snapshot()

    def _audit(
        self,
        task: Mapping[str, Any],
        node_state: Mapping[str, Any],
        gpu: Mapping[str, Any],
        context: Mapping[str, Any],
        legacy_score: Tuple[Any, ...] | None,
    ) -> Dict[str, Any]:
        features = self._features(task, node_state, gpu, context, legacy_score)
        binding = bind_service(
            task,
            features,
            cache=self._cache,
            admission_mode=self.config.admission_mode,
        )
        queue, queue_meta = queue_vector_for_task(
            task,
            binding.workload_key,
            queue_path=context.get("queue_file"),
            ttl_s=self.config.queue_ttl_s,
            admission_mode=self.config.admission_mode,
        )
        lower_vec = binding.lower_service_vector()
        q_weight = max(0.0, float(queue.get(binding.workload_key, 0.0))) if binding.workload_key else 0.0
        penalty = self._penalty_units(binding, q_weight=q_weight)
        robust_score = q_weight * float(binding.lower_service) - penalty
        theorem_ready = bool(binding.certified and lower_vec and queue)
        return {
            "candidate_bucket": features.get("candidate_bucket"),
            "class_key": features.get("class_key"),
            "regime_key": features.get("regime_key"),
            "finite": dict(features.get("finite") or {}),
            "features": {
                "post_task_count": features.get("post_task_count"),
                "post_vram_frac": features.get("post_vram_frac"),
                "util_pct": features.get("util_pct"),
                "running_task_count": features.get("running_task_count"),
            },
            "service_binding": binding.snapshot(),
            "lower_service": lower_vec,
            "selected_class_lower_service": float(binding.lower_service),
            "penalty_units": float(penalty),
            "queue_vector": queue,
            "queue_meta": queue_meta,
            "robust_maxweight_score": float(robust_score),
            "score_semantics": (
                "robust_maxweight_lower_service"
                if theorem_ready else
                "scheduler_sort_key_minimization"
            ),
            "best_score_exact_over_candidate_set": theorem_ready,
            "theorem_ready": theorem_ready,
            "bounded_penalty": True,
            "uncertified_mode": self.config.uncertified_mode,
            "fallback_reason": "" if theorem_ready else binding.reason,
        }

    def _features(
        self,
        task: Mapping[str, Any],
        node_state: Mapping[str, Any],
        gpu: Mapping[str, Any],
        context: Mapping[str, Any],
        legacy_score: Tuple[Any, ...] | None = None,
    ) -> Dict[str, Any]:
        ctx = dict(context or {})
        if self.config.sweet_spot_tasks_per_gpu:
            ctx["sweet_spot_tasks_per_gpu"] = self.config.sweet_spot_tasks_per_gpu
        if legacy_score is not None:
            ctx["legacy_score"] = legacy_score
        return gpu_candidate_features(task, node_state, gpu, ctx)

    def _penalty_units(self, binding: ServiceBinding, *, q_weight: float = 0.0) -> float:
        if not binding.certified:
            return 0.0
        reference = max(1, int(self.config.profile_penalty_reference or 1))
        extra = max(0, int(binding.profile) - reference)
        base = max(0.0, float(self.config.penalty_per_extra_profile)) * float(extra)
        backlog_ref = max(0.0, float(self.config.backlog_penalty_reference or 0.0))
        if base <= 0.0 or backlog_ref <= 0.0:
            return base
        min_fraction = min(1.0, max(0.0, float(self.config.backlog_penalty_min_fraction)))
        if q_weight <= backlog_ref:
            return base
        scale = max(min_fraction, backlog_ref / max(1e-12, float(q_weight)))
        return base * scale

    def _audit_key(
        self,
        task: Mapping[str, Any],
        node_state: Mapping[str, Any],
        gpu: Mapping[str, Any],
    ) -> Tuple[str, str, str]:
        gpu_idx = gpu.get("idx")
        return (
            str(task.get("id") or task.get("signature") or task.get("cmd") or ""),
            str(node_state.get("name") or ""),
            str(gpu_idx if gpu_idx is not None else ""),
        )


def theorem_policy_config(name: str = "theorem_maxweight_v1") -> TheoremPolicyConfig:
    mode = str(os.environ.get("SCHEDULEURM_THEOREM_UNCERTIFIED_MODE") or "legacy").strip().lower()
    if mode not in {"legacy", "block", "trace_only"}:
        mode = "legacy"
    return TheoremPolicyConfig(
        name=name,
        max_tasks_per_gpu=_optional_int_env("SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"),
        max_post_vram_frac=_optional_float_env("SCHEDULEURM_ALGO_MAX_POST_VRAM_FRAC"),
        max_gpu_util_pct=_optional_int_env("SCHEDULEURM_ALGO_MAX_GPU_UTIL_PCT"),
        sweet_spot_tasks_per_gpu=_optional_int_env("SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS"),
        uncertified_mode=mode,
        queue_ttl_s=max(0.0, _optional_float_env("SCHEDULEURM_THEOREM_QUEUE_TTL_S") or 1.0),
        penalty_per_extra_profile=max(
            0.0,
            _optional_float_env("SCHEDULEURM_THEOREM_PROFILE_PENALTY") or 0.0,
        ),
        profile_penalty_reference=max(
            1,
            _optional_int_env("SCHEDULEURM_THEOREM_PROFILE_PENALTY_REFERENCE") or 1,
        ),
        backlog_penalty_reference=max(
            0.0,
            _optional_float_env("SCHEDULEURM_THEOREM_BACKLOG_PENALTY_REFERENCE") or 0.0,
        ),
        backlog_penalty_min_fraction=min(
            1.0,
            max(
                0.0,
                _optional_float_env("SCHEDULEURM_THEOREM_BACKLOG_PENALTY_MIN_FRACTION") or 0.25,
            ),
        ),
        admission_mode=str(os.environ.get("SCHEDULEURM_THEOREM_ADMISSION_MODE") or ""),
    )
