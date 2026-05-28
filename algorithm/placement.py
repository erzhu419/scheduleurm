"""Placement policy hooks for scheduleurm.

This module is intentionally dependency-free.  scheduler.py owns probing,
state mutation, process launch, claims, rollback, and safety checks.  Policy
objects only contribute optional placement scoring and extra admission gates.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from .features import as_float as _as_float
from .features import as_int as _as_int
from .features import gpu_candidate_features
from .scoring import RobustScoreWeights, score_gpu_candidate, weights_from_env


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return _as_int(raw, 0)


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    return _as_float(raw, 0.0)


def _tuple_score(score: Any) -> Tuple[Any, ...]:
    if isinstance(score, tuple):
        return score
    if isinstance(score, list):
        return tuple(score)
    return (score,)


@dataclass(frozen=True)
class PlacementPolicyConfig:
    name: str
    sweet_spot_tasks_per_gpu: int | None = None
    max_tasks_per_gpu: int | None = None
    max_post_vram_frac: float | None = None
    max_gpu_util_pct: int | None = None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sweet_spot_tasks_per_gpu": self.sweet_spot_tasks_per_gpu,
            "max_tasks_per_gpu": self.max_tasks_per_gpu,
            "max_post_vram_frac": self.max_post_vram_frac,
            "max_gpu_util_pct": self.max_gpu_util_pct,
        }


class BasePlacementPolicy:
    """Placement policy interface.

    Returning an empty string from `gpu_fit_block_reason` means the policy does
    not add a block.  Returning `legacy_score` from `gpu_score` preserves the
    scheduler's original candidate order.
    """

    def __init__(self, config: PlacementPolicyConfig):
        self.config = config
        self.name = config.name

    def snapshot(self) -> Dict[str, Any]:
        return self.config.snapshot()

    def gpu_fit_block_reason(
        self,
        task: Dict[str, Any],
        gpu: Dict[str, Any],
        node_info: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        return ""

    def gpu_score(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        legacy_score: Tuple[Any, ...],
        context: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        return legacy_score

    def selected_gpu_audit(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {}


class LegacyPlacementPolicy(BasePlacementPolicy):
    def __init__(self):
        super().__init__(PlacementPolicyConfig(name="legacy"))


class SweetSpotPlacementPolicy(BasePlacementPolicy):
    """Interference-aware co-location policy.

    This is an experiment hook, not a theorem by itself.  It lets us sweep
    admission/scoring parameters while keeping scheduler.py's existing safety
    checks in place.
    """

    def __init__(self, config: PlacementPolicyConfig):
        super().__init__(config)
        self.score_weights: RobustScoreWeights = weights_from_env(
            config.sweet_spot_tasks_per_gpu)
        self._gpu_audit_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def snapshot(self) -> Dict[str, Any]:
        out = self.config.snapshot()
        out["score_weights"] = self.score_weights.snapshot()
        return out

    def _audit_key(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
    ) -> Tuple[str, str, str]:
        gpu_idx = gpu.get("idx")
        return (
            str(task.get("id") or task.get("signature") or task.get("cmd") or ""),
            str(node_state.get("name") or ""),
            str(gpu_idx if gpu_idx is not None else ""),
        )

    def _features(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        context: Dict[str, Any],
        legacy_score: Tuple[Any, ...] | None = None,
    ) -> Dict[str, Any]:
        ctx = dict(context or {})
        ctx["sweet_spot_tasks_per_gpu"] = self.score_weights.sweet_spot_tasks_per_gpu
        if legacy_score is not None:
            ctx["legacy_score"] = legacy_score
        return gpu_candidate_features(task, node_state, gpu, ctx)

    def _audit_from_features(self, features: Dict[str, Any], breakdown) -> Dict[str, Any]:
        return {
            "candidate_bucket": features.get("candidate_bucket"),
            "class_key": features.get("class_key"),
            "regime_key": features.get("regime_key"),
            "finite": dict(features.get("finite") or {}),
            "score": breakdown.snapshot(),
            "bounded_penalty": True,
            "calibration_required": True,
        }

    def gpu_fit_block_reason(
        self,
        task: Dict[str, Any],
        gpu: Dict[str, Any],
        node_info: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        count = _as_int(gpu.get("running_task_count"), 0)
        max_tasks = self.config.max_tasks_per_gpu
        if max_tasks is not None and max_tasks > 0 and count >= max_tasks:
            return f"algorithm:{self.name}: task_count {count}/{max_tasks}"

        util_cap = self.config.max_gpu_util_pct
        used_mb = _as_int(gpu.get("used_mb"), 0)
        empty_used_mb = _as_int(context.get("gpu_empty_used_mb"), 100)
        if (
            util_cap is not None
            and util_cap >= 0
            and used_mb > empty_used_mb
            and _as_int(gpu.get("util_pct"), 0) >= util_cap
        ):
            return f"algorithm:{self.name}: util {_as_int(gpu.get('util_pct'), 0)}%>={util_cap}%"

        max_frac = self.config.max_post_vram_frac
        if max_frac is not None and max_frac > 0:
            total = max(1, _as_int(gpu.get("total_mb"), 1))
            need = max(0, _as_int(task.get("est_vram_mb"), 0))
            post_frac = float(used_mb + need) / float(total)
            if post_frac > max_frac:
                return (
                    f"algorithm:{self.name}: post_vram_frac "
                    f"{post_frac:.3f}>{max_frac:.3f}"
                )
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
        features = self._features(task, node_state, gpu, context, legacy_tuple)
        breakdown = score_gpu_candidate(features, self.score_weights)
        audit = self._audit_from_features(features, breakdown)
        self._gpu_audit_cache[self._audit_key(task, node_state, gpu)] = audit
        return breakdown.sort_prefix + legacy_tuple

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
        features = self._features(task, node_state, gpu, context)
        breakdown = score_gpu_candidate(features, self.score_weights)
        return self._audit_from_features(features, breakdown)


def _sweetspot_config(name: str) -> PlacementPolicyConfig:
    return PlacementPolicyConfig(
        name=name,
        sweet_spot_tasks_per_gpu=_optional_int_env("SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS"),
        max_tasks_per_gpu=_optional_int_env("SCHEDULEURM_ALGO_MAX_TASKS_PER_GPU"),
        max_post_vram_frac=_optional_float_env("SCHEDULEURM_ALGO_MAX_POST_VRAM_FRAC"),
        max_gpu_util_pct=_optional_int_env("SCHEDULEURM_ALGO_MAX_GPU_UTIL_PCT"),
    )


def available_policies() -> Iterable[str]:
    return ("legacy", "sweetspot_v1", "interference_v1")


def load_placement_policy(name: str | None = None) -> BasePlacementPolicy:
    raw = (name or os.environ.get("SCHEDULEURM_ALGORITHM") or "legacy").strip()
    selected = raw.lower().replace("-", "_")
    if selected in ("", "legacy", "default"):
        return LegacyPlacementPolicy()
    if selected in ("sweetspot_v1", "sweetspot", "interference_v1", "interference"):
        return SweetSpotPlacementPolicy(_sweetspot_config(selected))
    choices = ", ".join(available_policies())
    raise ValueError(f"unknown scheduleurm algorithm {raw!r}; choices: {choices}")
