"""Robust candidate scoring for scheduleurm placement experiments.

Scores are minimization keys because scheduler.py sorts candidates ascending.
The component names mirror the proof route: queue/service reward, bounded
interference penalty, and a beta-style slack consumer.  Calibration of the
weights is experimental; the score surface itself is deterministic and auditable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .features import as_float, as_int


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return as_float(raw, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return as_int(raw, default)


@dataclass(frozen=True)
class RobustScoreWeights:
    sweet_spot_tasks_per_gpu: int = 0
    vram_pressure: float = 120.0
    util_pressure: float = 8.0
    colocation: float = 3.0
    sweet_gap: float = 2.0
    over_sweet: float = 25.0
    runtime: float = 0.20
    runtime_unknown: float = 15.0
    priority_reward: float = 3.0
    queue_age_reward: float = 0.20
    beta_penalty: float = 0.0
    bounded_penalty_cap: float = 1000.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sweet_spot_tasks_per_gpu": self.sweet_spot_tasks_per_gpu,
            "vram_pressure": self.vram_pressure,
            "util_pressure": self.util_pressure,
            "colocation": self.colocation,
            "sweet_gap": self.sweet_gap,
            "over_sweet": self.over_sweet,
            "runtime": self.runtime,
            "runtime_unknown": self.runtime_unknown,
            "priority_reward": self.priority_reward,
            "queue_age_reward": self.queue_age_reward,
            "beta_penalty": self.beta_penalty,
            "bounded_penalty_cap": self.bounded_penalty_cap,
        }


@dataclass(frozen=True)
class RobustScoreBreakdown:
    score: float
    sort_prefix: Tuple[Any, ...]
    components: Dict[str, float]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "sort_prefix": list(self.sort_prefix),
            "components": {k: round(v, 6) for k, v in self.components.items()},
        }


def weights_from_env(sweet_spot_tasks_per_gpu: int | None = None) -> RobustScoreWeights:
    sweet = _env_int(
        "SCHEDULEURM_ALGO_GPU_SWEET_SPOT_TASKS",
        int(sweet_spot_tasks_per_gpu or 0),
    )
    return RobustScoreWeights(
        sweet_spot_tasks_per_gpu=max(0, sweet),
        vram_pressure=_env_float("SCHEDULEURM_ALGO_WEIGHT_VRAM", 120.0),
        util_pressure=_env_float("SCHEDULEURM_ALGO_WEIGHT_UTIL", 8.0),
        colocation=_env_float("SCHEDULEURM_ALGO_WEIGHT_COLOCATION", 3.0),
        sweet_gap=_env_float("SCHEDULEURM_ALGO_WEIGHT_SWEET_GAP", 2.0),
        over_sweet=_env_float("SCHEDULEURM_ALGO_WEIGHT_OVER_SWEET", 25.0),
        runtime=_env_float("SCHEDULEURM_ALGO_WEIGHT_RUNTIME", 0.20),
        runtime_unknown=_env_float("SCHEDULEURM_ALGO_WEIGHT_RUNTIME_UNKNOWN", 15.0),
        priority_reward=_env_float("SCHEDULEURM_ALGO_WEIGHT_PRIORITY_REWARD", 3.0),
        queue_age_reward=_env_float("SCHEDULEURM_ALGO_WEIGHT_QUEUE_AGE_REWARD", 0.20),
        beta_penalty=_env_float("SCHEDULEURM_ALGO_BETA_PENALTY", 0.0),
        bounded_penalty_cap=_env_float("SCHEDULEURM_ALGO_BOUNDED_PENALTY_CAP", 1000.0),
    )


def _cap(value: float, cap: float) -> float:
    if cap <= 0:
        return value
    return max(-cap, min(cap, value))


def _log_runtime_minutes(seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return math.log1p(seconds) / math.log(60.0)


def score_gpu_candidate(
    features: Mapping[str, Any],
    weights: RobustScoreWeights,
) -> RobustScoreBreakdown:
    cap = max(0.0, float(weights.bounded_penalty_cap))
    post_vram_frac = max(0.0, as_float(features.get("post_vram_frac"), 0.0))
    util_frac = max(0.0, as_float(features.get("util_pct"), 0.0) / 100.0)
    running = max(0, as_int(features.get("running_task_count"), 0))
    over_sweet = max(0, as_int(features.get("over_sweet_tasks"), 0))
    sweet_gap = max(0, as_int(features.get("sweet_gap_tasks"), 0))
    runtime_unknown = max(0, as_int(features.get("legacy_runtime_unknown"), 1))
    runtime_s = max(0.0, as_float(features.get("legacy_runtime_s"), 0.0))
    priority = max(0.0, as_float(features.get("priority_weight"), 1.0))
    queue_age_min = max(0.0, as_float(features.get("queue_age_s"), 0.0) / 60.0)

    components = {
        "vram_pressure": weights.vram_pressure * post_vram_frac * post_vram_frac,
        "util_pressure": weights.util_pressure * util_frac * util_frac,
        "colocation_penalty": weights.colocation * float(running),
        "sweet_gap_penalty": weights.sweet_gap * float(sweet_gap),
        "over_sweet_penalty": weights.over_sweet * float(over_sweet),
        "runtime_penalty": weights.runtime * _log_runtime_minutes(runtime_s),
        "runtime_unknown_penalty": weights.runtime_unknown * float(1 if runtime_unknown else 0),
        "priority_reward": -weights.priority_reward * priority,
        "queue_age_reward": -weights.queue_age_reward * min(queue_age_min, 240.0),
        "beta_penalty": weights.beta_penalty,
    }
    bounded = {k: _cap(float(v), cap) for k, v in components.items()}
    score = sum(bounded.values())
    if not math.isfinite(score):
        score = cap if cap > 0 else 1e9
    sort_prefix = (
        round(score, 6),
        over_sweet,
        sweet_gap,
        round(post_vram_frac, 6),
        round(util_frac, 6),
        running,
        runtime_unknown,
        round(runtime_s, 3),
    )
    return RobustScoreBreakdown(score=score, sort_prefix=sort_prefix, components=bounded)
