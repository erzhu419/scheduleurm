"""Optional adaptive wrapper around theorem MaxWeight placement.

The live scheduler exposes a per-candidate scoring hook, not a batch action-set
callback.  This wrapper implements the same deterministic active-bucket forced
exploration model in that interface by prepending an exploration priority to
the theorem MaxWeight sort key.  It is opt-in only and is not the default
scheduler policy.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Tuple

from .adaptive_learning import TwoWindowMeanShiftDetector
from .features import as_float, as_int
from .theorem_dispatch import TheoremMaxWeightPlacementPolicy, theorem_policy_config


def _optional_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return as_int(raw, int(default))


def _optional_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return as_float(raw, float(default))


class AdaptiveTheoremPlacementPolicy:
    """Theorem policy with deterministic bucket exploration and regime detector."""

    def __init__(self, name: str = "adaptive_theorem_maxweight_v1") -> None:
        self.name = name
        self.base = TheoremMaxWeightPlacementPolicy(theorem_policy_config("theorem_maxweight_v1"))
        self.min_samples_per_bucket = max(
            1,
            _optional_int_env("SCHEDULEURM_ADAPTIVE_MIN_SAMPLES_PER_BUCKET", 128),
        )
        self.detector = TwoWindowMeanShiftDetector(
            window_size=max(1, _optional_int_env("SCHEDULEURM_ADAPTIVE_DETECTOR_WINDOW", 512)),
            threshold=max(0.0, _optional_float_env("SCHEDULEURM_ADAPTIVE_DETECTOR_THRESHOLD", 0.20)),
            min_shift=max(0.0, _optional_float_env("SCHEDULEURM_ADAPTIVE_DETECTOR_MIN_SHIFT", 0.40)),
        )
        self._bucket_counts: dict[str, int] = {}
        self._score_cache: dict[Tuple[str, str, str], dict[str, Any]] = {}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base.snapshot(),
            "sampler": {
                "model": "per-candidate deterministic forced active-bucket exploration",
                "min_samples_per_bucket": self.min_samples_per_bucket,
                "bucket_counts": dict(self._bucket_counts),
            },
            "detector": self.detector.tail_certificate(switching_count=1, failure_budget=1.0),
        }

    def gpu_fit_block_reason(
        self,
        task: Dict[str, Any],
        gpu: Dict[str, Any],
        node_info: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        return self.base.gpu_fit_block_reason(task, gpu, node_info, context)

    def gpu_score(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        legacy_score: Tuple[Any, ...],
        context: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        base_score = self.base.gpu_score(task, node_state, gpu, legacy_score, context)
        audit = self.base.selected_gpu_audit(task, node_state, gpu, context)
        bucket = str(audit.get("candidate_bucket") or "unknown")
        count = int(self._bucket_counts.get(bucket, 0))
        forced = count < self.min_samples_per_bucket
        adaptive_prefix = (0 if forced else 1, count, bucket)
        self._score_cache[self._audit_key(task, node_state, gpu)] = {
            "candidate_bucket": bucket,
            "bucket_count_before": count,
            "forced_exploration": forced,
            "base_score": base_score,
            "score_semantics": "adaptive_active_bucket_theorem_maxweight",
        }
        return adaptive_prefix + tuple(base_score)

    def selected_gpu_audit(
        self,
        task: Dict[str, Any],
        node_state: Dict[str, Any],
        gpu: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_audit = self.base.selected_gpu_audit(task, node_state, gpu, context)
        key = self._audit_key(task, node_state, gpu)
        cached = self._score_cache.get(key, {})
        bucket = str(cached.get("candidate_bucket") or base_audit.get("candidate_bucket") or "unknown")
        before = int(self._bucket_counts.get(bucket, 0))
        self._bucket_counts[bucket] = before + 1
        feedback = _normalized_feedback(base_audit)
        event = self.detector.update(str(base_audit.get("regime_key") or "unknown"), feedback)
        out = dict(base_audit)
        out["adaptive_sampler"] = {
            "model": "per_candidate_deterministic_forced_active_bucket_exploration",
            "candidate_bucket": bucket,
            "bucket_count_before": before,
            "bucket_count_after": self._bucket_counts[bucket],
            "required_min_samples_per_bucket": self.min_samples_per_bucket,
            "forced_exploration": bool(cached.get("forced_exploration", before < self.min_samples_per_bucket)),
            "bucket_counts": dict(self._bucket_counts),
        }
        out["regime_detector"] = {
            "model": "bounded_two_window_mean_shift_detector",
            "normalized_feedback": feedback,
            "change_event": event.__dict__ if event is not None else None,
        }
        out["score_semantics"] = base_audit.get("score_semantics")
        out["adaptive_score_semantics"] = "adaptive_active_bucket_theorem_maxweight"
        return out

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


def _normalized_feedback(audit: Mapping[str, Any]) -> float:
    lower = max(0.0, float(audit.get("selected_class_lower_service") or 0.0))
    # The detector only needs bounded feedback.  The scale is conservative and
    # does not enter the MaxWeight proof constants.
    return min(1.0, lower / max(1.0, lower + float(audit.get("penalty_units") or 0.0)))
