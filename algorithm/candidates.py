"""Candidate-set utilities for active-bucket experiments.

The scheduler still owns its legacy candidate enumeration.  This module provides
the proof-facing reduction layer: keep exact candidates when requested, or keep
the best representative per finite active bucket for approximation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _tuple_score(score: Any) -> Tuple[Any, ...]:
    if isinstance(score, tuple):
        return score
    if isinstance(score, list):
        return tuple(score)
    return (score,)


@dataclass(frozen=True)
class CandidateRecord:
    node: str
    gpu_idx: int | None
    score: Tuple[Any, ...]
    features: Mapping[str, Any]
    audit: Mapping[str, Any]

    @property
    def bucket(self) -> str:
        return str(self.features.get("candidate_bucket") or "bucket:unknown")

    def sort_key(self) -> Tuple[Any, ...]:
        return _tuple_score(self.score)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "gpu_idx": self.gpu_idx,
            "score": list(self.score),
            "bucket": self.bucket,
            "class_key": self.features.get("class_key"),
            "regime_key": self.features.get("regime_key"),
            "audit": dict(self.audit),
        }


def active_bucket_representatives(
    candidates: Iterable[CandidateRecord],
    max_per_bucket: int = 1,
) -> List[CandidateRecord]:
    """Return the best candidates per active finite bucket.

    `max_per_bucket=1` is the theorem-facing representative mode.  Larger values
    are useful for stress tests that interpolate between active-bucket and full
    exact action enumeration.
    """
    limit = max(1, int(max_per_bucket or 1))
    by_bucket: Dict[str, List[CandidateRecord]] = {}
    for cand in candidates:
        bucket = cand.bucket
        by_bucket.setdefault(bucket, []).append(cand)
    selected: List[CandidateRecord] = []
    for bucket in sorted(by_bucket):
        ordered = sorted(by_bucket[bucket], key=lambda c: c.sort_key())
        selected.extend(ordered[:limit])
    return sorted(selected, key=lambda c: c.sort_key())


def approximation_report(
    full: Sequence[CandidateRecord],
    selected: Sequence[CandidateRecord],
) -> Dict[str, Any]:
    full_buckets = {c.bucket for c in full}
    selected_buckets = {c.bucket for c in selected}
    missing = sorted(full_buckets - selected_buckets)
    return {
        "full_count": len(full),
        "selected_count": len(selected),
        "full_bucket_count": len(full_buckets),
        "selected_bucket_count": len(selected_buckets),
        "missing_bucket_count": len(missing),
        "missing_buckets": missing[:100],
    }
