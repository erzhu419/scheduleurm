"""Serializable action objects for candidate-set experiments.

Scheduler dispatch currently launches one task at a time, but the theorem route
is stated for global actions.  This module provides the bridge: each local
placement candidate can be embedded into a complete, versioned action object,
and future generators can combine multiple assignments without changing the
schema.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Tuple


def _stable_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def stable_action_id(data: Mapping[str, Any], prefix: str = "a") -> str:
    digest = hashlib.sha1(_stable_json(data).encode("utf-8", "ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class ActionAssignment:
    task_id: str
    node: str
    gpu_idx: int | None
    class_key: str
    candidate_bucket: str
    resource: Mapping[str, Any] = field(default_factory=dict)
    feasible: bool = True
    infeasible_reason: str = ""

    def snapshot(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "node": self.node,
            "gpu_idx": self.gpu_idx,
            "class_key": self.class_key,
            "candidate_bucket": self.candidate_bucket,
            "resource": dict(self.resource),
            "feasible": bool(self.feasible),
            "infeasible_reason": self.infeasible_reason,
        }


@dataclass(frozen=True)
class GlobalAction:
    assignments: Tuple[ActionAssignment, ...]
    source: str
    score: Tuple[Any, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "global_action:v1"

    @property
    def action_id(self) -> str:
        return stable_action_id(self.snapshot_without_id())

    def snapshot_without_id(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "score": list(self.score),
            "assignments": [a.snapshot() for a in self.assignments],
            "metadata": dict(self.metadata),
        }

    def snapshot(self) -> Dict[str, Any]:
        out = self.snapshot_without_id()
        out["action_id"] = self.action_id
        return out

    def bucket_signature(self) -> Tuple[str, ...]:
        return tuple(sorted(a.candidate_bucket for a in self.assignments))

    def class_signature(self) -> Tuple[str, ...]:
        return tuple(sorted(a.class_key for a in self.assignments))

    def is_feasible(self) -> bool:
        return all(a.feasible for a in self.assignments)


def assignment_from_gpu_candidate(
    task: Mapping[str, Any],
    node: str,
    gpu_idx: int | None,
    features: Mapping[str, Any],
    feasible: bool = True,
    infeasible_reason: str = "",
) -> ActionAssignment:
    return ActionAssignment(
        task_id=str(task.get("id") or task.get("signature") or task.get("cmd") or ""),
        node=str(node),
        gpu_idx=gpu_idx,
        class_key=str(features.get("class_key") or ""),
        candidate_bucket=str(features.get("candidate_bucket") or ""),
        resource={
            "need_vram_mb": features.get("need_vram_mb"),
            "post_used_mb": features.get("post_used_mb"),
            "post_vram_frac": features.get("post_vram_frac"),
            "post_task_count": features.get("post_task_count"),
            "util_pct": features.get("util_pct"),
        },
        feasible=feasible,
        infeasible_reason=infeasible_reason,
    )


def single_assignment_action(
    assignment: ActionAssignment,
    score: Iterable[Any] = (),
    source: str = "scheduleurm_single_dispatch",
    metadata: Mapping[str, Any] | None = None,
) -> GlobalAction:
    return GlobalAction(
        assignments=(assignment,),
        source=source,
        score=tuple(score),
        metadata=dict(metadata or {}),
    )


def combine_assignments(
    assignments: Iterable[ActionAssignment],
    score: Iterable[Any] = (),
    source: str = "scheduleurm_global_candidate",
    metadata: Mapping[str, Any] | None = None,
) -> GlobalAction:
    ordered = tuple(sorted(
        assignments,
        key=lambda a: (a.task_id, a.node, -1 if a.gpu_idx is None else int(a.gpu_idx)),
    ))
    return GlobalAction(
        assignments=ordered,
        source=source,
        score=tuple(score),
        metadata=dict(metadata or {}),
    )


def action_candidate_jsonl_row(
    slot_id: str,
    action: GlobalAction,
    chosen: bool = False,
) -> Dict[str, Any]:
    out = action.snapshot()
    out["slot_id"] = slot_id
    out["chosen"] = bool(chosen)
    out["bucket_signature"] = list(action.bucket_signature())
    out["class_signature"] = list(action.class_signature())
    out["feasible"] = action.is_feasible()
    return out
