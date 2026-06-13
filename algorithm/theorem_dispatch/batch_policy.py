"""Optional batch-level robust-MaxWeight placement construction.

This module deliberately does not mutate scheduler state or launch work.  It
turns a bounded snapshot of queued tasks and node/GPU candidates into the same
candidate-row format used by ``global_dispatch.select_global_action``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .global_dispatch import GlobalActionResult, select_global_action
from .policy import TheoremMaxWeightPlacementPolicy, theorem_policy_config


@dataclass(frozen=True)
class BatchPlacementResult:
    placements: dict[str, tuple[str, int | None]]
    action: GlobalActionResult
    candidate_rows: tuple[dict[str, Any], ...]
    queue_vector: dict[str, float]
    scheduler_hook_ready: bool

    def snapshot(self) -> dict[str, Any]:
        return {
            "placements": {
                task_id: {"node": node, "gpu_idx": gpu_idx}
                for task_id, (node, gpu_idx) in sorted(self.placements.items())
            },
            "action": self.action.snapshot(),
            "candidate_rows": list(self.candidate_rows),
            "queue_vector": dict(self.queue_vector),
            "scheduler_hook_ready": bool(self.scheduler_hook_ready),
        }


def select_global_batch_placements(
    tasks: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
    policy: TheoremMaxWeightPlacementPolicy | None = None,
    max_batch_size: int = 4,
    max_configurations: int = 10000,
) -> BatchPlacementResult:
    selected_policy = policy or TheoremMaxWeightPlacementPolicy(
        theorem_policy_config("global_theorem_maxweight_v1")
    )
    ctx = dict(context or {})
    rows = tuple(_candidate_rows(tasks, nodes, selected_policy, ctx))
    queue = _merged_queue_vector(rows)
    action = select_global_action(
        rows,
        queue,
        max_batch_size=max_batch_size,
        max_configurations=max_configurations,
        lookahead_weight=float(ctx.get("global_lookahead_weight") or 0.0),
    )
    by_action = {str(row.get("action_id") or ""): row for row in rows}
    placements: dict[str, tuple[str, int | None]] = {}
    for action_id in action.selected_action_ids:
        row = by_action.get(action_id)
        if not row:
            continue
        placements[str(row.get("task_id") or "")] = (
            str(row.get("node") or ""),
            _optional_int(row.get("gpu_idx")),
        )
    return BatchPlacementResult(
        placements=placements,
        action=action,
        candidate_rows=rows,
        queue_vector=queue,
        scheduler_hook_ready=bool(rows and placements),
    )


def _candidate_rows(
    tasks: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    policy: TheoremMaxWeightPlacementPolicy,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for node in nodes:
            node_name = str(node.get("name") or "")
            if not node.get("alive", True):
                continue
            for gpu in node.get("gpus") or []:
                block = policy.gpu_fit_block_reason(dict(task), dict(gpu), dict(node.get("node_info") or {}), dict(context))
                if block:
                    continue
                legacy_score = (0, node_name, _optional_int(gpu.get("idx")) or -1)
                policy.gpu_score(dict(task), dict(node), dict(gpu), legacy_score, dict(context))
                audit = policy.selected_gpu_audit(dict(task), dict(node), dict(gpu), dict(context))
                gpu_idx = _optional_int(gpu.get("idx"))
                action_id = f"task={task.get('id') or task.get('signature')};node={node_name};gpu={gpu_idx}"
                row = {
                    **audit,
                    "task_id": str(task.get("id") or task.get("signature") or action_id),
                    "action_id": action_id,
                    "resource_id": f"{node_name}:gpu:{gpu_idx}",
                    "resource_ids": (f"{node_name}:gpu:{gpu_idx}",),
                    "workload_key": str((audit.get("service_binding") or {}).get("workload_key") or ""),
                    "service_workload_key": str((audit.get("service_binding") or {}).get("workload_key") or ""),
                    "eta_lcb_s": _optional_float(task.get("eta_lcb_s") or task.get("eta_s")),
                    "oldest_wait_s": _optional_float(task.get("oldest_wait_s") or task.get("age_s")),
                    "node": node_name,
                    "gpu_idx": gpu_idx,
                    "scheduler_hint_only": True,
                }
                rows.append(row)
    return rows


def _merged_queue_vector(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in rows:
        queue = row.get("queue_vector") or {}
        if not isinstance(queue, Mapping):
            continue
        for key, value in queue.items():
            merged[str(key)] = max(float(value or 0.0), merged.get(str(key), 0.0))
    return merged


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
