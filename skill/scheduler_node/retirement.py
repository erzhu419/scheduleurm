"""Pure state migration for permanently retired scheduler nodes."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable


ACTIVE_STATUSES = frozenset({"queued", "launching", "running"})


def _task_node(task: dict) -> str:
    return str(task.get("node") or task.get("assigned_node") or "")


def reconcile_retired_nodes(
    state: dict,
    retired_nodes: Iterable[str],
    *,
    now: float,
    reason: str,
) -> dict:
    """Terminalize lost running work and remove retired mixed-node routes.

    Queued tasks restricted exclusively to retired nodes stay queued and retain
    their explicit restriction. Clearing the list would accidentally make them
    runnable everywhere.
    """
    retired = {str(node) for node in retired_nodes if str(node)}
    report = {
        "retired_nodes": sorted(retired),
        "reason": reason,
        "terminalized": [],
        "rerouted_queued": [],
        "blocked_queued": [],
        "cleared_preferences": [],
    }
    for task in state.get("tasks", []):
        if task.get("status") not in ACTIVE_STATUSES:
            continue
        task_id = str(task.get("id") or "")
        node = _task_node(task)
        if task.get("status") in {"running", "launching"} and node in retired:
            previous_status = str(task.get("status"))
            task["status"] = "failed"
            task["finished_at"] = now
            task["exit_code"] = None
            task["alive_pids"] = []
            task["current_vram_mb"] = 0
            task["current_ram_mb"] = 0
            task["current_pcpu"] = 0.0
            task["last_block_reason"] = f"{reason}: {node}"
            task["node_retirement"] = {
                "node": node,
                "reconciled_at": now,
                "status_before": previous_status,
                "reason": reason,
                "remote_termination_attempted": False,
            }
            task["_diagnosis"] = {
                "is_crash": True,
                "reason": task["last_block_reason"],
                "tail": "remote node permanently unavailable; liveness cannot be recovered",
                "lifetime_s": int(max(0.0, now - float(task.get("started_at") or now))),
                "log_path": task.get("log_path"),
            }
            report["terminalized"].append(task_id)
            continue

        if task.get("status") != "queued":
            continue
        preferred = str(task.get("preferred_node") or "")
        if preferred in retired:
            task["retired_preferred_node"] = preferred
            task.pop("preferred_node", None)
            report["cleared_preferences"].append(task_id)

        allowed = task.get("allowed_nodes")
        if isinstance(allowed, list) and retired.intersection(map(str, allowed)):
            remaining = [node for node in allowed if str(node) not in retired]
            if remaining:
                task["allowed_nodes"] = remaining
                submitted = task.get("allowed_nodes_submitted")
                if isinstance(submitted, list):
                    task["allowed_nodes_submitted"] = [
                        node for node in submitted if str(node) not in retired
                    ]
                task["retired_nodes_removed_from_allowed"] = sorted(
                    retired.intersection(map(str, allowed))
                )
                report["rerouted_queued"].append(task_id)
            else:
                task["last_block_reason"] = (
                    f"{reason}: every explicitly allowed node is retired"
                )
                task["retired_node_block"] = {
                    "nodes": sorted(map(str, allowed)),
                    "reconciled_at": now,
                    "reason": reason,
                }
                report["blocked_queued"].append(task_id)
        elif str(task.get("require_node") or "") in retired:
            task["last_block_reason"] = (
                f"{reason}: required node {task.get('require_node')} is retired"
            )
            task["retired_node_block"] = {
                "nodes": [str(task.get("require_node"))],
                "reconciled_at": now,
                "reason": reason,
            }
            report["blocked_queued"].append(task_id)

    report["counts"] = {
        key: len(report[key])
        for key in (
            "terminalized",
            "rerouted_queued",
            "blocked_queued",
            "cleared_preferences",
        )
    }
    return report


def preview_retired_node_reconciliation(
    state: dict,
    retired_nodes: Iterable[str],
    *,
    now: float,
    reason: str,
) -> tuple[dict, dict]:
    preview_state = deepcopy(state)
    report = reconcile_retired_nodes(
        preview_state,
        retired_nodes,
        now=now,
        reason=reason,
    )
    return preview_state, report
