"""Helpers for collecting watcher claim-renewal inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


@dataclass
class ClaimRenewalInputs:
    running_by_node: dict[str, list[str]] = field(default_factory=dict)
    active_claim_ids_by_node: dict[str, set[str]] = field(default_factory=dict)
    claim_records_by_node: dict[str, dict[str, dict]] = field(default_factory=dict)
    live_pid_by_node: dict[str, dict[str, int]] = field(default_factory=dict)


def collect_claim_renewal_inputs(
    tasks: Iterable[dict],
    *,
    enabled_for_node: Callable[[str], bool],
    claim_record_for_task: Callable[[dict], dict],
) -> ClaimRenewalInputs:
    """Collect per-node claim renewal data from scheduler task state."""
    out = ClaimRenewalInputs()
    for task in tasks:
        node = task.get("node")
        if not node or not enabled_for_node(node):
            continue
        status = task.get("status")
        if status in ("running", "launching"):
            out.active_claim_ids_by_node.setdefault(node, set()).add(task["id"])
        if status != "running":
            continue
        out.running_by_node.setdefault(node, []).append(task["id"])
        out.claim_records_by_node.setdefault(node, {})[task["id"]] = (
            claim_record_for_task(task)
        )
        pids = task.get("remote_pids") or []
        if pids:
            out.live_pid_by_node.setdefault(node, {})[task["id"]] = int(pids[0])
    return out


def stale_claim_task_ids(
    claims: Iterable[dict],
    *,
    own_scheduler_id: str,
    active_task_ids: Iterable[str],
) -> list[str]:
    """Return own claim task IDs that are no longer represented in state."""
    active = set(active_task_ids or [])
    stale: list[str] = []
    for claim in claims:
        if claim.get("scheduler_id") != own_scheduler_id:
            continue
        tid = claim.get("task_id")
        if not tid or tid in active:
            continue
        stale.append(tid)
    return stale


def claim_pid_updates(
    claims: Iterable[dict],
    desired_pid_by_task_id: Mapping[str, int],
) -> list[tuple[str, int]]:
    """Return claim PID updates needed to match live task remote_pids."""
    updates: list[tuple[str, int]] = []
    for claim in claims:
        tid = claim.get("task_id")
        want_pid = desired_pid_by_task_id.get(tid)
        if want_pid is None:
            continue
        if claim.get("pid") == want_pid:
            continue
        updates.append((tid, want_pid))
    return updates
