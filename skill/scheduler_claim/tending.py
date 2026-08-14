"""Watcher-side claim tending for cross-scheduler resource claims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass
class ClaimTendingDeps:
    enabled_for_node: Callable[[str], bool]
    collect_claim_renewal_inputs: Callable[..., Any]
    claim_record_for_task: Callable[[dict], dict]
    scheduler_id: Callable[[], str]
    enumerate_claims: Callable[[str], list[dict]]
    stale_claim_task_ids: Callable[..., list[str]]
    release_claim: Callable[[str, str], bool]
    renew_many: Callable[[str, list[str], Mapping[str, dict]], int]
    claim_pid_updates: Callable[..., list[tuple[str, int]]]
    update_pid: Callable[[str, str, int], Any]
    gc_stale: Callable[[str], int]
    notify: Callable[..., Any]


@dataclass
class ClaimTendingResult:
    released_by_node: dict[str, list[str]] = field(default_factory=dict)
    renewed_by_node: dict[str, list[str]] = field(default_factory=dict)
    pid_updates_by_node: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    gc_nodes: list[str] = field(default_factory=list)
    error_nodes: dict[str, str] = field(default_factory=dict)


def _iter_node_names(nodes: Iterable[str] | Mapping[str, Any]) -> list[str]:
    if isinstance(nodes, Mapping):
        return list(nodes.keys())
    return list(nodes)


def tend_scheduler_claims(
    tasks: Iterable[dict],
    nodes: Iterable[str] | Mapping[str, Any],
    *,
    deps: ClaimTendingDeps,
) -> ClaimTendingResult:
    """Renew, release, and reconcile scheduler claims for one watcher tick."""
    result = ClaimTendingResult()
    claim_inputs = deps.collect_claim_renewal_inputs(
        tasks,
        enabled_for_node=deps.enabled_for_node,
        claim_record_for_task=deps.claim_record_for_task,
    )
    own_sid = deps.scheduler_id()

    for node in _iter_node_names(nodes):
        if not deps.enabled_for_node(node):
            continue
        try:
            stale_released: list[str] = []
            try:
                current_claims = deps.enumerate_claims(node)
            except Exception:
                current_claims = []
            active_ids = claim_inputs.active_claim_ids_by_node.get(node, set())
            for tid in deps.stale_claim_task_ids(
                current_claims,
                own_scheduler_id=own_sid,
                active_task_ids=active_ids,
            ):
                try:
                    if deps.release_claim(node, tid):
                        stale_released.append(tid)
                except Exception:
                    pass
            if stale_released:
                result.released_by_node[node] = stale_released
                deps.notify(
                    "claims_released_inactive",
                    {
                        "node": node,
                        "task_ids": stale_released[:50],
                        "count": len(stale_released),
                    },
                    feishu_enabled=False,
                )

            running_ids = claim_inputs.running_by_node.get(node, [])
            if running_ids:
                deps.renew_many(
                    node,
                    running_ids,
                    claim_inputs.claim_records_by_node.get(node, {}),
                )
                result.renewed_by_node[node] = list(running_ids)

                pid_map = claim_inputs.live_pid_by_node.get(node) or {}
                if pid_map:
                    try:
                        current_claims = deps.enumerate_claims(node)
                    except Exception:
                        current_claims = []
                    updates = deps.claim_pid_updates(current_claims, pid_map)
                    for tid, want_pid in updates:
                        try:
                            deps.update_pid(node, tid, want_pid)
                        except Exception:
                            pass
                    if updates:
                        result.pid_updates_by_node[node] = list(updates)
            else:
                deps.gc_stale(node)
                result.gc_nodes.append(node)
        except Exception as exc:
            msg = str(exc)[:200]
            result.error_nodes[node] = msg
            deps.notify(
                "claims_tend_error",
                {"node": node, "error": msg},
                feishu_enabled=False,
            )
    return result
