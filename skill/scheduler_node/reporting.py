"""Node, claim, and heartbeat reporting helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


NODE_SUMMARY_HIDDEN_NAMES = frozenset({"zhengliang-hpc"})


@dataclass(frozen=True)
class NodeReportingDeps:
    node_configs: dict
    node_display_name: Callable[[str], str]
    format_mem_gb: Callable[[Any], str]
    format_node_ram_summary: Callable[[dict], str]
    scheduler_id: Callable[[], str]
    cpu_ownership_snapshot: Callable[[dict, list], Any]
    now: Callable[[], float] = time.time
    print_fn: Callable[..., Any] = print


def node_summary_visible(node: dict, *, hidden_names=NODE_SUMMARY_HIDDEN_NAMES) -> bool:
    return str(node.get("name") or "") not in hidden_names


def node_summary_sort_key(node: dict, *, deps: NodeReportingDeps) -> tuple:
    name = str(node.get("name") or "")
    display = deps.node_display_name(name)
    gpu_order = {
        "local": 0,
        "jtl110gpu": 1,
        "jtl110gpu2": 2,
        "jtl311linux": 3,
        "node007": 4,
    }
    cpu_order = {
        "jtl110cpu": 0,
        "jtl110cpu2": 1,
        "node001": 10,
        "node002": 11,
        "node003": 12,
        "node004": 13,
        "node005": 14,
        "node006": 15,
    }
    if name in gpu_order:
        return (0, gpu_order[name], display)
    if name in cpu_order:
        return (1, cpu_order[name], display)
    if node.get("gpus"):
        return (0, 50, display)
    if name.startswith("node"):
        return (1, 50, display)
    return (8, 0, display)


def iter_node_summary_nodes(nodes: list, *, deps: NodeReportingDeps) -> list:
    return sorted(
        (node for node in nodes if node_summary_visible(node)),
        key=lambda node: node_summary_sort_key(node, deps=deps),
    )


def print_node_summary(nodes: list, *, deps: NodeReportingDeps) -> None:
    deps.print_fn("=== nodes ===")
    for node in iter_node_summary_nodes(nodes, deps=deps):
        display_name = deps.node_display_name(str(node.get("name") or "?"))
        if not node["alive"]:
            deps.print_fn(f"  {display_name:11s} DOWN ({node.get('error','?')})")
            continue
        gpu_parts = []
        for gpu in node["gpus"]:
            mem_pct = int(round(100 * gpu["used_mb"] / max(gpu["total_mb"], 1)))
            gpu_parts.append(
                f"GPU{gpu['idx']}={deps.format_mem_gb(gpu['used_mb'])}/"
                f"{deps.format_mem_gb(gpu['total_mb'])}"
                f"(free={deps.format_mem_gb(gpu.get('free_mb', 0))}, "
                f"mem:{mem_pct}%, util:{gpu['util_pct']}%)"
            )
        gpu_str = ", ".join(gpu_parts) if gpu_parts else "CPU-only"
        load = node.get("loadavg", 0)
        reserve = int(deps.node_configs.get(node.get("name"), {}).get("reserved_cpu_cores") or 0)
        cpu_parts = []
        if isinstance(load, (int, float)):
            cpu_parts.append(f"load {load:.1f}")
        host_cpu = node.get("host_cpu_load_pct")
        if host_cpu is not None:
            wsl_load = node.get("wsl_loadavg")
            if isinstance(wsl_load, (int, float)):
                cpu_parts[-1:] = [f"wsl_load {wsl_load:.1f}"]
            cpu_parts.append(f"host {int(host_cpu)}%")
        if node.get("probe_fallback"):
            cpu_parts.append(str(node.get("probe_fallback")))
        if reserve:
            cpu_parts.append(f"reserve {reserve}")
        cpu_tail = f"({', '.join(cpu_parts)})" if cpu_parts else ""
        cpu_str = f"cpu={node.get('free_cpu', '?')}/{node.get('total_cpu', '?')}{cpu_tail}"
        ram_str = deps.format_node_ram_summary(node)
        claim_str = format_node_claim_summary(node, deps=deps)
        deps.print_fn(f"  {display_name:11s} {gpu_str}  {cpu_str}  {ram_str}{claim_str}")


def claim_wait_s(claim: dict, *, deps: NodeReportingDeps) -> int:
    try:
        ts = float(claim.get("intent_at") or claim.get("claimed_at") or 0)
    except Exception:
        ts = 0
    return max(0, int(deps.now() - ts)) if ts else 0


def format_claim_record(claim: dict, *, deps: NodeReportingDeps) -> str:
    gpu = claim.get("gpu_idx")
    loc = "CPU" if gpu is None else f"GPU{gpu}"
    owner = claim.get("owner") or "?"
    return (
        f"{owner}/{claim.get('scheduler_id','?')}/{claim.get('task_id','?')}:{loc} "
        f"vram={deps.format_mem_gb(claim.get('vram_mb') or 0)} "
        f"cpu={int(claim.get('cpu_cores') or 0)} "
        f"ram={deps.format_mem_gb(claim.get('ram_mb') or 0)} "
        f"age={claim_wait_s(claim, deps=deps)}s"
    )


def format_node_claim_summary(node: dict, *, deps: NodeReportingDeps) -> str:
    pending = node.get("pending_claims") or []
    active = node.get("active_claims") or []
    intents = node.get("claim_intents") or []
    err = node.get("claim_snapshot_error")
    parts = []
    if active:
        parts.append(f"active_claims={len(active)}")
    if pending:
        parts.append(f"pending_claims={len(pending)}")
    if intents:
        head = sorted(
            intents,
            key=lambda claim: (
                float(claim.get("intent_at", 0) or 0),
                str(claim.get("scheduler_id")),
                str(claim.get("task_id")),
            ),
        )[0]
        parts.append(f"intents={len(intents)} head={head.get('task_id','?')}@{claim_wait_s(head, deps=deps)}s")
    if err:
        parts.append(f"claims_error={err[:60]}")
    try:
        own_sid = deps.scheduler_id()
        other = [
            claim
            for claim in (active + pending + intents)
            if claim.get("scheduler_id") and claim.get("scheduler_id") != own_sid
        ]
    except Exception:
        other = []
    if other:
        owners = ",".join(sorted({str(claim.get("owner") or "?") for claim in other})[:3])
        parts.append(f"other_schedulers={len(other)} owner={owners}")
    return ("  claims(" + ", ".join(parts) + ")") if parts else ""


def format_claim_intent_hint_for_task(
    task: dict,
    node: str,
    snap: dict,
    *,
    deps: NodeReportingDeps,
) -> str:
    del node
    intents = list(snap.get("intents") or [])
    if not intents:
        return ""
    intents.sort(
        key=lambda claim: (
            float(claim.get("intent_at", 0) or 0),
            str(claim.get("scheduler_id")),
            str(claim.get("task_id")),
        )
    )
    key = (deps.scheduler_id(), task.get("id"))
    pos = None
    for i, claim in enumerate(intents):
        if (claim.get("scheduler_id"), claim.get("task_id")) == key:
            pos = i + 1
            break
    head = intents[0]
    if pos is not None:
        return (
            f"claim-intent: position {pos}/{len(intents)}; "
            f"head={format_claim_record(head, deps=deps)}"
        )
    return f"claim-intents: {len(intents)} queued; head={format_claim_record(head, deps=deps)}"


def build_heartbeat_payload(state: dict, nodes: list, *, deps: NodeReportingDeps) -> dict:
    running = sum(1 for task in state["tasks"] if task["status"] == "running")
    launching = sum(1 for task in state["tasks"] if task["status"] == "launching")
    queued = sum(1 for task in state["tasks"] if task["status"] == "queued")
    node_strs = []
    for node in nodes:
        if not node["alive"]:
            node_strs.append(f"{node['name']}:DOWN")
            continue
        gpu_brief = "/".join(deps.format_mem_gb(gpu.get("used_mb", 0)) for gpu in node["gpus"])
        node_strs.append(f"{node['name']}:{gpu_brief}")
    return {
        "running": running,
        "launching": launching,
        "queued": queued,
        "nodes": node_strs,
        "cpu_accounting": deps.cpu_ownership_snapshot(state, nodes),
    }
