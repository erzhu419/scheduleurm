from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ClaimProbeFoldDeps:
    claim_enabled_for: Callable[[str], bool]
    claim_snapshot: Callable[[str], dict]
    notify: Callable[..., Any]


def _claim_totals(claims: list[dict]) -> tuple[int, int, dict[int, int]]:
    cpu = sum(
        0 if claim.get("ignore_cpu_capacity") else int(claim.get("cpu_cores") or 0)
        for claim in claims
    )
    ram = sum(int(claim.get("ram_mb") or 0) for claim in claims)
    per_gpu = {}
    for claim in claims:
        gpu_idx = claim.get("gpu_idx")
        if gpu_idx is None:
            continue
        idx = int(gpu_idx)
        per_gpu[idx] = per_gpu.get(idx, 0) + int(claim.get("vram_mb") or 0)
    return cpu, ram, per_gpu


def fold_claims_into_probe(nodes: list, *, deps: ClaimProbeFoldDeps) -> list:
    """Fold shared claim budgets into the local probe view."""
    for node in nodes:
        if not node.get("alive"):
            continue
        name = node.get("name")
        if not name or not deps.claim_enabled_for(name):
            continue
        try:
            snap = deps.claim_snapshot(name)
            claims = list(snap.get("claims") or [])
            node["claim_intents"] = list(snap.get("intents") or [])
            node["claim_snapshot_error"] = snap.get("error") if not snap.get("ok", True) else None
        except Exception as exc:
            try:
                deps.notify(
                    "claims_probe_fold_error",
                    {"node": name, "error": str(exc)[:200]},
                    feishu_enabled=False,
                )
            except Exception:
                pass
            continue

        pending = [claim for claim in claims if not claim.get("pid")]
        active = [claim for claim in claims if claim.get("pid")]
        node["pending_claims"] = pending
        node["active_claims"] = active
        for gpu in node.get("gpus") or []:
            gpu.setdefault("observed_used_mb", int(gpu.get("used_mb") or 0))
            gpu.setdefault("observed_free_mb", int(gpu.get("free_mb") or 0))

        active_cpu, active_ram, per_gpu_active = _claim_totals(active)
        pending_cpu, pending_ram, per_gpu_pending = _claim_totals(pending)

        if active_cpu or pending_cpu:
            total_cpu = int(node.get("total_cpu") or node.get("cores") or 0)
            if total_cpu > 0:
                observed_used_cpu = max(0, total_cpu - int(node.get("free_cpu") or 0))
                budget_used_cpu = max(observed_used_cpu, active_cpu) + pending_cpu
                node["free_cpu"] = max(0, total_cpu - budget_used_cpu)

        if active_ram or pending_ram:
            total_ram = int(node.get("total_ram_mb") or 0)
            if total_ram > 0:
                observed_used_ram = max(0, total_ram - int(node.get("free_ram_mb") or 0))
                budget_used_ram = max(observed_used_ram, active_ram) + pending_ram
                node["free_ram_mb"] = max(0, total_ram - budget_used_ram)

        for gpu in node.get("gpus") or []:
            active_claimed = per_gpu_active.get(int(gpu["idx"]), 0)
            pending_claimed = per_gpu_pending.get(int(gpu["idx"]), 0)
            if active_claimed or pending_claimed:
                total = int(gpu.get("total_mb") or 0)
                used = max(int(gpu.get("used_mb") or 0), active_claimed) + pending_claimed
                gpu["used_mb"] = used
                if total > 0:
                    gpu["free_mb"] = min(int(gpu.get("free_mb") or 0), max(0, total - used))
    return nodes
