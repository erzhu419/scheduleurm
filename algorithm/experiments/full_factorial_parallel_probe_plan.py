"""Build one maximum, node-locked wave of pending full-factorial probes.

The planner never launches work.  It assigns at most one controlled experiment
to each physical node, emits durable row/node lock keys for the orchestrator,
and routes rows only to the dedicated experiment runners.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_DESIGN = (
    ARTIFACT_ROOT
    / "full_factorial_eta_design_rows_schemafixed_pre_native_20260727.csv"
)

T0_GPU_RUNNER = "algorithm.experiments.full_factorial_eta_probe_runner"
T0_CPU_RUNNER = "algorithm.experiments.full_factorial_cpu_eta_probe_runner"
T2_MARGINAL_RUNNER = "algorithm.experiments.live_marginal_under_load_matrix"
T0_STATES = frozenset({"empty", "half_loaded", "full_loaded"})
T2_STATES = frozenset({"high_vram_resident", "cpu_resident", "mixed_colocation"})
ACTIVE_LEASE_STATES = frozenset({"leased", "launching", "running", "recovery_required"})


@dataclass(frozen=True)
class LaneRule:
    lane_id: str
    node: str
    physical_node: str
    node_bucket: str
    resource_kinds: tuple[str, ...]
    hardware_groups: tuple[str, ...]
    availability_nodes: tuple[str, ...]
    notes: str

    @property
    def resource_kind(self) -> str:
        return self.resource_kinds[0] if len(self.resource_kinds) == 1 else "node"


@dataclass(frozen=True)
class AvailabilityDecision:
    allowed: bool
    lane_mode: str
    availability_class: str = ""
    observed_resident_state: str = ""
    external_process_count: int = 0
    reason: str = ""


LANE_RULES: tuple[LaneRule, ...] = (
    LaneRule(
        "gpu_jtl110_representative",
        "jtl110gpu",
        "jtl110gpu",
        "jtl110gpu:gpu_3080ti_12gb_dual",
        ("gpu",),
        ("gpu_3080ti_12gb_dual",),
        ("jtl110gpu",),
        "First 2x3080Ti lane; it may run a different factor row concurrently with jtl110gpu2.",
    ),
    LaneRule(
        "gpu_jtl110_spillover",
        "jtl110gpu2",
        "jtl110gpu2",
        "jtl110gpu2:gpu_3080ti_12gb_dual",
        ("gpu",),
        ("gpu_3080ti_12gb_dual",),
        ("jtl110gpu2",),
        "Second homogeneous 2x3080Ti lane; duplicate factor rows are locked out.",
    ),
    LaneRule(
        "node_jtl311linux",
        "jtl311linux",
        "jtl311linux",
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        ("gpu", "cpu"),
        ("gpu_rtx2080_8gb_dual_cpu_fast",),
        ("jtl311linux",),
        "One node-level lane shared by jtl311linux CPU and GPU probes.",
    ),
    LaneRule(
        "node_node007",
        "node007",
        "node007",
        "node007:gpu_node007_4x12gb",
        ("gpu", "cpu"),
        ("gpu_node007_4x11gb",),
        ("node007", "node007-direct"),
        "One node-level lane shared by all node007 CPU/GPU resources.",
    ),
    *tuple(
        LaneRule(
            f"cpu_hpc_192c_{node}",
            node,
            node,
            f"{node}:cpu_hpc_192c",
            ("cpu",),
            ("cpu_hpc_192c",),
            (node,),
            "One of six homogeneous 192-core CPU lanes; each lane receives a distinct factor row.",
        )
        for node in ("node001", "node002", "node003", "node004", "node005", "node006")
    ),
)


def build_parallel_probe_plan(
    *,
    design_csv: Path = DEFAULT_DESIGN,
    max_rows_per_lane: int = 1,
    include_tiers: set[str] | None = None,
    include_states: set[str] | None = None,
    pending_statuses: set[str] | None = None,
    busy_nodes: set[str] | None = None,
    busy_row_ids: set[str] | None = None,
    availability_json: Path | None = None,
    lease_state_json: Path | None = None,
    now: float | None = None,
    run_prefix: str = "ff_parallel_eta_20260701",
    assignment_dir: Path | None = None,
) -> dict[str, Any]:
    assignment_dir = assignment_dir or (design_csv.parent / "parallel_probe_assignments_20260701")
    rows = list(_read_rows(design_csv))
    availability = _read_availability(availability_json)
    statuses = (
        pending_statuses
        if pending_statuses is not None
        else {"pending_probe", "partial_pending_probe"}
    )
    pending = [row for row in rows if str(row.get("status") or "") in statuses]
    if include_tiers:
        pending = [row for row in pending if str(row.get("tier") or "") in include_tiers]
    if include_states:
        pending = [row for row in pending if str(row.get("resource_state") or "") in include_states]

    lease_rows, lease_nodes, lease_factors = _read_active_lease_reservations(
        lease_state_json,
        now=time.time() if now is None else float(now),
    )
    supplied_busy_rows = {str(x).strip() for x in (busy_row_ids or set()) if str(x).strip()}
    reserved_row_ids = supplied_busy_rows | lease_rows
    reserved_factor_keys = set(lease_factors)
    reserved_factor_keys.update(
        _factor_key(row)
        for row in rows
        if _row_identity(row) in reserved_row_ids
    )
    supplied_busy_nodes = {_canonical_node(str(x).strip()) for x in (busy_nodes or set()) if str(x).strip()}
    reserved_nodes = supplied_busy_nodes | {_canonical_node(node) for node in lease_nodes}
    rows_per_lane = 1 if int(max_rows_per_lane) > 0 else 0

    lane_candidates: dict[str, list[tuple[Mapping[str, str], AvailabilityDecision]]] = {}
    lane_skip_reasons: dict[str, str] = {}
    for lane in LANE_RULES:
        if _canonical_node(lane.physical_node) in reserved_nodes:
            lane_candidates[lane.lane_id] = []
            lane_skip_reasons[lane.lane_id] = "node_busy_or_leased"
            continue
        if not rows_per_lane:
            lane_candidates[lane.lane_id] = []
            lane_skip_reasons[lane.lane_id] = "wave_disabled"
            continue
        by_factor: dict[tuple[str, ...], tuple[Mapping[str, str], AvailabilityDecision]] = {}
        for row in pending:
            if _row_identity(row) in reserved_row_ids:
                continue
            factor = _factor_key(row)
            if factor in reserved_factor_keys or lane not in _candidate_lanes(row):
                continue
            if not _runner_ready(row):
                continue
            decision = _availability_decision(row, lane, availability)
            if not decision.allowed:
                continue
            candidate = (row, decision)
            current = by_factor.get(factor)
            if current is None or _candidate_priority(candidate) < _candidate_priority(current):
                by_factor[factor] = candidate
        lane_candidates[lane.lane_id] = sorted(by_factor.values(), key=_candidate_priority)

    assignments = _maximum_lane_matching(lane_candidates)
    lanes: list[dict[str, Any]] = []
    for lane in LANE_RULES:
        chosen = [assignments[lane.lane_id]] if lane.lane_id in assignments else []
        skip_reason = lane_skip_reasons.get(lane.lane_id, "")
        if not chosen and not skip_reason:
            skip_reason = (
                "factor_reserved_by_maximum_wave"
                if lane_candidates.get(lane.lane_id)
                else "no_runnable_pending_row"
            )
        lanes.append(
            _lane_report(
                lane,
                chosen,
                run_prefix=run_prefix,
                assignment_dir=assignment_dir,
                skip_reason=skip_reason,
            )
        )

    selected_raw = [row for row, _decision in assignments.values()]
    selected_row_ids = {_row_identity(row) for row in selected_raw}
    blocked: list[dict[str, Any]] = []
    runnable_not_selected: list[dict[str, Any]] = []
    availability_blocked: list[dict[str, Any]] = []
    runner_not_ready: list[dict[str, Any]] = []
    for row in pending:
        row_id = _row_identity(row)
        if row_id in selected_row_ids or row_id in reserved_row_ids:
            continue
        candidate_lanes = _candidate_lanes(row)
        if not candidate_lanes:
            blocked.append(_blocked_row(row))
            # Preserve the legacy diagnostic bucket for rows with no configured
            # physical lane while making harness-ready T2 rows explicit.
            runner_not_ready.append(_pending_row(row, blocker="no_configured_physical_lane"))
            continue
        if not _runner_ready(row):
            runner_not_ready.append(_pending_row(row, blocker="runner_not_ready_for_state"))
            continue
        decisions = [_availability_decision(row, lane, availability) for lane in candidate_lanes]
        if any(decision.allowed for decision in decisions):
            runnable_not_selected.append(_pending_row(row))
        else:
            availability_blocked.append(_availability_blocked_row(row, candidate_lanes, decisions))

    selected = [row for lane in lanes for row in lane["rows"]]
    wave_id = _wave_id(design_csv, selected)
    report = {
        "gate": "full_factorial_parallel_probe_plan",
        "status": "PARALLEL_PROBE_PLAN_READY",
        "wave_id": wave_id,
        "design_csv": str(design_csv),
        "pending_statuses": sorted(statuses),
        "requested_max_rows_per_lane": int(max_rows_per_lane),
        "max_rows_per_lane": rows_per_lane,
        "include_tiers": sorted(include_tiers or []),
        "include_states": sorted(include_states or []),
        "run_prefix": run_prefix,
        "busy_nodes": sorted(supplied_busy_nodes),
        "busy_row_ids": sorted(supplied_busy_rows),
        "availability_json": str(availability_json) if availability_json else "",
        "lease_state_json": str(lease_state_json) if lease_state_json else "",
        "active_lease_row_ids": sorted(lease_rows),
        "active_lease_nodes": sorted(lease_nodes),
        "assignment_dir": str(assignment_dir),
        "lane_count": len(lanes),
        "wave_capacity": sum(
            1 for lane in LANE_RULES if _canonical_node(lane.physical_node) not in reserved_nodes
        ),
        "selected_row_count": len(selected),
        "pending_row_count": len(pending),
        "observed_resident_lane_count": sum(
            1 for lane in lanes if lane.get("lane_mode") == "observed-resident"
        ),
        "lanes": lanes,
        "pending_rows": [_pending_row(row) for row in pending],
        "selected_rows": selected,
        "runnable_not_selected": runnable_not_selected,
        "availability_blocked_rows": availability_blocked,
        "runner_not_ready_rows": runner_not_ready,
        "blocked_rows": blocked,
        "resource_lock_policy": (
            "Each wave has at most one row per physical node.  jtl311linux and node007 use one shared "
            "CPU/GPU node lock; node001-node006 each provide one CPU lane.  Homogeneous nodes may run "
            "different rows concurrently, but the canonical factor lock prevents duplicate rows."
        ),
        "availability_policy": (
            "Explicit external load may be used only by a compatible T2 row and is labelled as an "
            "observed-resident lane.  It is never interpreted as an empty or T0 lane."
        ),
        "lease_policy": (
            "Every selected row carries row, factor, and node lock keys.  The parallel orchestrator "
            "must acquire durable leases before an explicitly authorized controlled launch."
        ),
        "launch_policy": (
            "Planning does not launch work.  Only the three controlled full-factorial experiment "
            "modules are emitted; ordinary scheduler/user-task launch paths are not used."
        ),
        "claim_boundary": (
            "This is a launch plan, not a measurement artifact.  Rows enter theorem-facing service "
            "cache only after their runner writes task-native tqdm/progress stable-rate summaries."
        ),
        "automatic_launch": False,
    }
    return report


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _runner_route(row: Mapping[str, str]) -> str | None:
    tier = str(row.get("tier") or "")
    state = str(row.get("resource_state") or "")
    kind = str(row.get("resource_kind") or "")
    if tier == "T2_load_state" and state in T2_STATES:
        return T2_MARGINAL_RUNNER
    if state in T0_STATES and kind == "cpu":
        return T0_CPU_RUNNER
    if state in T0_STATES and kind == "gpu":
        return T0_GPU_RUNNER
    return None


def _runner_ready(row: Mapping[str, str]) -> bool:
    return _runner_route(row) is not None


def _covered_by_lane(row: Mapping[str, str]) -> bool:
    return bool(_candidate_lanes(row))


def _candidate_lanes(row: Mapping[str, str]) -> tuple[LaneRule, ...]:
    kind = str(row.get("resource_kind") or "")
    hardware_group = _canonical_hardware_group(str(row.get("hardware_group") or ""))
    return tuple(
        lane
        for lane in LANE_RULES
        if kind in lane.resource_kinds
        and hardware_group in {_canonical_hardware_group(group) for group in lane.hardware_groups}
    )


def _read_availability(path: Path | None) -> dict[str, Mapping[str, Any]] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, Mapping[str, Any]] = {}
    for row in data.get("nodes") or []:
        node = str(row.get("node") or "")
        if node:
            parsed = dict(row)
            parsed.update(row.get("parsed") or {})
            out[node] = parsed
            out.setdefault(_canonical_node(node), parsed)
    return out


def _availability_allows(
    row: Mapping[str, str],
    lane: LaneRule,
    availability: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    return _availability_decision(row, lane, availability).allowed


def _availability_decision(
    row: Mapping[str, str],
    lane: LaneRule,
    availability: Mapping[str, Mapping[str, Any]] | None,
) -> AvailabilityDecision:
    if availability is None:
        return AvailabilityDecision(True, "unverified", reason="availability_not_supplied")
    parsed = next((availability.get(node) for node in lane.availability_nodes if availability.get(node)), None)
    if not parsed:
        return AvailabilityDecision(False, "blocked", reason="missing_availability_audit")

    availability_class = str(parsed.get("availability_class") or "")
    cpu_explicit = "cpu_busy" in parsed
    gpu_explicit = "gpu_busy" in parsed
    cpu_busy = _as_bool(parsed.get("cpu_busy"))
    gpu_busy = _as_bool(parsed.get("gpu_busy"))
    external_count = _external_process_count(parsed)
    has_external_load = (
        external_count > 0
        or bool(parsed.get("external_busy_processes"))
        or _as_bool(parsed.get("observed_external_load"))
    )

    # Full audit records carry explicit CPU/GPU busy flags.  Those flags are
    # authoritative for the physical-node non-overlap rule.  Older compact
    # audits without the flags retain their explicit safe_for_* decisions.
    if (cpu_explicit or gpu_explicit) and (cpu_busy or gpu_busy):
        if has_external_load and _observed_resident_matches(row, cpu_busy=cpu_busy, gpu_busy=gpu_busy):
            return AvailabilityDecision(
                True,
                "observed-resident",
                availability_class=availability_class,
                observed_resident_state=_observed_resident_state(cpu_busy=cpu_busy, gpu_busy=gpu_busy),
                external_process_count=external_count,
                reason="compatible_external_load_observed",
            )
        return AvailabilityDecision(
            False,
            "blocked",
            availability_class=availability_class,
            external_process_count=external_count,
            reason="physical_node_cpu_or_gpu_busy",
        )

    if has_external_load:
        inferred_cpu, inferred_gpu = _busy_kinds_from_class(availability_class)
        if _observed_resident_matches(row, cpu_busy=inferred_cpu, gpu_busy=inferred_gpu):
            return AvailabilityDecision(
                True,
                "observed-resident",
                availability_class=availability_class,
                observed_resident_state=_observed_resident_state(cpu_busy=inferred_cpu, gpu_busy=inferred_gpu),
                external_process_count=external_count,
                reason="compatible_external_load_observed",
            )
        return AvailabilityDecision(
            False,
            "blocked",
            availability_class=availability_class,
            external_process_count=external_count,
            reason="external_load_not_compatible_with_factor_row",
        )

    safety_field = _required_safety_field(row)
    if _as_bool(parsed.get(safety_field)) or (
        availability_class == "clean_idle" and safety_field not in parsed
    ):
        return AvailabilityDecision(
            True,
            "clean",
            availability_class=availability_class,
            reason=f"{safety_field}_true",
        )
    return AvailabilityDecision(
        False,
        "blocked",
        availability_class=availability_class,
        reason=f"{safety_field}_false",
    )


def _required_safety_field(row: Mapping[str, str]) -> str:
    kind = str(row.get("resource_kind") or "")
    state = str(row.get("resource_state") or "")
    if kind == "cpu":
        return "safe_for_cpu_probe"
    if state in {"cpu_resident", "mixed_colocation"} or _row_requires_hybrid_clean(row):
        return "safe_for_hybrid_probe"
    return "safe_for_gpu_only_probe"


def _observed_resident_matches(row: Mapping[str, str], *, cpu_busy: bool, gpu_busy: bool) -> bool:
    if str(row.get("tier") or "") != "T2_load_state":
        return False
    state = str(row.get("resource_state") or "")
    resident_mix = str(row.get("resident_mix") or "")
    kind = str(row.get("resource_kind") or "")
    if state == "cpu_resident":
        return cpu_busy and not gpu_busy
    if state == "high_vram_resident":
        return gpu_busy and not cpu_busy
    if state == "mixed_colocation":
        if resident_mix == "cpu_plus_gpu_target":
            return cpu_busy and kind == "gpu"
        return gpu_busy or (cpu_busy and kind == "gpu")
    return False


def _observed_resident_state(*, cpu_busy: bool, gpu_busy: bool) -> str:
    if cpu_busy and gpu_busy:
        return "mixed_colocation"
    if gpu_busy:
        return "high_vram_resident"
    if cpu_busy:
        return "cpu_resident"
    return ""


def _busy_kinds_from_class(availability_class: str) -> tuple[bool, bool]:
    value = str(availability_class or "").lower()
    return ("cpu_busy" in value or "cpu_and_gpu" in value, "gpu_busy" in value or "gpu_and_cpu" in value)


def _external_process_count(parsed: Mapping[str, Any]) -> int:
    try:
        return max(0, int(parsed.get("external_busy_process_count") or 0))
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _row_requires_hybrid_clean(row: Mapping[str, str]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("workload_key", "workload_env", "resource_kind", "resource_state", "resident_mix")
    )
    return any(token in text for token in ("hybrid_rl", "resac", "bapr", "ant", "halfcheetah", "hopper", "walker2d"))


def _availability_blocked_row(
    row: Mapping[str, str],
    lanes: Iterable[LaneRule],
    decisions: Iterable[AvailabilityDecision],
) -> dict[str, Any]:
    out = _pending_row(row)
    reasons = [
        f"{lane.physical_node}:{decision.availability_class or 'missing_audit'}:{decision.reason}"
        for lane, decision in zip(lanes, decisions)
    ]
    out["blocker"] = "availability_not_clean"
    out["availability_reasons"] = ";".join(reasons)
    return out


def _candidate_priority(
    candidate: tuple[Mapping[str, str], AvailabilityDecision],
) -> tuple[int, int, str, str, int, str]:
    row, decision = candidate
    mode_rank = {"clean": 0, "unverified": 1, "observed-resident": 2}.get(decision.lane_mode, 9)
    return (*_priority_key(row), mode_rank, _row_identity(row))


def _maximum_lane_matching(
    candidates: Mapping[str, list[tuple[Mapping[str, str], AvailabilityDecision]]],
) -> dict[str, tuple[Mapping[str, str], AvailabilityDecision]]:
    """Return a deterministic maximum-cardinality lane/factor matching."""

    factor_owner: dict[tuple[str, ...], str] = {}
    assignment: dict[str, tuple[Mapping[str, str], AvailabilityDecision]] = {}
    lane_order = {lane.lane_id: index for index, lane in enumerate(LANE_RULES)}

    def augment(lane_id: str, seen_factors: set[tuple[str, ...]], seen_lanes: set[str]) -> bool:
        if lane_id in seen_lanes:
            return False
        seen_lanes.add(lane_id)
        for candidate in candidates.get(lane_id, []):
            factor = _factor_key(candidate[0])
            if factor in seen_factors:
                continue
            seen_factors.add(factor)
            other_lane = factor_owner.get(factor)
            if other_lane is None or augment(other_lane, seen_factors, seen_lanes):
                factor_owner[factor] = lane_id
                assignment[lane_id] = candidate
                return True
        return False

    ordered_lanes = sorted(
        candidates,
        key=lambda lane_id: (len(candidates.get(lane_id, [])), lane_order.get(lane_id, 999)),
    )
    for lane_id in ordered_lanes:
        augment(lane_id, set(), set())
    return assignment


def _read_active_lease_reservations(
    path: Path | None,
    *,
    now: float,
) -> tuple[set[str], set[str], set[tuple[str, ...]]]:
    if not path or not Path(path).exists():
        return set(), set(), set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    row_ids: set[str] = set()
    nodes: set[str] = set()
    factors: set[tuple[str, ...]] = set()
    for lease in data.get("leases") or []:
        status = str(lease.get("status") or "")
        if status not in ACTIVE_LEASE_STATES:
            continue
        if status == "leased" and float(lease.get("expires_at") or 0.0) <= now:
            continue
        row_id = str(lease.get("row_id") or "")
        node = str(lease.get("node_lock") or "")
        factor_values = lease.get("factor_values") or []
        if row_id:
            row_ids.add(row_id)
        if node:
            nodes.add(_canonical_node(node))
        if isinstance(factor_values, list) and factor_values:
            factors.add(tuple(str(value) for value in factor_values))
    return row_ids, nodes, factors


def _priority_key(row: Mapping[str, str]) -> tuple[int, int, str, str]:
    tier_rank = {
        "T0_core_curve": 0,
        "T1_equivalence": 1,
        "T2_load_state": 2,
        "T3_admission": 3,
    }.get(str(row.get("tier") or ""), 9)
    state_rank = {
        "empty": 0,
        "half_loaded": 1,
        "full_loaded": 2,
        "high_vram_resident": 3,
        "cpu_resident": 4,
        "mixed_colocation": 5,
    }.get(str(row.get("resource_state") or ""), 9)
    return (tier_rank, state_rank, str(row.get("quadrant") or ""), str(row.get("workload_key") or ""))


def _factor_key(row: Mapping[str, str]) -> tuple[str, ...]:
    hardware_group = _canonical_hardware_group(str(row.get("hardware_group") or ""))
    return (
        hardware_group,
        str(row.get("resource_kind") or ""),
        str(row.get("resource_state") or ""),
        str(row.get("resident_mix") or ""),
        str(row.get("workload_key") or ""),
        str(row.get("workload_env") or ""),
    )


def _canonical_hardware_group(hardware_group: str) -> str:
    value = str(hardware_group or "")
    return value.removesuffix("_equiv") if value.endswith("_equiv") else value


def _canonical_node(node: str) -> str:
    value = str(node or "").strip()
    return "node007" if value == "node007-direct" else value


def _row_identity(row: Mapping[str, Any]) -> str:
    row_id = str(row.get("row_id") or "").strip()
    return row_id or "|".join(_factor_key(row))


def _lock_key(prefix: str, values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _row_lock_key(row: Mapping[str, Any]) -> str:
    return _lock_key("row", (_row_identity(row),))


def _factor_lock_key(row: Mapping[str, Any]) -> str:
    return _lock_key("factor", _factor_key(row))


def _wave_id(design_csv: Path, selected_rows: Iterable[Mapping[str, Any]]) -> str:
    assignments = sorted(
        (
            str(row.get("assignment_lane_id") or ""),
            str(row.get("node_lock") or row.get("physical_node") or ""),
            _row_identity(row),
        )
        for row in selected_rows
    )
    payload = json.dumps(
        {"design_csv": str(Path(design_csv).resolve()), "assignments": assignments},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"wave-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _lane_report(
    lane: LaneRule,
    assignments: list[tuple[Mapping[str, str], AvailabilityDecision]],
    *,
    run_prefix: str,
    assignment_dir: Path,
    skip_reason: str = "",
) -> dict[str, Any]:
    assigned_rows = [_assigned_row(row, lane, decision) for row, decision in assignments]
    assignment_paths = [
        _write_assignment_csv(assignment_dir, row, run_prefix=run_prefix, lane_id=lane.lane_id)
        for row in assigned_rows
    ]
    commands = [
        _command_for(row, run_prefix=run_prefix, lane_id=lane.lane_id, assignment_csv=path)
        for row, path in zip(assigned_rows, assignment_paths)
    ]
    decision = assignments[0][1] if assignments else None
    selected_kind = str(assigned_rows[0].get("resource_kind") or "") if assigned_rows else lane.resource_kind
    return {
        "lane_id": lane.lane_id,
        "assigned_node": lane.node,
        "physical_node": lane.physical_node,
        "node_lock": _canonical_node(lane.physical_node),
        "resource_lock": f"{_canonical_node(lane.physical_node)}:node",
        "node_bucket": lane.node_bucket,
        "hardware_groups": list(lane.hardware_groups),
        "resource_kinds": list(lane.resource_kinds),
        "resource_kind": selected_kind,
        "lane_mode": decision.lane_mode if decision else "idle",
        "observed_resident": bool(decision and decision.lane_mode == "observed-resident"),
        "availability_class": decision.availability_class if decision else "",
        "observed_resident_state": decision.observed_resident_state if decision else "",
        "selected_count": len(assignments),
        "rows": [_pending_row(row) for row in assigned_rows],
        "assignment_csvs": [str(path) for path in assignment_paths],
        "commands": commands,
        "command_argvs": [shlex.split(command) for command in commands],
        "notes": lane.notes,
        "skip_reason": skip_reason,
    }


def _assigned_row(
    row: Mapping[str, str],
    lane: LaneRule,
    decision: AvailabilityDecision,
) -> dict[str, Any]:
    out = dict(row)
    out["logical_node"] = str(row.get("node") or "")
    out["logical_node_bucket"] = str(row.get("node_bucket") or "")
    out["assigned_node"] = lane.node
    out["assigned_node_bucket"] = _assigned_node_bucket(lane, row)
    out["assignment_lane_id"] = lane.lane_id
    out["physical_node"] = _canonical_node(lane.physical_node)
    out["node_lock"] = _canonical_node(lane.physical_node)
    out["node"] = lane.node
    out["node_bucket"] = out["assigned_node_bucket"]
    out["equivalent_nodes"] = ""
    out["runner_route"] = _runner_route(row) or ""
    out["lane_mode"] = decision.lane_mode
    out["observed_resident"] = decision.lane_mode == "observed-resident"
    out["availability_class"] = decision.availability_class
    out["observed_resident_state"] = decision.observed_resident_state
    out["observed_external_process_count"] = decision.external_process_count
    out["availability_reason"] = decision.reason
    out["row_lock_key"] = _row_lock_key(row)
    out["factor_lock_key"] = _factor_lock_key(row)
    out["factor_values"] = json.dumps(list(_factor_key(row)), separators=(",", ":"))
    out["notes"] = (
        f"{row.get('notes') or ''} Assigned by full_factorial_parallel_probe_plan "
        f"from logical node {row.get('node')} to physical node {lane.physical_node}."
    ).strip()
    return out


def _assigned_node_bucket(lane: LaneRule, row: Mapping[str, str]) -> str:
    group = _canonical_hardware_group(str(row.get("hardware_group") or ""))
    primary_group = _canonical_hardware_group(lane.hardware_groups[0])
    return lane.node_bucket if group == primary_group else f"{lane.node}:{group}"


def _write_assignment_csv(assignment_dir: Path, row: Mapping[str, Any], *, run_prefix: str, lane_id: str) -> Path:
    assignment_dir.mkdir(parents=True, exist_ok=True)
    path = assignment_dir / f"{_safe_id(run_prefix + '_' + lane_id + '_' + str(row.get('row_id') or 'row'))}.csv"
    fieldnames = sorted(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(dict(row))
    return path


def _command_for(row: Mapping[str, str], *, run_prefix: str, lane_id: str, assignment_csv: Path) -> str:
    module = _runner_route(row)
    if not module:
        raise ValueError(f"no controlled runner for row {_row_identity(row)}")
    suffix = _safe_id(
        f"{run_prefix}_{lane_id}_{row.get('node')}_"
        f"{row.get('resource_state')}_{row.get('workload_key')}"
    )
    output_stem = f"full_factorial_parallel_probe_{suffix}"
    if module == T2_MARGINAL_RUNNER:
        args = [
            "python3",
            "-m",
            module,
            "--allow-launch",
            "--from-design",
            "--design-csv",
            str(assignment_csv),
            "--assigned-node",
            str(row.get("node") or ""),
            "--workloads",
            str(row.get("workload_key") or ""),
            "--states",
            str(row.get("resource_state") or ""),
            "--resident-mixes",
            str(row.get("resident_mix") or ""),
            "--max-rows",
            "1",
            "--run-id",
            suffix,
            "--output",
            f"md/experiment_artifacts/{output_stem}.json",
            "--cache-output",
            f"md/experiment_artifacts/service_cache_v2_{output_stem}.json",
            "--markdown-output",
            f"md/{output_stem}.md",
        ]
    else:
        args = [
            "python3",
            "-m",
            module,
            "--allow-launch",
            "--design-csv",
            str(assignment_csv),
            "--tier",
            str(row.get("tier") or ""),
            "--resource-state",
            str(row.get("resource_state") or ""),
            "--nodes",
            str(row.get("node") or ""),
            "--workloads",
            str(row.get("workload_key") or ""),
            "--max-rows",
            "1",
            "--run-prefix",
            suffix,
            "--output",
            f"md/experiment_artifacts/{output_stem}.json",
            "--cache-output",
            f"md/experiment_artifacts/service_cache_v2_{output_stem}.json",
            "--markdown-output",
            f"md/{output_stem}.md",
        ]
    return " ".join(shlex.quote(x) for x in args)


def _safe_id(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    value = "".join(out).strip("_")
    if len(value) <= 180:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"{value[:150].rstrip('_')}_{digest}"


def _pending_row(row: Mapping[str, Any], *, blocker: str = "") -> dict[str, Any]:
    keys = (
        "row_id",
        "tier",
        "quadrant",
        "node",
        "node_bucket",
        "hardware_group",
        "resource_kind",
        "resource_state",
        "resident_mix",
        "logical_node",
        "logical_node_bucket",
        "assigned_node",
        "assigned_node_bucket",
        "assignment_lane_id",
        "physical_node",
        "node_lock",
        "workload_key",
        "workload_env",
        "missing_profiles",
        "measured_profiles",
        "boundary_closed_profiles",
        "status",
        "runner",
        "runner_route",
        "lane_mode",
        "availability_class",
        "observed_resident_state",
        "availability_reason",
    )
    out = {key: str(row.get(key) or "") for key in keys}
    out["runner_route"] = out["runner_route"] or (_runner_route(row) or "")
    out["observed_resident"] = _as_bool(row.get("observed_resident"))
    try:
        out["observed_external_process_count"] = int(row.get("observed_external_process_count") or 0)
    except (TypeError, ValueError):
        out["observed_external_process_count"] = 0
    out["row_lock_key"] = str(row.get("row_lock_key") or _row_lock_key(row))
    out["factor_lock_key"] = str(row.get("factor_lock_key") or _factor_lock_key(row))
    factor_values = row.get("factor_values")
    if isinstance(factor_values, str):
        try:
            factor_values = json.loads(factor_values)
        except json.JSONDecodeError:
            factor_values = None
    out["factor_values"] = (
        [str(value) for value in factor_values]
        if isinstance(factor_values, list)
        else list(_factor_key(row))
    )
    out["eligible_nodes"] = ";".join(_eligible_nodes(row))
    if blocker:
        out["blocker"] = blocker
    return out


def _eligible_nodes(row: Mapping[str, str]) -> tuple[str, ...]:
    """Physical nodes that may measure this logical design row.

    Homogeneous nodes are listed as candidates for scheduling efficiency, but
    duplicate protection still happens through ``_factor_key``.  That means
    jtl110gpu and jtl110gpu2 can both be eligible for a 3080Ti row, while the
    planner will not select the same hardware/workload/load-state factor twice
    in one wave or while it is listed in ``busy_row_ids``.
    """

    candidates = [lane.node for lane in _candidate_lanes(row)]
    return tuple(dict.fromkeys(candidates))


def _blocked_row(row: Mapping[str, str]) -> dict[str, Any]:
    out = _pending_row(row)
    out["blocker"] = "no_parallel_lane_rule" if not _covered_by_lane(row) else "runner_not_ready_for_state"
    return out


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full-Factorial Parallel Probe Plan",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pending rows considered: `{report.get('pending_row_count')}`",
        f"- Selected rows: `{report.get('selected_row_count')}`",
        f"- Max rows per lane: `{report.get('max_rows_per_lane')}`",
        f"- Availability audit: `{report.get('availability_json') or 'not provided'}`",
        "",
        "## Resource Lock Policy",
        "",
        str(report.get("resource_lock_policy") or ""),
        "",
        "## Lanes",
        "",
        "| Lane | Node lock | Mode | Selected | Commands |",
        "|---|---|---|---:|---:|",
    ]
    for lane in report.get("lanes") or []:
        lines.append(
            f"| `{lane.get('lane_id')}` | `{lane.get('node_lock')}` | `{lane.get('lane_mode')}` | "
            f"{lane.get('selected_count')} | {len(lane.get('commands') or [])} |"
        )
    lines.extend(
        [
            "",
            "## Selected Rows",
            "",
            "| Lane | Workload | Node | State | Missing profiles | Command |",
            "|---|---|---|---|---|---|",
        ]
    )
    for lane in report.get("lanes") or []:
        commands = lane.get("commands") or []
        for row, cmd in zip(lane.get("rows") or [], commands):
            lines.append(
                f"| `{lane.get('lane_id')}` | `{row.get('workload_key')}` | `{row.get('node')}` | "
                f"`{row.get('resource_state')}` | `{row.get('missing_profiles')}` | `{cmd}` |"
            )
    lines.extend(["", "## Claim Boundary", "", str(report.get("claim_boundary") or ""), ""])
    runner_not_ready = report.get("runner_not_ready_rows") or []
    if runner_not_ready:
        lines.extend([
            "",
            "## Pending Rows Without A Runnable Lane",
            "",
            "| Workload | State | Eligible nodes | Reason |",
            "|---|---|---|---|",
        ])
        for row in runner_not_ready[:80]:
            lines.append(
                f"| `{row.get('workload_key')}` | `{row.get('resource_state')}/{row.get('resident_mix')}` | "
                f"`{row.get('eligible_nodes')}` | `{row.get('blocker') or 'runner_not_ready_for_state'}` |"
            )
    availability_blocked = report.get("availability_blocked_rows") or []
    if availability_blocked:
        lines.extend([
            "",
            "## Pending Rows Blocked By Current Availability",
            "",
            "| Workload | State | Eligible nodes | Availability reasons |",
            "|---|---|---|---|",
        ])
        for row in availability_blocked[:80]:
            lines.append(
                f"| `{row.get('workload_key')}` | `{row.get('resource_state')}/{row.get('resident_mix')}` | "
                f"`{row.get('eligible_nodes')}` | `{row.get('availability_reasons')}` |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-csv", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--max-rows-per-lane", type=int, default=1)
    parser.add_argument("--tiers", default="")
    parser.add_argument("--states", default="")
    parser.add_argument("--pending-statuses", default="pending_probe")
    parser.add_argument("--busy-nodes", default="")
    parser.add_argument("--busy-row-ids", default="")
    parser.add_argument("--availability-json", type=Path, default=None)
    parser.add_argument("--lease-state-json", type=Path, default=None)
    parser.add_argument("--run-prefix", default="ff_parallel_eta_20260701")
    parser.add_argument("--assignment-dir", type=Path, default=ARTIFACT_ROOT / "parallel_probe_assignments_20260701")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "full_factorial_parallel_probe_plan_20260701.json",
    )
    parser.add_argument(
        "--selected-csv-output",
        type=Path,
        default=ARTIFACT_ROOT / "full_factorial_parallel_probe_plan_selected_20260701.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "md" / "full_factorial_parallel_probe_plan_20260701.md",
    )
    args = parser.parse_args()
    report = build_parallel_probe_plan(
        design_csv=args.design_csv,
        max_rows_per_lane=args.max_rows_per_lane,
        include_tiers={x.strip() for x in args.tiers.split(",") if x.strip()} or None,
        include_states={x.strip() for x in args.states.split(",") if x.strip()} or None,
        pending_statuses={x.strip() for x in args.pending_statuses.split(",") if x.strip()},
        busy_nodes={x.strip() for x in args.busy_nodes.split(",") if x.strip()} or None,
        busy_row_ids={x.strip() for x in args.busy_row_ids.split(";;") if x.strip()} or None,
        availability_json=args.availability_json,
        lease_state_json=args.lease_state_json,
        run_prefix=str(args.run_prefix),
        assignment_dir=args.assignment_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.selected_csv_output, list(report["selected_rows"]))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
