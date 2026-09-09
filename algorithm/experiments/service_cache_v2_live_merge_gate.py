"""Merge live empty and under-load ETA probes into one v2 service cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache, records_from_summary_file


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


EMPTY_GATE = ARTIFACT_ROOT / "live_eta_service_cache_v2_gate_20260629.json"
MARGINAL_GATES = (
    ARTIFACT_ROOT / "live_marginal_under_load_core_v2_20260629.json",
    ARTIFACT_ROOT / "live_marginal_under_load_jtl311_half_cnn_v2_20260629.json",
    ARTIFACT_ROOT / "live_marginal_under_load_full_mixed_20260629.json",
    ARTIFACT_ROOT / "live_marginal_under_load_mixed_cnn_rl_llm_v5_20260629.json",
    ARTIFACT_ROOT / "live_marginal_under_load_cpu_idle_nodes_20260629.json",
)
EXTRA_GATES = (
    ARTIFACT_ROOT / "live_extra_eta_jtl311_resac_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_jtl311_resac_hopper_walker_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_jtl110_resac_more_envs_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_jtl110gpu2_equiv_retry_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_jtl110gpu2_cnn_p2_noearly_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_node007_llm_p1_20260629.json",
    ARTIFACT_ROOT / "live_extra_eta_node007_resac_ant_p1_20260629.json",
)
EXTRA_CACHES = (
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl311_resac_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl311_resac_hopper_walker_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl110_resac_more_envs_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl110gpu2_equiv_retry_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl110gpu2_cnn_retry2_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_jtl110gpu2_cnn_p2_noearly_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_node007_llm_p1_20260629.json",
    ARTIFACT_ROOT / "service_cache_v2_live_extra_node007_resac_ant_p1_20260629.json",
)

RL_ROWS = (
    {
        "run_id": "cross_node_eta_matrix_live_resac_ant_jtl110_p1_20260629",
        "profile": 1,
        "node_bucket": "jtl110gpu:gpu_3080ti_12gb_dual",
        "hardware_class": "gpu_3080ti_12gb_dual",
        "workload_key": "hybrid_rl_resac_ant",
        "workload_env": "ant",
        "resource_kind": "hybrid_rl",
        "total_units": 80.0,
        "resource_state": "empty",
    },
    {
        "run_id": "live_cycle_jtl110_resac_ant_p2_20260629",
        "profile": 2,
        "node_bucket": "jtl110gpu:gpu_3080ti_12gb_dual",
        "hardware_class": "gpu_3080ti_12gb_dual",
        "workload_key": "hybrid_rl_resac_ant",
        "workload_env": "ant",
        "resource_kind": "hybrid_rl",
        "total_units": 80.0,
        "resource_state": "empty",
        "stable_cycle_units": 5,
        "service_note": "phase-aware train/eval cycle-average stable ETA",
    },
    {
        "run_id": "cross_node_eta_matrix_live_resac_halfcheetah_jtl110_p1_20260629",
        "profile": 1,
        "node_bucket": "jtl110gpu:gpu_3080ti_12gb_dual",
        "hardware_class": "gpu_3080ti_12gb_dual",
        "workload_key": "hybrid_rl_resac_halfcheetah",
        "workload_env": "halfcheetah",
        "resource_kind": "hybrid_rl",
        "total_units": 80.0,
        "resource_state": "empty",
    },
    {
        "run_id": "live_cycle_jtl110_resac_halfcheetah_p2_short20_20260629",
        "profile": 2,
        "node_bucket": "jtl110gpu:gpu_3080ti_12gb_dual",
        "hardware_class": "gpu_3080ti_12gb_dual",
        "workload_key": "hybrid_rl_resac_halfcheetah",
        "workload_env": "halfcheetah",
        "resource_kind": "hybrid_rl",
        "total_units": 20.0,
        "resource_state": "empty",
        "stable_cycle_units": 5,
        "service_note": "phase-aware train/eval cycle-average stable ETA",
    },
    {
        "run_id": "live_cycle_jtl311_resac_halfcheetah_p2_short20_20260629",
        "profile": 2,
        "node_bucket": "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "hardware_class": "gpu_rtx2080_8gb_dual_cpu_fast",
        "workload_key": "hybrid_rl_resac_halfcheetah",
        "workload_env": "halfcheetah",
        "resource_kind": "hybrid_rl",
        "total_units": 20.0,
        "resource_state": "empty",
        "stable_cycle_units": 5,
        "service_note": "phase-aware train/eval cycle-average stable ETA",
    },
)


def build_service_cache_v2_live_merge_gate() -> dict[str, Any]:
    cache = ServiceRateCache()
    rows: list[dict[str, Any]] = []
    rows.extend(_empty_rows(cache))
    rows.extend(_rl_rows(cache))
    rows.extend(_marginal_rows(cache))
    rows.extend(_extra_rows(cache))
    admitted = [row for row in rows if row.get("measurement_valid")]
    boundaries = [row for row in rows if row.get("launched") and not row.get("measurement_valid")]
    missing = [row for row in rows if row.get("missing")]
    return {
        "gate": "service_cache_v2_live_merge_gate",
        "pass": bool(admitted) and not missing,
        "status": "SERVICE_CACHE_V2_LIVE_MERGED" if admitted and not missing else "SERVICE_CACHE_V2_LIVE_OPEN",
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "missing_count": len(missing),
        "rows": rows,
        "admitted_rows": admitted,
        "boundary_rows": boundaries,
        "missing_rows": missing,
        "workload_envs": sorted({str(row.get("workload_env")) for row in admitted if row.get("workload_env")}),
        "resource_states": sorted({str(row.get("resource_state")) for row in admitted if row.get("resource_state")}),
        "hardware_classes": sorted({str(row.get("hardware_class")) for row in admitted if row.get("hardware_class")}),
        "service_cache_snapshot": cache.snapshot(),
        "claim_boundary": (
            "Merged theorem-facing live rows use task-native progress and stable-rate gates. "
            "They cover measured empty and selected under-load states only; they do not "
            "certify all future workloads, all mixed co-location states, or legacy scheduler "
            "hard-rule behavior."
        ),
    }


def _empty_rows(cache: ServiceRateCache) -> list[dict[str, Any]]:
    if not EMPTY_GATE.exists():
        return [{"source_gate": str(EMPTY_GATE), "missing": True, "measurement_valid": False}]
    gate = json.loads(EMPTY_GATE.read_text(encoding="utf-8"))
    rows = []
    for row in gate.get("admitted_rows") or []:
        out = {
            **row,
            "source_gate": str(EMPTY_GATE),
            "service_semantics": "empty_profile_aggregate",
            "launched": True,
            "measurement_valid": True,
        }
        rows.append(out)
    _add_snapshot_to_cache(cache, gate.get("service_cache_snapshot") or {})
    return rows


def _rl_rows(cache: ServiceRateCache) -> list[dict[str, Any]]:
    rows = []
    for spec in RL_ROWS:
        path = RUN_ROOT / str(spec["run_id"]) / "reports" / f"profile_{int(spec['profile'])}_per_gpu_summary.json"
        if not path.exists():
            rows.append({**spec, "path": str(path), "missing": True, "measurement_valid": False})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        valid = bool(summary.get("measurement_valid"))
        row = {
            **spec,
            "path": str(path),
            "missing": False,
            "launched": True,
            "measurement_valid": valid,
            "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
            "running_count": int(summary.get("running_count") or 0),
            "aggregate_stable_rate_unit_s": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
            "mean_stable_rate_unit_s": float(summary.get("mean_stable_rate_unit_s") or 0.0),
            "eta_source": "tqdm/progress",
            "service_semantics": "empty_profile_aggregate",
            "hard_rule_mode": "clean_bench_direct_remote",
            "stable_cycle_units": int(spec.get("stable_cycle_units") or 0),
            "service_note": str(spec.get("service_note") or ""),
        }
        rows.append(row)
        if valid:
            for record in records_from_summary_file(
                path,
                workload_key=str(spec["workload_key"]),
                command_fingerprint=f"live_rl_env:{spec['workload_env']}",
                resource_kind=str(spec["resource_kind"]),
                total_units=float(spec["total_units"]),
                node_bucket=str(spec["node_bucket"]),
                workload_env=str(spec["workload_env"]),
                resource_state=str(spec["resource_state"]),
                eta_source="tqdm/progress",
                hardware_class=str(spec["hardware_class"]),
            ):
                cache.add(record, force_replace=True)
    return rows


def _marginal_rows(cache: ServiceRateCache) -> list[dict[str, Any]]:
    rows = []
    for path in MARGINAL_GATES:
        if not path.exists():
            rows.append({"source_gate": str(path), "missing": True, "measurement_valid": False})
            continue
        gate = json.loads(path.read_text(encoding="utf-8"))
        rows.extend({**row, "source_gate": str(path)} for row in gate.get("rows") or [])
        _add_snapshot_to_cache(cache, gate.get("service_cache_snapshot") or {})
    return rows


def _extra_rows(cache: ServiceRateCache) -> list[dict[str, Any]]:
    rows = []
    for path in EXTRA_GATES:
        if not path.exists():
            continue
        gate = json.loads(path.read_text(encoding="utf-8"))
        rows.extend({**row, "source_gate": str(path)} for row in gate.get("rows") or [])
    for path in EXTRA_CACHES:
        if not path.exists():
            continue
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_full_factorial_eta_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_full_factorial_cpu_eta_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_full_factorial_parallel_probe_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_marginal_under_load_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_marginal_from_design_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    for path in sorted(ARTIFACT_ROOT.glob("service_cache_v2_live_marginal_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _add_snapshot_to_cache(cache, snapshot)
    return rows


def _add_snapshot_to_cache(cache: ServiceRateCache, snapshot: Mapping[str, Any]) -> None:
    for record in list(snapshot.get("records") or []) + list(snapshot.get("capacity_boundaries") or []):
        rec = ProfileRecord.from_snapshot(record)
        if rec.capacity_boundary or (rec.stable_rate_ready and rec.aggregate_rate > 0.0):
            cache.add(rec, force_replace=True)


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Service Cache v2 Live Merge Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        f"- Boundary rows: `{report.get('boundary_count')}`",
        f"- Missing rows: `{report.get('missing_count')}`",
        f"- Workload envs: `{', '.join(report.get('workload_envs') or [])}`",
        f"- Resource states: `{', '.join(report.get('resource_states') or [])}`",
        "",
        "| Workload | Env | Node bucket | State | Semantics | Rate | Valid |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in report.get("admitted_rows") or []:
        rate = row.get("aggregate_stable_rate_unit_s")
        if rate is None:
            rate = row.get("stable_rate_unit_s")
        lines.append(
            f"| `{row.get('workload_key')}` | `{row.get('workload_env')}` | "
            f"`{row.get('node_bucket')}` | `{row.get('resource_state')}` | "
            f"`{row.get('service_semantics')}` | {float(rate or 0.0):.6g} | true |"
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_live_merge_gate_20260629.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_live_merged_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "service_cache_v2_live_merge_gate_20260629.md")
    args = parser.parse_args()
    report = build_service_cache_v2_live_merge_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(json.dumps(report["service_cache_snapshot"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
