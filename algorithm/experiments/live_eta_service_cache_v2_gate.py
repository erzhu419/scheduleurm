"""Build a v2 service-cache certificate from live task-native ETA probes."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache, records_from_summary_file


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


@dataclass(frozen=True)
class LiveSummarySpec:
    run_id: str
    profile: int
    node_bucket: str
    hardware_class: str
    workload_key: str = "gpu_cnn_torch_resnet50"
    workload_env: str = "cnn"
    resource_kind: str = "gpu_cnn"
    total_units: float = 80.0
    resource_state: str = "empty"

    @property
    def path(self) -> Path:
        return RUN_ROOT / self.run_id / "reports" / f"profile_{self.profile}_per_gpu_summary.json"


LIVE_SUMMARIES = (
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl110_cnn_20260629_gpu_3080ti_12gb_dual_cnn_resnet50",
        1,
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "gpu_3080ti_12gb_dual",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl110_cnn_20260629_gpu_3080ti_12gb_dual_cnn_resnet50",
        2,
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "gpu_3080ti_12gb_dual",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl110_cnn_p4_long_20260629",
        4,
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "gpu_3080ti_12gb_dual",
        total_units=480.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl311_cnn_20260629_gpu_rtx2080_8gb_dual_cpu_fast_cnn_resnet50",
        1,
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "gpu_rtx2080_8gb_dual_cpu_fast",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl311_cnn_20260629_gpu_rtx2080_8gb_dual_cpu_fast_cnn_resnet50",
        2,
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "gpu_rtx2080_8gb_dual_cpu_fast",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl311_cnn_p4_long_20260629",
        4,
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "gpu_rtx2080_8gb_dual_cpu_fast",
        total_units=480.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_node007_cnn_rerun_p4_20260629_gpu_node007_4x12gb_cnn_resnet50",
        1,
        "node007:gpu_node007_4x12gb",
        "gpu_node007_4x12gb",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_node007_cnn_rerun_p4_20260629_gpu_node007_4x12gb_cnn_resnet50",
        2,
        "node007:gpu_node007_4x12gb",
        "gpu_node007_4x12gb",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_node007_cnn_rerun_p4_20260629_gpu_node007_4x12gb_cnn_resnet50",
        4,
        "node007:gpu_node007_4x12gb",
        "gpu_node007_4x12gb",
        total_units=240.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl110_llm_20260629_gpu_3080ti_12gb_dual_llm_distilgpt2",
        1,
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "gpu_3080ti_12gb_dual",
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        total_units=96.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_noearly_jtl110_llm_20260629_gpu_3080ti_12gb_dual_llm_distilgpt2",
        2,
        "jtl110gpu:gpu_3080ti_12gb_dual",
        "gpu_3080ti_12gb_dual",
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        total_units=96.0,
    ),
    LiveSummarySpec(
        "cross_node_eta_matrix_live_llm_jtl311_cachefixed_20260629_gpu_rtx2080_8gb_dual_cpu_fast_llm_distilgpt2",
        1,
        "jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        "gpu_rtx2080_8gb_dual_cpu_fast",
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        total_units=64.0,
    ),
)


def build_live_eta_service_cache_v2_gate() -> dict[str, Any]:
    cache = ServiceRateCache()
    rows: list[dict[str, Any]] = []
    for spec in LIVE_SUMMARIES:
        row = _row_from_spec(spec)
        rows.append(row)
        if row["path_exists"]:
            for record in records_from_summary_file(
                spec.path,
                workload_key=spec.workload_key,
                command_fingerprint="live_tqdm_service_cache_v2_noearly_20260629",
                resource_kind=spec.resource_kind,
                total_units=spec.total_units,
                node_bucket=spec.node_bucket,
                workload_env=spec.workload_env,
                resource_state=spec.resource_state,
                eta_source="tqdm/progress",
                hardware_class=spec.hardware_class,
            ):
                cache.add(record, force_replace=True)
    snapshot = cache.snapshot()
    admitted = [row for row in rows if row.get("measurement_valid")]
    boundaries = [row for row in rows if row.get("path_exists") and not row.get("measurement_valid")]
    missing = [row for row in rows if not row.get("path_exists")]
    hardware_classes = sorted({str(row.get("hardware_class")) for row in rows if row.get("hardware_class")})
    workload_envs = sorted({str(row.get("workload_env")) for row in rows if row.get("workload_env")})
    return {
        "gate": "live_eta_service_cache_v2_gate",
        "pass": bool(admitted) and not missing,
        "status": "LIVE_ETA_SERVICE_CACHE_V2_READY" if admitted and not missing else "LIVE_ETA_SERVICE_CACHE_V2_OPEN",
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "capacity_boundary_count": len(boundaries),
        "missing_count": len(missing),
        "hardware_classes": hardware_classes,
        "workload_envs": workload_envs,
        "admitted_rows": admitted,
        "capacity_boundary_rows": boundaries,
        "missing_rows": missing,
        "rows": rows,
        "service_cache_snapshot": snapshot,
        "scope": (
            "Task-native tqdm/ScheduleurmStableRate CNN and LLM rows for "
            "jtl110gpu, jtl311linux, and node007 where reachable. "
            "Invalid p3/p4 rows are retained as capacity/unstable boundaries "
            "and are not theorem-facing service rows."
        ),
    }


def _row_from_spec(spec: LiveSummarySpec) -> dict[str, Any]:
    path = spec.path
    base = {
        "run_id": spec.run_id,
        "profile": spec.profile,
        "path": str(path),
        "path_exists": path.exists(),
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "workload_key": spec.workload_key,
        "workload_env": spec.workload_env,
        "resource_state": spec.resource_state,
    }
    if not path.exists():
        return {**base, "measurement_valid": False, "stable_rate_ready_count": 0, "running_count": 0}
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        **base,
        "measurement_valid": bool(summary.get("measurement_valid")),
        "placement_valid": bool(summary.get("placement_valid")),
        "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
        "running_count": int(summary.get("running_count") or 0),
        "aggregate_stable_rate_unit_s": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
        "mean_stable_rate_unit_s": float(summary.get("mean_stable_rate_unit_s") or 0.0),
        "status_counts": summary.get("status_counts") or {},
        "eta_source": "tqdm/progress",
        "theorem_facing_admitted": bool(summary.get("measurement_valid")),
        "boundary_reason": "" if summary.get("measurement_valid") else "capacity_or_unstable_profile",
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live ETA Service Cache v2 Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        f"- Capacity/unstable boundary rows: `{report.get('boundary_count')}`",
        f"- Missing rows: `{report.get('missing_count')}`",
        "",
        "| Node bucket | Profile | Valid | Ready/running | Aggregate stable rate | Boundary reason |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row['node_bucket']}` | {int(row['profile'])} | "
            f"{str(bool(row.get('measurement_valid'))).lower()} | "
            f"{int(row.get('stable_rate_ready_count') or 0)}/{int(row.get('running_count') or 0)} | "
            f"{float(row.get('aggregate_stable_rate_unit_s') or 0.0):.6g} | "
            f"`{row.get('boundary_reason') or ''}` |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "live_eta_service_cache_v2_gate_20260629.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_live_eta_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "live_eta_service_cache_v2_gate_20260629.md")
    args = parser.parse_args()
    report = build_live_eta_service_cache_v2_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(json.dumps(report["service_cache_snapshot"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
