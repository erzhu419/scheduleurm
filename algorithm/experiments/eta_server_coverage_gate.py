"""Server/workload ETA coverage audit for live-v2 service-cache rows.

The gate is intentionally stricter than the replay gates.  It answers a
different question: which server/workload/load-state combinations have
task-native stable ETA rows, which are only equivalence checks, and which are
still pending.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = ARTIFACT_ROOT / "service_cache_v2_live_merged_20260629.json"


@dataclass(frozen=True)
class ExpectedRow:
    node_bucket: str
    workload_key: str
    workload_env: str
    resource_state: str
    resident_mix: str
    profiles: tuple[int, ...]
    requirement: str
    rationale: str


EXPECTED_ROWS = (
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "empty", "", (1, 2), "representative_full", "jtl110gpu is the representative 3080Ti node."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "half_loaded", "same_workload_half_capacity", (2,), "representative_full", "CNN marginal ETA under one resident CNN."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "full_loaded", "same_workload_to_capacity_boundary", (4,), "representative_full", "CNN marginal ETA under high same-workload occupancy."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "high_vram_resident", "llm_or_memory_resident", (1,), "representative_full", "CNN marginal ETA under high-VRAM resident pressure."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "mixed_colocation", "cnn_plus_llm", (2,), "representative_full", "CNN marginal ETA under LLM resident pressure."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "mixed_colocation", "cnn_plus_llm_plus_hybrid_rl", (3,), "representative_full", "CNN marginal ETA under LLM+RL resident pressure."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "gpu_llm_distilgpt2", "llm", "empty", "", (1, 2), "representative_full", "Small LLM forward-pass ETA on the representative 3080Ti node."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "hybrid_rl_resac_ant", "ant", "empty", "", (1, 2), "representative_full", "RE-SAC Ant env-specific ETA."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "hybrid_rl_resac_halfcheetah", "halfcheetah", "empty", "", (1, 2), "representative_full", "RE-SAC HalfCheetah env-specific ETA."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "hybrid_rl_resac_hopper", "hopper", "empty", "", (1,), "representative_full", "RE-SAC Hopper env-specific ETA."),
    ExpectedRow("jtl110gpu:gpu_3080ti_12gb_dual", "hybrid_rl_resac_walker2d", "walker2d", "empty", "", (1,), "representative_full", "RE-SAC Walker2d env-specific ETA."),
    ExpectedRow("jtl110gpu2:gpu_3080ti_12gb_dual", "gpu_cnn_torch_resnet50", "cnn", "empty", "", (1, 2), "homogeneous_equivalence", "jtl110gpu2 should match the 3080Ti representative up to equivalence tolerance."),
    ExpectedRow("jtl110gpu2:gpu_3080ti_12gb_dual", "gpu_llm_distilgpt2", "llm", "empty", "", (1,), "homogeneous_equivalence", "LLM smoke/equivalence row for the second 3080Ti node."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "gpu_cnn_torch_resnet50", "cnn", "empty", "", (1, 2), "node_specific", "jtl311linux has a different GPU and must not share jtl110 rates."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "gpu_cnn_torch_resnet50", "cnn", "half_loaded", "same_workload_half_capacity", (2,), "node_specific", "jtl311linux add-one CNN under resident CNN."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "gpu_llm_distilgpt2", "llm", "empty", "", (1,), "node_specific", "jtl311linux LLM row; p2 remains optional because of 8GB VRAM."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "hybrid_rl_resac_ant", "ant", "empty", "", (1,), "node_specific", "jtl311linux Ant row captures CPU-fast/GPU-weaker behavior."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "hybrid_rl_resac_halfcheetah", "halfcheetah", "empty", "", (1, 2), "node_specific", "jtl311linux HalfCheetah p1/p2 rows test env and node awareness."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "hybrid_rl_resac_hopper", "hopper", "empty", "", (1,), "node_specific_pending", "Still needed if paper claims all four RE-SAC envs on jtl311linux."),
    ExpectedRow("jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast", "hybrid_rl_resac_walker2d", "walker2d", "empty", "", (1,), "node_specific_pending", "Still needed if paper claims all four RE-SAC envs on jtl311linux."),
    ExpectedRow("node007:gpu_node007_4x12gb", "gpu_cnn_torch_resnet50", "cnn", "empty", "", (1, 2, 4), "node_specific", "node007 has four 12GB GPUs; p4 should be measured or marked boundary."),
    ExpectedRow("node007:gpu_node007_4x12gb", "gpu_llm_distilgpt2", "llm", "empty", "", (1,), "node_specific_pending", "Needed before claiming node007 LLM ETA."),
    ExpectedRow("node007:gpu_node007_4x12gb", "hybrid_rl_resac_ant", "ant", "empty", "", (1,), "node_specific_pending", "Needed before claiming node007 RL ETA."),
    ExpectedRow("node001:cpu_hpc_192c", "cpu_heavy_local_bench", "cpu", "cpu_resident", "cpu_worker_resident", (1,), "cpu_representative", "Representative CPU-resident add-one row."),
    ExpectedRow("node003:cpu_hpc_192c", "cpu_heavy_local_bench", "cpu", "cpu_resident", "cpu_worker_resident", (1,), "cpu_representative", "Full-resident 192-core CPU representative row."),
    ExpectedRow("node005:cpu_hpc_192c", "cpu_heavy_local_bench", "cpu", "cpu_resident", "cpu_worker_resident", (1,), "cpu_equivalence", "Second CPU-node equivalence row."),
)


def build_eta_server_coverage_gate(*, cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    cache = ServiceRateCache.load(cache_path)
    rows = []
    for expected in EXPECTED_ROWS:
        present_profiles = []
        missing_profiles = []
        for profile in expected.profiles:
            rec = cache.get_statewise(
                expected.workload_key,
                int(profile),
                node_bucket=expected.node_bucket,
                resource_state=expected.resource_state,
                resident_mix=expected.resident_mix,
                workload_env=expected.workload_env,
                strict=True,
            )
            if rec is not None and rec.aggregate_rate > 0 and rec.stable_rate_ready:
                present_profiles.append(int(profile))
            else:
                missing_profiles.append(int(profile))
        status = "measured" if not missing_profiles else "partial" if present_profiles else "missing"
        rows.append({
            "node_bucket": expected.node_bucket,
            "workload_key": expected.workload_key,
            "workload_env": expected.workload_env,
            "resource_state": expected.resource_state,
            "resident_mix": expected.resident_mix,
            "profiles": list(expected.profiles),
            "present_profiles": present_profiles,
            "missing_profiles": missing_profiles,
            "requirement": expected.requirement,
            "status": status,
            "theorem_facing_ready": status == "measured" and not expected.requirement.endswith("_pending"),
            "rationale": expected.rationale,
        })
    required_rows = [row for row in rows if not str(row["requirement"]).endswith("_pending")]
    pending_rows = [row for row in rows if row["status"] != "measured"]
    required_missing = [
        row for row in required_rows
        if row["status"] != "measured"
    ]
    return {
        "gate": "eta_server_coverage_gate",
        "cache_path": str(cache_path),
        "row_count": len(rows),
        "measured_count": sum(1 for row in rows if row["status"] == "measured"),
        "partial_count": sum(1 for row in rows if row["status"] == "partial"),
        "missing_count": sum(1 for row in rows if row["status"] == "missing"),
        "required_missing_count": len(required_missing),
        "status": (
            "ETA_SERVER_COVERAGE_REQUIRED_READY"
            if not required_missing
            else "ETA_SERVER_COVERAGE_OPEN"
        ),
        "pass": not required_missing,
        "rows": rows,
        "required_missing_rows": required_missing,
        "pending_rows": pending_rows,
        "claim_boundary": (
            "This gate audits measured ETA coverage.  A measured representative or "
            "equivalence row does not imply every hardware/workload/load-state "
            "combination is globally certified."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# ETA Server Coverage Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Measured rows: `{report.get('measured_count')}`",
        f"- Partial rows: `{report.get('partial_count')}`",
        f"- Missing rows: `{report.get('missing_count')}`",
        f"- Required missing rows: `{report.get('required_missing_count')}`",
        "",
        "| Node bucket | Workload | Env | State | Resident mix | Required profiles | Present | Missing | Requirement | Status |",
        "|---|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row['node_bucket']}` | `{row['workload_key']}` | `{row['workload_env']}` | "
            f"`{row['resource_state']}` | `{row['resident_mix']}` | `{row['profiles']}` | `{row['present_profiles']}` | "
            f"`{row['missing_profiles']}` | `{row['requirement']}` | `{row['status']}` |"
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "eta_server_coverage_gate_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "eta_server_coverage_gate_20260629.md")
    args = parser.parse_args()
    report = build_eta_server_coverage_gate(cache_path=args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
