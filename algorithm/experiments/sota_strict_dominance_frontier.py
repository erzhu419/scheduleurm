"""Measured-cache external-policy frontier certificate.

The candidate-union gate is intentionally tolerance based because replay traces
contain sampled service variation.  This companion certificate records the
remaining scenarios where the fixed Pareto-slack policy has not strictly
matched the best external-policy envelope in both metrics.  It is not a direct
full-stack SOTA superiority certificate: rows here require either a finer
phase-switch action or a new measured service profile before strict
measured-cache policy-semantics noninferiority can be claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .sota_candidate_union_gate import PARETO_TOLERANCE, build_sota_candidate_union_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_sota_strict_dominance_frontier(
    *,
    strict_eps: float = 1e-12,
) -> dict[str, Any]:
    gate = build_sota_candidate_union_gate()
    policy_rows = list(gate.get("policy_matrix") or [])
    frontier = []
    for row in gate.get("scenarios") or []:
        if not row.get("replayable", True):
            continue
        ms_ratio = _candidate_vs_sota_best_makespan(row)
        flow_ratio = _candidate_vs_sota_best_mean_flow(row)
        if ms_ratio >= 1.0 - strict_eps and flow_ratio >= 1.0 - strict_eps:
            continue
        frontier.append(_frontier_row(row, policy_rows))
    frontier_closed = not frontier
    return {
        "gate": "sota_strict_dominance_frontier",
        "status": (
            "MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_OPEN"
            if frontier else "MEASURED_CACHE_EXTERNAL_POLICY_FRONTIER_CLOSED"
        ),
        "gate_pass": bool(frontier_closed),
        "scoped_claim_ready": bool(frontier_closed),
        "strong_claim_ready": False,
        "pass_meaning": (
            "strict measured-cache external-policy noninferiority, not direct "
            "full-stack SOTA binary superiority"
            if frontier_closed else
            "remaining measured-cache external-policy frontier rows require bridge actions"
        ),
        "strict_pareto_ready": bool(frontier_closed),
        "within_tolerance_ready": bool(
            (gate.get("aggregate") or {}).get("fixed_online_policy_pareto_dominates_sota_style_all")
        ),
        "pareto_tolerance": float(PARETO_TOLERANCE),
        "frontier_count": len(frontier),
        "frontier": frontier,
        "next_probe_order": _next_probe_order(frontier),
        "scope": (
            "This certificate only diagnoses strict measured-cache policy-semantics "
            "gaps against the registered external-policy envelope. It does not "
            "claim direct full-stack SOTA binary superiority."
        ),
        "source_gate_status": gate.get("status"),
        "pass": bool(frontier_closed),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Measured-Cache External-Policy Frontier",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `strict_pareto_ready` | {str(bool(report.get('strict_pareto_ready'))).lower()} |",
        f"| `within_tolerance_ready` | {str(bool(report.get('within_tolerance_ready'))).lower()} |",
        f"| `frontier_count` | {report.get('frontier_count', 0)} |",
        f"| `pareto_tolerance` | {float(report.get('pareto_tolerance') or 0.0):.6g} |",
        "",
        "## Open Frontier Rows",
        "",
        "| Taskset | Arrival | Makespan ratio | Flow ratio | Best makespan policy | Best flow policy | Diagnosis | Required action |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in report.get("frontier") or []:
        lines.append(
            "| `{taskset}` | `{arrival}` | {ms:.9g} | {flow:.9g} | `{best_ms}` | `{best_flow}` | {diagnosis} | {action} |".format(
                taskset=row.get("taskset"),
                arrival=row.get("arrival_mode"),
                ms=float(row.get("makespan_ratio") or 0.0),
                flow=float(row.get("mean_flow_ratio") or 0.0),
                best_ms=row.get("best_makespan_policy"),
                best_flow=row.get("best_mean_flow_policy"),
                diagnosis=row.get("diagnosis"),
                action=row.get("required_action"),
            )
        )
    lines.extend([
        "",
        "## Probe Order",
        "",
    ])
    for idx, item in enumerate(report.get("next_probe_order") or [], start=1):
        lines.append(f"{idx}. {item}")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _frontier_row(row: Mapping[str, Any], policy_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    taskset = str(row.get("taskset") or "")
    arrival = str(row.get("arrival_mode") or "")
    scoped = [
        item for item in policy_rows
        if item.get("taskset") == taskset and item.get("arrival_mode") == arrival
    ]
    best_ms = min(scoped, key=lambda item: float(item.get("makespan_s") or float("inf")))
    best_flow = min(scoped, key=lambda item: float(item.get("mean_flow_s") or float("inf")))
    diagnosis, required = _diagnose(row, best_ms, best_flow)
    return {
        "taskset": taskset,
        "arrival_mode": arrival,
        "makespan_ratio": _candidate_vs_sota_best_makespan(row),
        "mean_flow_ratio": _candidate_vs_sota_best_mean_flow(row),
        "pareto_slack_profiles": row.get("pareto_slack_union_profiles") or {},
        "candidate_profiles": row.get("candidate_profiles") or {},
        "best_makespan_policy": best_ms.get("policy"),
        "best_makespan_s": float(best_ms.get("makespan_s") or 0.0),
        "best_makespan_profiles": best_ms.get("profiles") or {},
        "best_mean_flow_policy": best_flow.get("policy"),
        "best_mean_flow_s": float(best_flow.get("mean_flow_s") or 0.0),
        "best_mean_flow_profiles": best_flow.get("profiles") or {},
        "diagnosis": diagnosis,
        "required_action": required,
    }


def _candidate_vs_sota_best_makespan(row: Mapping[str, Any]) -> float:
    candidate = float(row.get("candidate_makespan_s") or 0.0)
    best = float(row.get("best_sota_makespan_s") or 0.0)
    return best / candidate if candidate > 0.0 else 0.0


def _candidate_vs_sota_best_mean_flow(row: Mapping[str, Any]) -> float:
    candidate = float(row.get("candidate_mean_flow_s") or 0.0)
    best = float(row.get("best_sota_mean_flow_s") or 0.0)
    return best / candidate if candidate > 0.0 else 0.0


def _diagnose(
    scenario: Mapping[str, Any],
    best_ms: Mapping[str, Any],
    best_flow: Mapping[str, Any],
) -> tuple[str, str]:
    taskset = str(scenario.get("taskset") or "")
    if taskset == "q01_gpu_bound_cnn_resnet50":
        return (
            "same measured CNN profile but different statewise drain trajectory; current phase-switch/SJF/LPT bridge candidates improve flow but fail the strict makespan guard",
            "measure a new CNN tail/co-location service point or admit a bridge only after it improves flow without exceeding the finish-time makespan envelope",
        )
    if taskset == "q01_gpu_model_portfolio":
        return (
            "portfolio inherits the ResNet-50 static tail-drain gap at a much smaller scale",
            "close the CNN tail/co-location bridge first, then rerun the portfolio frontier on the same measured cache",
        )
    if taskset == "hybrid_research_portfolio":
        return (
            "hybrid RL profile-3 minimizes makespan while profile-2 minimizes flow",
            "measure or admit a hybrid RL co-location profile between current p2/p3, or add a critical-path-aware p3-to-p2 switch with a verified makespan guard",
        )
    if best_ms.get("policy") != best_flow.get("policy"):
        return (
            "best makespan and best mean-flow are supplied by different policy families",
            "add a finite candidate action that interpolates the two policies and certify it with the same service cache",
        )
    return (
        "strict gap remains inside one policy family",
        "increase replay samples and inspect whether the gap is sampling noise or requires a new measured profile",
    )


def _next_probe_order(frontier: list[Mapping[str, Any]]) -> list[str]:
    tasks = {str(row.get("taskset") or "") for row in frontier}
    order = []
    if "hybrid_research_portfolio" in tasks:
        order.append(
            "Hybrid RL p2/p3 bridge: run RE-SAC/BAPR-like probes at 2-3 concurrent tasks under empty and high-VRAM-resident states; admit only if the LCB gives profile-3 makespan with profile-2 tail flow."
        )
    if "q01_gpu_bound_cnn_resnet50" in tasks:
        order.append(
            "CNN tail/co-location bridge: current cache-only phase-switch, SJF, critical-SJF, and LPT-static bridge variants improve flow but miss the strict makespan envelope; run controlled ResNet-50 tail and CNN+LLM/CNN+CUDA service probes and admit only if the LCB improves flow without exceeding the finish-time makespan envelope."
        )
    if "q01_gpu_model_portfolio" in tasks:
        order.append(
            "q01 portfolio strict gap: rerun after the CNN bridge closes; do not add a separate portfolio-only claim unless the per-workload CNN service point is certified."
        )
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build",), nargs="?", default="build")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "sota_strict_dominance_frontier_20260613.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=REPO_ROOT / "md" / "sota_strict_dominance_frontier_20260613.md",
    )
    args = parser.parse_args()
    report = build_sota_strict_dominance_frontier()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
