"""Same-workload native Gavel simulator comparison gate.

This gate compares usable native Gavel simulator rows with Scheduleurm replay
rows only when the exported Gavel case covers the full Scheduleurm taskset job
count.  It is stronger than a smoke test and narrower than direct full-stack
binary superiority: Gavel's simulator still uses its own throughput tables and
does not launch the real workload binaries on the Scheduleurm cluster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from simulation.defaults import build_default_cache, calibrated_candidate_policy, legacy_policy
from simulation.sota_baselines import sota_baseline_specs, sota_candidate_union_policy
from simulation.tasksets import taskset_by_name
from simulation.trace_benchmark import build_task_trace, replay_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_GAVEL_ARTIFACT = ARTIFACT_ROOT / "gavel_native_q01_resnet50_64_scs_20260613.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "gavel_native_same_workload_comparison_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "gavel_native_same_workload_comparison_20260613.md"


def build_gavel_native_same_workload_comparison(
    *,
    gavel_artifact: str | Path = DEFAULT_GAVEL_ARTIFACT,
) -> dict[str, Any]:
    artifact_path = Path(gavel_artifact).expanduser()
    gavel = _load_json(artifact_path)
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in gavel.get("rows") or []:
        if not row.get("usable_native_performance_sample"):
            continue
        taskset = str(row.get("taskset") or "")
        job_count = int(row.get("job_count") or 0)
        if taskset and job_count > 0:
            grouped.setdefault((taskset, job_count), []).append(row)

    rows = [
        _comparison_row(taskset_name=taskset, job_count=job_count, gavel_rows=list(rows))
        for (taskset, job_count), rows in sorted(grouped.items())
    ]
    eligible = [row for row in rows if row.get("eligible_same_workload_full_taskset")]
    raw_wins = [
        row for row in eligible
        if row.get("scheduleurm_raw_native_simulator_pareto_superior")
    ]
    raw_ready = bool(eligible) and len(raw_wins) == len(eligible)
    return {
        "gate": "gavel_native_same_workload_comparison",
        "gavel_artifact": str(artifact_path),
        "gavel_solver": gavel.get("solver"),
        "row_count": len(rows),
        "eligible_row_count": len(eligible),
        "raw_native_simulator_superiority_count": len(raw_wins),
        "same_workload_native_simulator_superiority_ready": raw_ready,
        "direct_fullstack_binary_superiority_ready": False,
        "strong_claim_ready": False,
        "scoped_claim_ready": raw_ready,
        "pass": bool(rows),
        "rows": rows,
        "scope": (
            "Compares Scheduleurm measured-cache replay to Gavel's native simulator "
            "on full-taskset same-workload trace exports.  A ready result supports "
            "a scoped native-Gavel-simulator baseline claim.  It is not direct "
            "full-stack binary superiority because Gavel's simulator uses its own "
            "throughput tables and does not launch the real workload binaries on "
            "the Scheduleurm cluster."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gavel Native Same-Workload Comparison",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `same_workload_native_simulator_superiority_ready` | {str(bool(report.get('same_workload_native_simulator_superiority_ready'))).lower()} |",
        f"| `direct_fullstack_binary_superiority_ready` | {str(bool(report.get('direct_fullstack_binary_superiority_ready'))).lower()} |",
        f"| `eligible_row_count` | {report.get('eligible_row_count', 0)} |",
        f"| `raw_native_simulator_superiority_count` | {report.get('raw_native_simulator_superiority_count', 0)} |",
        "",
        "| Taskset | Jobs | Gavel best policy | Gavel makespan | Gavel mean JCT | Scheduleurm policy | Scheduleurm makespan | Scheduleurm mean flow | Raw superior | Eligible |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{taskset}` | {jobs} | `{gavel_policy}` | {gavel_ms:.6g} | {gavel_flow:.6g} | `{sched_policy}` | {sched_ms:.6g} | {sched_flow:.6g} | {superior} | {eligible} |".format(
                taskset=row.get("taskset"),
                jobs=int(row.get("job_count") or 0),
                gavel_policy=row.get("gavel_best_pareto_policy") or "",
                gavel_ms=float(row.get("gavel_best_makespan_s") or 0.0),
                gavel_flow=float(row.get("gavel_best_mean_jct_s") or 0.0),
                sched_policy=row.get("scheduleurm_best_pareto_policy") or "",
                sched_ms=float(row.get("scheduleurm_best_makespan_s") or 0.0),
                sched_flow=float(row.get("scheduleurm_best_mean_flow_s") or 0.0),
                superior=str(bool(row.get("scheduleurm_raw_native_simulator_pareto_superior"))).lower(),
                eligible=str(bool(row.get("eligible_same_workload_full_taskset"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _comparison_row(
    *,
    taskset_name: str,
    job_count: int,
    gavel_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    taskset = taskset_by_name(taskset_name)
    expected_jobs = sum(int(member.task_count) for member in taskset.members)
    eligible = int(job_count) == int(expected_jobs)
    cache = build_default_cache()
    trace = build_task_trace(taskset, arrival_mode="static", seed=42)
    specs = trace.workload_specs()
    policies = [
        legacy_policy(),
        calibrated_candidate_policy(cache, specs),
        sota_candidate_union_policy(cache, specs, selection_objective="pareto_slack"),
        sota_candidate_union_policy(cache, specs, selection_objective="makespan"),
        sota_candidate_union_policy(cache, specs, selection_objective="mean_flow"),
        *(spec.policy for spec in sota_baseline_specs()),
    ]
    sched_results = [replay_trace(cache, trace, policy, seed=7) for policy in policies]
    scheduleurm_best = min(
        sched_results,
        key=lambda result: (float(result.makespan_s), float(result.mean_flow_s), result.policy),
    )
    scheduleurm_best_flow = min(
        sched_results,
        key=lambda result: (float(result.mean_flow_s), float(result.makespan_s), result.policy),
    )
    gavel_best_ms = min(
        gavel_rows,
        key=lambda row: (float(row.get("makespan_s") or float("inf")), float(row.get("average_jct_s") or float("inf"))),
    )
    gavel_best_flow = min(
        gavel_rows,
        key=lambda row: (float(row.get("average_jct_s") or float("inf")), float(row.get("makespan_s") or float("inf"))),
    )
    sched_ms = min(float(scheduleurm_best.makespan_s), float(scheduleurm_best_flow.makespan_s))
    sched_flow = min(float(scheduleurm_best.mean_flow_s), float(scheduleurm_best_flow.mean_flow_s))
    gavel_ms = min(float(gavel_best_ms.get("makespan_s") or 0.0), float(gavel_best_flow.get("makespan_s") or 0.0))
    gavel_flow = min(float(gavel_best_ms.get("average_jct_s") or 0.0), float(gavel_best_flow.get("average_jct_s") or 0.0))
    superior = eligible and sched_ms <= gavel_ms and sched_flow <= gavel_flow
    return {
        "taskset": taskset_name,
        "job_count": int(job_count),
        "expected_full_taskset_job_count": int(expected_jobs),
        "eligible_same_workload_full_taskset": bool(eligible),
        "gavel_usable_policy_count": len(gavel_rows),
        "gavel_best_makespan_s": gavel_ms,
        "gavel_best_mean_jct_s": gavel_flow,
        "gavel_best_pareto_policy": str(gavel_best_ms.get("policy") or gavel_best_flow.get("policy") or ""),
        "scheduleurm_best_makespan_s": sched_ms,
        "scheduleurm_best_mean_flow_s": sched_flow,
        "scheduleurm_best_pareto_policy": scheduleurm_best.policy,
        "scheduleurm_best_flow_policy": scheduleurm_best_flow.policy,
        "scheduleurm_raw_native_simulator_makespan_ratio": (gavel_ms / sched_ms if sched_ms > 0 else 0.0),
        "scheduleurm_raw_native_simulator_mean_flow_ratio": (gavel_flow / sched_flow if sched_flow > 0 else 0.0),
        "scheduleurm_raw_native_simulator_pareto_superior": bool(superior),
        "gavel_usable_policies": sorted(str(row.get("policy") or "") for row in gavel_rows),
        "scheduleurm_policy_count": len(sched_results),
        "note": (
            "eligible full-taskset native simulator comparison"
            if eligible else
            "skipped for superiority because Gavel job_count does not cover the full Scheduleurm taskset"
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gavel_native_same_workload_comparison(gavel_artifact=args.gavel_artifact)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gavel_native_same_workload_comparison")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Compare usable native Gavel simulator rows with Scheduleurm replay")
    build.add_argument("--gavel-artifact", default=str(DEFAULT_GAVEL_ARTIFACT))
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
