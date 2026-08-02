"""Strict q00/q10 SOTA-policy replay on the new phase-aware CPU cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from algorithm.experiments.statewise_phase_sota_replay_gate import (
    _freeze_lower_service_candidate,
)
from simulation.service_cache import ServiceRateCache
from simulation.sota_baselines import sota_baseline_specs
from simulation.statewise_replay import ExactReplayProfile, build_exact_replay_views
from simulation.tasksets import TaskSet, TaskSetMember
from simulation.trace_benchmark import build_task_trace, replay_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = ARTIFACT_ROOT / "service_cache_v2_critical_phase_20260802.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_q00_q10_phase_sota_replay_20260802.json"
SCENARIOS = {
    "q00": {
        "workload_key": "light_control_local",
        "workload_env": "light_control",
        "node": "node003",
        "profiles": (1, 13),
        "task_count": 52,
        "total_units": 300.0,
        "quadrant": "low_cpu_low_gpu",
        "resource_kind": "critical_q00_cpu_light",
    },
    "q10": {
        "workload_key": "cpu_heavy_local_bench",
        "workload_env": "cpu",
        "node": "node001",
        "profiles": (1, 8, 9),
        "task_count": 45,
        "total_units": 120.0,
        "quadrant": "high_cpu_low_gpu",
        "resource_kind": "critical_q10_cpu_heavy",
    },
}


def build_critical_q00_q10_phase_sota_replay_gate(
    *,
    cache_path: Path = DEFAULT_CACHE,
    arrivals: Iterable[str] = ("static", "poisson"),
    trace_seed: int = 42,
    replay_seed: int = 7,
) -> dict[str, Any]:
    source_path = Path(cache_path).resolve()
    source = ServiceRateCache.load(source_path)
    rows = []
    projections = {}
    for quadrant, contract in SCENARIOS.items():
        requests = tuple(
            ExactReplayProfile(
                workload_key=contract["workload_key"],
                workload_env=contract["workload_env"],
                node_bucket=f"{contract['node']}:cpu_hpc_192c",
                resource_state=("empty" if profile == 1 else "controlled_colocation"),
                allocation_workers=1,
                colocation_count=profile,
            )
            for profile in contract["profiles"]
        )
        views = build_exact_replay_views(source, requests)
        projections[quadrant] = views.snapshot()
        taskset = _taskset(quadrant, contract)
        specs = taskset.workload_specs()
        for arrival in tuple(str(value) for value in arrivals):
            trace = build_task_trace(taskset, arrival_mode=arrival, seed=trace_seed)
            candidate = _freeze_lower_service_candidate(
                lower_cache=views.lower_service,
                specs=specs,
                arrival_mode=arrival,
                statewise_resource_kinds=(contract["resource_kind"],),
            )
            policies = (candidate, *(spec.policy for spec in sota_baseline_specs()))
            results = [
                replay_trace(views.completion_point, trace, policy, seed=replay_seed)
                for policy in policies
            ]
            ours, sota = results[0], results[1:]
            best_makespan = min(result.makespan_s for result in sota)
            best_flow = min(result.mean_flow_s for result in sota)
            rows.append(
                {
                    "quadrant": quadrant,
                    "arrival_mode": arrival,
                    "job_count": len(trace.jobs),
                    "candidate": ours.snapshot(),
                    "sota": [result.snapshot() for result in sota],
                    "best_sota_makespan_s": best_makespan,
                    "best_sota_mean_flow_s": best_flow,
                    "candidate_vs_best_sota_makespan": ours.makespan_s / best_makespan,
                    "candidate_vs_best_sota_mean_flow": ours.mean_flow_s / best_flow,
                    "candidate_pareto_dominated_by_sota": any(
                        result.makespan_s < ours.makespan_s
                        and result.mean_flow_s < ours.mean_flow_s
                        for result in sota
                    ),
                }
            )

    projection_rows = [row for value in projections.values() for row in value["rows"]]
    checks = {
        "exact_projection_row_count": len(projection_rows) == 5,
        "all_exact_rows_natural_completion_ready": all(
            int(row["completion_model_sample_count"]) > 0
            and float(row["completion_total_wall_s"]) > 0.0
            for row in projection_rows
        ),
        "no_history_fallback": all(
            "hist" not in str(row["eta_source"]).lower()
            for row in projection_rows
        ),
        "separate_lower_and_completion_views": all(
            abs(float(row["lower_service_aggregate_rate"]) - float(row["completion_point_aggregate_rate"])) > 1e-12
            for row in projection_rows
        ),
        "all_scenarios_complete": len(rows) == 4
        and all(
            row["candidate"]["completed_jobs"] == row["job_count"]
            and all(result["completed_jobs"] == row["job_count"] for result in row["sota"])
            for row in rows
        ),
        "candidate_selection_bound_to_lower_service": all(
            row["candidate"]["policy_config"].get("selection_service_view")
            == "lower_service"
            for row in rows
        ),
        "candidate_not_pareto_dominated": all(
            not row["candidate_pareto_dominated_by_sota"] for row in rows
        ),
    }
    passed = all(checks.values())
    return {
        "gate": "critical_q00_q10_phase_sota_replay_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "cache": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "checks": checks,
        "exact_replay_projections": projections,
        "scenarios": rows,
        "claim_boundary": (
            "PASS compares the fixed Scheduleurm lower-service-selected policy "
            "with registered SOTA-style policy semantics on the same exact q00 "
            "and q10 natural-completion views. It is not a direct external-binary "
            "comparison and does not cover the GPU-blocked q01/q11 measurements."
        ),
    }


def _taskset(quadrant: str, contract: dict[str, Any]) -> TaskSet:
    return TaskSet(
        name=f"critical_{quadrant}_phase_aware",
        purpose=f"Exact statewise {quadrant} natural-completion replay.",
        arrival_model="static and Poisson",
        members=(
            TaskSetMember(
                workload_key=contract["workload_key"],
                resource_kind=contract["resource_kind"],
                task_count=int(contract["task_count"]),
                total_units=float(contract["total_units"]),
                resource_count=1,
                variation_cv=0.10,
                quadrant=contract["quadrant"],
                role=f"phase-aware {quadrant} replay",
                benchmark_source="actual-I/O natural-completion matrix",
                required_profiles=tuple(contract["profiles"]),
                empirical_status="real",
                note="Exact node/state co-location axis with one worker per task.",
            ),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--arrivals", default="static,poisson")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_q00_q10_phase_sota_replay_gate(
        cache_path=args.cache,
        arrivals=tuple(value.strip() for value in args.arrivals.split(",") if value.strip()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
