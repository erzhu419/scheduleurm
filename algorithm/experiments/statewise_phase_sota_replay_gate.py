"""SOTA-policy replay on exact statewise natural-completion service rows.

This gate is deliberately separate from the historical default-cache replay.
It selects the Scheduleurm action union on calibrated lower service, then
evaluates that frozen action and every SOTA-style policy on an isolated point
completion view derived from the same exact full-tuple records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from simulation.fast_forward import WorkloadSpec
from simulation.service_cache import ServiceRateCache
from simulation.sota_baselines import (
    sota_baseline_specs,
    sota_candidate_union_policy,
)
from simulation.statewise_replay import (
    ExactReplayProfile,
    FrozenActionUnionPolicy,
    FrozenReplayAction,
    build_exact_replay_views,
)
from simulation.tasksets import TaskSet, TaskSetMember
from simulation.trace_benchmark import build_task_trace, replay_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_native_all_state_lcb_20260727.json"
)
DEFAULT_OUTPUT = ARTIFACT_ROOT / "statewise_phase_sota_replay_cpu_20260802.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "statewise_phase_sota_replay_cpu_20260802.md"
DEFAULT_NODE = "node001"
DEFAULT_PROFILES = (1, 2, 4)
DEFAULT_ARRIVALS = ("static", "poisson")
WORKLOADS = (
    ("freqduet_cpu_native", "freqduet"),
    ("sumo_eval_cpu_native", "sumo"),
)


def build_statewise_phase_sota_replay_gate(
    *,
    cache_path: Path = DEFAULT_CACHE,
    node: str = DEFAULT_NODE,
    profiles: Iterable[int] = DEFAULT_PROFILES,
    arrivals: Iterable[str] = DEFAULT_ARRIVALS,
    task_count: int = 48,
    total_units_per_task: float = 6.0,
    trace_seed: int = 42,
    replay_seed: int = 7,
) -> dict[str, Any]:
    source_path = Path(cache_path).resolve()
    source = ServiceRateCache.load(source_path)
    selected_profiles = tuple(sorted({int(value) for value in profiles}))
    requests = _native_cpu_requests(node=node, profiles=selected_profiles)
    views = build_exact_replay_views(source, requests)
    taskset = _native_cpu_taskset(
        task_count=int(task_count),
        total_units_per_task=float(total_units_per_task),
        profiles=selected_profiles,
    )
    specs = taskset.workload_specs()

    scenario_rows = []
    for arrival in tuple(str(value) for value in arrivals):
        trace = build_task_trace(taskset, arrival_mode=arrival, seed=trace_seed)
        candidate = _freeze_lower_service_candidate(
            lower_cache=views.lower_service,
            specs=specs,
            arrival_mode=arrival,
        )
        policies = (
            candidate,
            *(spec.policy for spec in sota_baseline_specs()),
        )
        results = [
            replay_trace(
                views.completion_point,
                trace,
                policy,
                seed=replay_seed,
            )
            for policy in policies
        ]
        candidate_result = results[0]
        sota_results = results[1:]
        best_sota_makespan = min(row.makespan_s for row in sota_results)
        best_sota_flow = min(row.mean_flow_s for row in sota_results)
        scenario_rows.append(
            {
                "arrival_mode": arrival,
                "job_count": len(trace.jobs),
                "candidate": candidate_result.snapshot(),
                "sota": [row.snapshot() for row in sota_results],
                "best_sota_makespan_s": best_sota_makespan,
                "best_sota_mean_flow_s": best_sota_flow,
                "candidate_vs_best_sota_makespan": _ratio(
                    candidate_result.makespan_s,
                    best_sota_makespan,
                ),
                "candidate_vs_best_sota_mean_flow": _ratio(
                    candidate_result.mean_flow_s,
                    best_sota_flow,
                ),
                "candidate_pareto_dominated_by_sota": any(
                    row.makespan_s < candidate_result.makespan_s
                    and row.mean_flow_s < candidate_result.mean_flow_s
                    for row in sota_results
                ),
            }
        )

    projection = views.snapshot()
    checks = {
        "exact_projection_row_count": len(projection["rows"])
        == len(requests),
        "all_exact_rows_natural_completion_ready": all(
            int(row["completion_model_sample_count"]) > 0
            and float(row["completion_total_wall_s"]) > 0.0
            for row in projection["rows"]
        ),
        "no_history_fallback": all(
            "history" not in str(row["eta_source"]).lower()
            and "hist" not in str(row["eta_source"]).lower()
            for row in projection["rows"]
        ),
        "separate_lower_and_completion_views": any(
            abs(
                float(row["lower_service_aggregate_rate"])
                - float(row["completion_point_aggregate_rate"])
            )
            > 1e-12
            for row in projection["rows"]
        ),
        "all_scenarios_complete": bool(scenario_rows)
        and all(
            int(row["candidate"]["completed_jobs"]) == int(row["job_count"])
            and all(
                int(sota["completed_jobs"]) == int(row["job_count"])
                for sota in row["sota"]
            )
            for row in scenario_rows
        ),
        "candidate_selection_bound_to_lower_service": all(
            row["candidate"]["policy_config"].get("selection_service_view")
            == "lower_service"
            for row in scenario_rows
        ),
    }
    passed = all(checks.values())
    return {
        "gate": "statewise_phase_sota_replay_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "cache": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "node": node,
        "profiles": list(selected_profiles),
        "profile_axis": "colocation_count",
        "allocation_workers": 1,
        "task_count_per_workload": int(task_count),
        "total_units_per_task": float(total_units_per_task),
        "checks": checks,
        "exact_replay_projection": projection,
        "scenarios": scenario_rows,
        "claim_boundary": (
            "PASS certifies a fail-closed, phase-aware SOTA-policy replay for "
            "the exact native FreqDuet/SUMO rows on the declared CPU node and "
            "profiles. Scheduleurm actions are selected on lower service and "
            "evaluated on natural-completion point service. It does not cover "
            "GPU profiles, other load states, unregistered workloads, or direct "
            "external scheduler binaries. Performance ratios are reported, not "
            "used as infrastructure-gate pass conditions."
        ),
    }


def _native_cpu_requests(
    *,
    node: str,
    profiles: tuple[int, ...],
) -> tuple[ExactReplayProfile, ...]:
    requests = []
    for workload_key, workload_env in WORKLOADS:
        for profile in profiles:
            requests.append(
                ExactReplayProfile(
                    workload_key=workload_key,
                    workload_env=workload_env,
                    node_bucket=f"{node}:cpu_hpc_192c",
                    resource_state=(
                        "empty" if profile == 1 else "controlled_colocation"
                    ),
                    allocation_workers=1,
                    colocation_count=profile,
                )
            )
    return tuple(requests)


def _native_cpu_taskset(
    *,
    task_count: int,
    total_units_per_task: float,
    profiles: tuple[int, ...],
) -> TaskSet:
    members = tuple(
        TaskSetMember(
            workload_key=workload_key,
            resource_kind="cpu_sumo_transit",
            task_count=task_count,
            total_units=total_units_per_task,
            resource_count=1,
            variation_cv=0.10,
            quadrant="high_cpu_low_gpu",
            role="native phase-aware q10 replay",
            benchmark_source=(
                "Scheduleurm native natural-completion split-conformal "
                "co-location matrix"
            ),
            required_profiles=profiles,
            empirical_status="real",
            note=(
                f"Exact statewise {workload_env} rows; profile is independent "
                "task co-location count and allocation_workers=1."
            ),
        )
        for workload_key, workload_env in WORKLOADS
    )
    return TaskSet(
        name="q10_cpu_native_phase_aware",
        purpose=(
            "Natural-completion q10 validation with native FreqDuet and SUMO "
            "tasks on exact statewise service rows."
        ),
        arrival_model="static and Poisson",
        members=members,
    )


def _freeze_lower_service_candidate(
    *,
    lower_cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    arrival_mode: str,
    statewise_resource_kinds: tuple[str, ...] = ("cpu_sumo_transit",),
) -> FrozenActionUnionPolicy:
    selector = sota_candidate_union_policy(
        lower_cache,
        specs,
        selection_objective="online_pareto_slack",
    )
    actions = []
    for spec in specs:
        trace_delegate = selector.trace_action_policy_for(
            lower_cache,
            spec,
            arrival_mode=arrival_mode,
        )
        if trace_delegate is None:
            selected = selector.selected_action_row(lower_cache, spec)
            delegate = selected["policy"]
            profile = int(selected["profile"])
            family_name = str(selected.get("family_name") or delegate.name)
        else:
            delegate = trace_delegate
            profile = int(delegate.select_profile(lower_cache, spec).profile)
            family_name = delegate.name
        actions.append(
            FrozenReplayAction(
                workload_key=spec.workload_key,
                profile=profile,
                family_name=family_name,
                delegate=delegate,
            )
        )
    return FrozenActionUnionPolicy(
        name="scheduleurm_statewise_lower_service_frozen_union",
        statewise=True,
        statewise_resource_kinds=statewise_resource_kinds,
        frozen_actions=tuple(actions),
        selection_cache=lower_cache,
    )


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Statewise Phase-Aware CPU SOTA Replay",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Node: `{report.get('node')}`",
        f"- Profiles: `{report.get('profiles')}`",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for name, value in (report.get("checks") or {}).items():
        lines.append(f"| `{name}` | {str(bool(value)).lower()} |")
    lines.extend(
        [
            "",
            "| Arrival | Ours / best SOTA makespan | Ours / best SOTA mean flow | Pareto dominated |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report.get("scenarios") or []:
        lines.append(
            "| `{}` | {:.6f} | {:.6f} | {} |".format(
                row.get("arrival_mode"),
                float(row.get("candidate_vs_best_sota_makespan") or 0.0),
                float(row.get("candidate_vs_best_sota_mean_flow") or 0.0),
                str(bool(row.get("candidate_pareto_dominated_by_sota"))).lower(),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--node", default=DEFAULT_NODE)
    parser.add_argument("--profiles", default="1,2,4")
    parser.add_argument("--arrivals", default="static,poisson")
    parser.add_argument("--task-count", type=int, default=48)
    parser.add_argument("--total-units-per-task", type=float, default=6.0)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_statewise_phase_sota_replay_gate(
        cache_path=args.cache,
        node=args.node,
        profiles=(
            int(value.strip())
            for value in str(args.profiles).split(",")
            if value.strip()
        ),
        arrivals=(
            value.strip()
            for value in str(args.arrivals).split(",")
            if value.strip()
        ),
        task_count=args.task_count,
        total_units_per_task=args.total_units_per_task,
        trace_seed=args.trace_seed,
        replay_seed=args.replay_seed,
    )
    _atomic_write_json(args.output, report)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
