"""SOTA-policy replay on the available hardware-local GPU completion rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.experiments.critical_gpu_available_lcb_cache_merge import (
    AVAILABLE_NODES,
    DEFAULT_CACHE_OUTPUT,
    EFFECTIVE_NODE_BUCKETS,
)
from algorithm.experiments.critical_gpu_completion_campaign import (
    NODE_SPECS,
    TRAINING_WAVES,
    campaign_cells,
)
from algorithm.experiments.statewise_phase_sota_replay_gate import (
    _freeze_lower_service_candidate,
)
from simulation.fast_forward import ReplayPolicy, WorkloadSpec
from simulation.service_cache import ServiceRateCache
from simulation.sota_baselines import sota_baseline_specs
from simulation.statewise_replay import ExactReplayProfile, build_exact_replay_views
from simulation.statewise_replay import FrozenActionUnionPolicy, FrozenReplayAction
from simulation.tasksets import TaskSet, TaskSetMember
from simulation.trace_benchmark import build_task_trace, replay_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_gpu_statewise_sota_replay_20260808.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "critical_gpu_statewise_sota_replay_20260808.md"
ARRIVALS = ("static", "poisson")
SCOPES = ("q01", "q11", "portfolio")
TOLERANCE = 0.005
WORKLOAD_RESOURCE_KINDS = {
    "gpu_heavy_jax_matmul": "gpu_heavy",
    "gpu_cnn_torch_resnet50": "gpu_cnn",
    "gpu_llm_distilgpt2": "gpu_llm",
    "hybrid_rl_resac_ant": "hybrid_rl",
}


def build_critical_gpu_statewise_sota_replay_gate(
    *,
    cache_path: Path = DEFAULT_CACHE_OUTPUT,
    task_count_per_workload: int = 24,
    trace_seed: int = 42,
    replay_seed: int = 7,
) -> dict[str, Any]:
    source_path = Path(cache_path).resolve()
    source = ServiceRateCache.load(source_path)
    scenario_rows = []
    projections = {}
    for node in AVAILABLE_NODES:
        cells = campaign_cells(node=node, wave=TRAINING_WAVES[0])
        requests = tuple(
            ExactReplayProfile(
                workload_key=str(cell["workload_key"]),
                workload_env=str(cell["workload_env"]),
                node_bucket=EFFECTIVE_NODE_BUCKETS[node],
                resource_state="empty",
                allocation_workers=1,
                colocation_count=int(cell["profile"]),
            )
            for cell in cells
        )
        views = build_exact_replay_views(source, requests)
        projections[node] = views.snapshot()
        for scope in SCOPES:
            taskset = _taskset(
                node=node,
                cells=cells,
                scope=scope,
                task_count_per_workload=int(task_count_per_workload),
            )
            specs = taskset.workload_specs()
            for arrival in ARRIVALS:
                trace = build_task_trace(taskset, arrival_mode=arrival, seed=trace_seed)
                candidate = _freeze_lower_service_candidate(
                    lower_cache=views.lower_service,
                    specs=specs,
                    arrival_mode=arrival,
                )
                policies = (
                    candidate,
                    *(
                        _freeze_policy_on_lower_service(
                            lower_cache=views.lower_service,
                            specs=specs,
                            policy=baseline.policy,
                            arrival_mode=arrival,
                        )
                        for baseline in sota_baseline_specs()
                    ),
                )
                results = [
                    replay_trace(views.completion_point, trace, policy, seed=replay_seed)
                    for policy in policies
                ]
                ours, sota = results[0], results[1:]
                comparisons = [
                    {
                        "policy": result.policy,
                        "ours_to_sota_makespan_ratio": _ratio(
                            ours.makespan_s,
                            result.makespan_s,
                        ),
                        "ours_to_sota_mean_flow_ratio": _ratio(
                            ours.mean_flow_s,
                            result.mean_flow_s,
                        ),
                        "ours_strictly_pareto_dominates": _dominates(
                            ours.makespan_s,
                            ours.mean_flow_s,
                            result.makespan_s,
                            result.mean_flow_s,
                            tolerance=0.0,
                        ),
                        "ours_tolerance_pareto_dominates": _dominates(
                            ours.makespan_s,
                            ours.mean_flow_s,
                            result.makespan_s,
                            result.mean_flow_s,
                            tolerance=TOLERANCE,
                        ),
                        "ours_0p5pct_weakly_noninferior": _weakly_noninferior(
                            ours.makespan_s,
                            ours.mean_flow_s,
                            result.makespan_s,
                            result.mean_flow_s,
                            tolerance=TOLERANCE,
                        ),
                        "exact_metric_equivalence": bool(
                            abs(ours.makespan_s - result.makespan_s) <= 1e-12
                            and abs(ours.mean_flow_s - result.mean_flow_s) <= 1e-12
                        ),
                        "sota_strictly_pareto_dominates": _dominates(
                            result.makespan_s,
                            result.mean_flow_s,
                            ours.makespan_s,
                            ours.mean_flow_s,
                            tolerance=0.0,
                        ),
                    }
                    for result in sota
                ]
                best_makespan = min(result.makespan_s for result in sota)
                best_flow = min(result.mean_flow_s for result in sota)
                scenario_rows.append(
                    {
                        "node": node,
                        "operational_execution_class": EFFECTIVE_NODE_BUCKETS[node],
                        "scope": scope,
                        "arrival_mode": arrival,
                        "job_count": len(trace.jobs),
                        "candidate": ours.snapshot(),
                        "sota": [result.snapshot() for result in sota],
                        "comparisons": comparisons,
                        "best_sota_makespan_s": best_makespan,
                        "best_sota_mean_flow_s": best_flow,
                        "ours_to_best_sota_makespan_ratio": _ratio(
                            ours.makespan_s,
                            best_makespan,
                        ),
                        "ours_to_best_sota_mean_flow_ratio": _ratio(
                            ours.mean_flow_s,
                            best_flow,
                        ),
                        "ours_not_pareto_dominated": not any(
                            row["sota_strictly_pareto_dominates"]
                            for row in comparisons
                        ),
                        "ours_strictly_dominates_every_sota_policy": all(
                            row["ours_strictly_pareto_dominates"]
                            for row in comparisons
                        ),
                        "ours_tolerance_dominates_every_sota_policy": all(
                            row["ours_tolerance_pareto_dominates"]
                            for row in comparisons
                        ),
                        "ours_0p5pct_weakly_noninferior_to_every_sota_policy": all(
                            row["ours_0p5pct_weakly_noninferior"]
                            for row in comparisons
                        ),
                        "exactly_equivalent_to_every_sota_policy": all(
                            row["exact_metric_equivalence"]
                            for row in comparisons
                        ),
                    }
                )

    projection_rows = [
        row
        for projection in projections.values()
        for row in projection.get("rows") or []
    ]
    infrastructure_checks = {
        "three_separate_operational_execution_classes": (
            len(set(EFFECTIVE_NODE_BUCKETS.values())) == 3
        ),
        "all_27_exact_rows_projected": len(projection_rows) == 27,
        "all_rows_natural_completion_ready": all(
            int(row["completion_model_sample_count"]) == 12
            and float(row["completion_total_wall_s"]) > 0.0
            for row in projection_rows
        ),
        "no_history_fallback": all(
            "hist" not in str(row["eta_source"]).lower()
            for row in projection_rows
        ),
        "lower_service_and_completion_views_separate": all(
            abs(
                float(row["lower_service_aggregate_rate"])
                - float(row["completion_point_aggregate_rate"])
            )
            > 1e-12
            for row in projection_rows
        ),
        "all_18_scenarios_complete": len(scenario_rows) == 18
        and all(
            int(row["candidate"]["completed_jobs"]) == int(row["job_count"])
            and all(
                int(result["completed_jobs"]) == int(row["job_count"])
                for result in row["sota"]
            )
            for row in scenario_rows
        ),
        "candidate_selected_only_on_lower_service": all(
            row["candidate"]["policy_config"].get("selection_service_view")
            == "lower_service"
            for row in scenario_rows
        ),
        "all_sota_actions_selected_only_on_lower_service": all(
            all(
                result["policy_config"].get("selection_service_view")
                == "lower_service"
                for result in row["sota"]
            )
            for row in scenario_rows
        ),
    }
    performance_diagnostics = {
        "candidate_not_pareto_dominated_in_any_scenario": all(
            row["ours_not_pareto_dominated"] for row in scenario_rows
        ),
        "candidate_strictly_dominates_every_sota_policy_in_every_scenario": all(
            row["ours_strictly_dominates_every_sota_policy"]
            for row in scenario_rows
        ),
        "candidate_0p5pct_tolerance_dominates_every_sota_policy_in_every_scenario": all(
            row["ours_tolerance_dominates_every_sota_policy"]
            for row in scenario_rows
        ),
    }
    performance_requirements = {
        "candidate_not_pareto_dominated_in_any_scenario": (
            performance_diagnostics[
                "candidate_not_pareto_dominated_in_any_scenario"
            ]
        ),
        "candidate_0p5pct_weakly_noninferior_in_every_scenario": all(
            row["ours_0p5pct_weakly_noninferior_to_every_sota_policy"]
            for row in scenario_rows
        ),
        "candidate_tolerance_dominates_all_sota_in_every_static_scenario": all(
            row["ours_tolerance_dominates_every_sota_policy"]
            for row in scenario_rows
            if row["arrival_mode"] == "static"
        ),
        "strict_all_sota_improvement_observed_on_each_available_node": all(
            any(
                row["node"] == node
                and row["ours_strictly_dominates_every_sota_policy"]
                for row in scenario_rows
            )
            for node in AVAILABLE_NODES
        ),
    }
    infrastructure_pass = all(infrastructure_checks.values())
    performance_pass = all(performance_requirements.values())
    status = (
        "PASS"
        if infrastructure_pass and performance_pass
        else (
            "PASS_INFRASTRUCTURE_PERFORMANCE_PENDING"
            if infrastructure_pass
            else "FAIL_INFRASTRUCTURE"
        )
    )
    return {
        "gate": "critical_gpu_statewise_sota_replay_gate",
        "schema_version": 1,
        "status": status,
        "pass": status == "PASS",
        "infrastructure_pass": infrastructure_pass,
        "performance_validation_pass": performance_pass,
        "cache": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "available_nodes": list(AVAILABLE_NODES),
        "pending_nodes": ["jtl311linux"],
        "operational_execution_classes": dict(EFFECTIVE_NODE_BUCKETS),
        "task_count_per_workload": int(task_count_per_workload),
        "arrivals": list(ARRIVALS),
        "scopes": list(SCOPES),
        "sota_policy_count": len(sota_baseline_specs()),
        "tolerance": TOLERANCE,
        "infrastructure_checks": infrastructure_checks,
        "performance_requirements": performance_requirements,
        "performance_diagnostics": performance_diagnostics,
        "exact_replay_projections": projections,
        "scenarios": scenario_rows,
        "claim_boundary": (
            "Both Scheduleurm and every registered SOTA-style policy freeze "
            "their profile and event-level trajectory decisions on the same "
            "lower-service information set; the untouched natural-completion "
            "point view is used only for evaluation. PASS means no registered "
            "policy Pareto-dominates Scheduleurm in any of 18 scenarios, all "
            "comparisons are 0.5%-weakly-noninferior, every static scenario "
            "0.5%-dominates every registered policy, and every measured node "
            "has at least one strict all-policy improvement. Sparse p1 ties are "
            "reported as ties, not strict wins. This is not a direct external "
            "binary comparison, does not pool the two RTX 3080 Ti hosts, and "
            "does not impute jtl311linux or unmeasured resource states."
        ),
    }


def _taskset(
    *,
    node: str,
    cells: list[dict[str, Any]],
    scope: str,
    task_count_per_workload: int,
) -> TaskSet:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell["workload_key"]), []).append(cell)
    selected = []
    for workload, rows in sorted(grouped.items()):
        quadrant = str(rows[0]["quadrant"])
        if scope == "q01" and quadrant != "q01":
            continue
        if scope == "q11" and quadrant != "q11":
            continue
        selected.append(
            TaskSetMember(
                workload_key=workload,
                resource_kind=WORKLOAD_RESOURCE_KINDS[workload],
                task_count=int(task_count_per_workload),
                total_units=float(rows[0]["max_iters"]),
                resource_count=len(NODE_SPECS[node].gpus),
                variation_cv=0.10,
                quadrant=(
                    "high_cpu_high_gpu" if quadrant == "q11" else "low_cpu_high_gpu"
                ),
                role=f"{scope} hardware-local natural-completion replay",
                benchmark_source="task-native progress split-conformal GPU campaign",
                required_profiles=tuple(sorted(int(row["profile"]) for row in rows)),
                empirical_status="real",
                note=(
                    f"Exact empty-state actions on {node}; profiles are tasks/GPU."
                ),
            )
        )
    if not selected:
        raise ValueError(f"scope {scope!r} selected no workloads")
    return TaskSet(
        name=f"critical_gpu_{node}_{scope}",
        purpose="Hardware-local lower-service action selection and completion replay.",
        arrival_model="static and Poisson",
        members=tuple(selected),
    )


def _freeze_policy_on_lower_service(
    *,
    lower_cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    policy: ReplayPolicy,
    arrival_mode: str,
) -> FrozenActionUnionPolicy:
    actions = []
    for spec in specs:
        trace_delegate = policy.trace_action_policy_for(
            lower_cache,
            spec,
            arrival_mode=arrival_mode,
        )
        delegate = trace_delegate or policy
        record = delegate.select_profile(lower_cache, spec)
        actions.append(
            FrozenReplayAction(
                workload_key=spec.workload_key,
                profile=int(record.profile),
                family_name=delegate.name,
                delegate=delegate,
            )
        )
    return FrozenActionUnionPolicy(
        name=policy.name,
        statewise=True,
        statewise_workload_keys=tuple(spec.workload_key for spec in specs),
        frozen_actions=tuple(actions),
        selection_cache=lower_cache,
    )


def _dominates(
    left_makespan: float,
    left_flow: float,
    right_makespan: float,
    right_flow: float,
    *,
    tolerance: float,
) -> bool:
    upper = 1.0 + float(tolerance)
    noninferior = bool(
        left_makespan <= right_makespan * upper
        and left_flow <= right_flow * upper
    )
    strictly_better = bool(
        left_makespan < right_makespan * (1.0 - 1e-12)
        or left_flow < right_flow * (1.0 - 1e-12)
    )
    return noninferior and strictly_better


def _weakly_noninferior(
    left_makespan: float,
    left_flow: float,
    right_makespan: float,
    right_flow: float,
    *,
    tolerance: float,
) -> bool:
    upper = 1.0 + float(tolerance)
    return bool(
        left_makespan <= right_makespan * upper
        and left_flow <= right_flow * upper
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0.0 else 0.0


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU Statewise SOTA Replay",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Performance validation: `{bool(report.get('performance_validation_pass'))}`",
        "",
        "| Node | Scope | Arrival | Ours / best SOTA makespan | Ours / best SOTA flow | Strict all-SOTA | 0.5% dominance | 0.5% weak NI | Exact all-policy tie |",
        "|---|---|---|---:|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in report.get("scenarios") or []:
        lines.append(
            "| `{}` | `{}` | `{}` | {:.6f} | {:.6f} | {} | {} | {} | {} |".format(
                row.get("node"),
                row.get("scope"),
                row.get("arrival_mode"),
                float(row.get("ours_to_best_sota_makespan_ratio") or 0.0),
                float(row.get("ours_to_best_sota_mean_flow_ratio") or 0.0),
                str(bool(row.get("ours_strictly_dominates_every_sota_policy"))).lower(),
                str(bool(row.get("ours_tolerance_dominates_every_sota_policy"))).lower(),
                str(bool(row.get("ours_0p5pct_weakly_noninferior_to_every_sota_policy"))).lower(),
                str(bool(row.get("exactly_equivalent_to_every_sota_policy"))).lower(),
            )
        )
    lines.extend(["", "## Performance requirements", ""])
    for name, value in (report.get("performance_requirements") or {}).items():
        lines.append(f"- `{name}`: `{str(bool(value)).lower()}`")
    lines.extend(["", "## Diagnostics", ""])
    for name, value in (report.get("performance_diagnostics") or {}).items():
        lines.append(f"- `{name}`: `{str(bool(value)).lower()}`")
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_OUTPUT)
    parser.add_argument("--task-count", type=int, default=24)
    parser.add_argument("--trace-seed", type=int, default=42)
    parser.add_argument("--replay-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_statewise_sota_replay_gate(
        cache_path=args.cache,
        task_count_per_workload=args.task_count,
        trace_seed=args.trace_seed,
        replay_seed=args.replay_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
