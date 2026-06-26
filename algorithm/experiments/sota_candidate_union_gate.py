"""SOTA policy-family union gate.

This gate is stronger than treating SOTA schedulers as after-the-fact baselines.
It replays three populations on identical measured Scheduleurm service curves:

* the current Scheduleurm candidate;
* each fixed SOTA-style policy family;
* selectors whose action set is the union of the Scheduleurm candidate action
  and the SOTA-style policy-family configuration-trajectory actions.

The metric-specific union rows are not external binary executions.  They certify
the theorem-facing statement that the robust candidate set can contain SOTA
policy actions, including statewise admission/drain semantics, and select over
that enlarged family in the same service units.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from simulation.defaults import build_default_cache, legacy_policy
from simulation.fast_forward import ReplayPolicy
from simulation.sota_baselines import (
    sota_baseline_specs,
    sota_candidate_union_policy,
    sota_candidate_union_policies,
)
from simulation.tasksets import taskset_by_name
from simulation.trace_benchmark import build_task_trace, replay_trace


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q01_gpu_bound_cnn_resnet50",
    "q01_gpu_bound_llm_inference",
    "q01_gpu_model_portfolio",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)
DEFAULT_ARRIVALS = ("static", "poisson")
PARETO_TOLERANCE = 0.005
CANDIDATE_POLICY_NAME = "scheduleurm_sota_union_online_pareto_slack"


def build_sota_candidate_union_gate(
    *,
    tasksets: Sequence[str] = DEFAULT_TASKSETS,
    arrivals: Sequence[str] = DEFAULT_ARRIVALS,
    trace_seed: int = 42,
    replay_seed: int = 7,
) -> dict[str, Any]:
    cache = build_default_cache()
    scenario_rows = []
    policy_rows = []
    for taskset_name in tasksets:
        taskset = taskset_by_name(taskset_name)
        missing = taskset.missing_measurements(cache)
        if missing:
            scenario_rows.append(
                {
                    "taskset": taskset_name,
                    "arrival_mode": "",
                    "replayable": False,
                    "missing_measurements": missing,
                }
            )
            continue
        specs = taskset.workload_specs()
        candidate = sota_candidate_union_policy(
            cache,
            specs,
            selection_objective="online_pareto_slack",
        )
        policies: tuple[ReplayPolicy, ...] = (
            legacy_policy(),
            candidate,
            *(spec.policy for spec in sota_baseline_specs()),
            *sota_candidate_union_policies(cache, specs),
        )
        for arrival in arrivals:
            trace = build_task_trace(taskset, arrival_mode=arrival, seed=trace_seed)
            results = {
                result.policy: result
                for result in (
                    replay_trace(cache, trace, policy, seed=replay_seed)
                    for policy in policies
                )
            }
            row = _scenario_row(taskset_name, arrival, results)
            scenario_rows.append(row)
            policy_rows.extend(_policy_rows(taskset_name, arrival, results))
    aggregate = _aggregate_rows(scenario_rows, policy_rows)
    return {
        "gate": "sota_candidate_union_gate",
        "status": "SOTA_CANDIDATE_UNION_PASS" if aggregate["gate_pass"] else "SOTA_CANDIDATE_UNION_NEEDS_REVIEW",
        "tasksets": list(tasksets),
        "arrivals": list(arrivals),
        "trace_seed": int(trace_seed),
        "replay_seed": int(replay_seed),
        "pareto_tolerance": PARETO_TOLERANCE,
        "scenario_count": len([row for row in scenario_rows if row.get("replayable", True)]),
        "scenarios": scenario_rows,
        "policy_matrix": policy_rows,
        "aggregate": aggregate,
        "scoped_claim_ready": bool(aggregate["gate_pass"]),
        "strong_claim_ready": False,
        "pass": bool(aggregate["gate_pass"]),
        "scope": (
            "Policy-semantics gate on identical measured service curves.  It "
            "certifies SOTA-policy-family configuration-trajectory action-union "
            "dominance/envelope rows, "
            "not direct full-stack external binary superiority."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    aggregate = report.get("aggregate") or {}
    lines = [
        "# SOTA Candidate-Union Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `scenario_count` | {report.get('scenario_count', 0)} |",
        f"| `union_makespan_beats_sota_envelope_all` | {str(bool(aggregate.get('union_makespan_beats_sota_envelope_all'))).lower()} |",
        f"| `union_mean_flow_beats_sota_envelope_all` | {str(bool(aggregate.get('union_mean_flow_beats_sota_envelope_all'))).lower()} |",
        f"| `guarded_union_not_pareto_dominated_all` | {str(bool(aggregate.get('guarded_union_not_pareto_dominated_all'))).lower()} |",
        f"| `single_guarded_union_beats_both_envelopes_all` | {str(bool(aggregate.get('single_guarded_union_beats_both_envelopes_all'))).lower()} |",
        f"| `adaptive_scalarized_union_not_pareto_dominated_all` | {str(bool(aggregate.get('adaptive_scalarized_union_not_pareto_dominated_all'))).lower()} |",
        f"| `adaptive_scalarized_union_beats_both_envelopes_all` | {str(bool(aggregate.get('adaptive_scalarized_union_beats_both_envelopes_all'))).lower()} |",
        f"| `pareto_slack_union_not_pareto_dominated_all` | {str(bool(aggregate.get('pareto_slack_union_not_pareto_dominated_all'))).lower()} |",
        f"| `pareto_slack_union_beats_both_envelopes_all` | {str(bool(aggregate.get('pareto_slack_union_beats_both_envelopes_all'))).lower()} |",
        f"| `fixed_online_policy_pareto_dominates_sota_style_all` | {str(bool(aggregate.get('fixed_online_policy_pareto_dominates_sota_style_all'))).lower()} |",
        "",
        "## Scenario Envelope",
        "",
        "| Taskset | Arrival | Best SOTA makespan | Pareto-slack makespan | Ratio | Best SOTA flow | Pareto-slack flow | Ratio | Pareto-slack dominated |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("scenarios") or []:
        if not row.get("replayable", True):
            lines.append(
                f"| `{row.get('taskset')}` | missing |  |  |  |  |  |  | true |"
            )
            continue
        lines.append(
            "| `{taskset}` | `{arrival}` | {best_ms:.6g} | {union_ms:.6g} | {ms_ratio:.6g} | {best_flow:.6g} | {union_flow:.6g} | {flow_ratio:.6g} | {dom} |".format(
                taskset=row.get("taskset"),
                arrival=row.get("arrival_mode"),
                best_ms=float(row.get("best_sota_makespan_s") or 0.0),
                union_ms=float(row.get("pareto_slack_union_makespan_s") or 0.0),
                ms_ratio=float(row.get("pareto_slack_union_vs_sota_best_makespan") or 0.0),
                best_flow=float(row.get("best_sota_mean_flow_s") or 0.0),
                union_flow=float(row.get("pareto_slack_union_mean_flow_s") or 0.0),
                flow_ratio=float(row.get("pareto_slack_union_vs_sota_best_mean_flow") or 0.0),
                dom=str(bool(row.get("pareto_slack_union_pareto_dominated_by_sota"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        "| Policy | Family | Systems | Sum makespan | Weighted mean flow | Candidate vs policy makespan | Candidate vs policy flow |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in aggregate.get("policy_aggregate") or []:
        lines.append(
            "| `{policy}` | `{family}` | {systems} | {makespan:.6g} | {flow:.6g} | {ms_ratio:.6g} | {flow_ratio:.6g} |".format(
                policy=row.get("policy"),
                family=row.get("policy_family"),
                systems=", ".join(row.get("representative_systems") or []),
                makespan=float(row.get("sum_makespan_s") or 0.0),
                flow=float(row.get("job_weighted_mean_flow_s") or 0.0),
                ms_ratio=float(row.get("candidate_vs_policy_sum_makespan") or 0.0),
                flow_ratio=float(row.get("candidate_vs_policy_job_weighted_mean_flow") or 0.0),
            )
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _scenario_row(
    taskset_name: str,
    arrival: str,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _candidate_policy(results)
    guarded_union = results["scheduleurm_sota_union_guarded_mean_flow"]
    adaptive_union = results["scheduleurm_sota_union_adaptive_scalarized"]
    pareto_slack_union = results["scheduleurm_sota_union_pareto_slack"]
    union_makespan = results["scheduleurm_sota_union_makespan"]
    union_mean_flow = results["scheduleurm_sota_union_mean_flow"]
    sota_rows = [
        result for name, result in results.items()
        if name.startswith("sota_")
    ]
    best_sota_makespan = min(sota_rows, key=lambda result: result.makespan_s)
    best_sota_flow = min(sota_rows, key=lambda result: result.mean_flow_s)
    dominated_by = []
    adaptive_dominated_by = []
    pareto_slack_dominated_by = []
    floor = 1.0 - PARETO_TOLERANCE
    for result in sota_rows:
        ms = _ratio(result.makespan_s, guarded_union.makespan_s)
        flow = _ratio(result.mean_flow_s, guarded_union.mean_flow_s)
        if ms < floor and flow < floor:
            dominated_by.append(
                {
                    "policy": result.policy,
                    "guarded_union_vs_policy_makespan": ms,
                    "guarded_union_vs_policy_mean_flow": flow,
                    "sota_profiles": result.profiles,
                    "guarded_union_profiles": guarded_union.profiles,
                }
            )
        adaptive_ms = _ratio(result.makespan_s, adaptive_union.makespan_s)
        adaptive_flow = _ratio(result.mean_flow_s, adaptive_union.mean_flow_s)
        if adaptive_ms < floor and adaptive_flow < floor:
            adaptive_dominated_by.append(
                {
                    "policy": result.policy,
                    "adaptive_union_vs_policy_makespan": adaptive_ms,
                    "adaptive_union_vs_policy_mean_flow": adaptive_flow,
                    "sota_profiles": result.profiles,
                    "adaptive_union_profiles": adaptive_union.profiles,
                }
            )
        pareto_slack_ms = _ratio(result.makespan_s, pareto_slack_union.makespan_s)
        pareto_slack_flow = _ratio(result.mean_flow_s, pareto_slack_union.mean_flow_s)
        if pareto_slack_ms < floor and pareto_slack_flow < floor:
            pareto_slack_dominated_by.append(
                {
                    "policy": result.policy,
                    "pareto_slack_union_vs_policy_makespan": pareto_slack_ms,
                    "pareto_slack_union_vs_policy_mean_flow": pareto_slack_flow,
                    "sota_profiles": result.profiles,
                    "pareto_slack_union_profiles": pareto_slack_union.profiles,
                }
            )
    return {
        "taskset": taskset_name,
        "arrival_mode": arrival,
        "replayable": True,
        "candidate_policy": candidate.policy,
        "candidate_makespan_s": candidate.makespan_s,
        "candidate_mean_flow_s": candidate.mean_flow_s,
        "best_sota_makespan_policy": best_sota_makespan.policy,
        "best_sota_makespan_s": best_sota_makespan.makespan_s,
        "best_sota_mean_flow_policy": best_sota_flow.policy,
        "best_sota_mean_flow_s": best_sota_flow.mean_flow_s,
        "guarded_union_makespan_s": guarded_union.makespan_s,
        "guarded_union_mean_flow_s": guarded_union.mean_flow_s,
        "adaptive_union_makespan_s": adaptive_union.makespan_s,
        "adaptive_union_mean_flow_s": adaptive_union.mean_flow_s,
        "pareto_slack_union_makespan_s": pareto_slack_union.makespan_s,
        "pareto_slack_union_mean_flow_s": pareto_slack_union.mean_flow_s,
        "union_makespan_s": union_makespan.makespan_s,
        "union_mean_flow_s": union_mean_flow.mean_flow_s,
        "union_vs_sota_envelope_makespan": _ratio(
            best_sota_makespan.makespan_s,
            union_makespan.makespan_s,
        ),
        "union_vs_sota_envelope_mean_flow": _ratio(
            best_sota_flow.mean_flow_s,
            union_mean_flow.mean_flow_s,
        ),
        "guarded_union_vs_sota_best_makespan": _ratio(
            best_sota_makespan.makespan_s,
            guarded_union.makespan_s,
        ),
        "guarded_union_vs_sota_best_mean_flow": _ratio(
            best_sota_flow.mean_flow_s,
            guarded_union.mean_flow_s,
        ),
        "adaptive_union_vs_sota_best_makespan": _ratio(
            best_sota_makespan.makespan_s,
            adaptive_union.makespan_s,
        ),
        "adaptive_union_vs_sota_best_mean_flow": _ratio(
            best_sota_flow.mean_flow_s,
            adaptive_union.mean_flow_s,
        ),
        "pareto_slack_union_vs_sota_best_makespan": _ratio(
            best_sota_makespan.makespan_s,
            pareto_slack_union.makespan_s,
        ),
        "pareto_slack_union_vs_sota_best_mean_flow": _ratio(
            best_sota_flow.mean_flow_s,
            pareto_slack_union.mean_flow_s,
        ),
        "single_guarded_union_beats_both_envelopes": (
            _ratio(best_sota_makespan.makespan_s, guarded_union.makespan_s)
            >= 1.0 - PARETO_TOLERANCE
            and _ratio(best_sota_flow.mean_flow_s, guarded_union.mean_flow_s)
            >= 1.0 - PARETO_TOLERANCE
        ),
        "adaptive_scalarized_union_beats_both_envelopes": (
            _ratio(best_sota_makespan.makespan_s, adaptive_union.makespan_s)
            >= 1.0 - PARETO_TOLERANCE
            and _ratio(best_sota_flow.mean_flow_s, adaptive_union.mean_flow_s)
            >= 1.0 - PARETO_TOLERANCE
        ),
        "pareto_slack_union_beats_both_envelopes": (
            _ratio(best_sota_makespan.makespan_s, pareto_slack_union.makespan_s)
            >= 1.0 - PARETO_TOLERANCE
            and _ratio(best_sota_flow.mean_flow_s, pareto_slack_union.mean_flow_s)
            >= 1.0 - PARETO_TOLERANCE
        ),
        "guarded_union_pareto_dominated_by_sota": bool(dominated_by),
        "guarded_union_dominated_by": dominated_by,
        "adaptive_union_pareto_dominated_by_sota": bool(adaptive_dominated_by),
        "adaptive_union_dominated_by": adaptive_dominated_by,
        "pareto_slack_union_pareto_dominated_by_sota": bool(pareto_slack_dominated_by),
        "pareto_slack_union_dominated_by": pareto_slack_dominated_by,
        "candidate_profiles": candidate.profiles,
        "guarded_union_profiles": guarded_union.profiles,
        "adaptive_union_profiles": adaptive_union.profiles,
        "pareto_slack_union_profiles": pareto_slack_union.profiles,
        "union_makespan_profiles": union_makespan.profiles,
        "union_mean_flow_profiles": union_mean_flow.profiles,
    }


def _policy_rows(
    taskset_name: str,
    arrival: str,
    results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate = _candidate_policy(results)
    rows = []
    for result in results.values():
        meta = _policy_metadata(result.policy)
        rows.append(
            {
                "taskset": taskset_name,
                "arrival_mode": arrival,
                "policy": result.policy,
                "policy_family": meta["policy_family"],
                "baseline_name": meta["baseline_name"],
                "representative_systems": meta["representative_systems"],
                "profiles": result.profiles,
                "makespan_s": result.makespan_s,
                "mean_flow_s": result.mean_flow_s,
                "p90_flow_s": result.p90_flow_s,
                "completed_jobs": result.completed_jobs,
                "candidate_policy": candidate.policy,
                "candidate_vs_policy_makespan": _ratio(
                    result.makespan_s,
                    candidate.makespan_s,
                ),
                "candidate_vs_policy_mean_flow": _ratio(
                    result.mean_flow_s,
                    candidate.mean_flow_s,
                ),
            }
        )
    return rows


def _aggregate_rows(
    scenario_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    replayable = [row for row in scenario_rows if row.get("replayable", True)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in policy_rows:
        grouped.setdefault(str(row["policy"]), []).append(row)
    policy_aggregate = []
    candidate_aggregate = None
    for policy, rows in sorted(grouped.items()):
        jobs = sum(int(row["completed_jobs"]) for row in rows)
        weighted_flow = (
            sum(float(row["mean_flow_s"]) * int(row["completed_jobs"]) for row in rows)
            / max(1, jobs)
        )
        sum_makespan = sum(float(row["makespan_s"]) for row in rows)
        candidate_sum_makespan = sum(
            float(_candidate_row_for(row, policy_rows)["makespan_s"])
            for row in rows
        )
        candidate_weighted_flow = (
            sum(
                float(_candidate_row_for(row, policy_rows)["mean_flow_s"])
                * int(row["completed_jobs"])
                for row in rows
            )
            / max(1, jobs)
        )
        first = rows[0]
        out = {
            "policy": policy,
            "policy_family": first["policy_family"],
            "baseline_name": first["baseline_name"],
            "representative_systems": first["representative_systems"],
            "scenario_count": len(rows),
            "completed_jobs": jobs,
            "sum_makespan_s": sum_makespan,
            "job_weighted_mean_flow_s": weighted_flow,
            "candidate_vs_policy_sum_makespan": _ratio(
                sum_makespan,
                candidate_sum_makespan,
            ),
            "candidate_vs_policy_job_weighted_mean_flow": _ratio(
                weighted_flow,
                candidate_weighted_flow,
            ),
        }
        if policy == CANDIDATE_POLICY_NAME:
            candidate_aggregate = out
        policy_aggregate.append(out)
    union_makespan_beats = all(
        float(row.get("union_vs_sota_envelope_makespan") or 0.0)
        >= 1.0 - PARETO_TOLERANCE
        for row in replayable
    )
    union_flow_beats = all(
        float(row.get("union_vs_sota_envelope_mean_flow") or 0.0)
        >= 1.0 - PARETO_TOLERANCE
        for row in replayable
    )
    guarded_not_dominated = all(
        not bool(row.get("guarded_union_pareto_dominated_by_sota"))
        for row in replayable
    )
    single_guarded_both = all(
        bool(row.get("single_guarded_union_beats_both_envelopes"))
        for row in replayable
    )
    adaptive_not_dominated = all(
        not bool(row.get("adaptive_union_pareto_dominated_by_sota"))
        for row in replayable
    )
    adaptive_both = all(
        bool(row.get("adaptive_scalarized_union_beats_both_envelopes"))
        for row in replayable
    )
    pareto_slack_not_dominated = all(
        not bool(row.get("pareto_slack_union_pareto_dominated_by_sota"))
        for row in replayable
    )
    pareto_slack_both = all(
        bool(row.get("pareto_slack_union_beats_both_envelopes"))
        for row in replayable
    )
    fixed_online_both = all(
        _ratio(
            float(row.get("best_sota_makespan_s") or 0.0),
            float(row.get("candidate_makespan_s") or 0.0),
        ) >= 1.0 - PARETO_TOLERANCE
        and _ratio(
            float(row.get("best_sota_mean_flow_s") or 0.0),
            float(row.get("candidate_mean_flow_s") or 0.0),
        ) >= 1.0 - PARETO_TOLERANCE
        for row in replayable
    )
    fixed_online_not_dominated = all(
        not _candidate_pareto_dominated_by_sota(row, policy_rows)
        for row in replayable
    )
    return {
        "policy_aggregate": policy_aggregate,
        "candidate_aggregate": candidate_aggregate,
        "union_makespan_beats_sota_envelope_all": union_makespan_beats,
        "union_mean_flow_beats_sota_envelope_all": union_flow_beats,
        "guarded_union_not_pareto_dominated_all": guarded_not_dominated,
        "single_guarded_union_beats_both_envelopes_all": single_guarded_both,
        "adaptive_scalarized_union_not_pareto_dominated_all": adaptive_not_dominated,
        "adaptive_scalarized_union_beats_both_envelopes_all": adaptive_both,
        "pareto_slack_union_not_pareto_dominated_all": pareto_slack_not_dominated,
        "pareto_slack_union_beats_both_envelopes_all": pareto_slack_both,
        "fixed_online_policy_pareto_dominates_sota_style_all": (
            fixed_online_not_dominated and fixed_online_both
        ),
        "fixed_online_policy_beats_both_envelopes_all": fixed_online_both,
        "fixed_online_policy_not_pareto_dominated_all": fixed_online_not_dominated,
        "gate_pass": (
            fixed_online_not_dominated
            and fixed_online_both
            and guarded_not_dominated
            and pareto_slack_not_dominated
        ),
    }


def _candidate_row_for(row: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    for candidate in rows:
        if (
            candidate["taskset"] == row["taskset"]
            and candidate["arrival_mode"] == row["arrival_mode"]
            and str(candidate["policy"]) == CANDIDATE_POLICY_NAME
        ):
            return candidate
    return dict(row)


def _candidate_policy(results: Mapping[str, Any]) -> Any:
    if CANDIDATE_POLICY_NAME in results:
        return results[CANDIDATE_POLICY_NAME]
    for name, result in results.items():
        if str(name).startswith("calibrated_"):
            return result
    raise KeyError(f"missing candidate policy {CANDIDATE_POLICY_NAME!r}")


def _candidate_pareto_dominated_by_sota(
    scenario: Mapping[str, Any],
    policy_rows: list[dict[str, Any]],
) -> bool:
    taskset = scenario.get("taskset")
    arrival = scenario.get("arrival_mode")
    candidate = next(
        (
            row for row in policy_rows
            if row.get("taskset") == taskset
            and row.get("arrival_mode") == arrival
            and row.get("policy") == CANDIDATE_POLICY_NAME
        ),
        None,
    )
    if candidate is None:
        return True
    floor = 1.0 - PARETO_TOLERANCE
    for row in policy_rows:
        if row.get("taskset") != taskset or row.get("arrival_mode") != arrival:
            continue
        if row.get("policy_family") != "sota_style":
            continue
        ms = _ratio(float(row.get("makespan_s") or 0.0), float(candidate.get("makespan_s") or 0.0))
        flow = _ratio(float(row.get("mean_flow_s") or 0.0), float(candidate.get("mean_flow_s") or 0.0))
        if ms < floor and flow < floor:
            return True
    return False


def _policy_metadata(policy: str) -> dict[str, Any]:
    if policy == "legacy_fixed_caps":
        return {
            "policy_family": "legacy",
            "baseline_name": "Scheduleurm legacy",
            "representative_systems": [],
        }
    if policy == CANDIDATE_POLICY_NAME:
        return {
            "policy_family": "candidate",
            "baseline_name": "Scheduleurm theorem-facing online candidate",
            "representative_systems": [],
        }
    if policy.startswith("scheduleurm_sota_union_"):
        return {
            "policy_family": "candidate_union",
            "baseline_name": "Scheduleurm + SOTA candidate action union",
            "representative_systems": [
                "Gavel",
                "Pollux",
                "Sia",
                "IADeep",
                "Salus",
                "SRPT",
                "Gittins",
            ],
        }
    for spec in sota_baseline_specs():
        if spec.policy.name == policy:
            return {
                "policy_family": "sota_style",
                "baseline_name": spec.name,
                "representative_systems": list(spec.representative_systems),
            }
    return {
        "policy_family": "unknown",
        "baseline_name": policy,
        "representative_systems": [],
    }


def _ratio(baseline_value: float, candidate_value: float) -> float:
    return float(baseline_value) / float(candidate_value) if float(candidate_value) > 0 else 0.0


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "taskset",
        "arrival_mode",
        "policy",
        "policy_family",
        "baseline_name",
        "representative_systems",
        "makespan_s",
        "mean_flow_s",
        "p90_flow_s",
        "completed_jobs",
        "candidate_vs_policy_makespan",
        "candidate_vs_policy_mean_flow",
        "profiles",
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), sort_keys=True)
                        if key in {"representative_systems", "profiles"}
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_candidate_union_gate(
        tasksets=tuple(x.strip() for x in args.tasksets.split(",") if x.strip()),
        arrivals=tuple(x.strip() for x in args.arrivals.split(",") if x.strip()),
        trace_seed=args.trace_seed,
        replay_seed=args.replay_seed,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    if args.csv_output:
        _write_csv(args.csv_output, report.get("policy_matrix") or [])
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_candidate_union_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build the SOTA candidate-union policy-semantics gate")
    build.add_argument("--tasksets", default=",".join(DEFAULT_TASKSETS))
    build.add_argument("--arrivals", default=",".join(DEFAULT_ARRIVALS))
    build.add_argument("--trace-seed", type=int, default=42)
    build.add_argument("--replay-seed", type=int, default=7)
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "sota_candidate_union_gate_20260613.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "sota_candidate_union_gate_20260613.md"),
    )
    build.add_argument(
        "--csv-output",
        default=str(ARTIFACT_ROOT / "sota_candidate_union_policy_matrix_20260613.csv"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
