"""Export paper-facing summaries from the final schema-v2 replay.

The exporter is deliberately downstream of measurement and replay.  It never
launches work and never recomputes service.  Its only job is to bind paper
numbers to passing, hash-audited artifacts and to keep legacy, SOTA-style,
ablation, migration, queue, and slack summaries on one population.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_REPLAY = ARTIFACT_ROOT / "unified_hardware_or_replay_20260809.json"
DEFAULT_SLACK = ARTIFACT_ROOT / "critical_gpu_all_hardware_statewise_slack_20260810.json"
DEFAULT_LOADED_MERGE = ARTIFACT_ROOT / "critical_gpu_loaded_action_ledger_merge_20260809.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "unified_hardware_paper_results_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "unified_hardware_paper_results_20260810.md"
DEFAULT_TEX = REPO_ROOT / "paper" / "generated" / "unified_hardware_results.tex"

SOTA_POLICY_LABELS = {
    "sota_gavel_finish_time_fairness": "Gavel finish-time fairness",
    "sota_gavel_pollux_sia_table_goodput": "Gavel/Pollux/Sia table goodput",
    "sota_iadeep_salus_interference_guard": "IADeep/Salus interference guard",
    "sota_quadrant_composite": "Quadrant composite",
    "sota_salus_iadeep_packing_guard": "Salus/IADeep packing guard",
    "sota_sia_pollux_resource_adaptive": "Sia/Pollux resource-adaptive",
    "sota_srpt_gittins_mean_flow_oracle": "SRPT/Gittins delay oracle",
}

ABLATION_LABELS = {
    "ablation_delay_only": "Delay-only score",
    "ablation_high_profile_tiebreak": "High-profile tie-break",
    "ablation_no_bounded_penalty": "No bounded penalty",
    "ablation_no_loaded_ledger": "No loaded-action ledger",
    "ablation_no_migration": "No migration actions",
    "ablation_support_only": "Support-only score",
}


def build_unified_hardware_paper_results(
    *,
    replay_path: Path = DEFAULT_REPLAY,
    slack_path: Path = DEFAULT_SLACK,
    loaded_merge_path: Path = DEFAULT_LOADED_MERGE,
) -> dict[str, Any]:
    replay_source = Path(replay_path).resolve()
    slack_source = Path(slack_path).resolve()
    merge_source = Path(loaded_merge_path).resolve()
    common = {
        "gate": "unified_hardware_paper_results",
        "schema_version": 1,
        "status": "WAIT_INPUTS",
        "pass": False,
        "replay_path": str(replay_source),
        "slack_path": str(slack_source),
        "loaded_merge_path": str(merge_source),
        "errors": [],
    }
    missing = [
        str(path)
        for path in (replay_source, slack_source, merge_source)
        if not path.is_file()
    ]
    if missing:
        return {**common, "wait_reasons": [{"code": "MISSING_INPUT", "paths": missing}]}
    try:
        replay_raw = replay_source.read_bytes()
        slack_raw = slack_source.read_bytes()
        merge_raw = merge_source.read_bytes()
        replay = json.loads(replay_raw.decode("utf-8"))
        slack = json.loads(slack_raw.decode("utf-8"))
        loaded_merge = json.loads(merge_raw.decode("utf-8"))
        _validate_replay(replay)
        _validate_slack(slack)
        _validate_hash_chain(replay, slack, loaded_merge)
        summary = _summarize(replay, slack)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            **common,
            "status": "FAIL_INPUT_CONTRACT",
            "errors": [{"code": "INVALID_INPUT", "detail": f"{type(exc).__name__}: {exc}"}],
        }
    return {
        **common,
        "status": "PASS",
        "pass": True,
        "replay_sha256": hashlib.sha256(replay_raw).hexdigest(),
        "slack_sha256": hashlib.sha256(slack_raw).hexdigest(),
        "loaded_merge_sha256": hashlib.sha256(merge_raw).hexdigest(),
        **summary,
        "claim_boundary": (
            "These are paper-facing summaries of the exact schema-v2 measured-cache "
            "policy-semantics population. Legacy is fixed-cap semantics on the same "
            "cache; external policies are not full-stack binary executions. Performance "
            "diagnostics are reported even when superiority is false and are never used "
            "as input-gate assumptions."
        ),
    }


def _validate_replay(replay: Mapping[str, Any]) -> None:
    required = {
        "gate": "unified_hardware_or_replay",
        "schema_version": 2,
        "status": "PASS",
        "pass": True,
        "comparison_kind": "same-cache_policy-semantics",
        "full_stack_external_binary_comparison": False,
    }
    for field, expected in required.items():
        if replay.get(field) != expected:
            raise ValueError(f"replay.{field} expected {expected!r}, got {replay.get(field)!r}")
    checks = replay.get("coverage_checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("replay coverage checks are not all true")
    if not replay.get("runs") or not replay.get("pareto_rows") or not replay.get("legacy_comparison_rows"):
        raise ValueError("replay result matrix is empty")


def _validate_slack(slack: Mapping[str, Any]) -> None:
    required = {
        "gate": "critical_gpu_all_hardware_statewise_slack_certificate",
        "status": "PASS",
        "pass": True,
    }
    for field, expected in required.items():
        if slack.get(field) != expected:
            raise ValueError(f"slack.{field} expected {expected!r}, got {slack.get(field)!r}")
    nodes = slack.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 4 or not all(row.get("ready") is True for row in nodes):
        raise ValueError("slack certificate does not contain four ready node rows")


def _validate_hash_chain(
    replay: Mapping[str, Any],
    slack: Mapping[str, Any],
    loaded_merge: Mapping[str, Any],
) -> None:
    required = {
        "gate": "critical_gpu_loaded_action_ledger_merge",
        "status": "PASS",
        "pass": True,
        "registered_loaded_action_ledger_ready": True,
        "unified_cache_ready": True,
    }
    for field, expected in required.items():
        if loaded_merge.get(field) != expected:
            raise ValueError(
                f"loaded_merge.{field} expected {expected!r}, got {loaded_merge.get(field)!r}"
            )
    manifest = replay.get("input_manifest") or {}
    cache_hash = str((manifest.get("unified_cache") or {}).get("sha256") or "")
    ledger_hash = str((manifest.get("loaded_action_ledger") or {}).get("sha256") or "")
    expected = {
        "base_cache_sha256": str(slack.get("cache_sha256") or ""),
        "cache_output_sha256": cache_hash,
        "ledger_output_sha256": ledger_hash,
    }
    for field, value in expected.items():
        if not value or str(loaded_merge.get(field) or "") != value:
            raise ValueError(f"loaded merge hash chain mismatch at {field}")


def _summarize(replay: Mapping[str, Any], slack: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = list(replay["scenarios"])
    runs = list(replay["runs"])
    scenario_metadata = {str(row["scenario_id"]): row for row in scenarios}
    hardware_ids = {
        str(row["scenario_id"])
        for row in scenarios
        if row.get("scenario_kind") == "hardware_local"
    }
    migration_ids = {
        str(row["scenario_id"])
        for row in scenarios
        if row.get("scenario_kind") == "controlled_migration_counterfactual"
    }
    hardware_ours = [
        row for row in runs
        if row["scenario_id"] in hardware_ids
        and row["migration_mode"] == "without_migration"
        and row["policy_category"] == "ours"
    ]
    if not hardware_ours:
        raise ValueError("hardware-local Scheduleurm population is empty")
    legacy_rows = [
        row for row in replay["legacy_comparison_rows"]
        if row["scenario_id"] in hardware_ids
        and row["migration_mode"] == "without_migration"
    ]
    pareto_rows = [
        row for row in replay["pareto_rows"]
        if row["scenario_id"] in hardware_ids
        and row["migration_mode"] == "without_migration"
    ]
    return {
        "population": {
            "hardware_scenario_count": len(hardware_ids),
            "migration_scenario_count": len(migration_ids),
            "hardware_trace_count": len({str(row["trace_id"]) for row in hardware_ours}),
            "hardware_scheduleurm_run_count": len(hardware_ours),
            "complete_matrix_run_count": int(replay["run_count"]),
            "policy_count": int(replay["policy_count"]),
            "arrival_families": sorted({str(row["arrival_family"]) for row in hardware_ours}),
            "quadrants": sorted({str(row["quadrant"]) for row in hardware_ours}),
        },
        "legacy": _legacy_summary(legacy_rows),
        "online_legacy": _legacy_summary(
            [row for row in legacy_rows if row["arrival_family"] != "static"]
        ),
        "static_scenarios": _static_scenario_summary(
            runs,
            hardware_ids,
            scenario_metadata,
        ),
        "sota_style": _sota_summary(pareto_rows),
        "ablations": _ablation_summary(runs, hardware_ids),
        "migration": _migration_summary(runs, migration_ids),
        "queues": _queue_summary(hardware_ours),
        "slack": _slack_summary(slack),
    }


def _static_scenario_summary(
    runs: Sequence[Mapping[str, Any]],
    hardware_ids: set[str],
    scenario_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize like-for-like closed-batch comparisons by hardware scenario."""

    grouped = _group_runs(runs, scenario_ids=hardware_ids, migration_mode="without_migration")
    by_scenario: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for (scenario_id, _trace_id), rows in grouped.items():
        if not rows or str(rows[0]["arrival_family"]) != "static":
            continue
        by_scenario.setdefault(scenario_id, []).append(
            (_one(rows, "ours"), _one(rows, "legacy"))
        )

    missing = sorted(hardware_ids - set(by_scenario))
    if missing:
        raise ValueError(f"static comparison lacks hardware scenarios {missing!r}")

    output = []
    for scenario_id, pairs in sorted(by_scenario.items()):
        metadata = scenario_metadata[scenario_id]
        output.append(
            {
                "scenario_id": scenario_id,
                "quadrant": str(metadata["quadrant"]),
                "node_bucket": str(metadata["node_bucket"]),
                "workload_keys": [str(value) for value in metadata["workload_keys"]],
                "trace_count": len(pairs),
                "legacy_to_ours_makespan_geomean_ratio": _geomean(
                    [_ratio(legacy["makespan_s"], ours["makespan_s"]) for ours, legacy in pairs]
                ),
                "legacy_to_ours_mean_flow_geomean_ratio": _geomean(
                    [_ratio(legacy["mean_flow_s"], ours["mean_flow_s"]) for ours, legacy in pairs]
                ),
                "legacy_to_ours_p90_flow_geomean_ratio": _geomean(
                    [_ratio(legacy["p90_flow_s"], ours["p90_flow_s"]) for ours, legacy in pairs]
                ),
            }
        )
    return output


def _legacy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("legacy comparison population is empty")
    return {
        **_ratio_summary(rows),
        "ours_strictly_dominates_every_row": all(
            bool(row["ours_strictly_pareto_dominates_legacy"]) for row in rows
        ),
        "legacy_strictly_dominates_ours_count": sum(
            bool(row["legacy_strictly_pareto_dominates_ours"]) for row in rows
        ),
        "by_quadrant": {
            quadrant: _ratio_summary([row for row in rows if row["quadrant"] == quadrant])
            for quadrant in sorted({str(row["quadrant"]) for row in rows})
        },
        "by_arrival_family": {
            family: _ratio_summary([row for row in rows if row["arrival_family"] == family])
            for family in sorted({str(row["arrival_family"]) for row in rows})
        },
    }


def _ratio_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    makespan = [float(row["legacy_to_ours_makespan_ratio"]) for row in rows]
    flow = [float(row["legacy_to_ours_mean_flow_ratio"]) for row in rows]
    return {
        "comparison_count": len(rows),
        "legacy_to_ours_makespan_geomean_ratio": _geomean(makespan),
        "legacy_to_ours_mean_flow_geomean_ratio": _geomean(flow),
        "legacy_to_ours_makespan_median_ratio": _quantile(makespan, 0.5),
        "legacy_to_ours_mean_flow_median_ratio": _quantile(flow, 0.5),
        "legacy_to_ours_makespan_worst_ratio": min(makespan),
        "legacy_to_ours_mean_flow_worst_ratio": min(flow),
        "legacy_to_ours_makespan_p05_ratio": _quantile(makespan, 0.05),
        "legacy_to_ours_makespan_p95_ratio": _quantile(makespan, 0.95),
        "legacy_to_ours_mean_flow_p05_ratio": _quantile(flow, 0.05),
        "legacy_to_ours_mean_flow_p95_ratio": _quantile(flow, 0.95),
    }


def _sota_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("SOTA comparison population is empty")
    by_policy: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for comparison in row["comparisons"]:
            by_policy.setdefault(str(comparison["baseline_policy"]), []).append(comparison)
    policy_rows = {}
    for policy, comparisons in sorted(by_policy.items()):
        makespan = [1.0 / _positive(row["ours_to_baseline_makespan_ratio"], "SOTA ratio") for row in comparisons]
        flow = [1.0 / _positive(row["ours_to_baseline_mean_flow_ratio"], "SOTA ratio") for row in comparisons]
        policy_rows[policy] = {
            "comparison_count": len(comparisons),
            "baseline_to_ours_makespan_geomean_ratio": _geomean(makespan),
            "baseline_to_ours_mean_flow_geomean_ratio": _geomean(flow),
            "baseline_strictly_dominates_ours_count": sum(
                bool(row["baseline_strictly_pareto_dominates"]) for row in comparisons
            ),
            "ours_strictly_dominates_baseline_count": sum(
                bool(row["ours_strictly_pareto_dominates"]) for row in comparisons
            ),
        }
    return {
        "comparison_trace_count": len(rows),
        "ours_not_dominated_in_every_trace": all(
            bool(row["ours_not_pareto_dominated_by_any_sota_style_policy"])
            for row in rows
        ),
        "ours_strictly_dominates_every_policy_in_every_trace": all(
            bool(row["ours_strictly_dominates_every_sota_style_policy"])
            for row in rows
        ),
        "policies": policy_rows,
    }


def _ablation_summary(
    runs: Sequence[Mapping[str, Any]], hardware_ids: set[str]
) -> dict[str, Any]:
    grouped = _group_runs(runs, scenario_ids=hardware_ids, migration_mode="without_migration")
    by_policy: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for rows in grouped.values():
        ours = _one(rows, "ours")
        for row in rows:
            if row["policy_category"] == "ablation":
                by_policy.setdefault(str(row["policy"]), []).append((ours, row))
    out = {}
    for policy, pairs in sorted(by_policy.items()):
        makespan = [_ratio(ablation["makespan_s"], ours["makespan_s"]) for ours, ablation in pairs]
        flow = [_ratio(ablation["mean_flow_s"], ours["mean_flow_s"]) for ours, ablation in pairs]
        out[policy] = {
            "comparison_count": len(pairs),
            "ablation_to_full_makespan_geomean_ratio": _geomean(makespan),
            "ablation_to_full_mean_flow_geomean_ratio": _geomean(flow),
            "ablation_strictly_dominates_full_count": sum(
                _dominates(ablation, ours) for ours, ablation in pairs
            ),
        }
    return out


def _migration_summary(
    runs: Sequence[Mapping[str, Any]], migration_ids: set[str]
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in runs:
        if row["scenario_id"] not in migration_ids or row["policy_category"] != "ours":
            continue
        grouped.setdefault((str(row["scenario_id"]), str(row["trace_id"])), []).append(row)
    rows = []
    for (scenario_id, trace_id), values in sorted(grouped.items()):
        without = _one(values, "ours", migration_mode="without_migration")
        with_migration = _one(values, "ours", migration_mode="with_migration")
        rows.append(
            {
                "scenario_id": scenario_id,
                "trace_id": trace_id,
                "without_to_with_makespan_ratio": _ratio(without["makespan_s"], with_migration["makespan_s"]),
                "without_to_with_mean_flow_ratio": _ratio(without["mean_flow_s"], with_migration["mean_flow_s"]),
                "migration_action_selected": any(
                    str(action).startswith("migrate:")
                    for action in with_migration["selected_action_counts"]
                ),
            }
        )
    return {
        "comparison_count": len(rows),
        "migration_selected_count": sum(bool(row["migration_action_selected"]) for row in rows),
        "without_to_with_makespan_geomean_ratio": _geomean(
            [row["without_to_with_makespan_ratio"] for row in rows]
        ),
        "without_to_with_mean_flow_geomean_ratio": _geomean(
            [row["without_to_with_mean_flow_ratio"] for row in rows]
        ),
        "rows": rows,
    }


def _queue_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean_backlog = [float(row["mean_queue_backlog_jobs"]) for row in rows]
    max_backlog = [float(row["max_queue_backlog_jobs"]) for row in rows]
    mean_unfinished = [float(row["mean_unfinished_jobs"]) for row in rows]
    max_unfinished = [float(row["max_unfinished_jobs"]) for row in rows]
    return {
        "run_count": len(rows),
        "mean_of_time_weighted_queue_backlog_jobs": sum(mean_backlog) / len(mean_backlog),
        "maximum_queue_backlog_jobs": int(max(max_backlog)),
        "mean_of_time_weighted_unfinished_jobs": sum(mean_unfinished) / len(mean_unfinished),
        "maximum_unfinished_jobs": int(max(max_unfinished)),
        "distributions": {
            "time_weighted_queue_backlog_jobs": _distribution(mean_backlog),
            "maximum_queue_backlog_jobs": _distribution(max_backlog),
            "time_weighted_unfinished_jobs": _distribution(mean_unfinished),
            "maximum_unfinished_jobs": _distribution(max_unfinished),
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution population is empty")
    return {
        "count": len(values),
        "mean": sum(float(value) for value in values) / len(values),
        "minimum": min(float(value) for value in values),
        "p05": _quantile(values, 0.05),
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "maximum": max(float(value) for value in values),
    }


def _slack_summary(slack: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "node": str(row["node"]),
            "execution_class": str(row["operational_execution_class"]),
            "delta": float(row["delta"]),
            "eta": float(row["eta"]),
            "nominal_lcb_coverage": float(row["nominal_lcb_coverage"]),
        }
        for row in slack["nodes"]
    ]
    return {
        "load_fraction": float(slack["load_fraction"]),
        "minimum_eta": min(row["eta"] for row in rows),
        "simultaneous_four_node_union_bound_coverage": float(
            slack["simultaneous_four_node_union_bound_coverage"]
        ),
        "nodes": rows,
    }


def _group_runs(
    runs: Sequence[Mapping[str, Any]], *, scenario_ids: set[str], migration_mode: str
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in runs:
        if row["scenario_id"] in scenario_ids and row["migration_mode"] == migration_mode:
            grouped.setdefault((str(row["scenario_id"]), str(row["trace_id"])), []).append(row)
    return grouped


def _one(
    rows: Sequence[Mapping[str, Any]], category: str, *, migration_mode: str | None = None
) -> Mapping[str, Any]:
    selected = [
        row for row in rows
        if row["policy_category"] == category
        and (migration_mode is None or row["migration_mode"] == migration_mode)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {category}/{migration_mode} row, got {len(selected)}")
    return selected[0]


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        float(left["makespan_s"]) <= float(right["makespan_s"])
        and float(left["mean_flow_s"]) <= float(right["mean_flow_s"])
        and (
            float(left["makespan_s"]) < float(right["makespan_s"])
            or float(left["mean_flow_s"]) < float(right["mean_flow_s"])
        )
    )


def _ratio(numerator: Any, denominator: Any) -> float:
    return _positive(numerator, "ratio numerator") / _positive(denominator, "ratio denominator")


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _geomean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean population is empty")
    return math.exp(sum(math.log(_positive(value, "geomean value")) for value in values) / len(values))


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("quantile population is empty")
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def tex_macros(report: Mapping[str, Any]) -> str:
    population = report["population"]
    legacy = report["legacy"]
    online_legacy = report["online_legacy"]
    sota = report["sota_style"]
    ablations = report["ablations"]
    migration = report["migration"]
    queues = report["queues"]
    slack = report["slack"]
    queue_distributions = queues["distributions"]
    rows = {
        "UnifiedHardwareScenarioCount": population["hardware_scenario_count"],
        "UnifiedHardwareTraceCount": population["hardware_trace_count"],
        "UnifiedCompleteMatrixRunCount": population["complete_matrix_run_count"],
        "UnifiedPolicyCount": population["policy_count"],
        "UnifiedLegacyMakespanGeoRatio": _fmt(legacy["legacy_to_ours_makespan_geomean_ratio"]),
        "UnifiedLegacyMeanFlowGeoRatio": _fmt(legacy["legacy_to_ours_mean_flow_geomean_ratio"]),
        "UnifiedLegacyMakespanWorstRatio": _fmt(legacy["legacy_to_ours_makespan_worst_ratio"]),
        "UnifiedLegacyMeanFlowWorstRatio": _fmt(legacy["legacy_to_ours_mean_flow_worst_ratio"]),
        "UnifiedOnlineComparisonCount": online_legacy["comparison_count"],
        "UnifiedOnlineLegacyMakespanGeoRatio": _fmt(
            online_legacy["legacy_to_ours_makespan_geomean_ratio"]
        ),
        "UnifiedOnlineLegacyMeanFlowGeoRatio": _fmt(
            online_legacy["legacy_to_ours_mean_flow_geomean_ratio"]
        ),
        "UnifiedOnlineLegacyMakespanMedianRatio": _fmt(
            online_legacy["legacy_to_ours_makespan_median_ratio"]
        ),
        "UnifiedOnlineLegacyMeanFlowMedianRatio": _fmt(
            online_legacy["legacy_to_ours_mean_flow_median_ratio"]
        ),
        "UnifiedOnlineLegacyMakespanMinRatio": _fmt(
            online_legacy["legacy_to_ours_makespan_worst_ratio"]
        ),
        "UnifiedOnlineLegacyMeanFlowMinRatio": _fmt(
            online_legacy["legacy_to_ours_mean_flow_worst_ratio"]
        ),
        "UnifiedOnlineLegacyMakespanPFiveRatio": _fmt(
            online_legacy["legacy_to_ours_makespan_p05_ratio"]
        ),
        "UnifiedOnlineLegacyMeanFlowPFiveRatio": _fmt(
            online_legacy["legacy_to_ours_mean_flow_p05_ratio"]
        ),
        "UnifiedOnlineLegacyMakespanPNinetyFiveRatio": _fmt(
            online_legacy["legacy_to_ours_makespan_p95_ratio"]
        ),
        "UnifiedOnlineLegacyMeanFlowPNinetyFiveRatio": _fmt(
            online_legacy["legacy_to_ours_mean_flow_p95_ratio"]
        ),
        "UnifiedQueueMeanBacklogMean": _fmt(
            queue_distributions["time_weighted_queue_backlog_jobs"]["mean"]
        ),
        "UnifiedQueueMeanBacklogMedian": _fmt(
            queue_distributions["time_weighted_queue_backlog_jobs"]["median"]
        ),
        "UnifiedQueueMeanBacklogMin": _fmt(
            queue_distributions["time_weighted_queue_backlog_jobs"]["minimum"]
        ),
        "UnifiedQueueMeanBacklogPFive": _fmt(
            queue_distributions["time_weighted_queue_backlog_jobs"]["p05"]
        ),
        "UnifiedQueueMeanBacklogPNinetyFive": _fmt(
            queue_distributions["time_weighted_queue_backlog_jobs"]["p95"]
        ),
        "UnifiedQueueMaxBacklogMean": _fmt(
            queue_distributions["maximum_queue_backlog_jobs"]["mean"]
        ),
        "UnifiedQueueMaxBacklogMedian": _fmt(
            queue_distributions["maximum_queue_backlog_jobs"]["median"]
        ),
        "UnifiedQueueMaxBacklogMin": _fmt(
            queue_distributions["maximum_queue_backlog_jobs"]["minimum"]
        ),
        "UnifiedQueueMaxBacklogPFive": _fmt(
            queue_distributions["maximum_queue_backlog_jobs"]["p05"]
        ),
        "UnifiedQueueMaxBacklogPNinetyFive": _fmt(
            queue_distributions["maximum_queue_backlog_jobs"]["p95"]
        ),
        "UnifiedSotaNotDominated": "true" if sota["ours_not_dominated_in_every_trace"] else "false",
        "UnifiedMigrationScenarioCount": migration["comparison_count"],
        "UnifiedMigrationSelectedCount": migration["migration_selected_count"],
        "UnifiedMigrationMakespanGeoRatio": _fmt(
            migration["without_to_with_makespan_geomean_ratio"]
        ),
        "UnifiedMigrationMeanFlowGeoRatio": _fmt(
            migration["without_to_with_mean_flow_geomean_ratio"]
        ),
        "UnifiedMinimumHardwareEta": _fmt(slack["minimum_eta"]),
    }
    lines = ["% Generated by unified_hardware_paper_results.py; do not edit manually."]
    lines.extend(f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in rows.items())
    table_rows = {
        "UnifiedStaticLegacyRows": "\n".join(
            f"{_latex_escape(row['scenario_id'])} & {row['trace_count']} & "
            f"{_fmt(row['legacy_to_ours_makespan_geomean_ratio'])} & "
            f"{_fmt(row['legacy_to_ours_mean_flow_geomean_ratio'])} & "
            f"{_fmt(row['legacy_to_ours_p90_flow_geomean_ratio'])} \\\\"
            for row in report["static_scenarios"]
        ),
        "UnifiedSotaPolicyRows": "\n".join(
            f"{_latex_escape(SOTA_POLICY_LABELS.get(policy, policy))} & "
            f"{row['comparison_count']} & "
            f"{_fmt(row['baseline_to_ours_makespan_geomean_ratio'])} & "
            f"{_fmt(row['baseline_to_ours_mean_flow_geomean_ratio'])} & "
            f"{row['baseline_strictly_dominates_ours_count']} \\\\"
            for policy, row in sorted(sota["policies"].items())
        ),
        "UnifiedAblationRows": "\n".join(
            f"{_latex_escape(ABLATION_LABELS.get(policy, policy))} & "
            f"{row['comparison_count']} & "
            f"{_fmt(row['ablation_to_full_makespan_geomean_ratio'])} & "
            f"{_fmt(row['ablation_to_full_mean_flow_geomean_ratio'])} & "
            f"{row['ablation_strictly_dominates_full_count']} \\\\"
            for policy, row in sorted(ablations.items())
        ),
        "UnifiedHardwareSlackRows": "\n".join(
            f"{_latex_escape(row['node'])} & {_latex_escape(row['execution_class'])} & "
            f"{_fmt(row['delta'])} & {_fmt(row['eta'])} & "
            f"{_fmt(row['nominal_lcb_coverage'])} \\\\"
            for row in slack["nodes"]
        ),
    }
    lines.extend(
        f"\\providecommand{{\\{name}}}{{%\n{value}\n}}"
        for name, value in table_rows.items()
    )
    return "\n".join(lines) + "\n"


def markdown_report(report: Mapping[str, Any]) -> str:
    population = report.get("population") or {}
    legacy = report.get("legacy") or {}
    sota = report.get("sota_style") or {}
    migration = report.get("migration") or {}
    slack = report.get("slack") or {}
    lines = [
        "# Unified Hardware Paper Results",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Hardware scenarios: `{population.get('hardware_scenario_count', 0)}`",
        f"- Hardware traces: `{population.get('hardware_trace_count', 0)}`",
        f"- Complete matrix runs: `{population.get('complete_matrix_run_count', 0)}`",
        f"- Legacy/ours makespan geomean: `{legacy.get('legacy_to_ours_makespan_geomean_ratio', 0):.6f}`",
        f"- Legacy/ours mean-flow geomean: `{legacy.get('legacy_to_ours_mean_flow_geomean_ratio', 0):.6f}`",
        f"- SOTA non-dominance on every trace: `{str(bool(sota.get('ours_not_dominated_in_every_trace'))).lower()}`",
        f"- Migration comparisons: `{migration.get('comparison_count', 0)}`",
        f"- Minimum hardware-local eta: `{slack.get('minimum_eta', 0):.6f}`",
        "",
        "## SOTA-style policies",
        "",
        "| Policy | Comparisons | Makespan baseline/ours | Mean flow baseline/ours | Baseline dominates count |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, row in sorted((sota.get("policies") or {}).items()):
        lines.append(
            f"| `{policy}` | {row['comparison_count']} | "
            f"{row['baseline_to_ours_makespan_geomean_ratio']:.6f} | "
            f"{row['baseline_to_ours_mean_flow_geomean_ratio']:.6f} | "
            f"{row['baseline_strictly_dominates_ours_count']} |"
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    lines.extend(
        [
            "## Static hardware-local legacy comparison",
            "",
            "| Scenario | Traces | Makespan legacy/ours | Mean flow legacy/ours | P90 flow legacy/ours |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("static_scenarios") or []:
        lines.append(
            f"| `{row['scenario_id']}` | {row['trace_count']} | "
            f"{row['legacy_to_ours_makespan_geomean_ratio']:.6f} | "
            f"{row['legacy_to_ours_mean_flow_geomean_ratio']:.6f} | "
            f"{row['legacy_to_ours_p90_flow_geomean_ratio']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return f"{float(value):.6f}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_outputs(
    report: Mapping[str, Any], *, output: Path, markdown: Path, tex: Path
) -> None:
    _atomic_write(output, json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown, markdown_report(report))
    _atomic_write(tex, tex_macros(report))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--slack", type=Path, default=DEFAULT_SLACK)
    parser.add_argument("--loaded-merge", type=Path, default=DEFAULT_LOADED_MERGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_unified_hardware_paper_results(
        replay_path=args.replay,
        slack_path=args.slack,
        loaded_merge_path=args.loaded_merge,
    )
    if report["pass"]:
        write_outputs(report, output=args.output, markdown=args.markdown, tex=args.tex)
    else:
        _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
