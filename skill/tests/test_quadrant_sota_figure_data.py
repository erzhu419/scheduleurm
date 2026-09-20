import pytest

from md.figures.make_scheduleurm_quadrant_sota_grid import (
    POLICY_ID_TO_BASELINE,
    POLICY_LABELS,
    _validate_final_replay,
    figure_population,
    quadrant_points,
    scenario_best_policies,
)


def test_unified_replay_is_inverted_to_baseline_over_scheduleurm():
    report = _report()

    points = quadrant_points(report)

    point = points["q01"]["throughput_table_goodput"]
    assert point["x"] == 1.25
    assert point["y"] == 2.0
    assert point["scenario_count"] == 1


def test_controlled_migration_and_duplicate_migration_view_are_excluded():
    report = _report()
    report["pareto_rows"].extend(
        [
            _pareto_row("hardware:q01", "q01", "with_migration", 0.25, 0.25),
            _pareto_row("migration:q01:p25", "q01", "without_migration", 0.10, 0.10),
        ]
    )

    point = quadrant_points(report)["q01"]["throughput_table_goodput"]

    assert point["x"] == 1.25
    assert point["y"] == 2.0
    assert point["scenario_count"] == 1


def test_best_sota_uses_lower_baseline_over_ours_cost():
    report = _report()
    comparison = report["pareto_rows"][0]["comparisons"][0]
    second = dict(comparison)
    second["baseline_policy"] = "sota_srpt_gittins_mean_flow_oracle"
    second["ours_to_baseline_mean_flow_ratio"] = 1.0 / 1.10
    second["ours_to_baseline_makespan_ratio"] = 1.0 / 1.15
    report["pareto_rows"][0]["comparisons"].append(second)

    best = scenario_best_policies(report)

    assert best["q01"] == POLICY_ID_TO_BASELINE["sota_srpt_gittins_mean_flow_oracle"]


def test_final_figure_source_rejects_stale_or_incomplete_replay():
    report = _report()
    report.update(
        {
            "schema_version": 3,
            "status": "PASS",
            "pass": True,
            "comparison_kind": "same-cache_policy-semantics",
            "full_stack_external_binary_comparison": False,
            "run_count": 2,
            "policy_count": 8,
        }
    )

    _validate_final_replay(report)
    report["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        _validate_final_replay(report)


def test_incomplete_policy_excludes_the_trace_for_every_plotted_baseline():
    report = _report()
    report["pareto_rows"].append(
        _pareto_row("hardware:q01", "q01", "without_migration", 0.01, 0.01, trace_id="trace:two")
    )
    report["runs"].extend(
        [dict(row, trace_id="trace:two") for row in list(report["runs"])]
    )
    report["runs"].append(
        dict(report["runs"][0], trace_id="trace:two", policy="ablation", completion_status="incomplete_tail")
    )

    point = quadrant_points(report)["q01"]["throughput_table_goodput"]
    population = figure_population(report)

    assert point["x"] == 1.25
    assert point["y"] == 2.0
    assert point["scenario_count"] == 1
    assert population["registered_hardware_trace_count"] == 2
    assert population["included_complete_trace_count"] == 1
    assert population["excluded_incomplete_trace_count"] == 1
    assert population["excluded_incomplete_trace_count_by_quadrant"]["q01"] == 1


def test_policy_labels_describe_the_implemented_rules():
    assert POLICY_LABELS["throughput_table_goodput"] == "Target-support rule"
    assert POLICY_LABELS["finish_time_fairness"] == "Target-support rule (duplicate)"
    assert POLICY_LABELS["delay_oracle"] == "Lower-service delay rule"


def _report():
    return {
        "gate": "unified_hardware_or_replay",
        "scenarios": [
            {
                "scenario_id": "hardware:q01",
                "quadrant": "q01",
                "scenario_kind": "hardware_local",
            },
            {
                "scenario_id": "migration:q01:p25",
                "quadrant": "q01",
                "scenario_kind": "controlled_migration_counterfactual",
            },
        ],
        "pareto_rows": [
            _pareto_row("hardware:q01", "q01", "without_migration", 0.80, 0.50)
        ],
        "runs": [
            {
                "scenario_id": "hardware:q01",
                "trace_id": "trace:one",
                "migration_mode": "without_migration",
                "policy": policy,
                "completion_status": "complete",
            }
            for policy in ("scheduleurm_unified_robust_maxweight", "sota_gavel_pollux_sia_table_goodput")
        ],
    }


def _pareto_row(scenario_id, quadrant, migration_mode, flow_ratio, makespan_ratio, trace_id="trace:one"):
    return {
        "scenario_id": scenario_id,
        "trace_id": trace_id,
        "quadrant": quadrant,
        "migration_mode": migration_mode,
        "comparisons": [
            {
                "baseline_policy": "sota_gavel_pollux_sia_table_goodput",
                "ours_to_baseline_mean_flow_ratio": flow_ratio,
                "ours_to_baseline_makespan_ratio": makespan_ratio,
            }
        ],
    }
