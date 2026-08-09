from md.figures.make_scheduleurm_quadrant_sota_grid import (
    POLICY_ID_TO_BASELINE,
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
    }


def _pareto_row(scenario_id, quadrant, migration_mode, flow_ratio, makespan_ratio):
    return {
        "scenario_id": scenario_id,
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
