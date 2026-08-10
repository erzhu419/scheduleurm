from __future__ import annotations

from pathlib import Path

from algorithm.experiments.mmrcpsp_class_balance_holdout import (
    PARETO_COST_COORDINATES,
    compact_class_balance_report,
    run_class_balance_holdout,
)
from algorithm.experiments.mmrcpsp_renewal_stream_benchmark import (
    ArrivalScenario,
    POLICY_ORDER,
    run_mmrcpsp_renewal_stream_benchmark,
)


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_PATHS = (
    ROOT / "tests" / "data" / "mmrcpsp_core.mm",
    ROOT / "tests" / "data" / "mmrcpsp_nonrenewable_reserve.mm",
)
SOURCE_SCENARIO = (
    ArrivalScenario("source", "static", "none", 0.0, 72, 2),
)
HOLDOUT_SCENARIOS = (
    ArrivalScenario("new_poisson", "poisson", "poisson", 0.65, 96, 1),
    ArrivalScenario(
        "new_bursty", "bursty", "markov_bursty", 0.65, 96, 1
    ),
)


def _source_artifact() -> dict:
    return {
        "report": run_mmrcpsp_renewal_stream_benchmark(
            INSTANCE_PATHS,
            scenarios=SOURCE_SCENARIO,
            seeds=(7,),
        )
    }


def test_new_tapes_preserve_pathwise_gates_and_class_balance_identity():
    report = run_class_balance_holdout(
        _source_artifact(), scenarios=HOLDOUT_SCENARIOS, seeds=(101,)
    )

    assert report["gate"]["pass"] is True
    assert report["aggregate"]["run_count"] == len(HOLDOUT_SCENARIOS) * len(
        POLICY_ORDER
    )
    assert report["aggregate"]["ours_exact_oracle_run_count"] == len(
        HOLDOUT_SCENARIOS
    )
    assert report["pareto_cost_coordinates"] == list(PARETO_COST_COORDINATES)
    for run in report["runs"]:
        assert run["gate"]["pass"] is True
        assert run["class_balance"]["total_time_average_queue"] == run[
            "performance"
        ]["time_average_queue"]


def test_compact_report_keeps_prospectivity_and_metric_boundaries():
    report = run_class_balance_holdout(
        _source_artifact(), scenarios=HOLDOUT_SCENARIOS, seeds=(101,)
    )
    compact = compact_class_balance_report(report)

    assert compact["claim_boundary"]["v1_total_cost_result_retained"] is True
    assert compact["claim_boundary"][
        "metric_family_registered_after_v1_total_cost_result"
    ] is True
    assert compact["claim_boundary"][
        "arrival_tapes_observed_before_metric_freeze"
    ] is False
    assert compact["claim_boundary"]["new_structural_instance_holdout"] is False
    assert compact["aggregate"]["run_count"] == len(compact["rows"])
