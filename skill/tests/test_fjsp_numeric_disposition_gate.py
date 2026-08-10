from decimal import Decimal
from pathlib import Path

from algorithm.experiments.fjsp_numeric_disposition_gate import (
    build_disposition,
    exact_completion_ledger,
    exact_relation,
)


def _schedule(*, end: int, machine: int = 0):
    return [
        {
            "job_id": 0,
            "operation_id": 0,
            "machine_id": machine,
            "start_time": 0,
            "end_time": end,
        }
    ]


def test_same_schedule_is_same_action_without_float_tolerance() -> None:
    ours = exact_completion_ledger(_schedule(end=10), job_count=1)
    baseline = exact_completion_ledger(_schedule(end=10), job_count=1)
    assert exact_relation(ours, baseline) == "same_action"
    assert Decimal(ours["sum_completion_ticks"]) == Decimal(10)


def test_exact_relation_distinguishes_dominance_and_distinct_tie() -> None:
    fast = exact_completion_ledger(_schedule(end=9), job_count=1)
    slow = exact_completion_ledger(_schedule(end=10), job_count=1)
    tied_other = exact_completion_ledger(_schedule(end=9, machine=1), job_count=1)
    assert exact_relation(fast, slow) == "ours_dominates"
    assert exact_relation(slow, fast) == "baseline_dominates"
    assert exact_relation(fast, tied_other) == "metric_tie_distinct_action"


def test_frozen_artifacts_reclassify_only_two_false_dominance_rows() -> None:
    report = build_disposition()
    assert report["gate"]["pass"] is True
    assert len(report["numeric_false_dominance_rows"]) == 2
    assert {
        (row["group"], row["relative_path"])
        for row in report["numeric_false_dominance_rows"]
    } == {
        ("first_family_holdout", "ChambersBarnes1996/setb4xx.txt"),
        ("hurink_confirmation", "vdata/la10.txt"),
    }
    for group in report["groups"].values():
        assert group["exact_ledger_pareto_nondominated_count"] == 20
