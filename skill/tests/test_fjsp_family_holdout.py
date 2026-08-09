from __future__ import annotations

from pathlib import Path

from algorithm.experiments.fjsp_family_holdout import evaluate_instance
from algorithm.experiments.fjsp_instances import load_fjsp_instance


ROOT = Path(__file__).resolve().parents[2]


def test_family_evaluation_separates_protocol_from_superiority():
    instance = load_fjsp_instance(ROOT / "tests" / "data" / "fjsp_brandimarte_tiny.fjs")
    row = evaluate_instance(
        instance,
        lower_bound=1.0,
        upper_bound=100.0,
        run_cp_sat=False,
        fixed_reference_budget_s=1.0,
    )

    assert row["ours"]["selection"]["exact_generated_family_argmax"] is True
    assert row["ours"]["performance_superiority_claim_ready"] is False
    assert row["ours_pareto_nondominated"] is True
    assert row["reference_ready"] is True
    assert set(row["baseline_relations"])


def test_family_evaluation_uses_both_flow_and_makespan_relations():
    instance = load_fjsp_instance(ROOT / "tests" / "data" / "fjsp_brandimarte_tiny.fjs")
    row = evaluate_instance(
        instance,
        lower_bound=1.0,
        upper_bound=100.0,
        run_cp_sat=False,
        fixed_reference_budget_s=1.0,
    )

    assert all(
        relation in {"ours_dominates", "baseline_dominates", "tradeoff", "tie"}
        for relation in row["baseline_relations"].values()
    )
    assert row["ours_makespan_over_public_upper_bound"] > 0
