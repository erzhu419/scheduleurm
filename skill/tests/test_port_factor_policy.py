from algorithm.experiments.port_factor_policy_development import (
    cell_key,
    load_candidate_plans,
    minimax_regret_selection,
)
from algorithm.experiments.port_factor_policy_holdout import verify_preregistration
from algorithm.experiments.port_public_external_holdout import verify_suite_manifest


def _row(a, b):
    return {
        "policy_metrics": {
            "plan|a": {
                "quay_makespan": a[0],
                "mean_quay_flow_time": a[1],
                "source_objective_cost": a[2],
            },
            "plan|b": {
                "quay_makespan": b[0],
                "mean_quay_flow_time": b[1],
                "source_objective_cost": b[2],
            },
        }
    }


def test_cell_key_is_unambiguous():
    assert cell_key(5, "160-0.2", "1.5") == "Q5|S160-0.2|D1.5"


def test_minimax_regret_prefers_robust_plan_not_best_mean_only():
    rows = [
        _row((10.0, 10.0, 10.0), (9.0, 9.0, 14.0)),
        _row((10.0, 10.0, 10.0), (9.0, 9.0, 14.0)),
    ]
    selected = minimax_regret_selection(rows, ("plan|a", "plan|b"))
    assert selected["selected_plan_id"] == "plan|a"
    assert selected["exact_finite_family_argmin"] is True


def test_minimax_regret_tie_is_deterministic():
    selected = minimax_regret_selection(
        [_row((10.0, 10.0, 10.0), (10.0, 10.0, 10.0))],
        ("plan|b", "plan|a"),
    )
    assert selected["selected_plan_id"] == "plan|a"


def test_registered_candidate_family_is_hash_bound_and_complete():
    plans = load_candidate_plans()
    assert len(plans) == 31
    assert len({row.plan_id for row in plans}) == 31


def test_r85_r86_confirmation_is_committed_disjoint_and_preregistered():
    suite = verify_suite_manifest(
        "tests/data/port_bacasp_s_factor_holdout_r85_r86/source_manifest.json",
        expected_replications=(85, 86),
    )
    freeze = verify_preregistration()
    assert suite["ready"] is True
    assert suite["instance_count"] == 54
    assert suite["factor_cell_count"] == 27
    assert suite["development_holdout_path_disjoint"] is True
    assert suite["development_holdout_hash_disjoint"] is True
    assert freeze["ready"] is True
    assert len(freeze["factor_policy"]) == 27
