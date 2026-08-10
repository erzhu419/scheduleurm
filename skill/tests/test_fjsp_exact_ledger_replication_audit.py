from __future__ import annotations

from copy import deepcopy

from algorithm.experiments.fjsp_exact_ledger_replication_audit import (
    build_replication_audit,
)


def _compact() -> dict:
    return {
        "source_contract": {"source": "frozen"},
        "execution_contract": {"protocol": "committed"},
        "candidate_family_aggregate": {"instance_count": 1},
        "cp_sat_reference_aggregate": {"feasible_reference_count": 1},
        "rows": [
            {
                "relative_path": "edata/example.txt",
                "selected_exact_ledger": {"makespan_ticks": 10},
                "exact_nondominated_by_fixed_policy_seeds": True,
                "fixed_policy_dominators": [],
                "same_action_policies": [],
                "cp_sat_status": "FEASIBLE",
                "cp_sat_reference_feasible": True,
                "cp_sat_optimality_proved": False,
                "cp_sat_exact_relation": "comparator_dominates",
                "candidate_family_minus_reference": {
                    "makespan_ticks": 2,
                    "sum_completion_ticks": 3,
                },
            }
        ],
    }


def test_numeric_cp_sat_variation_does_not_hide_candidate_reproducibility():
    primary = _compact()
    replication = deepcopy(primary)
    replication["rows"][0]["candidate_family_minus_reference"]["makespan_ticks"] = 1

    report = build_replication_audit(primary, replication)

    assert report["gate"]["status"] == "FJSP_REPLICATION_AUDIT_PASS_WITH_CP_SAT_VARIATION"
    assert report["candidate_exact_ledger_reproducibility_ready"] is True
    assert report["cp_sat_qualitative_disposition_reproducibility_ready"] is True
    assert report["cp_sat_wall_clock_numeric_reproducibility_ready"] is False
    assert report["strict_full_artifact_reproducibility_ready"] is False
