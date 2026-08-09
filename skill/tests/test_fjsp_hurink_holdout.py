from pathlib import Path

from algorithm.experiments.fjsp_hurink_holdout import (
    compact_certificate,
    markdown_report,
)
from algorithm.experiments.fjsp_hurink_holdout_freeze import stratified_indices


def test_stratified_indices_span_subfamily():
    assert stratified_indices(9) == (0, 2, 4, 6, 8)


def test_compact_certificate_preserves_separate_gates():
    report = {
        "gate": {
            "pass": True,
            "status": "FJSP_HURINK_HOLDOUT_PASS",
            "complete_cp_sat_reference_coverage_ready": False,
            "performance_nondominance_ready": False,
        },
        "source": {},
        "protocol": {},
        "aggregate": {
            "instance_count": 1,
            "subfamily_counts": {"edata": 1},
            "ours_pareto_nondominated_count": 0,
            "ours_strict_every_baseline_count": 0,
            "geomean_ours_makespan_over_public_upper_bound": 1.2,
            "reference_ready_count": 0,
            "fixed_budget_cp_sat_feasible_count": 0,
            "fixed_budget_cp_sat_optimal_count": 0,
        },
        "rows": [
            {
                "relative_path": "edata/x.txt",
                "subfamily": "edata",
                "result": {
                    "instance": {"name": "x"},
                    "ours_makespan_over_public_upper_bound": 1.2,
                    "ours_pareto_nondominated": False,
                    "ours_strictly_dominates_every_baseline": False,
                    "reference_ready": False,
                    "exact_reference": {"fixed_budget": {"status": "UNKNOWN"}},
                },
            }
        ],
        "claim_boundary": {
            "protocol_does_not_depend_on_performance": True,
            "protocol_does_not_depend_on_cp_sat_returning_a_solution": True,
        },
    }
    compact = compact_certificate(report, "a" * 64)
    assert compact["gate"]["pass"] is True
    assert compact["gate"]["complete_cp_sat_reference_coverage_ready"] is False
    rendered = markdown_report(compact)
    assert "UNKNOWN solver rows remain missing references" in rendered
