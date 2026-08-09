from pathlib import Path

from algorithm.experiments.mmrcpsp_family_holdout import (
    evaluate_instance,
    instance_shape,
    markdown_report,
)
from algorithm.experiments.mmrcpsp_family_holdout_freeze import (
    instance_identifier,
    parse_optimum_table,
    stratified_indices,
)
from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm


ROOT = Path(__file__).resolve().parents[2]
INSTANCE = ROOT / "skill" / "tests" / "data" / "psplib_mm_suite" / "j1010_1.mm"


def test_stratified_indices_cover_endpoints_and_middle():
    assert stratified_indices(9) == (0, 2, 4, 6, 8)


def test_parse_public_optimum_table_and_instance_identifier(tmp_path):
    optimum = tmp_path / "j12.opt"
    optimum.write_text(
        "header\n1 1 16384 0.00\n10 8 49 0.50\n", encoding="ascii"
    )
    assert parse_optimum_table(optimum) == {(1, 1): 16384, (10, 8): 49}
    assert instance_identifier("j12", "j1210_8") == (10, 8)


def test_family_evaluation_separates_protocol_from_superiority():
    instance = parse_psplib_mm(INSTANCE)
    provisional = evaluate_instance(
        instance,
        public_optimal_makespan=1,
        run_cp_sat=False,
        fixed_reference_budget_s=0.1,
    )
    # The guard rejects an impossible public optimum only when ours is below it;
    # use the observed makespan as an exact test anchor on the second call.
    observed = int(provisional["ours"]["metrics"]["makespan"])
    report = evaluate_instance(
        instance,
        public_optimal_makespan=observed,
        run_cp_sat=False,
        fixed_reference_budget_s=0.1,
    )
    assert report["reference_ready"] is True
    assert report["ours_attains_public_optimum"] is True
    assert report["ours"]["selection"]["exact_generated_family_argmax"] is True
    assert instance_shape(instance)["real_job_count"] == 10


def test_markdown_states_claim_boundary():
    report = {
        "aggregate": {
            "instance_count": 1,
            "family_counts": {"j12": 1},
            "ours_pareto_nondominated_count": 1,
            "ours_strict_every_baseline_count": 0,
            "ours_attains_public_optimum_count": 0,
            "geomean_ours_makespan_over_public_optimum": 1.1,
        },
        "gate": {"pass": True},
        "rows": [
            {
                "family": "j12",
                "result": {
                    "instance": {"name": "j1210_8"},
                    "ours_makespan_over_optimum": 1.1,
                    "ours_pareto_nondominated": True,
                    "ours_strictly_dominates_every_baseline": False,
                },
            }
        ],
    }
    rendered = markdown_report(report)
    assert "Protocol validity does not imply performance superiority" in rendered
    assert "time-bounded CP-SAT" in rendered
