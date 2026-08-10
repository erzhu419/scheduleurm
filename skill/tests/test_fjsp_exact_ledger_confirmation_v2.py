from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from algorithm.experiments.fjsp_benchmark import POLICY_ORDER, build_schedule
from algorithm.experiments.fjsp_exact_ledger_confirmation_v2 import (
    FJSPExactLedgerConfirmationError,
    build_preregistration_draft,
    exact_completion_ledger,
    exact_ledger_relation,
    run_exact_ledger_confirmation,
    select_unused_instances,
    verify_frozen_contract,
    write_preregistration_draft,
)


INSTANCE_A = """\
2 2
2 2 1 3 2 2 1 1 2
2 2 1 2 2 4 1 2 3
"""

INSTANCE_B = """\
2 2
2 2 1 4 2 2 1 1 3
2 2 1 3 2 5 1 2 2
"""


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "family_a").mkdir(parents=True)
    (root / "family_a" / "a.txt").write_text(INSTANCE_A, encoding="utf-8")
    (root / "family_a" / "b.txt").write_text(INSTANCE_B, encoding="utf-8")
    return root


def _frozen_contract(
    tmp_path: Path, *, run_cp_sat_reference: bool = False
) -> tuple[Path, str, Path, Path]:
    source_root = _source_tree(tmp_path)
    selection = select_unused_instances(
        source_root,
        ["family_a/a.txt", "family_a/b.txt"],
        used_relative_paths=["family_a/b.txt"],
    )
    implementation_root = tmp_path / "implementation"
    implementation_root.mkdir()
    (implementation_root / "selector.py").write_text(
        "FROZEN_POLICY = 'v2'\n", encoding="utf-8"
    )
    preregistration = build_preregistration_draft(
        selection,
        source_identity={"repository": "fixture", "commit": "frozen"},
        implementation_root=implementation_root,
        implementation_paths=["selector.py"],
        registration_label="prospective-fixture-v2",
        run_cp_sat_reference=run_cp_sat_reference,
        cp_sat_time_limit_s=0.1,
    )
    preregistration_path = tmp_path / "preregistration.json"
    frozen = write_preregistration_draft(preregistration, preregistration_path)
    return (
        preregistration_path,
        frozen["sha256"],
        source_root,
        implementation_root,
    )


def test_unused_selection_is_outcome_blind_and_excludes_registered_source(
    tmp_path: Path,
) -> None:
    root = _source_tree(tmp_path)
    used_hash = sha256((root / "family_a" / "a.txt").read_bytes()).hexdigest()
    report = select_unused_instances(
        root,
        ["family_a/b.txt", "family_a/a.txt"],
        used_source_sha256=[used_hash],
        per_family=1,
    )
    assert report["outcomes_observed_during_selection"] is False
    assert report["selection_rule"]["uses_algorithm_outcomes"] is False
    assert report["selection_rule"]["uses_cp_sat_outcomes"] is False
    assert [row["relative_path"] for row in report["selected_instances"]] == [
        "family_a/b.txt"
    ]
    assert set(report["selected_instances"][0]) == {
        "relative_path",
        "family",
        "file_sha256",
        "canonical_source_sha256",
        "size_bytes",
        "job_count",
        "machine_count",
        "operation_count",
        "selection_scale",
    }


def test_source_preregistration_and_implementation_hashes_fail_closed(
    tmp_path: Path,
) -> None:
    preregistration, digest, source_root, implementation_root = _frozen_contract(
        tmp_path
    )
    verified = verify_frozen_contract(
        preregistration,
        expected_preregistration_sha256=digest,
        source_root=source_root,
        implementation_root=implementation_root,
    )
    assert verified["ready"] is True

    with pytest.raises(
        FJSPExactLedgerConfirmationError, match="preregistration hash mismatch"
    ):
        verify_frozen_contract(
            preregistration,
            expected_preregistration_sha256="0" * 64,
            source_root=source_root,
            implementation_root=implementation_root,
        )

    source_path = source_root / "family_a" / "a.txt"
    original_source = source_path.read_bytes()
    source_path.write_bytes(original_source + b"\n")
    with pytest.raises(FJSPExactLedgerConfirmationError, match="source hash mismatch"):
        verify_frozen_contract(
            preregistration,
            expected_preregistration_sha256=digest,
            source_root=source_root,
            implementation_root=implementation_root,
        )
    source_path.write_bytes(original_source)

    implementation = implementation_root / "selector.py"
    implementation.write_text("FROZEN_POLICY = 'drifted'\n", encoding="utf-8")
    with pytest.raises(
        FJSPExactLedgerConfirmationError, match="implementation hash mismatch"
    ):
        verify_frozen_contract(
            preregistration,
            expected_preregistration_sha256=digest,
            source_root=source_root,
            implementation_root=implementation_root,
        )


def test_exact_integer_ledger_distinguishes_same_action_and_distinct_tie(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)
    from algorithm.experiments.fjsp_instances import load_fjsp_instance

    instance = load_fjsp_instance(source_root / "family_a" / "a.txt")
    schedule = build_schedule(instance, POLICY_ORDER[0])["schedule"]
    first = exact_completion_ledger(instance, schedule)
    repeated = exact_completion_ledger(instance, schedule)
    assert exact_ledger_relation(first, repeated) == "same_action"
    assert isinstance(first["makespan_ticks"], int)
    assert isinstance(first["sum_completion_ticks"], int)

    fractional = [dict(row) for row in schedule]
    fractional[0]["start_time"] = 0.5
    with pytest.raises(FJSPExactLedgerConfirmationError, match="fractional"):
        exact_completion_ledger(instance, fractional)


def test_confirmation_keeps_candidate_family_and_cp_sat_gap_separate(
    tmp_path: Path,
) -> None:
    preregistration, digest, source_root, implementation_root = _frozen_contract(
        tmp_path, run_cp_sat_reference=True
    )

    def reference_runner(instance, **kwargs):
        del kwargs
        reference = build_schedule(instance, POLICY_ORDER[0])
        return {
            "status": "FEASIBLE",
            "feasible_solution": True,
            "optimality_proved": False,
            "verifier": reference["feasibility"],
            "schedule": reference["schedule"],
        }

    report = run_exact_ledger_confirmation(
        preregistration,
        expected_preregistration_sha256=digest,
        source_root=source_root,
        implementation_root=implementation_root,
        cp_sat_runner=reference_runner,
    )
    assert report["gate"]["pass"] is True
    assert report["gate"]["global_fjsp_optimality_claim_ready"] is False
    row = report["rows"][0]
    candidate = row["candidate_family_confirmation"]
    reference = row["cp_sat_reference_gap"]
    assert set(candidate["fixed_policy_exact_relations"]) == set(POLICY_ORDER)
    assert "cp_sat" not in candidate["fixed_policy_exact_relations"]
    assert reference["reference_outside_generated_candidate_family"] is True
    assert reference["reference_optimality_proved"] is False
    assert reference["global_optimality_claimed"] is False
    assert set(reference["candidate_family_minus_reference"]) == {
        "makespan_ticks",
        "sum_completion_ticks",
    }
    assert report["cp_sat_reference_aggregate"][
        "kept_separate_from_candidate_family_gate"
    ] is True
