from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v2_baseline_confirmation_v3 as baseline


def _registrations(tmp_path: Path):
    canonical_protocol = tmp_path / "canonical-v2-protocol.json"
    canonical_fixed = tmp_path / "canonical-v2-fixed.json"
    baseline.canonical_v2.register(canonical_protocol, canonical_fixed)
    protocol = tmp_path / "baseline-v2-protocol.json"
    fixed = tmp_path / "baseline-v2-fixed.json"
    return canonical_protocol, protocol, fixed


def test_registration_is_v2_only_outcome_blind_and_reuses_corrected_v3_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical_protocol, protocol_path, fixed_path = _registrations(tmp_path)
    future_capacity = tmp_path / "future-capacity.json"
    future_canonical = tmp_path / "future-canonical.json"
    forbidden_paths = {future_capacity.resolve(), future_canonical.resolve()}
    original = baseline._read_object
    reads: list[Path] = []

    def guarded_read(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved in forbidden_paths or baseline.canonical_v2.DEFAULT_RESULTS_DIR.resolve() in resolved.parents:
            raise AssertionError("registration read v2 empirical evidence")
        return original(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("registration invoked trajectory or selector work")

    monkeypatch.setattr(baseline, "_read_object", guarded_read)
    monkeypatch.setattr(baseline.base, "_build_tasks", forbidden)
    monkeypatch.setattr(baseline.base, "_run_case_seed", forbidden)
    monkeypatch.setattr(baseline.base, "select_exact_renewal_mw", forbidden)
    frozen = baseline.register(
        protocol_path,
        fixed_path,
        canonical_registration=canonical_protocol,
        canonical_result=future_canonical,
        capacity_result=future_capacity,
    )

    assert protocol_path.is_file() and fixed_path.is_file()
    assert not (set(reads) & forbidden_paths)
    assert frozen["trajectory_contract"]["arrival_seeds"] == list(range(12041, 12061))
    assert frozen["trajectory_contract"]["case_seed_tapes"] == 120
    assert frozen["trajectory_contract"]["trajectory_runs"] == 480
    assert frozen["statistics"]["primary_family_size"] == 18
    assert frozen["v2_data_gate"]["score_only_static_result_required"] is False
    assert frozen["registration_scope"]["v1_score_only_static_protocol_read"] is False
    assert frozen["registration_scope"]["v1_score_only_static_result_read"] is False
    assert frozen["v1_contribution"] == {
        "joint_units": 0,
        "joint_scores": 0,
        "capacity_rows": 0,
        "holdout_rows": 0,
        "score_only_static_dependency": False,
    }


def test_run_refuses_incomplete_v2_before_results_tapes_or_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical_protocol, protocol_path, fixed_path = _registrations(tmp_path)
    future_capacity = tmp_path / "future-capacity.json"
    future_canonical = tmp_path / "future-canonical.json"
    output = tmp_path / "baseline-v2-result.json"
    baseline.register(
        protocol_path,
        fixed_path,
        canonical_registration=canonical_protocol,
        canonical_result=future_canonical,
        capacity_result=future_capacity,
    )
    original = baseline._read_object

    def guarded_read(path):
        if Path(path).resolve() in {future_capacity.resolve(), future_canonical.resolve()}:
            raise AssertionError("incomplete v2 gate read a prerequisite result")
        return original(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete v2 gate began scientific work")

    monkeypatch.setattr(baseline, "_read_object", guarded_read)
    monkeypatch.setattr(baseline.base, "_build_tasks", forbidden)
    monkeypatch.setattr(baseline.base, "_run_case_seed", forbidden)
    monkeypatch.setattr(baseline.base, "select_exact_renewal_mw", forbidden)
    monkeypatch.setattr(
        baseline.canonical_v2.v2_dispatcher,
        "load_registered_manifest",
        lambda path: {},
    )
    monkeypatch.setattr(
        baseline.canonical_v2.v2_dispatcher,
        "collection_status",
        lambda manifest, results: {
            "status": "READY_CALIBRATION",
            "errors": [],
            "successful_calibration_count": 0,
            "holdout_released": False,
            "holdout_success": False,
            "next_wave": 34,
        },
    )

    with pytest.raises(baseline.V2BaselineRefusal, match="incomplete at the data gate"):
        baseline.run_registered(protocol_path, fixed_path, output)
    assert not output.exists()


def test_adapter_points_to_corrected_independent_exact_mw_implementation():
    assert baseline.base.EXPERIMENT == "aligned_hardware_joint95_baseline_confirmation_v3"
    assert baseline.base.ARRIVAL_SEEDS == tuple(range(12041, 12061))
    assert baseline.base.POLICIES[1] == "exact_renewal_mw_milp"
    assert baseline.base.PRIMARY_FAMILY_SIZE == 18
