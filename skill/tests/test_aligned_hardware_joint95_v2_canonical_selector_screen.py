from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v2_canonical_selector_screen as screen


def test_reused_roster_is_exactly_the_frozen_seed13041_set():
    contract = screen._reused_roster_contract()
    queues = screen._load_reused_queues(contract)

    assert contract["seed"] == 13041
    assert contract["size"] == 512
    assert contract["candidate_draws"] == 545
    assert len(queues) == len(set(queues)) == 512
    assert list(queues[:3]) == [(26, 6, 13), (47, 19, 28), (21, 2, 6)]


def test_registration_reads_no_v2_outcome_and_invokes_no_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration = tmp_path / "canonical-v2-protocol.json"
    fixed = tmp_path / "canonical-v2-fixed.json"
    forbidden_paths = {
        screen.DEFAULT_CAPACITY_RESULT.resolve(),
        screen.DEFAULT_RESULT.resolve(),
    }
    original = screen._read_object
    reads: list[Path] = []

    def guarded_read(path):
        resolved = Path(path).resolve()
        reads.append(resolved)
        if resolved in forbidden_paths or screen.DEFAULT_RESULTS_DIR.resolve() in resolved.parents:
            raise AssertionError("registration read v2 empirical evidence")
        return original(path)

    def forbidden_selector(*args, **kwargs):
        raise AssertionError("registration invoked the canonical selector")

    monkeypatch.setattr(screen, "_read_object", guarded_read)
    monkeypatch.setattr(screen.base, "_screen_lane", forbidden_selector)
    protocol = screen.register(registration, fixed)

    assert registration.is_file() and fixed.is_file()
    assert not screen.DEFAULT_RESULT.exists()
    assert not (set(reads) & forbidden_paths)
    assert protocol["selector_contract"]["fixed_budget_by_lane"] == {"8": 36, "16": 72}
    assert protocol["selector_contract"]["cold_canonical_reference_budget"] == 4096
    assert protocol["queue_roster_contract"]["seed"] == 13041
    assert protocol["queue_roster_contract"]["size"] == 512
    assert protocol["registration_scope"]["v2_results_directory_read"] is False
    assert protocol["registration_scope"]["v2_capacity_result_read"] is False
    assert protocol["registration_scope"]["roster_rng_draws"] == 0
    assert protocol["v1_contribution"] == {
        "joint_units": 0,
        "joint_scores": 0,
        "capacity_rows": 0,
        "holdout_rows": 0,
        "result_directory_read_during_registration": False,
    }


def test_run_refuses_incomplete_v2_before_capacity_result_or_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registration = tmp_path / "canonical-v2-protocol.json"
    fixed = tmp_path / "canonical-v2-fixed.json"
    output = tmp_path / "canonical-v2-result.json"
    screen.register(registration, fixed)
    original = screen._read_object

    def guarded_read(path):
        if Path(path).resolve() == screen.DEFAULT_CAPACITY_RESULT.resolve():
            raise AssertionError("incomplete v2 gate read capacity result")
        return original(path)

    def forbidden_selector(*args, **kwargs):
        raise AssertionError("incomplete v2 gate invoked selector work")

    monkeypatch.setattr(screen, "_read_object", guarded_read)
    monkeypatch.setattr(screen.base, "_screen_lane", forbidden_selector)
    monkeypatch.setattr(screen.v2_dispatcher, "load_registered_manifest", lambda path: {})
    monkeypatch.setattr(
        screen.v2_dispatcher,
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

    with pytest.raises(screen.V2CanonicalScreenRefusal, match="incomplete at the data gate"):
        screen.run_registered(registration, fixed, output)
    assert not output.exists()


def test_success_validation_requires_exact_v2_complete_state(monkeypatch: pytest.MonkeyPatch):
    protocol = {
        "source_paths": {
            "collection_manifest": str(screen.DEFAULT_COLLECTION_MANIFEST),
            "joint_results_directory": str(screen.DEFAULT_RESULTS_DIR),
        }
    }
    monkeypatch.setattr(screen.v2_dispatcher, "load_registered_manifest", lambda path: {})
    monkeypatch.setattr(
        screen.v2_dispatcher,
        "collection_status",
        lambda manifest, results: {
            "status": "COMPLETE",
            "errors": [],
            "successful_calibration_count": 19,
            "holdout_released": True,
            "holdout_success": True,
            "next_wave": None,
        },
    )

    state = screen.require_v2_complete(protocol)
    assert state["status"] == "COMPLETE"
