from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments import aligned_hardware_joint95_v3_capacity_closure as old_capacity
from algorithm.experiments import aligned_hardware_joint95_v3_canonical_selector_screen as old_canonical
from algorithm.experiments import aligned_hardware_joint95_v5_baseline_confirmation_v3 as baseline
from algorithm.experiments import aligned_hardware_joint95_v5_canonical_selector_screen as canonical
from algorithm.experiments import aligned_hardware_joint95_v5_capacity_closure as capacity
from algorithm.experiments import aligned_hardware_joint95_v5_postcampaign_runner as postcampaign
from algorithm.experiments import critical_gpu_joint95_v3_registration as old_campaign
from algorithm.experiments import critical_gpu_joint95_v5_dispatcher as dispatcher
from algorithm.experiments import critical_gpu_joint95_v5_registration as campaign


def test_capacity_rejects_v3_collection():
    with pytest.raises(capacity.ClosureRefusal):
        capacity.build_registration(
            collection_manifest_path=old_campaign.DEFAULT_OUTPUT,
            results_dir=old_campaign.DEFAULT_RESULTS_DIR,
        )


def test_canonical_rejects_v3_capacity():
    with pytest.raises(canonical.V5CanonicalScreenRefusal):
        canonical._capacity_contract(canonical._read_object(old_capacity.DEFAULT_REGISTRATION))


def test_baseline_rejects_v3_canonical():
    source = baseline._read_object(old_canonical.DEFAULT_REGISTRATION)
    with pytest.raises(baseline.V5BaselineRefusal):
        baseline._canonical_contract(source, source.get("capacity_registration_contract") or {})


def test_postcampaign_rejects_v3_collection():
    with pytest.raises((postcampaign.PostcampaignRefusal, dispatcher.DispatchRefusal)):
        postcampaign.build_registration(
            manifest_path=old_campaign.DEFAULT_OUTPUT,
            results_dir=old_campaign.DEFAULT_RESULTS_DIR,
        )


def test_postcampaign_waits_for_v5_complete_without_running_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest = {
        "results_contract": {"results_directory": str(tmp_path / "results")}
    }
    monkeypatch.setattr(
        postcampaign,
        "_load_registration",
        lambda _path: (
            {"source_paths": {"queue_status": str(tmp_path / "queue.json")}},
            manifest,
            [],
        ),
    )
    monkeypatch.setattr(
        postcampaign, "_active_v5_collection_tasks", lambda _path: []
    )
    monkeypatch.setattr(
        dispatcher,
        "collection_status",
        lambda actual, _results: {
            "status": "READY_CALIBRATION",
            "errors": [],
        }
        if actual is manifest
        else (_ for _ in ()).throw(AssertionError("manifest changed")),
    )
    monkeypatch.setattr(
        postcampaign,
        "_run_stage",
        lambda _stage: (_ for _ in ()).throw(
            AssertionError("downstream stage must not run")
        ),
    )
    report = postcampaign.postcampaign_step(Path("unused"), dry_run=False)
    assert report["status"] == "WAIT_COLLECTION"
    assert report["stages_run"] == []


def test_frozen_20260919_artifacts_validate_completed_campaign():
    assert campaign.DEFAULT_OUTPUT.name.endswith("20260919.json")
    manifest = dispatcher.load_registered_manifest(campaign.DEFAULT_OUTPUT)
    status = dispatcher.collection_status(
        manifest, campaign.DEFAULT_RESULTS_DIR
    )
    assert status["status"] == "COMPLETE"
    assert status["successful_calibration_waves"] == list(range(94, 113))
    assert status["successful_calibration_count"] == 19
    assert status["holdout_success"] is True
    assert status["next_wave"] is None
