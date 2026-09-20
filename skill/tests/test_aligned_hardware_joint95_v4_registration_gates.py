from __future__ import annotations

import pytest

from algorithm.experiments import aligned_hardware_joint95_v3_capacity_closure as old_capacity
from algorithm.experiments import aligned_hardware_joint95_v3_canonical_selector_screen as old_canonical
from algorithm.experiments import aligned_hardware_joint95_v4_baseline_confirmation_v3 as baseline
from algorithm.experiments import aligned_hardware_joint95_v4_canonical_selector_screen as canonical
from algorithm.experiments import aligned_hardware_joint95_v4_capacity_closure as capacity
from algorithm.experiments import aligned_hardware_joint95_v4_postcampaign_runner as postcampaign
from algorithm.experiments import critical_gpu_joint95_v3_registration as old_campaign
from algorithm.experiments import critical_gpu_joint95_v4_dispatcher as dispatcher


def test_capacity_rejects_v3_collection():
    with pytest.raises(capacity.ClosureRefusal):
        capacity.build_registration(
            collection_manifest_path=old_campaign.DEFAULT_OUTPUT,
            results_dir=old_campaign.DEFAULT_RESULTS_DIR,
        )


def test_canonical_rejects_v3_capacity():
    with pytest.raises(canonical.V4CanonicalScreenRefusal):
        canonical._capacity_contract(canonical._read_object(old_capacity.DEFAULT_REGISTRATION))


def test_baseline_rejects_v3_canonical():
    source = baseline._read_object(old_canonical.DEFAULT_REGISTRATION)
    with pytest.raises(baseline.V4BaselineRefusal):
        baseline._canonical_contract(source, source.get("capacity_registration_contract") or {})


def test_postcampaign_rejects_v3_collection():
    with pytest.raises((postcampaign.PostcampaignRefusal, dispatcher.DispatchRefusal)):
        postcampaign.build_registration(
            manifest_path=old_campaign.DEFAULT_OUTPUT,
            results_dir=old_campaign.DEFAULT_RESULTS_DIR,
        )


def test_postcampaign_waits_for_v4_complete_without_running_stages():
    report = postcampaign.postcampaign_step(dry_run=False)
    assert report["status"] in {"WAIT_COLLECTION", "WAIT_COLLECTION_CONTROLLER"}
    assert report["stages_run"] == []
