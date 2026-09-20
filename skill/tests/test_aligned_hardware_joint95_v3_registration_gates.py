from __future__ import annotations

import pytest

from algorithm.experiments import aligned_hardware_joint95_v3_baseline_confirmation_v3 as baseline
from algorithm.experiments import aligned_hardware_joint95_v3_canonical_selector_screen as canonical
from algorithm.experiments import aligned_hardware_joint95_v3_capacity_closure as capacity
from algorithm.experiments import aligned_hardware_joint95_v3_postcampaign_runner as postcampaign
from algorithm.experiments import critical_gpu_joint95_v2_registration as v2
from algorithm.experiments import critical_gpu_joint95_v3_dispatcher as dispatcher


def test_capacity_cannot_register_against_v2_collection():
    with pytest.raises(capacity.ClosureRefusal):
        capacity.build_registration(
            collection_manifest_path=v2.DEFAULT_OUTPUT,
            results_dir=v2.DEFAULT_RESULTS_DIR,
        )


def test_canonical_cannot_consume_v2_capacity_registration():
    from algorithm.experiments import aligned_hardware_joint95_v2_capacity_closure as old

    with pytest.raises(canonical.V3CanonicalScreenRefusal):
        canonical._capacity_contract(canonical._read_object(old.DEFAULT_REGISTRATION))


def test_baseline_cannot_consume_v2_canonical_registration():
    from algorithm.experiments import aligned_hardware_joint95_v2_canonical_selector_screen as old

    source = baseline._read_object(old.DEFAULT_REGISTRATION)
    with pytest.raises(baseline.V3BaselineRefusal):
        baseline._canonical_contract(source, source.get("capacity_registration_contract") or {})


def test_postcampaign_cannot_register_against_v2_collection():
    with pytest.raises((postcampaign.PostcampaignRefusal, dispatcher.DispatchRefusal)):
        postcampaign.build_registration(
            manifest_path=v2.DEFAULT_OUTPUT,
            results_dir=v2.DEFAULT_RESULTS_DIR,
        )
