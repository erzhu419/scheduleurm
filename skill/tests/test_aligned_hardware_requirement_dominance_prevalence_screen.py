import copy

import pytest

from algorithm.experiments import (
    aligned_hardware_requirement_dominance_prevalence_screen as screen,
)


def test_current_roster_stops_at_one_catalogue_topology():
    result = screen.run_screen(screen.build_protocol("2026-09-20T00:00:00+00:00"))

    assert result["status"] == "PASS"
    assert result["scientific_decision"] == "SINGLE_CATALOGUE_TOPOLOGY_STOP"
    assert result["snapshot_count"] == 17
    assert result["catalogue_topology_count"] == 1
    assert result["duration_variant_count"] == 4
    assert result["qualified_catalogue_topology_count"] == 1
    assert result["development_advance"] is False
    assert result["scaling_launched"] is False
    assert result["paper_claim_authorized"] is False
    assert all(result["checks"].values())


def test_all_duration_variants_have_the_same_natural_canonical_quotient():
    result = screen.run_screen(screen.build_protocol("2026-09-20T00:00:00+00:00"))

    variants = result["duration_variant_groups"]
    assert len(variants) == 4
    assert {variant["option_count"] for variant in variants} == {9}
    assert {variant["retained_option_count"] for variant in variants} == {8}
    assert {tuple(variant["removed_original_indices"]) for variant in variants} == {
        (1,)
    }
    assert {tuple(map(tuple, variant["canonical_lift_pairs"])) for variant in variants} == {
        ((1, 0),)
    }
    assert all(variant["canonical_lift_valid"] for variant in variants)
    assert all(variant["strict_requirement_duration_pair_count"] == 1 for variant in variants)


def test_roster_members_are_reported_without_result_filtering():
    result = screen.run_screen(screen.build_protocol("2026-09-20T00:00:00+00:00"))

    observed = {row["snapshot_id"] for row in result["snapshots"]}
    registered = {source.snapshot_id for source in screen.SOURCES}
    grouped = {
        snapshot_id
        for group in result["duration_variant_groups"]
        for snapshot_id in group["snapshot_ids"]
    }
    assert observed == registered == grouped
    assert result["work_accounting"] == {
        "server_pulls": 0,
        "rng_draws": 0,
        "selector_calls": 0,
        "trajectory_runs": 0,
        "catalogue_snapshots_read": 17,
        "duration_variants_reduced": 4,
    }


@pytest.mark.parametrize("mutation", ["decision_rule", "catalogue_contract"])
def test_registered_protocol_rejects_rule_or_catalogue_mutation(mutation):
    protocol = screen.build_protocol("2026-09-20T00:00:00+00:00")
    changed = copy.deepcopy(protocol)
    if mutation == "decision_rule":
        changed["decision_rule"]["SINGLE_CATALOGUE_TOPOLOGY_STOP"] = "changed"
    else:
        changed["catalogue_contract"][0]["model"]["options"][0]["duration"] += 1.0

    with pytest.raises(
        ValueError,
        match="protocol constants or normalized catalogue contents changed",
    ):
        screen.run_screen(changed)
