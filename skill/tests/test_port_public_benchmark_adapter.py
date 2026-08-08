from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from algorithm.experiments.port_public_benchmark_adapter import (
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
    PublicBacaspSPortSimulator,
    adapt_bacasp_s_to_port_instance,
    build_port_public_benchmark,
    load_source_manifest,
    parse_bacasp_s_instance,
    verify_public_fixture,
)
from algorithm.experiments.port_scheduling_benchmark import ALL_POLICIES, THEOREM_POLICY


EXPECTED_FIXTURE_SHA256 = "20f7610929fa26726535f0dedbb95971ecead4994e16fb5f4633aa55b4029a36"
EXPECTED_REPOSITORY_COMMIT = "e8eda86ec0435105ca313afe85987e4f4ab2cbd7"


@pytest.fixture(scope="module")
def source_manifest():
    return load_source_manifest(DEFAULT_MANIFEST)


@pytest.fixture(scope="module")
def raw_instance(source_manifest):
    return parse_bacasp_s_instance(
        DEFAULT_FIXTURE,
        expected_sha256=source_manifest["fixture"]["sha256"],
    )


@pytest.fixture(scope="module")
def adapted_instance(raw_instance):
    return adapt_bacasp_s_to_port_instance(raw_instance)


@pytest.fixture(scope="module")
def public_report():
    return build_port_public_benchmark(determinism_repeats=2)


def test_fixture_is_unmodified_pinned_public_source(source_manifest):
    payload = DEFAULT_FIXTURE.read_bytes()
    fixture = source_manifest["fixture"]
    dataset = source_manifest["dataset"]

    assert hashlib.sha256(payload).hexdigest() == EXPECTED_FIXTURE_SHA256
    assert fixture["sha256"] == EXPECTED_FIXTURE_SHA256
    assert len(payload) == fixture["byte_count"] == 1324
    assert len(payload.splitlines()) == fixture["line_count"] == 32
    assert fixture["raw_bytes_modified"] is False
    assert fixture["raw_url"].startswith(
        "https://raw.githubusercontent.com/juanfdata/bacasp_setup/"
        + EXPECTED_REPOSITORY_COMMIT
    )
    assert dataset["repository_commit"] == EXPECTED_REPOSITORY_COMMIT
    assert dataset["license"]["spdx"] == "CC-BY-SA-4.0"
    assert len(dataset["license"]["sha256"]) == 64
    assert len(dataset["paper"]["institutional_pdf_sha256"]) == 64


def test_authenticity_gate_checks_hash_size_lines_commit_and_license():
    result = verify_public_fixture()

    assert result["ready"] is True
    assert result["actual_sha256"] == EXPECTED_FIXTURE_SHA256
    assert all(result["checks"].values())
    assert result["repository_commit"] == EXPECTED_REPOSITORY_COMMIT
    assert result["license"]["spdx"] == "CC-BY-SA-4.0"


def test_parser_recovers_large_mb_header_and_exact_u_iq(raw_instance):
    assert raw_instance.quay_length == 50.0
    assert raw_instance.horizon == 300.0
    assert raw_instance.crane_count == 10
    assert raw_instance.vessel_count == len(raw_instance.vessels) == 30
    assert raw_instance.crane_speed == 160.0
    assert raw_instance.crane_setup_time == 0.2
    assert raw_instance.vessels[0].position_cost == 200.0
    assert raw_instance.vessels[0].waiting_cost == 1000.0
    assert raw_instance.vessels[0].delay_cost == 2000.0
    assert raw_instance.vessels[0].processing_times == ((1, 10.0), (2, 5.4))
    assert raw_instance.vessels[1].processing_times == (
        (4, 14.6),
        (5, 12.0),
        (6, 10.2),
    )
    # This boundary row verifies the source's one-based b_i <= L + 1 - l_i convention.
    assert raw_instance.vessels[22].length == 12.0
    assert raw_instance.vessels[22].desired_position == 39.0


def test_parser_rejects_changed_bytes_and_nonintegral_counts(tmp_path):
    changed = tmp_path / "changed.dat"
    changed.write_bytes(DEFAULT_FIXTURE.read_bytes().replace(b"20 4 12.0", b"20 4 12.1", 1))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        parse_bacasp_s_instance(changed, expected_sha256=EXPECTED_FIXTURE_SHA256)

    malformed = tmp_path / "nonintegral_header.dat"
    lines = DEFAULT_FIXTURE.read_text(encoding="ascii").splitlines()
    lines[0] = "50 300 10.5 30"
    malformed.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="crane_count must be an integer"):
        parse_bacasp_s_instance(malformed)


def test_mapping_separates_source_core_from_synthetic_tail(adapted_instance):
    instance = adapted_instance.port_instance
    augmentation = adapted_instance.augmentation

    assert len(instance.berths) == 2
    assert len(instance.quay_cranes) == 10
    assert len(instance.vessels) == 30
    assert adapted_instance.berth_intervals == {
        "public_berth_01": (0.0, 35.0),
        "public_berth_02": (35.0, 50.0),
    }
    assert augmentation["continuous_quay_projection"]["conservative_subset"] is True
    assert augmentation["neutral_draft"]["source_field"] is False
    assert augmentation["yard"]["kind"] == "synthetic_augmentation"
    assert augmentation["yard"]["source_field"] is False
    assert augmentation["gate"]["kind"] == "synthetic_augmentation"
    assert augmentation["gate"]["source_field"] is False
    assert instance.max_reberths_per_vessel == 0
    assert instance.max_crane_reassignments_per_vessel == 0
    assert instance.max_yard_rehandles_per_vessel == 0


def test_candidate_family_enforces_source_q_bounds_and_specific_consecutive_cranes(
    adapted_instance,
):
    simulator = PublicBacaspSPortSimulator(adapted_instance, THEOREM_POLICY)
    simulator.clock = 5.0
    simulator._release_arrivals()
    actions = [
        row for row in simulator.feasible_atomic_actions() if row.action_type == "start_quay"
    ]

    vessel_one = [row for row in actions if row.vessel_id == "public_v01"]
    vessel_two = [row for row in actions if row.vessel_id == "public_v02"]
    assert {len(row.metadata["crane_ids"]) for row in vessel_one} == {1, 2}
    assert {len(row.metadata["crane_ids"]) for row in vessel_two} == {4, 5, 6}
    assert any(
        tuple(row.metadata["crane_ids"]) == ("qc_05", "qc_06", "qc_07", "qc_08")
        for row in vessel_two
    )
    for row in actions:
        numbers = sorted(int(value.rsplit("_", 1)[-1]) for value in row.metadata["crane_ids"])
        assert numbers == list(range(numbers[0], numbers[-1] + 1))


def test_quay_rate_uses_public_u_iq_exactly(adapted_instance):
    simulator = PublicBacaspSPortSimulator(adapted_instance, THEOREM_POLICY)
    runtime = simulator.vessels["public_v02"]

    rate = simulator._quay_rate(
        runtime,
        "public_berth_01",
        ("qc_01", "qc_02", "qc_03", "qc_04"),
        lower=False,
    )
    assert rate == pytest.approx(1.0 / 14.6)


def test_complete_public_matrix_is_deterministic_and_resource_feasible(public_report):
    assert public_report["pass"] is True
    assert public_report["status"] == "PORT_PUBLIC_BACASP_S_ADAPTER_PASS"
    assert public_report["source_authenticity_ready"] is True
    assert public_report["parser_ready"] is True
    assert public_report["same_policy_matrix_ready"] is True
    assert public_report["same_metric_schema_ready"] is True
    assert public_report["deterministic_reproduction_ready"] is True
    assert public_report["resource_feasibility_ready"] is True
    assert public_report["source_time_invariant_assignment_ready"] is True
    assert public_report["mid_service_reconfiguration_admitted"] is False
    assert tuple(public_report["policy_results"]) == tuple(ALL_POLICIES)

    for policy in ALL_POLICIES:
        result = public_report["policy_results"][policy]
        audit = result["public_source_feasibility_audit"]
        assert result["pass"] is True
        assert result["public_source_core_metrics"]["completed_source_vessels"] == 30
        assert audit["source_min_max_cranes_ready"] is True
        assert audit["source_specific_cranes_consecutive_ready"] is True
        assert audit["source_u_iq_processing_time_ready"] is True
        assert audit["rail_endpoint_order_ready"] is True
        assert audit["initial_crane_position_observed"] is False
        assert audit["zero_transition_on_first_crane_use_ready"] is True
        assert audit["resource_feasibility_ready"] is True
        assert result["public_source_core_metrics"]["crane_setup_travel_duration"] > 0.0
        assert result["action_counts"].get("reberth", 0) == 0
        assert result["action_counts"].get("crane_reassignment", 0) == 0
        assert result["action_counts"].get("yard_rehandle", 0) == 0
        assert public_report["determinism"][policy]["deterministic"] is True


def test_selected_joint_actions_are_resource_disjoint(public_report):
    for result in public_report["policy_results"].values():
        for slot in result["decision_audits"]:
            resources = [
                resource
                for action in slot["selected_actions"]
                for resource in action["resource_ids"]
            ]
            tasks = [action["task_id"] for action in slot["selected_actions"]]
            assert len(resources) == len(set(resources))
            assert len(tasks) == len(set(tasks))


def test_report_keeps_scoped_validity_separate_from_strong_claims(public_report):
    assert public_report["scoped_claim_ready"] is True
    assert public_report["continuous_quay_reproduction_ready"] is False
    assert public_report["physical_port_claim_ready"] is False
    assert public_report["published_optimum_comparison_ready"] is False
    assert isinstance(public_report["ours_source_core_pareto_nondominated"], bool)
    assert public_report["claim_boundary"]["does_not_support"]
    wording = public_report["claim_boundary"]["augmentation_wording"]
    assert "schema augmentations" in wording
    assert "source-core metrics stop at quay completion" in wording
