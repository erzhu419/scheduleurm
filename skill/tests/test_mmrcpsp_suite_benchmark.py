from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from algorithm.experiments.mmrcpsp_benchmark import OURS_POLICY
from algorithm.experiments.mmrcpsp_suite_benchmark import (
    MANIFEST_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SuiteIntegrityError,
    build_psplib_mm_suite_benchmark,
    write_artifacts,
)


SUITE_ROOT = Path(__file__).resolve().parent / "data" / "psplib_mm_suite"
MANIFEST = SUITE_ROOT / "source_manifest.json"
INSTANCE = SUITE_ROOT / "j1010_1.mm"
OFFICIAL_INSTANCE_SHA256 = (
    "73f6489376462981d935bbcda80ba231e12651ea1eeebb50745f1d413150f75c"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "31776e005269a8a8324fe73312ab01c1330b3623692020c282ca769cb2a5393d"
)


def _copy_suite(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "suite"
    root.mkdir()
    shutil.copy2(INSTANCE, root / INSTANCE.name)
    shutil.copy2(MANIFEST, root / MANIFEST.name)
    return root, root / MANIFEST.name


def test_official_fixture_is_byte_locked_and_manifested():
    payload = INSTANCE.read_bytes()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert hashlib.sha256(payload).hexdigest() == OFFICIAL_INSTANCE_SHA256
    assert len(payload) == 2956
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["archive_sha256"] == OFFICIAL_ARCHIVE_SHA256
    assert manifest["archive_size_bytes"] == 426593
    assert manifest["archive_url"] == (
        "https://www.om-db.wi.tum.de/psplib/"
        "download_dataset.php?set=j10&mode=mm&format=zip"
    )
    assert manifest["coverage"] == "representative_official_instance_only"
    assert manifest["instances"] == [
        {
            "archive_member": "j1010_1.mm",
            "file": "j1010_1.mm",
            "horizon": 77,
            "jobs_including_dummies": 12,
            "nonrenewable_capacities": [42, 17],
            "renewable_capacities": [11, 9],
            "sha256": OFFICIAL_INSTANCE_SHA256,
            "size_bytes": 2956,
        }
    ]


def test_suite_report_has_feasibility_metrics_pareto_and_aggregates():
    report = build_psplib_mm_suite_benchmark(MANIFEST)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["gate"] == {
        "status": "OFFICIAL_PSPLIB_MM_SUITE_PASS",
        "pass": True,
        "source_integrity_verified": True,
        "all_policy_schedules_feasible": True,
        "zero_resource_violation": True,
        "zero_precedence_violation": True,
        "dominance_not_required_for_pass": True,
    }
    assert report["verified_instances"] == [
        {
            "file": "j1010_1.mm",
            "archive_member": "j1010_1.mm",
            "sha256": OFFICIAL_INSTANCE_SHA256,
            "size_bytes": 2956,
            "jobs_including_dummies": 12,
            "horizon": 77,
            "renewable_capacities": [11, 9],
            "nonrenewable_capacities": [42, 17],
        }
    ]

    instance = report["instances"]["j1010_1"]
    for result in instance["policy_results"].values():
        assert result["feasible"] is True
        assert result["metrics"]["schedule_feasible"] is True
        assert result["metrics"]["resource_violation_units"] == 0.0
        assert result["metrics"]["precedence_violation_count"] == 0
        assert result["metrics"]["makespan"] > 0
        assert result["metrics"]["mean_flow_time"] > 0

    pareto = report["per_instance_pareto"]["j1010_1"]
    assert OURS_POLICY in pareto["pareto_policies"]
    assert "shortest_processing_time" not in pareto["pareto_policies"]
    assert pareto["best_makespan"] == 20.0
    assert pareto["best_mean_flow_time"] == 7.7
    assert pareto["ours_comparison"][
        "ours_strictly_dominates_every_baseline"
    ] is False

    ours = report["suite_aggregate"][OURS_POLICY]
    assert ours["feasible_instance_count"] == 1
    assert ours["pareto_instance_count"] == 1
    assert ours["geometric_mean_makespan_ratio_to_instance_best"] == 1.0
    assert ours["geometric_mean_flow_ratio_to_instance_best"] == 1.0


def test_dominance_is_reported_honestly_and_never_used_as_gate():
    report = build_psplib_mm_suite_benchmark(MANIFEST)
    boundary = report["claim_boundary"]

    assert report["gate"]["pass"] is True
    assert report["gate"]["dominance_not_required_for_pass"] is True
    assert boundary[
        "ours_strictly_dominates_every_baseline_on_every_instance"
    ] is False
    assert boundary["dominance_is_descriptive_not_a_gate"] is True
    assert boundary["full_psplib_j10_suite_claim_ready"] is False
    assert boundary["exact_or_state_of_the_art_mmrcpsp_claim_ready"] is False
    assert boundary["strong_claim_ready"] is False
    assert "not the complete official" in boundary["strong_claim_blockers"][0]


def test_coverage_label_alone_cannot_fabricate_a_complete_suite_claim(tmp_path):
    root, manifest_path = _copy_suite(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = "complete_official_j10_multi_mode_archive"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_psplib_mm_suite_benchmark(manifest_path, instance_root=root)

    assert report["gate"]["pass"] is True
    assert report["claim_boundary"]["full_psplib_j10_suite_claim_ready"] is False
    assert "not the complete official" in report["claim_boundary"][
        "strong_claim_blockers"
    ][0]


def test_modified_official_instance_fails_closed_before_benchmark(tmp_path):
    root, manifest = _copy_suite(tmp_path)
    path = root / INSTANCE.name
    path.write_bytes(path.read_bytes().replace(b"horizon                       :  77", b"horizon                       :  78"))

    with pytest.raises(SuiteIntegrityError, match="SHA-256 mismatch"):
        build_psplib_mm_suite_benchmark(manifest)


def test_semantic_metadata_mismatch_fails_even_with_valid_file_hash(tmp_path):
    root, manifest_path = _copy_suite(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instances"][0]["renewable_capacities"] = [99, 9]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SuiteIntegrityError, match="semantic metadata mismatch"):
        build_psplib_mm_suite_benchmark(manifest_path, instance_root=root)


@pytest.mark.parametrize("failure", ["missing", "unregistered"])
def test_manifest_and_directory_file_sets_must_match_exactly(tmp_path, failure):
    root, manifest = _copy_suite(tmp_path)
    if failure == "missing":
        (root / INSTANCE.name).unlink()
    else:
        (root / "unregistered.mm").write_bytes(INSTANCE.read_bytes())

    with pytest.raises(SuiteIntegrityError, match="suite file set differs"):
        build_psplib_mm_suite_benchmark(manifest)


def test_duplicate_and_path_traversal_manifest_entries_fail_closed(tmp_path):
    root, manifest_path = _copy_suite(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instances"].append(dict(manifest["instances"][0]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SuiteIntegrityError, match="duplicate registered instance"):
        build_psplib_mm_suite_benchmark(manifest_path, instance_root=root)

    manifest["instances"] = [dict(manifest["instances"][0])]
    manifest["instances"][0]["file"] = "../j1010_1.mm"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SuiteIntegrityError, match="unsafe or invalid"):
        build_psplib_mm_suite_benchmark(manifest_path, instance_root=root)


def test_json_and_markdown_artifacts_are_byte_deterministic(tmp_path):
    report = build_psplib_mm_suite_benchmark(MANIFEST)
    json_path = tmp_path / "suite.json"
    markdown_path = tmp_path / "suite.md"

    write_artifacts(report, json_path=json_path, markdown_path=markdown_path)
    first_json = json_path.read_bytes()
    first_markdown = markdown_path.read_bytes()
    write_artifacts(report, json_path=json_path, markdown_path=markdown_path)

    assert json_path.read_bytes() == first_json
    assert markdown_path.read_bytes() == first_markdown
    parsed = json.loads(first_json)
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert b"OFFICIAL_PSPLIB_MM_SUITE_PASS" in first_markdown
    assert b"Dominance is descriptive" in first_markdown
