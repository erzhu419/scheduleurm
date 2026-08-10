from __future__ import annotations

import gzip
import json
from pathlib import Path

from algorithm.experiments.port_trajectory_refresh_artifact import (
    build_refresh_artifact,
    markdown_report,
    write_artifacts,
)


def test_registered_refresh_preserves_protocol_and_negative_boundaries():
    full, summary = build_refresh_artifact()

    assert full["artifact_hash"] == summary["ledger"]["full_artifact_hash"]
    assert summary["gate"]["pass"] is True
    assert summary["gate"]["prospective_holdout_claim_ready"] is False
    assert summary["gate"]["physical_port_claim_ready"] is False
    assert summary["result_summary"]["trajectory_context_pair_count"] == 124
    assert summary["result_summary"]["ours_synthetic_pareto_nondominated_count"] == 3
    assert summary["result_summary"]["migration_analogues_in_candidate_family"] == [
        "crane_reassignment",
        "reberth",
        "yard_rehandle",
    ]


def test_artifact_writer_is_path_independent_and_reproducible(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a = write_artifacts(
        full_output=first / "a.json.gz",
        compact_output=first / "a.json",
        markdown_output=first / "a.md",
    )
    b = write_artifacts(
        full_output=second / "b.json.gz",
        compact_output=second / "b.json",
        markdown_output=second / "b.md",
    )

    assert (first / "a.json.gz").read_bytes() == (second / "b.json.gz").read_bytes()
    assert gzip.decompress((first / "a.json.gz").read_bytes()) == gzip.decompress(
        (second / "b.json.gz").read_bytes()
    )
    assert a["full_gzip_sha256"] == b["full_gzip_sha256"]
    assert a["artifact_sha256_excluding_self"] == b["artifact_sha256_excluding_self"]
    assert json.loads((first / "a.json").read_text())["gate"]["pass"] is True
    assert "not a prospective holdout" in markdown_report(a)
