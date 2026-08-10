from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_renewal_stream_v1_artifact import (
    MMRCPSPRenewalArtifactError,
    _performance_dispositions,
    write_outputs,
)


def _run(scenario: str, seed: int, policy: str, backlog: int, queue: float) -> dict:
    return {
        "scenario_id": scenario,
        "seed": seed,
        "policy": policy,
        "final_backlog": backlog,
        "time_average_queue": queue,
    }


def test_performance_disposition_uses_both_queue_costs():
    rows = [
        _run("s", 1, "duration_normalized_robust_maxweight", 2, 4.0),
        _run("s", 1, "shortest_registered_trajectory", 3, 5.0),
        _run("s", 1, "longest_queue_then_shortest", 1, 6.0),
    ]

    report = _performance_dispositions(rows)

    assert report["ours_nondominated_count"] == 1
    assert report["ours_strict_every_baseline_count"] == 0
    assert report["baseline_dominates_count"] == 0
    assert report["rows"][0]["ours_disposition"] == "ours_nondominated_tradeoff_or_tie"


def test_writer_refuses_to_overwrite_result(tmp_path: Path):
    full = tmp_path / "full.json.gz"
    compact = tmp_path / "compact.json"
    markdown = tmp_path / "report.md"
    full.write_bytes(b"occupied")

    with pytest.raises(MMRCPSPRenewalArtifactError, match="refusing to overwrite"):
        write_outputs(
            {},
            full_path=full,
            compact_path=compact,
            markdown_path=markdown,
        )
