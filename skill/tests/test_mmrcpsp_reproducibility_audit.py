from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_reproducibility_audit import (
    MMRCPSPReproducibilityAuditError,
    audit_pair,
    build_audit,
    markdown_report,
)


def _write_gzip(path: Path, raw: bytes, *, header_name: str) -> None:
    with path.open("wb") as handle:
        with gzip.GzipFile(
            filename=header_name, mode="wb", fileobj=handle, mtime=0
        ) as zipped:
            zipped.write(raw)


def test_audit_separates_gzip_header_from_semantic_result(tmp_path: Path):
    full = b'{"result":1}\n'
    primary_gzip = tmp_path / "primary.json.gz"
    reproduction_gzip = tmp_path / "reproduction.json.gz"
    _write_gzip(primary_gzip, full, header_name="primary.json")
    _write_gzip(reproduction_gzip, full, header_name="reproduction.json")
    base = {"schema_version": "test.schema", "result": 1}
    primary_compact = tmp_path / "primary.json"
    reproduction_compact = tmp_path / "reproduction.json"
    primary_compact.write_text(
        json.dumps(
            {
                **base,
                "full_artifact_gzip_sha256": "primary",
                "artifact_sha256_excluding_self": "primary-self",
            }
        ),
        encoding="utf-8",
    )
    reproduction_compact.write_text(
        json.dumps(
            {
                **base,
                "full_artifact_gzip_sha256": "reproduction",
                "artifact_sha256_excluding_self": "reproduction-self",
            }
        ),
        encoding="utf-8",
    )
    primary_md = tmp_path / "primary.md"
    reproduction_md = tmp_path / "reproduction.md"
    primary_md.write_text("same\n", encoding="utf-8")
    reproduction_md.write_text("same\n", encoding="utf-8")

    row = audit_pair(
        name="synthetic",
        expected_schema="test.schema",
        primary_gzip=primary_gzip,
        primary_compact=primary_compact,
        primary_markdown=primary_md,
        reproduction_gzip=reproduction_gzip,
        reproduction_compact=reproduction_compact,
        reproduction_markdown=reproduction_md,
    )

    assert row["semantic_reproducibility_ready"] is True
    assert row["decompressed_full_artifact_equal"] is True
    assert row["semantic_compact_artifact_equal"] is True
    assert row["raw_gzip_container_equal"] is False
    assert row["gzip_header_original_filename"] == {
        "primary": "primary.json",
        "reproduction": "reproduction.json",
    }


def test_audit_fails_on_semantic_drift(tmp_path: Path):
    primary_gzip = tmp_path / "primary.json.gz"
    reproduction_gzip = tmp_path / "reproduction.json.gz"
    _write_gzip(primary_gzip, b'{"result":1}', header_name="a")
    _write_gzip(reproduction_gzip, b'{"result":2}', header_name="b")
    for name, result in (("primary", 1), ("reproduction", 2)):
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"schema_version": "test.schema", "result": result}),
            encoding="utf-8",
        )
        (tmp_path / f"{name}.md").write_text("same", encoding="utf-8")

    with pytest.raises(
        MMRCPSPReproducibilityAuditError, match="semantic reproduction failed"
    ):
        audit_pair(
            name="synthetic",
            expected_schema="test.schema",
            primary_gzip=primary_gzip,
            primary_compact=tmp_path / "primary.json",
            primary_markdown=tmp_path / "primary.md",
            reproduction_gzip=reproduction_gzip,
            reproduction_compact=tmp_path / "reproduction.json",
            reproduction_markdown=tmp_path / "reproduction.md",
        )


def test_real_mmrcpsp_reruns_are_semantically_identical():
    report = build_audit()

    assert report["gate"]["pass"] is True
    assert report["gate"]["semantic_reproducibility_ready_count"] == 2
    assert report["gate"]["raw_gzip_container_match_count"] == 0
    assert report["gate"]["bitwise_path_independent_gzip_claim_ready"] is False
    assert all(
        row["decompressed_full_artifact_equal"] for row in report["pairs"]
    )
    assert "Raw gzip equality" in markdown_report(report)
