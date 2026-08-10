from __future__ import annotations

from pathlib import Path

import pytest

from algorithm.experiments.mmrcpsp_class_balance_holdout_v1_artifact import (
    MMRCPSPClassBalanceArtifactError,
    write_outputs,
)


def test_writer_refuses_to_overwrite_frozen_result(tmp_path: Path):
    full = tmp_path / "full.json.gz"
    full.write_bytes(b"occupied")

    with pytest.raises(
        MMRCPSPClassBalanceArtifactError, match="refusing to overwrite"
    ):
        write_outputs(
            {},
            full_path=full,
            compact_path=tmp_path / "compact.json",
            markdown_path=tmp_path / "report.md",
        )
