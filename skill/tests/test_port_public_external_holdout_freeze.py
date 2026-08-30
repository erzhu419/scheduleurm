from pathlib import Path

import pytest

from algorithm.experiments.port_public_external_holdout_freeze import (
    EXPECTED_DEADLINE,
    EXPECTED_Q,
    EXPECTED_REPLICATIONS,
    EXPECTED_SPEED_SETUP,
    _select_sources,
)


def _write_factor_grid(root: Path) -> None:
    root.mkdir(parents=True)
    for replication in EXPECTED_REPLICATIONS:
        for q_max in EXPECTED_Q:
            for speed in EXPECTED_SPEED_SETUP:
                for deadline in EXPECTED_DEADLINE:
                    name = (
                        "instance_Gen_Meisel2009_10m_V30_L50_"
                        f"{replication}_Q{q_max}_S{speed}_D{deadline}.dat"
                    )
                    (root / name).write_text(
                        f"{replication} {q_max} {speed} {deadline}\n",
                        encoding="ascii",
                    )


def test_select_sources_requires_complete_disjoint_factor_grid(tmp_path: Path) -> None:
    root = tmp_path / "instances" / "LargeMB"
    _write_factor_grid(root)

    rows = _select_sources(root)

    assert len(rows) == 54
    assert {row["replication"] for row in rows} == set(EXPECTED_REPLICATIONS)
    assert len({row["sha256"] for row in rows}) == 54

    Path(rows[0]["_source"]).unlink()
    with pytest.raises(ValueError, match="complete 54-row factor grid"):
        _select_sources(root)
