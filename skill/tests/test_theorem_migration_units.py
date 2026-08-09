from __future__ import annotations

import pytest

from algorithm.theorem_dispatch.migration import (
    MigrationCost,
    build_migration_action_row,
    migration_is_beneficial,
    migration_time_savings,
)


def test_migration_benefit_compares_seconds_to_seconds():
    cost = MigrationCost(checkpoint_flush_s=4.0, sync_s=3.0, resume_warmup_s=3.0)

    assert migration_time_savings(
        remaining_work=100.0, current_rate=1.0, target_rate=2.0
    ) == pytest.approx(50.0)
    assert migration_is_beneficial(
        remaining_work=100.0, current_rate=1.0, target_rate=2.0, cost=cost
    ) is True
    assert migration_is_beneficial(
        remaining_work=10.0, current_rate=1.0, target_rate=2.0, cost=cost
    ) is False


@pytest.mark.parametrize(
    ("remaining", "old_rate", "new_rate"),
    ((0.0, 1.0, 2.0), (100.0, 0.0, 2.0), (100.0, 2.0, 2.0), (100.0, 2.0, 1.0)),
)
def test_nonpositive_or_nonimproving_rate_has_no_benefit(remaining, old_rate, new_rate):
    assert migration_time_savings(
        remaining_work=remaining, current_rate=old_rate, target_rate=new_rate
    ) == 0.0


def test_action_row_exposes_auditable_time_decomposition():
    row = build_migration_action_row(
        task={
            "id": "controlled",
            "controlled_benchmark": True,
            "checkpoint_verified": True,
            "resume_verified": True,
        },
        workload_key="gpu_cnn_torch_resnet50",
        from_node="node007",
        to_node="jtl311linux",
        current_rate=1.0,
        target_lower_service=2.0,
        remaining_work=100.0,
        progress_fraction=0.5,
        cost=MigrationCost(checkpoint_flush_s=1.0, sync_s=1.0, resume_warmup_s=1.0),
    )

    assert row["keep_remaining_completion_s"] == pytest.approx(100.0)
    assert row["migrated_remaining_completion_s_before_cost"] == pytest.approx(50.0)
    assert row["migration_time_savings_before_cost_s"] == pytest.approx(50.0)
    assert row["migration_net_time_savings_s"] == pytest.approx(47.0)
    assert row["theorem_ready"] is True
