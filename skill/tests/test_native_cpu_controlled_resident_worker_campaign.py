from algorithm.experiments.native_cpu_controlled_resident_worker_campaign import (
    FULL_LOAD,
    HALF_LOAD,
    _build_plan,
)


def test_controlled_load_plan_is_complete_and_capacity_safe():
    plan = _build_plan(
        lane_specs={
            "node001": HALF_LOAD,
            "node002": HALF_LOAD,
            "node003": FULL_LOAD,
        },
        replicates=3,
    )
    half = [row for row in plan if row["load_class"] == HALF_LOAD]
    full = [row for row in plan if row["load_class"] == FULL_LOAD]
    assert len(half) == 8 * 3
    assert len(full) == 4 * 3
    assert all(row["resident_workers"] + row["workers"] <= 192 for row in plan)
    assert {
        (row["workers"], row["replicate"])
        for row in half
    } == {
        (profile, replicate)
        for profile in (1, 2, 4, 8, 16, 32, 64, 96)
        for replicate in range(3)
    }


def test_controlled_load_plan_supports_exact_repair_replicate():
    plan = _build_plan(
        lane_specs={"node001": HALF_LOAD},
        replicates=1,
        replicate_offset=8,
        load_specs={
            HALF_LOAD: {
                "resident_workers": 96,
                "profiles": (2,),
                "expected_regime": "cpu_external_heavy",
                "resource_state": "half_loaded",
            }
        },
    )

    assert len(plan) == 1
    assert plan[0]["workers"] == 2
    assert plan[0]["replicate"] == 8
