from algorithm.experiments.native_cpu_allocation_worker_state_campaign import (
    _build_plan,
    _safe_profiles_for_survey,
)


def test_idle_profile_plan_covers_every_profile_and_replicate_once():
    profiles = (2, 4, 8, 16)
    plan = _build_plan(
        nodes=("node001", "node002"),
        profiles=profiles,
        target_regimes={
            "node001": "cpu_external_idle",
            "node002": "cpu_external_idle",
        },
        safe_profiles_by_node={
            "node001": profiles,
            "node002": profiles,
        },
        idle_replicates=3,
        loaded_replicates=1,
    )
    assert len(plan) == len(profiles) * 3
    assert {
        (row["workers"], row["replicate"])
        for row in plan
    } == {
        (profile, replicate)
        for profile in profiles
        for replicate in range(3)
    }
    assert {row["node"] for row in plan} == {"node001", "node002"}


def test_replicate_offsets_keep_extension_disjoint_from_training():
    plan = _build_plan(
        nodes=("node001",),
        profiles=(2, 4),
        target_regimes={"node001": "cpu_external_idle"},
        safe_profiles_by_node={"node001": (2, 4)},
        idle_replicates=2,
        loaded_replicates=1,
        idle_replicate_offset=3,
    )
    assert {row["replicate"] for row in plan} == {3, 4}
    assert {row["cell_id"] for row in plan} == {
        "cpu_external_idle_w2_r3",
        "cpu_external_idle_w4_r3",
        "cpu_external_idle_w2_r4",
        "cpu_external_idle_w4_r4",
    }


def test_nonidle_capacity_filter_reserves_external_p95_and_headroom():
    profiles = (2, 64, 128, 180, 192)
    safe = _safe_profiles_for_survey(
        profiles,
        {
            "dispatch_external_cpu_regime": "cpu_external_moderate",
            "dispatch_external_cpu_fraction_p95": 0.25,
        },
        logical_cpus=192,
        safety_reserve_cores=8,
    )
    assert safe == (2, 64, 128)


def test_idle_capacity_filter_keeps_full_worker_ladder():
    profiles = (2, 64, 128, 180, 192)
    safe = _safe_profiles_for_survey(
        profiles,
        {
            "dispatch_external_cpu_regime": "cpu_external_idle",
            "dispatch_external_cpu_fraction_p95": 0.0,
        },
        logical_cpus=192,
        safety_reserve_cores=8,
    )
    assert safe == profiles
