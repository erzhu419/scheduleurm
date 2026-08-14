from skill.scheduler_resource.estimates import (
    description_resource_family_key,
    description_sibling_resource_estimate,
    description_sibling_resource_estimate_index,
    EffectiveResourceEstimateDeps,
    ResourceEstimateDeps,
    effective_est_ram,
    effective_est_vram,
    learned_resource_estimate_index,
    live_sibling_ram_floor,
    live_sibling_resource_estimate,
    live_sibling_resource_estimate_index,
    maybe_lower_explicit_resource_estimate,
    refresh_queued_resource_estimates,
    resource_family_key,
    resource_history_metadata,
    task_has_host_aggregate_vram,
    with_resource_slack,
)


def _deps(
    *,
    live_ram=0,
    live_resources=None,
    description_resources=None,
    effective_vram=0,
    effective_ram=0,
    now=123.0,
):
    live_resources = live_resources or {}
    description_resources = description_resources or {}

    def live_index(state):
        out = {}
        for task in state.get("tasks", []):
            family = resource_family_key(task)
            for kind, value in live_resources.items():
                if family and value:
                    out[(family, kind)] = value
        return out

    def description_index(state):
        out = {}
        for task in state.get("tasks", []):
            family = description_resource_family_key(task)
            for kind, value in description_resources.items():
                mode = "vram" if kind == "vram" else (
                    "gpu" if int(task.get("est_vram_mb") or 0) > 100 else "cpu"
                )
                if family and value:
                    out[(family, mode, kind)] = value
        return out

    def maybe_lower(task, key, new_value, *, min_mb, kind):
        if new_value and new_value >= min_mb and new_value < (task.get(key) or 0):
            task[key] = new_value
            task["last_resource_estimate_update"] = {
                "kind": kind,
                "new_value": new_value,
            }
            return True
        return False

    return ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower,
        effective_est_vram=lambda task, state, history: effective_vram,
        effective_est_ram=lambda task, state, history: effective_ram,
        live_sibling_ram_floor=lambda task, state: live_ram,
        live_sibling_resource_estimate_index=live_index,
        description_sibling_resource_estimate_index=description_index,
        now=lambda: now,
    )


def test_refresh_queued_resource_estimates_applies_exact_vram_history():
    task = {
        "id": "t1",
        "status": "queued",
        "signature": "sig",
        "est_vram_mb": 4096,
        "ram_mb": 2048,
    }
    refresh_queued_resource_estimates(
        {"tasks": [task]},
        {"sig": {"vram_mb": 768}},
        deps=_deps(),
    )

    assert task["est_vram_mb"] == 768
    assert task["ram_mb"] == 2048


def test_refresh_queued_resource_estimates_lowers_explicit_ram_from_live_family():
    task = {
        "id": "t2",
        "status": "queued",
        "signature": "sig",
        "est_vram_mb": 0,
        "ram_mb": 50000,
        "ram_mb_explicit": True,
    }
    refresh_queued_resource_estimates(
        {"tasks": [task]},
        {},
        deps=_deps(live_ram=4096, live_resources={"ram": 3000}, effective_ram=3000),
    )

    assert task["ram_mb"] == 3000
    assert task["last_resource_estimate_update"]["kind"] == "ram_live_family_lower"


def test_refresh_queued_resource_estimates_lowers_explicit_vram_from_exact_history():
    task = {
        "id": "t2v",
        "status": "queued",
        "signature": "sig",
        "est_vram_mb": 5000,
        "est_vram_mb_explicit": True,
        "ram_mb": 2048,
    }
    refresh_queued_resource_estimates(
        {"tasks": [task]},
        {"sig": {"vram_mb": 512}},
        deps=_deps(effective_vram=768),
    )

    assert task["est_vram_mb"] == 512
    assert task["last_resource_estimate_update"]["kind"] == "vram_exact_history_lower"


def test_live_family_supersedes_old_exact_history_for_queued_resources():
    task = {
        "id": "queued",
        "status": "queued",
        "project": "proj",
        "signature": "proj/run/shard_002_of_010",
        "est_vram_mb": 7000,
        "ram_mb": 9000,
    }
    running = {
        "id": "running",
        "status": "running",
        "project": "proj",
        "signature": "proj/run/shard_001_of_010",
        "current_vram_mb": 600,
        "peak_vram_mb": 650,
        "current_ram_mb": 900,
        "peak_ram_mb": 1000,
    }
    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=live_sibling_ram_floor,
        live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
        description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
        now=lambda: 456.0,
    )

    refresh_queued_resource_estimates(
        {"tasks": [task, running]},
        {task["signature"]: {"vram_mb": 6000, "ram_mb": 8000}},
        deps=deps,
    )

    assert task["est_vram_mb"] == 780
    assert task["ram_mb"] == 1200
    assert task["last_resource_estimate_update"]["kind"] == "ram_live_family_lower"


def test_explicit_resource_family_calibrates_file_gated_formal_from_smoke():
    queued = {
        "id": "formal",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/new/formal/seed-8",
        "vram_resource_family": "BAPR/new/gpu-runtime",
        "est_vram_mb": 2048,
        "est_vram_mb_explicit": True,
        "ram_mb": 8192,
    }
    smoke = {
        "id": "smoke",
        "status": "running",
        "project": "BAPR",
        "signature": "BAPR/new/smoke/seed-8",
        "vram_resource_family": "BAPR/new/gpu-runtime",
        "current_vram_mb": 1180,
        "peak_vram_mb": 1240,
        "current_ram_mb": 1400,
        "peak_ram_mb": 1500,
        "last_progress_line": "Iter 0/4",
    }
    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=live_sibling_ram_floor,
        live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
        description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
        now=lambda: 789.0,
    )

    refresh_queued_resource_estimates(
        {"tasks": [queued, smoke]},
        {},
        deps=deps,
    )

    assert queued["est_vram_mb"] == 1488
    assert queued["ram_mb"] == 8192
    assert queued["last_resource_estimate_update"]["kind"] == "vram_live_family_lower"


def test_host_dispatch_aggregate_vram_does_not_raise_worker_family_budget():
    queued = {
        "id": "host-a",
        "status": "queued",
        "project": "h2oplus",
        "signature": "H2Oplus/v7/sensitivity/host-a",
        "resource_family": "H2Oplus/v7/sensitivity-host",
        "ram_resource_family": "H2Oplus/v7/sensitivity-host",
        "vram_resource_family": "H2Oplus/v7/sensitivity-worker",
        "est_vram_mb": 8192,
        "est_vram_mb_explicit": True,
        "ram_mb": 32768,
    }
    running = {
        "id": "host-b",
        "status": "running",
        "project": "h2oplus",
        "signature": "H2Oplus/v7/sensitivity/host-b",
        "resource_family": "H2Oplus/v7/sensitivity-host",
        "ram_resource_family": "H2Oplus/v7/sensitivity-host",
        "vram_resource_family": "H2Oplus/v7/sensitivity-worker",
        "est_vram_mb": 8192,
        "current_vram_mb": 12468,
        "peak_vram_mb": 12644,
        "current_ram_mb": 31000,
        "last_progress_line": "epoch=14",
    }
    state = {"tasks": [queued, running]}

    assert task_has_host_aggregate_vram(running) is True
    assert not any(
        key[-1] == "vram"
        for key in live_sibling_resource_estimate_index(state)
    )

    refresh_queued_resource_estimates(state, {}, deps=_actual_resource_deps())

    assert queued["est_vram_mb"] == 8192
    assert not str(
        (queued.get("last_resource_estimate_update") or {}).get("kind") or ""
    ).startswith("vram_")


def test_host_aggregate_vram_is_marked_and_excluded_from_persisted_history():
    queued = {
        "id": "host-resume",
        "status": "queued",
        "project": "h2oplus",
        "signature": "H2Oplus/v7/sensitivity/host-a",
        "resource_family": "H2Oplus/v7/sensitivity-host",
        "ram_resource_family": "H2Oplus/v7/sensitivity-host",
        "vram_resource_family": "H2Oplus/v7/sensitivity-worker",
        "est_vram_mb": 8192,
        "est_vram_mb_explicit": True,
        "ram_mb": 32768,
    }
    metadata = resource_history_metadata(queued)
    history = {
        queued["signature"]: {
            **metadata,
            "vram_mb": 12644,
            "ram_mb": 31000,
        },
    }

    assert metadata["vram_observation_scope"] == "host_aggregate"
    assert not any(
        key[-1] == "vram"
        for key in learned_resource_estimate_index(
            {"tasks": [queued]},
            history,
        )
    )

    refresh_queued_resource_estimates(
        {"tasks": [queued]},
        history,
        deps=_actual_resource_deps(),
    )

    assert queued["est_vram_mb"] == 8192


def test_startup_sample_does_not_lower_queued_family_estimate():
    queued = {
        "id": "formal",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/new/formal/seed-8",
        "vram_resource_family": "BAPR/new/gpu-runtime",
        "est_vram_mb": 2048,
        "est_vram_mb_explicit": True,
        "ram_mb": 8192,
    }
    compiling = {
        "id": "smoke",
        "status": "running",
        "project": "BAPR",
        "signature": "BAPR/new/smoke/seed-8",
        "vram_resource_family": "BAPR/new/gpu-runtime",
        "est_vram_mb": 2048,
        "current_vram_mb": 124,
        "peak_vram_mb": 124,
    }

    refresh_queued_resource_estimates(
        {"tasks": [queued, compiling]},
        {},
        deps=ResourceEstimateDeps(
            maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
            effective_est_vram=lambda task, state, history: 0,
            effective_est_ram=lambda task, state, history: 0,
            live_sibling_ram_floor=live_sibling_ram_floor,
            live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
            description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
            now=lambda: 790.0,
        ),
    )

    assert queued["est_vram_mb"] == 2048


def test_refresh_queued_resource_estimates_raises_low_ram_from_live_sibling():
    task = {
        "id": "t3",
        "status": "queued",
        "signature": "sig",
        "est_vram_mb": 0,
        "ram_mb": 608,
    }
    refresh_queued_resource_estimates(
        {"tasks": [task]},
        {},
        deps=_deps(live_ram=4700, now=456.0),
    )

    assert task["ram_mb"] == 4700
    assert task["last_resource_estimate_update"] == {
        "ts": 456.0,
        "kind": "ram_live_sibling_floor",
        "old_ram_mb": 608,
        "new_ram_mb": 4700,
    }


def _effective_deps(*, default_vram=3500, default_ram=2000, untrusted_ids=None):
    untrusted_ids = set(untrusted_ids or [])
    return EffectiveResourceEstimateDeps(
        default_vram_mb=default_vram,
        default_ram_mb=default_ram,
        untrusted_startup_oom_sample=lambda task: task.get("id") in untrusted_ids,
    )


def test_effective_est_vram_prefers_exact_history_and_legacy_ints():
    task = {"id": "t1", "signature": "proj/run", "project": "proj", "est_vram_mb": 999}

    assert effective_est_vram(
        task,
        {"tasks": []},
        {"proj/run": {"vram_mb": 1234}},
        deps=_effective_deps(),
    ) == 1234
    assert effective_est_vram(
        task,
        {"tasks": []},
        {"proj/run": 2048},
        deps=_effective_deps(),
    ) == 2048


def test_effective_est_vram_uses_sibling_and_project_fallbacks_skipping_untrusted_oom():
    task = {
        "id": "target",
        "signature": "proj/new/s1",
        "project": "proj",
        "description": "train: seed=1",
        "est_vram_mb": 3500,
    }
    state = {"tasks": [
        task,
        {"id": "bad", "project": "proj", "description": "train: seed=2", "peak_vram_mb": 9000},
        {"id": "good", "project": "proj", "description": "train: seed=3", "peak_vram_mb": 1800},
        {"id": "other", "project": "proj", "description": "eval", "peak_vram_mb": 1200},
    ]}

    assert effective_est_vram(
        task,
        state,
        {},
        deps=_effective_deps(untrusted_ids={"bad"}),
    ) == 1800

    task2 = dict(task, id="target2", description="novel")
    assert effective_est_vram(
        task2,
        state,
        {},
        deps=_effective_deps(untrusted_ids={"bad"}),
    ) == 1800


def test_effective_est_ram_uses_prefix_history_project_history_then_defaults():
    task = {
        "id": "target",
        "signature": "proj/family/new",
        "project": "proj",
        "description": "novel",
        "ram_mb": 5000,
    }

    assert effective_est_ram(
        task,
        {"tasks": []},
        {"proj/family/old": {"ram_mb": 1500}},
        deps=_effective_deps(),
    ) == 1500
    assert effective_est_ram(
        {"id": "target2", "signature": "proj/other/new", "project": "proj"},
        {"tasks": []},
        {"proj/x": {"ram_mb": 1700}},
        deps=_effective_deps(default_ram=2200),
    ) == 1700
    assert effective_est_ram(
        {"id": "target3", "signature": "none", "project": "none"},
        {"tasks": []},
        {},
        deps=_effective_deps(default_ram=2200),
    ) == 2200


def test_live_sibling_ram_floor_prefers_desc_then_prefix_then_two_project_samples():
    task = {
        "id": "target",
        "project": "proj",
        "signature": "proj/family/new",
        "description": "train: seed=1",
    }
    state = {"tasks": [
        task,
        {"id": "queued", "status": "queued", "project": "proj", "peak_ram_mb": 9999},
        {"id": "desc", "status": "running", "project": "proj", "signature": "proj/other/x", "description": "train: seed=2", "peak_ram_mb": 3000},
        {"id": "prefix", "status": "running", "project": "proj", "signature": "proj/family/old", "description": "eval", "current_ram_mb": 2000},
        {"id": "project", "status": "running", "project": "proj", "signature": "proj/z", "description": "other", "peak_ram_mb": 5000},
    ]}

    assert live_sibling_ram_floor(task, state) == 3000

    task2 = dict(task, id="target2", description="novel")
    assert live_sibling_ram_floor(task2, state) == 2000

    task3 = dict(task, id="target3", signature="proj/nope/new", description="novel")
    assert live_sibling_ram_floor(task3, state) == 3000


def test_live_sibling_ram_floor_ignores_single_broad_project_sample():
    task = {"id": "target", "project": "proj", "signature": "proj/new", "description": "novel"}
    state = {"tasks": [
        task,
        {"id": "only", "status": "running", "project": "proj", "signature": "proj/other", "description": "other", "peak_ram_mb": 5000},
    ]}

    assert live_sibling_ram_floor(task, state) == 0


def test_live_resource_family_normalizes_only_shard_index_and_seed():
    target = {
        "id": "target",
        "status": "queued",
        "project": "Laplace-semi-MDP",
        "signature": "Laplace-semi-MDP/p0/reviewer/planner_shard_013_of_108/seed_2",
    }
    same = {
        "id": "same",
        "status": "running",
        "project": "Laplace-semi-MDP",
        "signature": "Laplace-semi-MDP/p0/reviewer/planner_shard_042_of_108/seed_9",
        "current_ram_mb": 1400,
        "peak_ram_mb": 1600,
        "current_vram_mb": 700,
    }
    different_total = dict(
        same,
        id="different-total",
        signature="Laplace-semi-MDP/p0/reviewer/planner_shard_042_of_216/seed_9",
        peak_ram_mb=9000,
    )
    different_stage = dict(
        same,
        id="different-stage",
        signature="Laplace-semi-MDP/p1/reviewer/planner_shard_042_of_108/seed_9",
        peak_ram_mb=8000,
    )
    state = {"tasks": [target, same, different_total, different_stage]}

    assert resource_family_key(target) == resource_family_key(same)
    assert resource_family_key(target) != resource_family_key(different_total)
    assert resource_family_key(target) != resource_family_key(different_stage)
    assert live_sibling_resource_estimate(target, state, "ram") == 1600
    assert live_sibling_resource_estimate(target, state, "vram") == 700


def test_resource_family_normalizes_slash_shard_syntax():
    first = {
        "project": "proj",
        "signature": "proj/run/shard=0/30/seed=1",
    }
    second = {
        "project": "proj",
        "signature": "proj/run/shard=29/30/seed=8",
    }

    assert resource_family_key(first) == resource_family_key(second)


def test_description_resource_family_normalizes_seed_but_keeps_other_parameters():
    first = {
        "project": "KG-SYNTH",
        "description": "transfer fairness InventorySupplyChain seed=0 d=200",
    }
    second = {
        "project": "KG-SYNTH",
        "description": "transfer fairness InventorySupplyChain seed=19 d=200",
    }
    different = {
        "project": "KG-SYNTH",
        "description": "transfer fairness InventorySupplyChain seed=2 d=1000",
    }

    assert description_resource_family_key(first) == description_resource_family_key(second)
    assert description_resource_family_key(first) != description_resource_family_key(different)


def test_description_sibling_estimate_requires_two_samples_and_uses_upper_observation():
    target = {
        "id": "target",
        "status": "queued",
        "project": "KG-SYNTH",
        "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=0",
        "est_vram_mb": 0,
    }
    first = {
        "id": "first",
        "status": "done",
        "project": "KG-SYNTH",
        "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=1",
        "peak_ram_mb": 482,
        "est_vram_mb": 0,
    }
    second = {
        "id": "second",
        "status": "running",
        "project": "KG-SYNTH",
        "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=2",
        "current_ram_mb": 489,
        "est_vram_mb": 0,
    }
    failed_outlier = dict(second, id="failed", status="failed", peak_ram_mb=32000)

    assert description_sibling_resource_estimate(
        target,
        {"tasks": [target, first]},
        "ram",
    ) == 0
    assert description_sibling_resource_estimate(
        target,
        {"tasks": [target, first, second, failed_outlier]},
        "ram",
    ) == 489


def test_refresh_lowers_explicit_ram_from_cross_signature_description_evidence():
    target = {
        "id": "t37000",
        "status": "queued",
        "project": "KG-SYNTH",
        "signature": "KG_op/new-run/seed0000",
        "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=0",
        "est_vram_mb": 0,
        "est_vram_mb_explicit": True,
        "ram_mb": 32768,
        "ram_mb_explicit": True,
    }
    siblings = [
        {
            "id": "old1",
            "status": "done",
            "project": "KG-SYNTH",
            "signature": "KG_op/old-run-a/seed0001",
            "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=1",
            "est_vram_mb": 0,
            "peak_ram_mb": 482,
        },
        {
            "id": "old2",
            "status": "done",
            "project": "KG-SYNTH",
            "signature": "KG_op/old-run-b/seed0002",
            "description": "transfer fairness official InventorySupplyChain malibo_cbo seed=2",
            "est_vram_mb": 0,
            "peak_ram_mb": 489,
        },
    ]
    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=live_sibling_ram_floor,
        live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
        description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
        now=lambda: 456.0,
    )

    refresh_queued_resource_estimates(
        {"tasks": [target, *siblings]},
        {},
        deps=deps,
    )

    assert target["ram_mb"] == 587
    assert target["last_resource_estimate_update"]["kind"] == "ram_description_sibling_lower"
    assert target["last_resource_estimate_update"]["observed_mb"] == 489


def test_refresh_lowers_explicit_vram_from_cross_signature_description_evidence():
    target = {
        "id": "gpu-target",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/new/seed0",
        "description": "BAPR evaluation seed=0",
        "est_vram_mb": 8000,
        "est_vram_mb_explicit": True,
        "ram_mb": 4096,
        "ram_mb_explicit": True,
    }
    siblings = [
        {
            "id": "gpu-old1",
            "status": "done",
            "project": "BAPR",
            "signature": "BAPR/old-a/seed1",
            "description": "BAPR evaluation seed=1",
            "est_vram_mb": 8000,
            "peak_vram_mb": 900,
        },
        {
            "id": "gpu-old2",
            "status": "done",
            "project": "BAPR",
            "signature": "BAPR/old-b/seed2",
            "description": "BAPR evaluation seed=2",
            "est_vram_mb": 8000,
            "peak_vram_mb": 1000,
        },
    ]
    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=live_sibling_ram_floor,
        live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
        description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
        now=lambda: 456.0,
    )

    refresh_queued_resource_estimates(
        {"tasks": [target, *siblings]},
        {},
        deps=deps,
    )

    assert target["est_vram_mb"] == 1200
    assert target["last_resource_estimate_update"]["kind"] == "vram_description_sibling_lower"


def test_refresh_caches_resource_queries_per_normalized_family():
    calls = []
    tasks = [
        {
            "id": f"queued-{seed}",
            "status": "queued",
            "project": "proj",
            "signature": f"proj/run/shard_{seed:03d}_of_010/seed{seed}",
            "description": f"evaluation shard_{seed:03d}_of_010 seed={seed}",
            "est_vram_mb": 0,
            "ram_mb": 4096,
        }
        for seed in range(5)
    ]
    def live_index(state):
        calls.append("live-index")
        return {}

    def description_index(state):
        calls.append("description-index")
        return {}

    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=lambda task, state: 0,
        live_sibling_resource_estimate_index=live_index,
        description_sibling_resource_estimate_index=description_index,
    )

    refresh_queued_resource_estimates({"tasks": tasks}, {}, deps=deps)

    assert calls == ["live-index", "description-index"]


def test_refresh_skips_broad_vram_scan_when_zero_cannot_be_lowered():
    calls = []
    task = {
        "id": "cpu-task",
        "status": "queued",
        "project": "cpu-project",
        "signature": "cpu-project/run",
        "description": "CPU evaluation",
        "est_vram_mb": 0,
        "ram_mb": 2048,
        "ram_mb_explicit": True,
    }
    deps = ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=(
            lambda task, state, history: calls.append("effective-vram") or 1024
        ),
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=lambda task, state: 0,
        live_sibling_resource_estimate_index=lambda state: {},
        description_sibling_resource_estimate_index=lambda state: {},
    )

    refresh_queued_resource_estimates({"tasks": [task]}, {}, deps=deps)

    assert calls == []
    assert task["est_vram_mb"] == 0


def test_maybe_lower_explicit_resource_estimate_adds_slack_and_threshold():
    task = {"ram_mb": 10000}

    assert with_resource_slack(1000, min_mb=512) == 1200
    assert maybe_lower_explicit_resource_estimate(
        task,
        "ram_mb",
        1000,
        min_mb=512,
        kind="test",
        now=lambda: 42.0,
    ) is True
    assert task["ram_mb"] == 1200
    assert task["last_resource_estimate_update"] == {
        "ts": 42.0,
        "kind": "test",
        "old_ram_mb": 10000,
        "new_ram_mb": 1200,
        "observed_mb": 1000,
    }

    near = {"ram_mb": 1300}
    assert maybe_lower_explicit_resource_estimate(
        near,
        "ram_mb",
        1000,
        min_mb=512,
        kind="near",
        now=lambda: 42.0,
    ) is False
    assert near["ram_mb"] == 1300



def _actual_resource_deps(now=123.0):
    return ResourceEstimateDeps(
        maybe_lower_explicit_resource_estimate=maybe_lower_explicit_resource_estimate,
        effective_est_vram=lambda task, state, history: 0,
        effective_est_ram=lambda task, state, history: 0,
        live_sibling_ram_floor=live_sibling_ram_floor,
        live_sibling_resource_estimate_index=live_sibling_resource_estimate_index,
        description_sibling_resource_estimate_index=description_sibling_resource_estimate_index,
        now=lambda: now,
    )


def test_explicit_bapr_ram_is_cold_start_hint_after_campaign_has_actuals():
    queued = {
        "id": "audit",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/v8-seed-validation/v1/audit/seed-0/event-seed-11100",
        "description": "Strict BAPR-v8 paired audit, training seed 0, event seed 11100",
        "est_vram_mb": 0,
        "est_vram_mb_explicit": True,
        "ram_mb": 12288,
        "ram_mb_explicit": True,
    }
    running = [
        {
            "id": "train-1",
            "status": "running",
            "project": "BAPR",
            "signature": "BAPR/v8-seed-validation/v1/train/seed-1/sac",
            "description": "BAPR-v8 independent training seed 1, sac",
            "est_vram_mb": 2048,
            "peak_vram_mb": 700,
            "peak_ram_mb": 1400,
            "last_progress_line": "Iter 10",
        },
        {
            "id": "train-2",
            "status": "running",
            "project": "BAPR",
            "signature": "BAPR/v8-seed-validation/v1/train/seed-2/escp",
            "description": "BAPR-v8 independent training seed 2, escp",
            "est_vram_mb": 2048,
            "peak_vram_mb": 900,
            "peak_ram_mb": 1800,
            "last_progress_line": "Iter 20",
        },
    ]

    refresh_queued_resource_estimates(
        {"tasks": [queued, *running]},
        {},
        deps=_actual_resource_deps(),
    )

    assert queued["ram_mb"] == 2160
    assert queued["est_vram_mb"] == 0
    assert queued["last_resource_estimate_update"]["kind"] == "ram_learned_prefix_lower"
    assert queued["last_resource_estimate_update"]["evidence_scope"] == "prefix"
    assert queued["last_resource_estimate_update"]["sample_count"] == 2


def test_new_explicit_vram_family_does_not_inherit_prefix_measurements():
    queued = {
        "id": "new-runtime",
        "status": "queued",
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run-v3/domain/saas/seed-80",
        "vram_resource_family": "KG-SYNTH/saas-nograd-v3",
        "description": "SAAS no-grad v3",
        "est_vram_mb": 9000,
        "est_vram_mb_explicit": True,
        "ram_mb": 16000,
    }
    old_runtime = {
        "id": "old-runtime",
        "status": "done",
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run-v2/domain/saas/seed-81",
        "description": "old SAAS runtime",
        "peak_vram_mb": 2000,
        "peak_ram_mb": 4000,
    }

    refresh_queued_resource_estimates(
        {"tasks": [queued, old_runtime]},
        {},
        deps=_actual_resource_deps(),
    )

    assert queued["est_vram_mb"] == 9000


def test_explicit_resource_family_keeps_seed_as_literal_isolation_key():
    first = {
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run/domain/saas/seed-80",
        "vram_resource_family": "KG-SYNTH/saas/domain/seed0080",
    }
    second = {
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run/domain/saas/seed-81",
        "vram_resource_family": "KG-SYNTH/saas/domain/seed0081",
    }

    assert resource_family_key(first, "vram") != resource_family_key(
        second, "vram")


def test_explicit_vram_family_does_not_inherit_description_siblings():
    queued = {
        "id": "new-runtime",
        "status": "queued",
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run-v3/domain/saas/seed-80",
        "vram_resource_family": "KG-SYNTH/saas/domain/seed0080",
        "description": "periodic SAAS domain seed=80",
        "est_vram_mb": 10500,
        "est_vram_mb_explicit": True,
        "ram_mb": 8192,
    }
    siblings = [
        {
            "id": f"old-{seed}",
            "status": "done",
            "project": "KG-SYNTH",
            "signature": f"KG_op/sota/old/domain/saas/seed-{seed}",
            "description": f"periodic SAAS domain seed={seed}",
            "peak_vram_mb": 12000,
        }
        for seed in (81, 82)
    ]

    refresh_queued_resource_estimates(
        {"tasks": [queued, *siblings]},
        {},
        deps=_actual_resource_deps(),
    )

    assert queued["est_vram_mb"] == 10500


def test_explicit_vram_family_rejects_mismatched_exact_signature_history():
    queued = {
        "id": "resume",
        "status": "queued",
        "project": "KG-SYNTH",
        "signature": "KG_op/sota/run/domain/saas/seed-80",
        "vram_resource_family": "KG-SYNTH/saas-nograd-v3",
        "description": "SAAS no-grad v3 resume",
        "est_vram_mb": 9000,
        "est_vram_mb_explicit": True,
        "ram_mb": 16000,
    }
    history = {
        queued["signature"]: {
            "vram_mb": 11200,
            "vram_resource_family_key": (
                "kg-synth|kg-synth/saas-autograd-v2"
            ),
        },
    }

    refresh_queued_resource_estimates(
        {"tasks": [queued]},
        history,
        deps=_actual_resource_deps(),
    )

    assert queued["est_vram_mb"] == 9000


def test_persisted_resource_family_history_lowers_future_explicit_estimate():
    queued = {
        "id": "future-audit",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/future/audit/seed-9",
        "ram_resource_family": "BAPR/audit/cpu-runtime",
        "description": "future audit seed 9",
        "est_vram_mb": 0,
        "ram_mb": 12000,
        "ram_mb_explicit": True,
    }
    family = resource_family_key(queued, "ram")
    history = {
        "BAPR/previous/audit/seed-1": {
            "ram_mb": 1500,
            "project": "bapr",
            "ram_resource_family_key": family,
            "resource_mode": "cpu",
        },
    }

    refresh_queued_resource_estimates(
        {"tasks": [queued]},
        history,
        deps=_actual_resource_deps(),
    )

    assert queued["ram_mb"] == 1800
    assert queued["last_resource_estimate_update"]["kind"] == "ram_learned_family_lower"
    assert queued["last_resource_estimate_update"]["sample_count"] == 1


def test_resource_history_metadata_normalizes_seed_and_preserves_mode():
    first = {
        "project": "BAPR",
        "signature": "BAPR/run/shard_001_of_010/seed-1",
        "description": "BAPR training seed 1",
        "est_vram_mb": 2048,
    }
    second = dict(
        first,
        signature="BAPR/run/shard_009_of_010/seed-8",
        description="BAPR training seed 8",
    )

    first_metadata = resource_history_metadata(first)
    second_metadata = resource_history_metadata(second)

    assert first_metadata["ram_resource_family_key"] == second_metadata["ram_resource_family_key"]
    assert first_metadata["vram_resource_family_key"] == second_metadata["vram_resource_family_key"]
    assert first_metadata["description_resource_family_key"] == second_metadata["description_resource_family_key"]
    assert first_metadata["resource_mode"] == "gpu"


def test_legacy_history_without_mode_is_not_used_as_vram_family_evidence():
    state = {
        "tasks": [{
            "id": "cpu",
            "status": "queued",
            "project": "BAPR",
            "signature": "BAPR/run/seed-3",
            "est_vram_mb": 0,
            "ram_mb": 4096,
        }],
    }
    history = {
        "BAPR/run/seed-1": {"vram_mb": 3000},
        "BAPR/run/seed-2": {"vram_mb": 3200},
    }

    index = learned_resource_estimate_index(state, history)

    assert not any(key[-1] == "vram" for key in index)



def test_explicit_zero_vram_remains_cpu_despite_family_probe_noise():
    queued = {
        "id": "cpu-queued",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR/audit/seed-1",
        "description": "BAPR audit seed 1",
        "est_vram_mb": 0,
        "est_vram_mb_explicit": True,
        "ram_mb": 4096,
    }
    running = {
        "id": "cpu-running",
        "status": "running",
        "project": "BAPR",
        "signature": "BAPR/audit/seed-2",
        "description": "BAPR audit seed 2",
        "est_vram_mb": 0,
        "est_vram_mb_explicit": True,
        "peak_vram_mb": 376,
        "peak_ram_mb": 1300,
        "last_progress_line": "audit 1/10",
    }

    refresh_queued_resource_estimates(
        {"tasks": [queued, running]},
        {},
        deps=_actual_resource_deps(),
    )

    assert queued["est_vram_mb"] == 0
    assert resource_history_metadata(running)["resource_mode"] == "cpu"
