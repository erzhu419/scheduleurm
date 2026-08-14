from __future__ import annotations

from dataclasses import replace

import pytest

from skill.scheduler_history import (
    HistoryCommandDeps,
    ResourceHistoryDeps,
    RuntimeHistoryLookupDeps,
    cmd_history,
    history_record,
    runtime_history_best,
    seed_pending_eta_from_history,
)


def _resource_deps(saved, *, history=None, max_entries=10, samples_per_sig=10, now=123.0):
    return ResourceHistoryDeps(
        load_history=lambda: dict(history or {}),
        save_history=lambda value: saved.append(value),
        max_entries=max_entries,
        samples_per_sig=samples_per_sig,
        percentile_value=80,
        now=lambda: now,
    )


def _runtime_deps(*, runtime_history=None, resource_history=None, now=456.0):
    runtime_history = runtime_history or {}
    resource_history = resource_history or {}

    def keys(task):
        sig = task.get("signature") or ""
        return [(f"sig:{sig}", "signature", task)] if sig else []

    return RuntimeHistoryLookupDeps(
        load_runtime_history=lambda: runtime_history,
        history_get=lambda sig: resource_history.get(sig),
        task_runtime_keys=keys,
        runtime_total_units_from_cmd=lambda cmd: (
            200 if "--steps 200" in (cmd or "")
            else 10 if "--N 10" in (cmd or "")
            else 20 if "--N 20" in (cmd or "")
            else 80 if "--N 80" in (cmd or "")
            else 160 if "--N 160" in (cmd or "")
            else 0
        ),
        queued_has_stale_live_eta=lambda task: task.get("eta_source") == "progress_rate",
        clear_live_eta_fields=lambda task, **kwargs: task.pop("eta_seconds", None) is not None,
        eta_confidence_for_source=lambda source: "medium" if source == "runtime_history" else "low",
        min_walltime_s=600,
        walltime_mult=1.2,
        closest_min_score=0.58,
        now=lambda: now,
    )


def test_history_record_uses_sliding_p80_and_lru_pruning():
    saved = []
    history = {f"old-{i}": {"last_seen": i, "ram_mb": 1} for i in range(5)}

    history_record(
        "sig",
        peak_ram_mb=9000,
        deps=_resource_deps(saved, history=history, max_entries=3, samples_per_sig=2, now=99.0),
    )

    first = saved[-1]
    assert "sig" in first
    assert len(first) == 3
    assert first["sig"]["ram_samples"] == [9000]
    assert first["sig"]["last_seen"] == 99

    saved.clear()
    history_record(
        "sig",
        peak_ram_mb=1500,
        deps=_resource_deps(saved, history=first, max_entries=3, samples_per_sig=2, now=100.0),
    )
    assert saved[-1]["sig"]["ram_samples"] == [9000, 1500]
    assert 7400 <= saved[-1]["sig"]["ram_mb"] <= 7600


def test_history_record_persists_only_supported_resource_metadata():
    saved = []

    history_record(
        "BAPR/run/seed-1",
        peak_ram_mb=1500,
        metadata={
            "project": "bapr",
            "ram_resource_family_key": "bapr|bapr/run/seed-*",
            "vram_resource_family_key": "bapr|gpu-runtime",
            "description_resource_family_key": "bapr|training seed=*",
            "resource_mode": "gpu",
            "ignored": "must-not-persist",
        },
        deps=_resource_deps(saved),
    )

    record = saved[-1]["BAPR/run/seed-1"]
    assert record["project"] == "bapr"
    assert record["ram_resource_family_key"] == "bapr|bapr/run/seed-*"
    assert record["resource_mode"] == "gpu"
    assert "ignored" not in record


def test_runtime_history_best_uses_exact_before_closest_and_scales_units():
    history = {
        "sig:exact": {"total_s": 50, "cmd": "python train.py --steps 100"},
        "other": {
            "total_s": 100,
            "unit_s": 2.0,
            "cmd": "python train.py --steps 100 --dataset bus",
            "project": "proj",
            "cwd": "/tmp/proj",
        },
    }
    deps = _runtime_deps(runtime_history=history)

    rec, _key, kind = runtime_history_best(
        {"signature": "exact", "cmd": "python whatever.py"},
        task_runtime_payload=lambda task: task,
        deps=deps,
    )
    assert kind == "signature"
    assert rec["total_s"] == 50

    rec, key, kind = runtime_history_best(
        {
            "signature": "new",
            "cmd": "python train.py --steps 200 --dataset bus",
            "project": "proj",
            "cwd": "/tmp/proj",
        },
        task_runtime_payload=lambda task: task,
        deps=deps,
    )
    assert kind == "closest"
    assert key == "closest:other"
    assert rec["total_s"] == 400
    assert rec["total_units"] == 200


def test_runtime_history_closest_scales_from_record_cmd_units_without_unit_s():
    history = {
        "old-kg": {
            "total_s": 960,
            "total_units": 1,
            "unit_s": 960.0,
            "cmd": "python performance/benchmark_sota.py --problem P --d 10000 --N 80 --baselines botorch_scbo",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-kg",
            "cmd": "python performance/benchmark_sota.py --problem P --d 10000 --N 160 --baselines botorch_scbo",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:old-kg"
    assert rec["total_s"] == 1920
    assert rec["total_units"] == 160
    assert rec["scaled_from_total_units"] == 80


def test_runtime_history_closest_never_crosses_method_identity():
    common = (
        "python performance/benchmark_transfer_fairness.py "
        "--implementation official --heldout InventorySupplyChain --N 20"
    )
    history = {
        "wrong-fast": {
            "total_s": 90,
            "cmd": common + " --method malibo_cbo",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        "right-slow": {
            "total_s": 7200,
            "cmd": common + " --method fsbo_cbo",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-transfer",
            "cmd": common + " --method fsbo_cbo --initial-design source_informed",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:right-slow"
    assert rec["total_s"] == 7200


def test_runtime_history_closest_kg_reuses_other_experiment_variant():
    common = (
        "python performance/run_lodo_manifest_shard.py "
        "--heldout InventorySupplyChain --line lodo --d 1000 "
        "--meta-source-d 50 --N 10 --n0 10"
    )
    history = {
        "prior-variant": {
            "total_s": 820,
            "cmd": common + " --experiment-variant prior/gate",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-kg",
            "cmd": common + " --experiment-variant new/gate",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:prior-variant"
    assert rec["total_s"] == 820


def test_runtime_history_closest_kg_prefers_same_numeric_budget():
    base = (
        "python performance/run_lodo_manifest_shard.py "
        "--heldout InventorySupplyChain --line lodo --d 1000 "
        "--meta-source-d 50 --n0 10 --experiment-variant old"
    )
    history = {
        "n20-newer": {
            "total_s": 1400,
            "last_seen": 200,
            "cmd": base + " --N 20",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        "n10-older": {
            "total_s": 800,
            "last_seen": 100,
            "cmd": base + " --N 10",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-kg",
            "cmd": base.replace("old", "new") + " --N 10",
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:n10-older"
    assert rec["total_s"] == 800


def test_runtime_history_closest_kg_keeps_heldout_hard_boundary():
    history = {
        "wrong-heldout": {
            "total_s": 90,
            "cmd": (
                "python performance/run_lodo_manifest_shard.py "
                "--heldout QueueResourceControl --line lodo --d 1000 "
                "--meta-source-d 50 --N 10 --n0 10 "
                "--experiment-variant old"
            ),
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-kg",
            "cmd": (
                "python performance/run_lodo_manifest_shard.py "
                "--heldout InventorySupplyChain --line lodo --d 1000 "
                "--meta-source-d 50 --N 10 --n0 10 "
                "--experiment-variant new"
            ),
            "project": "KG-SYNTH",
            "cwd": "/tmp/SC-OLH-KG",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert rec is None
    assert key is None
    assert kind is None


def test_runtime_history_closest_bapr_specialist_reuses_mode_not_env():
    module = (
        "python -u -m "
        "jax_experiments.analysis.launch_bapr_v3_stochastic_independent_specialist "
        "--profile structured_channel --family structured_channel "
    )
    history = {
        "same-env-mode-0": {
            "total_s": 7000,
            "cmd": module + "--env HalfCheetah-v2 --mode 0 --resume",
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
        "wrong-env-mode-1": {
            "total_s": 70,
            "cmd": module + "--env Ant-v2 --mode 1 --resume",
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-bapr",
            "cmd": module + "--env HalfCheetah-v2 --mode 1 --resume",
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:same-env-mode-0"
    assert rec["total_s"] == 7000


def test_runtime_history_closest_never_crosses_python_module_entrypoint():
    common = (
        "python -u -m jax_experiments.analysis.{} "
        "--family structured_channel --env Ant-v2 --seed 0 --resume"
    )
    history = {
        "wrong-module": {
            "total_s": 160,
            "cmd": common.format("run_bapr_v3_stochastic_headroom"),
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
        "right-module": {
            "total_s": 12000,
            "cmd": common.format("run_bapr_v3_structured_channel_headroom"),
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-bapr",
            "cmd": common.format("run_bapr_v3_structured_channel_headroom"),
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert kind == "closest"
    assert key == "closest:right-module"
    assert rec["total_s"] == 12000


def test_runtime_history_closest_never_crosses_finalize_only_switch():
    base = (
        "python -u -m jax_experiments.analysis.run_bapr_v3_stochastic_headroom "
        "--family packet_loss --env Ant-v2 --seed 0 --resume"
    )
    history = {
        "finalize-only": {
            "total_s": 160,
            "cmd": base + " --finalize-existing",
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
    }

    rec, key, kind = runtime_history_best(
        {
            "signature": "new-training-run",
            "cmd": base,
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(runtime_history=history),
    )

    assert rec is None
    assert key is None
    assert kind is None


def test_runtime_history_closest_rejects_unknown_entrypoint_for_module_task():
    history = {
        "smoke-wrapper": {
            "total_s": 72,
            "cmd": "python -c 'print(\"smoke\")' --family packet_loss --env Ant-v2",
            "project": "BAPR",
            "cwd": "/tmp/BAPR",
        },
    }
    task = {
        "signature": "new-training-run",
        "cmd": (
            "python -u -m "
            "jax_experiments.analysis.run_bapr_v3_structured_channel_headroom "
            "--family structured_channel --env Ant-v2 --seed 0 --resume"
        ),
        "project": "BAPR",
        "cwd": "/tmp/BAPR",
    }

    rec, key, kind = runtime_history_best(
        task,
        task_runtime_payload=lambda value: value,
        deps=_runtime_deps(runtime_history=history),
    )

    assert rec is None
    assert key is None
    assert kind is None


def test_seed_pending_eta_clears_stale_live_eta_then_uses_duration_history():
    task = {
        "id": "t1",
        "status": "queued",
        "signature": "sig",
        "eta_seconds": 999,
        "eta_source": "progress_rate",
    }

    changed = seed_pending_eta_from_history(
        {"tasks": [task]},
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(resource_history={"sig": {"dur_s_ewma": 321}}),
    )

    assert changed == 2
    assert task["eta_seconds"] == 321
    assert task["eta_source"] == "duration_ewma"
    assert task["eta_confidence"] == "low"
    assert task["eta_updated_at"] == 456


def test_seed_pending_eta_skips_history_load_when_every_pending_task_has_eta():
    deps = replace(
        _runtime_deps(),
        load_runtime_history=lambda: (_ for _ in ()).throw(
            AssertionError("runtime history should not be loaded")
        ),
    )
    state = {
        "tasks": [
            {
                "id": "t1",
                "status": "queued",
                "eta_seconds": 123,
                "eta_source": "runtime_history",
            },
            {"id": "t2", "status": "running", "eta_seconds": 0},
        ]
    }

    assert seed_pending_eta_from_history(
        state,
        task_runtime_payload=lambda task: task,
        deps=deps,
    ) == 0


def test_seed_pending_eta_uses_fresh_matching_running_peer_without_history():
    pending = {
        "id": "t-pending",
        "status": "queued",
        "signature": "train-seed-2",
        "project": "h2oplus",
        "cwd": "/work",
        "cmd": "python train.py --steps 200 --seed 2",
    }
    peer = {
        "id": "t-running",
        "status": "running",
        "signature": "train-seed-1",
        "project": "h2oplus",
        "cwd": "/work",
        "cmd": "python train.py --steps 200 --seed 1",
        "runtime_total_s_est": 1000,
        "runtime_current_unit": 80,
        "runtime_total_units": 200,
        "runtime_unit_s_est": 5.0,
        "runtime_est_source": "progress_window",
        "runtime_progress_at": 450,
    }

    changed = seed_pending_eta_from_history(
        {"tasks": [pending, peer]},
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(now=456),
    )

    assert changed == 1
    assert pending["eta_seconds"] == 1000
    assert pending["eta_source"] == "peer_progress"
    assert pending["eta_detail"] == "queued ETA seeded from peer_progress"


def test_seed_pending_eta_ignores_stale_running_peer_projection():
    pending = {
        "id": "t-pending",
        "status": "queued",
        "signature": "train-seed-2",
        "project": "h2oplus",
        "cwd": "/work",
        "cmd": "python train.py --steps 200 --seed 2",
    }
    peer = {
        "id": "t-running",
        "status": "running",
        "signature": "train-seed-1",
        "project": "h2oplus",
        "cwd": "/work",
        "cmd": "python train.py --steps 200 --seed 1",
        "runtime_total_s_est": 1000,
        "runtime_current_unit": 80,
        "runtime_total_units": 200,
        "runtime_unit_s_est": 5.0,
        "runtime_est_source": "progress_window",
        "runtime_progress_at": 100,
    }

    assert seed_pending_eta_from_history(
        {"tasks": [pending, peer]},
        task_runtime_payload=lambda task: task,
        deps=_runtime_deps(now=456),
    ) == 0
    assert "eta_seconds" not in pending


class _Args:
    drop = None
    set = None
    vram_mb = None
    ram_mb = None
    cpu = None


def test_cmd_history_drop_set_and_missing_field_errors():
    history = {"sig": {"vram_mb": 9000}}
    saved = []
    printed = []
    deps = HistoryCommandDeps(
        load_history=lambda: history,
        save_history=lambda value: saved.append(dict(value)),
        now=lambda: 12.0,
        print_fn=lambda *parts, **kwargs: printed.append(" ".join(str(part) for part in parts)),
        exit_fn=lambda message: (_ for _ in ()).throw(SystemExit(message)),
    )

    args = _Args()
    args.drop = "missing"
    with pytest.raises(SystemExit, match="not in history"):
        cmd_history(args, deps=deps)

    args = _Args()
    args.set = "sig"
    args.ram_mb = 2048
    cmd_history(args, deps=deps)
    assert history["sig"]["ram_mb"] == 2048
    assert history["sig"]["ram_samples"] == [2048]
    assert saved[-1]["sig"]["last_seen"] == 12

    args = _Args()
    args.drop = "sig"
    cmd_history(args, deps=deps)
    assert "sig" not in history
    assert any(line.startswith("dropped 'sig'") for line in printed)
