from __future__ import annotations

import json
import os

from skill.scheduler_runtime.history_runtime import (
    build_runtime_history_runtime_exports,
    load_runtime_history_cached,
)
from skill.scheduler_runtime_history import RuntimeHistoryRecordDeps, runtime_history_record


def _deps(saved, *, history=None, max_entries=10, samples_per_key=3,
          now=123.0, payload_cmd="python train.py"):
    payload = {
        "signature": "sig",
        "project": "proj",
        "description": "desc",
        "cwd": "/work",
        "env_spec": "none",
        "cmd": payload_cmd,
    }
    return RuntimeHistoryRecordDeps(
        load_runtime_history=lambda: dict(history or {}),
        save_runtime_history=lambda value: saved.append(value),
        task_runtime_keys=lambda task: [("exact:key", "exact", payload)],
        runtime_device_kind_for_task=lambda task, gpu_idx: "gpu",
        runtime_node_bucket_key=lambda node, device: f"{node}:{device}",
        percentile=lambda samples, percentile_value: max(samples) if samples else 0,
        max_entries=max_entries,
        samples_per_key=samples_per_key,
        percentile_value=80,
        min_walltime_s=60,
        walltime_mult=1.2,
        now=lambda: now,
    )


def test_runtime_history_records_progress_sample_and_node_bucket():
    saved = []
    history = {
        "exact:key": {
            "total_s_samples": [50],
            "unit_s_samples": [1.0],
            "runs": 1,
            "node_runtime": {
                "node001:gpu": {
                    "total_s_samples": [40],
                    "unit_s_samples": [0.8],
                    "runs": 1,
                }
            },
        }
    }
    task = {
        "runtime_total_s_est": 100,
        "runtime_est_source": "progress",
        "runtime_unit_s_est": 2.5,
        "runtime_total_units": 40,
        "node": "node001",
        "gpu_idx": 0,
    }

    runtime_history_record(task, deps=_deps(saved, history=history, samples_per_key=2))

    assert len(saved) == 1
    rec = saved[0]["exact:key"]
    assert rec["total_s_samples"] == [50, 100]
    assert rec["total_s"] == 100
    assert rec["walltime_s"] == 120
    assert rec["unit_s_samples"] == [1.0, 2.5]
    assert rec["unit_s"] == 2.5
    assert rec["total_units"] == 40
    assert rec["source"] == "progress"
    assert rec["kind"] == "exact"
    assert rec["signature"] == "sig"
    assert rec["node_runtime"]["node001:gpu"]["total_s_samples"] == [40, 100]
    assert rec["node_runtime"]["node001:gpu"]["unit_s_samples"] == [0.8, 2.5]
    assert rec["node_runtime"]["node001:gpu"]["device_kind"] == "gpu"


def test_runtime_history_uses_duration_fallback_and_prunes_old_entries():
    saved = []
    history = {
        "old": {"last_seen": 1},
        "older": {"last_seen": 2},
    }

    runtime_history_record(
        {"node": "node002"},
        duration_s=30,
        deps=_deps(saved, history=history, max_entries=1, now=50.0),
    )

    assert list(saved[0]) == ["exact:key"]
    assert saved[0]["exact:key"]["total_s"] == 30
    assert saved[0]["exact:key"]["source"] == "duration"
    assert saved[0]["exact:key"]["last_seen"] == 50


def test_completed_full_run_replaces_stale_projection_with_actual_duration():
    saved = []
    history = {
        "exact:key": {
            "total_s_samples": [8492],
            "total_s": 8492,
            "unit_s_samples": [4.246],
            "unit_s": 4.246,
            "source": "kg_inner_progress",
            "runs": 1,
            "node_runtime": {
                "node006:gpu": {
                    "total_s_samples": [8492],
                    "unit_s_samples": [4.246],
                    "runs": 1,
                }
            },
        }
    }
    task = {
        "runtime_total_s_est": 8492,
        "runtime_est_source": "kg_inner_progress",
        "runtime_unit_s_est": 4.246,
        "runtime_total_units": 2000,
        "retry_count": 0,
        "node": "node006",
        "gpu_idx": 0,
    }

    runtime_history_record(
        task,
        duration_s=1276,
        deps=_deps(saved, history=history),
    )

    rec = saved[0]["exact:key"]
    assert rec["total_s_samples"] == [1276]
    assert rec["total_s"] == 1276
    assert rec["unit_s_samples"] == [0.638]
    assert rec["unit_s"] == 0.638
    assert rec["source"] == "duration"
    assert rec["runs"] == 1
    assert rec["node_runtime"]["node006:gpu"]["total_s_samples"] == [1276]


def test_resumed_completion_keeps_full_run_projection():
    saved = []
    task = {
        "runtime_total_s_est": 2000,
        "runtime_est_source": "progress_rate",
        "runtime_unit_s_est": 2.0,
        "runtime_total_units": 1000,
        "retry_count": 1,
        "resume_from": "/checkpoints/latest.pkl",
        "node": "node001",
        "gpu_idx": 0,
    }

    runtime_history_record(task, duration_s=500, deps=_deps(saved))

    rec = saved[0]["exact:key"]
    assert rec["total_s"] == 2000
    assert rec["unit_s"] == 2.0
    assert rec["source"] == "progress_rate"


def test_runtime_history_skips_empty_samples_without_saving():
    saved = []

    runtime_history_record({}, deps=_deps(saved))
    runtime_history_record({"runtime_total_s_est": 0}, deps=_deps(saved))

    assert saved == []


def test_runtime_history_keeps_script_and_flags_after_long_env_prefix():
    saved = []
    command = (
        "env " + " ".join(f"LONG_ENV_{index}={'x' * 40}" for index in range(20))
        + " python performance/benchmark_transfer_fairness.py "
        + "--method fsbo_cbo --heldout InventorySupplyChain"
    )

    runtime_history_record(
        {"node": "node001"},
        duration_s=120,
        deps=_deps(saved, payload_cmd=command),
    )

    stored = saved[0]["exact:key"]["cmd"]
    assert "benchmark_transfer_fairness.py" in stored
    assert "--method fsbo_cbo" in stored
    assert stored == command


def test_runtime_history_read_cache_reloads_only_after_atomic_file_change(tmp_path):
    path = tmp_path / "runtime_history.json"
    path.write_text('{"version": 1}')
    calls = []
    cache = {}

    def load():
        calls.append(path.stat().st_ino)
        return json.loads(path.read_text())

    assert load_runtime_history_cached(path, load, cache) == {"version": 1}
    assert load_runtime_history_cached(path, load, cache) == {"version": 1}
    assert len(calls) == 1

    replacement = tmp_path / "runtime_history.next"
    replacement.write_text('{"version": 2}')
    os.replace(replacement, path)

    assert load_runtime_history_cached(path, load, cache) == {"version": 2}
    assert len(calls) == 2


def test_candidate_runtime_scoring_reuses_one_history_snapshot(tmp_path):
    path = tmp_path / "runtime_history.json"
    path.write_text("{}")
    loads = []
    index_builds = []

    def load_runtime_history():
        loads.append(1)
        return {"runtime": len(loads)}

    def candidate_impl(_task, _node, _gpu_idx, **kwargs):
        return kwargs["load_runtime_history"]()["runtime"]

    namespace = {
        "RUNTIME_FILE": path,
        "load_runtime_history": load_runtime_history,
        "_candidate_runtime_seconds_impl": candidate_impl,
        "_task_runtime_keys": lambda _task: [],
        "_runtime_device_kind_for_task": lambda _task, _gpu_idx=None: "cpu",
        "_runtime_node_bucket_keys": lambda _node, _device: [],
        "_runtime_history_closest_index_impl": lambda history: (
            index_builds.append(history) or [history["runtime"]]
        ),
    }
    exports = build_runtime_history_runtime_exports(namespace)

    scores = [
        exports["_candidate_runtime_seconds"]({}, "node001", None)
        for _ in range(500)
    ]

    assert scores == [1] * 500
    assert len(loads) == 1
    history = exports["_candidate_runtime_history"]()
    assert exports["_candidate_runtime_closest_index"](history) == [1]
    assert exports["_candidate_runtime_closest_index"](history) == [1]
    assert len(index_builds) == 1
