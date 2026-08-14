from dataclasses import replace

from skill.scheduler_resource.forensics import (
    ResourceForensicsDeps,
    build_crash_forensics_payload,
    build_oom_forensics_payload,
    ckpt_task_evict_protected,
    is_oom_like_forensics,
    log_progress_forensics,
    remember_running_resource_snapshots,
    resource_snapshot_for_placement,
    summarize_task_for_resource_log,
    task_evict_loss_protected,
    task_has_progress_evidence,
    task_progress_ratio,
    task_ram_pressure_mb,
)


class _EtaTracker:
    @staticmethod
    def parse_progress(text, *, cmd=None):
        if "Iter 4/10" in text:
            return 4, 10
        return None

    @staticmethod
    def runtime_projection(text, *, elapsed_s, cmd=None):
        if "Iter 4/10" in text:
            return {"source": "progress_rate", "eta_s": 60, "total_s": elapsed_s + 60, "unit_s": 15}
        return None


def _deps(*, artifacts=None, elapsed=0.0, now=1000.0, eta_tracker=None):
    return ResourceForensicsDeps(
        now=lambda: now,
        effective_elapsed_s=lambda task: elapsed,
        discover_result_artifacts=lambda task, **kwargs: artifacts or [],
        compact_resource_task_summary=lambda payload: {k: payload[k] for k in payload if k in {
            "id", "started_at", "ram_pressure_mb", "progress_ratio"
        }},
        gpu_threshold_snapshot=lambda gpu: None if gpu is None else {
            "idx": gpu.get("idx"),
            "used_mb": gpu.get("used_mb"),
            "total_mb": gpu.get("total_mb"),
            "over_one_third": gpu.get("used_mb", 0) > gpu.get("total_mb", 1) // 3,
            "over_full": gpu.get("used_mb", 0) >= gpu.get("total_mb", 1),
            "one_third_mb": gpu.get("total_mb", 0) // 3,
        },
        node_ram_snapshot=lambda node: None if node is None else {"free_mb": node.get("free_ram_mb")},
        last_progress_line=lambda text: text.strip().splitlines()[-1] if text.strip() else "",
        load_eta_tracker_module=lambda: eta_tracker,
        ram_evict_ckpt_protect_progress=0.25,
        ram_evict_ckpt_protect_min_age_s=300.0,
    )


def test_task_progress_and_ram_pressure_helpers():
    task = {
        "runtime_current_unit": 3,
        "runtime_total_units": 10,
        "current_ram_mb": 256,
        "peak_ram_mb": 512,
        "ram_mb": 384,
    }

    assert task_progress_ratio(task) == 0.3
    assert task_has_progress_evidence(task)
    assert task_ram_pressure_mb(task) == 512
    assert task_progress_ratio({"runtime_current_unit": 1, "runtime_total_units": 0}) is None


def test_ckpt_and_incremental_result_protection():
    ckpt_task = {"ckpt_dir": "/tmp/ckpt", "runtime_current_unit": 3, "runtime_total_units": 10}
    assert ckpt_task_evict_protected(ckpt_task, deps=_deps(elapsed=10.0))

    old_ckpt_task = {"ckpt_dir": "/tmp/ckpt"}
    assert ckpt_task_evict_protected(old_ckpt_task, deps=_deps(elapsed=400.0))

    resumable_ckpt = {"ckpt_dir": "/tmp/ckpt", "resume_from": "/tmp/ckpt/latest"}
    assert not ckpt_task_evict_protected(resumable_ckpt, deps=_deps(elapsed=400.0))

    eval_task = {"cmd": "python eval.py --skip_existing --out_csv results.csv"}
    assert task_evict_loss_protected(
        eval_task,
        deps=_deps(artifacts=[{"kind": "file", "path": "/tmp/results.csv"}]),
    )


def test_resource_snapshot_summarizes_same_gpu_tasks():
    target = {
        "id": "t3",
        "status": "launching",
        "node": "node001",
        "gpu_idx": 0,
        "started_at": 30,
        "current_ram_mb": 300,
    }
    state = {
        "tasks": [
            {"id": "t1", "status": "running", "node": "node001", "gpu_idx": 0, "started_at": 10, "ram_mb": 100},
            {"id": "t2", "status": "running", "node": "node001", "gpu_idx": 1, "started_at": 20, "ram_mb": 200},
            target,
            {"id": "tdone", "status": "done", "node": "node001", "gpu_idx": 0},
        ]
    }
    nodes = [{"name": "node001", "free_ram_mb": 10000, "gpus": [{"idx": 0, "used_mb": 9000, "total_mb": 12000}]}]

    snap = resource_snapshot_for_placement(
        state,
        nodes,
        target,
        "node001",
        0,
        "launch_pre",
        deps=_deps(),
    )

    assert snap["stage"] == "launch_pre"
    assert snap["gpu"]["over_one_third"] is True
    assert snap["node_ram"] == {"free_mb": 10000}
    assert snap["same_gpu_task_count"] == 2
    assert [task["id"] for task in snap["same_gpu_tasks"]] == ["t1", "t3"]
    assert snap["same_node_running_task_count"] == 3


def test_remember_running_resource_snapshots_summarizes_each_active_task_once():
    compacted = []
    deps = replace(
        _deps(),
        compact_resource_task_summary=lambda payload: (
            compacted.append(payload["id"]) or {
                "id": payload["id"],
                "started_at": payload["started_at"],
            }
        ),
    )
    tasks = [
        {
            "id": f"t{idx}",
            "status": "running",
            "node": "node001",
            "gpu_idx": 0,
            "started_at": idx + 1,
        }
        for idx in range(100)
    ]

    remember_running_resource_snapshots(
        {"tasks": tasks},
        [{
            "name": "node001",
            "free_ram_mb": 10000,
            "gpus": [{"idx": 0, "used_mb": 1000, "total_mb": 12000}],
        }],
        deps=deps,
    )

    assert len(compacted) == 100
    assert tasks[0]["last_resource_snapshot"]["same_node_running_task_count"] == 100
    assert tasks[0]["last_resource_snapshot"]["same_gpu_task_count"] == 100
    assert [
        item["id"]
        for item in tasks[0]["last_resource_snapshot"]["same_gpu_tasks"]
    ] == [f"t{idx}" for idx in range(88, 100)]


def test_log_progress_forensics_and_crash_payload():
    task = {
        "id": "tcrash",
        "project": "BAPR",
        "signature": "BAPR/run",
        "description": "CUDA OOM during train",
        "node": "node001",
        "gpu_idx": 0,
        "started_at": 100.0,
        "finished_at": 220.0,
        "peak_vram_mb": 4096,
        "_diagnosis": {
            "reason": "CUDA_ERROR_OUT_OF_MEMORY",
            "tail": "Starting training\nIter 4/10\n",
            "lifetime_s": 120,
            "log_path": "/tmp/t.log",
        },
        "last_resource_snapshot": {
            "gpu": {
                "over_one_third": True,
                "over_full": False,
                "used_mb": 9000,
                "total_mb": 12000,
                "one_third_mb": 4000,
            },
            "same_gpu_tasks": [
                {"id": "tcrash", "started_at": 100},
                {"id": "tnew", "started_at": 130},
            ],
        },
    }

    progress = log_progress_forensics(
        task,
        "Starting training\nIter 4/10\n",
        elapsed_s=120,
        deps=_deps(eta_tracker=_EtaTracker()),
    )
    assert progress["progress_ratio"] == 0.4
    assert progress["eta_source"] == "progress_rate"
    assert progress["failure_stage"] == "mid_training_or_after_progress"

    crash = build_crash_forensics_payload(
        task,
        {"tasks": [task, {"id": "tnew", "status": "running"}]},
        deps=_deps(eta_tracker=_EtaTracker()),
    )
    assert crash["gpu_over_one_third"] is True
    assert crash["per_task_vram_known"] is True
    assert is_oom_like_forensics(crash)

    oom = build_oom_forensics_payload(
        crash,
        {"tasks": [task, {"id": "tnew", "status": "running"}]},
    )
    assert oom["suspected_trigger_task"]["id"] == "tnew"
    assert oom["trigger_task_status_after_oom"] == "running"
    assert oom["oom_victim_status"] is None


def test_summarize_task_for_resource_log_includes_resume_flags():
    task = {
        "id": "tsum",
        "status": "running",
        "started_at": 900.0,
        "runtime_current_unit": 5,
        "runtime_total_units": 10,
        "current_ram_mb": 600,
        "cmd": "python eval.py --skip-existing",
    }

    summary = summarize_task_for_resource_log(
        task,
        deps=_deps(artifacts=[{"kind": "file", "path": "/tmp/out.csv"}]),
    )

    assert summary["elapsed_s"] == 100
    assert summary["progress_ratio"] == 0.5
    assert summary["ram_pressure_mb"] == 600
    assert summary["incremental_result_resume"] is True
    assert summary["evict_loss_protected"] is True
