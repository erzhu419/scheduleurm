from __future__ import annotations

import argparse
import json

import pytest

from skill.scheduler_commands.bulk_submit import (
    BulkSubmitDeps,
    CpuBatchSubmitDeps,
    build_bulk_submit_task,
    build_bulk_submit_tasks,
    build_cpu_batch_submit_spec,
    bulk_submit_result_conflict_message,
    load_submit_jsonl_specs_from_text,
    spec_bool,
    spec_int,
)


def _deps(seed_calls: list | None = None) -> BulkSubmitDeps:
    seed_calls = seed_calls if seed_calls is not None else []

    def parse_env(pairs):
        if isinstance(pairs, dict):
            return dict(pairs)
        return {
            key: value
            for key, value in (str(pair).split("=", 1) for pair in (pairs or []))
        }

    def identity(task: dict):
        return (
            task.get("signature"),
            task.get("cmd"),
            task.get("cwd"),
            tuple(task.get("allowed_nodes") or []),
        )

    return BulkSubmitDeps(
        known_nodes={"node001": {}, "node002": {}, "local": {}},
        default_vram_mb=3500,
        default_ram_mb=4096,
        default_cpu_cores=1,
        canonical_node_name=lambda node: str(node).replace("n001", "node001"),
        canonicalize_node_list=lambda nodes: [
            str(n).replace("n001", "node001") for n in (nodes or [])
        ],
        parse_env=parse_env,
        task_run_identity=identity,
        history_get=lambda sig: {"vram_mb": 222, "ram_mb": 333, "cpu_cores": 4}
        if sig == "hist/sig"
        else {},
        project_from_path=lambda _cwd: "project-from-path",
        allocate_task_id=lambda state: f"t{state.get('next_id', 1):05d}",
        local_user=lambda: "tester",
        local_host_short=lambda: "host",
        scheduler_id=lambda: "scheduler-id",
        seed_pending_eta_from_history=lambda *args, **kwargs: seed_calls.append(
            (args, kwargs)
        ),
        now=lambda: 123.5,
    )


def _spec(**overrides) -> dict:
    spec = {
        "description": "bulk task",
        "cmd": "echo ok",
        "cwd": "/tmp/work",
        "signature": "bulk/sig",
    }
    spec.update(overrides)
    return spec


def test_load_submit_jsonl_specs_accepts_json_list_and_jsonl():
    assert load_submit_jsonl_specs_from_text(
        '[{"signature":"a"}]', source="stdin"
    ) == [{"signature": "a"}]
    assert load_submit_jsonl_specs_from_text(
        '{"signature":"a"}\n\n{"signature":"b"}', source="jobs.jsonl"
    ) == [{"signature": "a"}, {"signature": "b"}]


def test_load_submit_jsonl_specs_rejects_non_object_items():
    with pytest.raises(ValueError, match="item 1 is not an object"):
        load_submit_jsonl_specs_from_text('[{"ok":true}, 3]', source="stdin")


def test_spec_coercion_helpers_match_scheduler_fast_path_semantics():
    assert spec_bool({"x": "YES"}, "x") is True
    assert spec_bool({"x": None}, "x", True) is True
    assert spec_bool({"x": "0"}, "x", True) is False
    assert spec_int({"n": "7"}, "n") == 7
    assert spec_int({"n": ""}, "n", 3) == 3


def test_build_cpu_batch_submit_spec_materializes_templates_and_env():
    args = argparse.Namespace(
        description="eval {node}",
        cmd_template="python eval.py --start {start} --end {end} --workers {workers}",
        cwd="/work/{node}",
        signature="Demo/{node}/{start}-{end}",
        ram_mb=256,
        priority="high",
        project="Demo",
        env=["BASE=1"],
        result_dir_template="/remote/out/{node}",
        local_result_dir_template="/local/out/{node}",
        wait_for_file_template=["/ready/{node}/{start}"],
        stage_exclude=["data/cache"],
        node_down_requeue_s=42,
        allow_cpu_training=True,
        cpu_training_justification="short cpu-only validation batch for scheduler regression",
        allow_remote_large_data=True,
        allow_duplicate=True,
        skip_launch_staging=True,
        env_spec="none",
        image="",
    )
    row = {
        "node": "node001",
        "start": 3,
        "end": 7,
        "items": 4,
        "workers": 2,
        "waves": 2,
        "physical_cores": 8,
        "total_physical_cores": 16,
        "last_wave_items": 1,
        "shard_index": 0,
        "num_shards": 2,
    }

    def fmt(text, p, *_args):
        if text is None:
            return None
        return str(text).format(**p)

    def rewrite(text, p, *_args):
        return str(text).format(**p)

    spec, vals = build_cpu_batch_submit_spec(
        args,
        row,
        total_items=10,
        logical_items=5,
        item_multiplier=2,
        node_names=["node001", "node002"],
        deps=CpuBatchSubmitDeps(
            cpu_parallel_template_values=lambda p, *_args: {"node": p["node"]},
            format_cpu_parallel_template=fmt,
            rewrite_cpu_parallel_cmd=rewrite,
        ),
    )

    assert vals == {"node": "node001"}
    assert spec["description"] == "eval node001"
    assert spec["cmd"] == "python eval.py --start 3 --end 7 --workers 2"
    assert spec["cwd"] == "/work/node001"
    assert spec["signature"] == "Demo/node001/3-7"
    assert spec["preferred_node"] == "node001"
    assert spec["allowed_nodes"] == ["node001", "node002"]
    assert spec["vram"] == 0
    assert spec["ram_mb"] == 256
    assert spec["cpu"] == 2
    assert spec["cpu_parallel_total_items"] == 10
    assert spec["cpu_parallel_logical_items"] == 5
    assert spec["cpu_parallel_item_multiplier"] == 2
    assert spec["cpu_parallel_start"] == 3
    assert spec["cpu_parallel_end"] == 7
    assert spec["wait_for_files"] == ["/ready/node001/3"]
    assert "BASE=1" in spec["env"]
    assert "SCHEDULEURM_CPU_SHARD_START=3" in spec["env"]
    assert "SCHEDULEURM_CPU_TOTAL_WORK_ITEMS=10" in spec["env"]
    assert spec["skip_launch_staging"] is True


def test_bulk_submit_result_conflict_message_detects_existing_destination():
    msg = bulk_submit_result_conflict_message(
        {
            "tasks": [
                {
                    "id": "t1",
                    "status": "done",
                    "result_dir": "/remote/out",
                    "local_result_dir": "/tmp/out",
                    "result_synced_at": None,
                }
            ]
        },
        [{"signature": "new", "local_result_dir": "/tmp/out/"}],
        allow_shared_result_dir=False,
        active_or_unsynced_result_status=lambda task: (
            task.get("status") in {"queued", "launching", "running"}
            or (task.get("status") == "done" and task.get("result_dir") and not task.get("result_synced_at"))
        ),
        conflict_path_key=lambda path: str(path or "").rstrip("/"),
        same_declared_path=lambda a, b: str(a or "").rstrip("/") == str(b or "").rstrip("/"),
    )

    assert "already in use" in msg
    assert "t1" in msg


def test_bulk_submit_result_conflict_message_detects_same_batch_destination():
    msg = bulk_submit_result_conflict_message(
        {"tasks": []},
        [
            {"signature": "first", "local_result_dir": "/tmp/out"},
            {"signature": "second", "result_dir": "/tmp/out/"},
        ],
        allow_shared_result_dir=False,
        active_or_unsynced_result_status=lambda _task: False,
        conflict_path_key=lambda path: str(path or "").rstrip("/"),
        same_declared_path=lambda a, b: str(a or "").rstrip("/") == str(b or "").rstrip("/"),
    )

    assert "share a local result destination" in msg
    assert "first" in msg
    assert "second" in msg
    assert bulk_submit_result_conflict_message(
        {"tasks": []},
        [{"signature": "first", "local_result_dir": "/tmp/out"}],
        allow_shared_result_dir=True,
        active_or_unsynced_result_status=lambda _task: False,
        conflict_path_key=lambda path: str(path or "").rstrip("/"),
        same_declared_path=lambda a, b: str(a or "").rstrip("/") == str(b or "").rstrip("/"),
    ) == ""


def test_build_bulk_submit_task_uses_explicit_resources_and_seed_cache_inputs():
    seed_calls = []
    task = build_bulk_submit_task(
        {"tasks": [], "next_id": 10},
        _spec(
            vram=0,
            ram_mb=128,
            cpu=2,
            allowed_nodes="n001,node002",
            cpu_parallel_items=5,
            cpu_batch_plan={"workers": 2, "waves": 3, "physical_cores": 4},
            parent_id="t00009",
            retry_count=2,
            skip_resume_scan=True,
            resource_family="bapr/new-controller",
        ),
        deps=_deps(seed_calls),
        task_id="t00010",
        resource_history={"bulk/sig": {"vram_mb": 999, "ram_mb": 999, "cpu_cores": 9}},
        runtime_history={"runtime": {}},
        runtime_closest_index=[{"key": "x"}],
    )

    assert task["id"] == "t00010"
    assert task["status"] == "queued"
    assert task["project"] == "project-from-path"
    assert task["allowed_nodes"] == ["node001", "node002"]
    assert task["est_vram_mb"] == 0
    assert task["ram_mb"] == 128
    assert task["cpu_cores"] == 2
    assert task["submitted_by"] == "tester"
    assert task["scheduler_id"] == "scheduler-id"
    assert task["submitted_at"] == 123.5
    assert task["cpu_parallel_items"] == 5
    assert task["cpu_auto_workers"] == 2
    assert task["parent_id"] == "t00009"
    assert task["retry_count"] == 2
    assert task["resource_family"] == "bapr/new-controller"
    assert task["skip_resume_scan"] is True
    assert seed_calls[0][1]["runtime_history_cache"] == {"runtime": {}}
    assert seed_calls[0][1]["runtime_closest_index"] == [{"key": "x"}]


def test_build_bulk_submit_task_uses_resource_history_cache_without_history_get():
    task = build_bulk_submit_task(
        {"tasks": [], "next_id": 10},
        _spec(signature="hist/sig"),
        deps=_deps(),
        resource_history={"hist/sig": {"vram_mb": 12, "ram_mb": 34, "cpu_cores": 5}},
    )

    assert task["est_vram_mb"] == 12
    assert task["ram_mb"] == 34
    assert task["cpu_cores"] == 5


def test_submit_jsonl_task_round_trip_preserves_v2_eta_metadata_and_environment():
    metadata = {
        "theorem_workload_key": "hybrid_rl_resac_halfcheetah",
        "workload_env": "halfcheetah",
        "allocation_workers": 8,
        "colocation_count": 1,
        "profile_axis": "allocation_workers",
        "resource_state": "cpu_resident",
        "resident_mix": "cpu_worker_resident",
    }
    first = build_bulk_submit_task(
        {"tasks": []},
        _spec(
            env=["FROM_ENV=present", "OVERRIDE=new"],
            extra_env={"FROM_RECORD": "present", "OVERRIDE": "old"},
            **metadata,
        ),
        deps=_deps(),
        task_id="t00001",
    )

    round_tripped_specs = load_submit_jsonl_specs_from_text(
        json.dumps(first),
        source="round-trip.jsonl",
    )
    second = build_bulk_submit_task(
        {"tasks": []},
        round_tripped_specs[0],
        deps=_deps(),
        task_id="t00002",
    )

    assert first["extra_env"] == {
        "FROM_RECORD": "present",
        "FROM_ENV": "present",
        "OVERRIDE": "new",
    }
    assert second["extra_env"] == first["extra_env"]
    assert {key: second[key] for key in metadata} == metadata


def test_build_bulk_submit_task_rejects_duplicate_active_identity():
    existing = _spec()
    existing.update({"id": "t00001", "status": "running", "allowed_nodes": None})

    with pytest.raises(ValueError, match="duplicate active task t00001"):
        build_bulk_submit_task({"tasks": [existing]}, _spec(), deps=_deps())


def test_build_bulk_submit_task_rejects_unknown_nodes():
    with pytest.raises(ValueError, match="unknown node"):
        build_bulk_submit_task(
            {"tasks": []},
            _spec(require_node="missing-node"),
            deps=_deps(),
        )


def test_build_bulk_submit_task_preserves_hard_gpu_pin():
    task = build_bulk_submit_task(
        {"tasks": []},
        _spec(require_node="node001", require_gpu_idx="1"),
        deps=_deps(),
    )

    assert task["require_node"] == "node001"
    assert task["require_gpu_idx"] == 1


def test_build_bulk_submit_task_preserves_gpu_packing_override():
    task = build_bulk_submit_task(
        {"tasks": []},
        _spec(allow_gpu_over_one_third=True),
        deps=_deps(),
    )

    assert task["allow_gpu_over_one_third"] is True


def test_build_bulk_submit_task_preserves_command_managed_resume():
    task = build_bulk_submit_task(
        {"tasks": []},
        _spec(resume_managed_by_cmd=True, resume_flag=""),
        deps=_deps(),
    )

    assert task["resume_managed_by_cmd"] is True
    assert task["resume_flag"] == ""


def test_build_bulk_submit_task_rejects_negative_gpu_pin():
    with pytest.raises(ValueError, match="non-negative"):
        build_bulk_submit_task(
            {"tasks": []},
            _spec(require_node="node001", require_gpu_idx=-1),
            deps=_deps(),
        )


def test_build_bulk_submit_tasks_appends_all_after_validation():
    state = {"tasks": []}

    submitted = build_bulk_submit_tasks(
        state,
        [_spec(signature="bulk/a"), _spec(signature="bulk/b")],
        deps=_deps(),
        task_ids=["t00001", "t00002"],
    )

    assert [t["id"] for t in submitted] == ["t00001", "t00002"]
    assert [t["id"] for t in state["tasks"]] == ["t00001", "t00002"]


def test_build_bulk_submit_tasks_keeps_state_unchanged_when_later_spec_fails():
    existing = _spec(signature="bulk/existing")
    existing.update({"id": "t00000", "status": "done"})
    state = {"tasks": [existing]}

    with pytest.raises(RuntimeError, match="bulk spec #1 failed"):
        build_bulk_submit_tasks(
            state,
            [_spec(signature="bulk/a"), _spec(signature="bulk/b", require_node="bad")],
            deps=_deps(),
            task_ids=["t00001", "t00002"],
        )

    assert state["tasks"] == [existing]


def test_build_bulk_submit_tasks_detects_duplicates_between_new_specs():
    state = {"tasks": []}

    with pytest.raises(RuntimeError, match="bulk spec #1 failed: duplicate active task t00001"):
        build_bulk_submit_tasks(
            state,
            [_spec(signature="bulk/a"), _spec(signature="bulk/a")],
            deps=_deps(),
            task_ids=["t00001", "t00002"],
        )

    assert state["tasks"] == []


def test_build_bulk_submit_tasks_rejects_task_id_count_mismatch():
    with pytest.raises(ValueError, match="id count mismatch"):
        build_bulk_submit_tasks(
            {"tasks": []},
            [_spec(signature="bulk/a"), _spec(signature="bulk/b")],
            deps=_deps(),
            task_ids=["t00001"],
        )
