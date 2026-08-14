from __future__ import annotations

from types import SimpleNamespace

from skill.scheduler_submit.task import build_submit_task_for_state
from skill.scheduler_submit.task_common import SubmitTaskDeps


def _parse_env(pairs) -> dict:
    return {
        key: value
        for key, value in (str(pair).split("=", 1) for pair in (pairs or []))
    }


def _deps(seed_calls: list) -> SubmitTaskDeps:
    return SubmitTaskDeps(
        known_nodes={"node001": {}},
        parse_env=_parse_env,
        task_run_identity=lambda task: (
            task.get("signature"),
            task.get("cmd"),
            task.get("cwd"),
        ),
        task_has_recorded_launch_artifacts=lambda _task: False,
        recorded_launch_safety_state=lambda _task: ("dead", ""),
        same_declared_path=lambda left, right: left == right,
        active_or_unsynced_result_status=lambda _task: False,
        history_get=lambda _signature: {},
        load_history=lambda: {},
        effective_est_vram=lambda _task, _state, _history: 3500,
        effective_est_ram=lambda _task, _state, _history: 4096,
        project_from_path=lambda _cwd: "project-from-path",
        cpu_worker_plan_for_items=lambda items, cores: {
            "workers": min(items, cores),
            "waves": 1,
            "physical_cores": cores,
            "last_wave_items": items,
        },
        node_physical_cores=lambda _node: 16,
        cpu_labor_node_names=lambda: ["node001"],
        allocate_task_id=lambda _state: "t00001",
        local_user=lambda: "tester",
        local_host_short=lambda: "host",
        scheduler_id=lambda: "scheduler-id",
        apply_test_log_runtime_profile=lambda _task, _path: False,
        seed_pending_eta_from_history=lambda state, **_kwargs: seed_calls.append(state),
        now=lambda: 123.5,
    )


def test_single_submit_preserves_v2_eta_metadata_and_submitted_environment():
    seed_calls = []
    args = SimpleNamespace(
        description="single task",
        cmd="python train.py --env HalfCheetah-v2",
        cwd="/tmp/work",
        signature="resac/halfcheetah",
        project=None,
        resource_family=None,
        vram_resource_family=None,
        ram_resource_family=None,
        vram=0,
        ram_mb=512,
        cpu=8,
        priority="normal",
        preferred_node=None,
        require_node=None,
        allowed_nodes=[],
        stage_excludes=[],
        stage_input_paths=[],
        git_repo=None,
        ckpt_dir=None,
        ckpt_glob="*",
        resume_flag="",
        result_dir=None,
        local_result_dir=None,
        env=["FROM_ENV=present", "OVERRIDE=new"],
        extra_env={"FROM_RECORD": "present", "OVERRIDE": "old"},
        allow_duplicate=False,
        allow_shared_ckpt_dir=False,
        allow_shared_result_dir=False,
        theorem_workload_key="hybrid_rl_resac_halfcheetah",
        workload_env="halfcheetah",
        allocation_workers=8,
        colocation_count=1,
        profile_axis="allocation_workers",
        resource_state="cpu_resident",
        resident_mix="cpu_worker_resident",
    )
    preflight = SimpleNamespace(
        ckpt_dir_was_inferred=False,
        inferred_ckpt_source="",
        inferred_resume_managed=False,
    )

    result = build_submit_task_for_state(
        {"tasks": []},
        args,
        preflight,
        deps=_deps(seed_calls),
        default_ram_mb=4096,
        default_cpu_cores=1,
    )

    assert result.task["extra_env"] == {
        "FROM_RECORD": "present",
        "FROM_ENV": "present",
        "OVERRIDE": "new",
    }
    assert {
        key: result.task[key]
        for key in (
            "theorem_workload_key",
            "workload_env",
            "allocation_workers",
            "colocation_count",
            "profile_axis",
            "resource_state",
            "resident_mix",
        )
    } == {
        "theorem_workload_key": "hybrid_rl_resac_halfcheetah",
        "workload_env": "halfcheetah",
        "allocation_workers": 8,
        "colocation_count": 1,
        "profile_axis": "allocation_workers",
        "resource_state": "cpu_resident",
        "resident_mix": "cpu_worker_resident",
    }
    assert result.task["resume_managed_by_cmd"] is False
    assert result.task["skip_resume_scan"] is False
    assert seed_calls == [{"tasks": [result.task]}]

    managed = build_submit_task_for_state(
        {"tasks": []},
        args,
        SimpleNamespace(
            ckpt_dir_was_inferred=False,
            inferred_ckpt_source="command-managed:runner.py",
            inferred_resume_managed=True,
        ),
        deps=_deps([]),
        default_ram_mb=4096,
        default_cpu_cores=1,
    ).task
    assert managed["resume_managed_by_cmd"] is True
    assert managed["skip_resume_scan"] is True
