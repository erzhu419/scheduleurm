import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithm.experiments.full_factorial_parallel_orchestrator import (
    LeaseConflict,
    UnsafeLaunchCommand,
    launch_leased_probe,
    prepare_or_resume_wave,
    recover_wave,
    validate_controlled_probe_command,
)
from algorithm.experiments.full_factorial_parallel_probe_plan import (
    DEFAULT_DESIGN,
    T0_CPU_RUNNER,
    T0_GPU_RUNNER,
    T2_MARGINAL_RUNNER,
    build_parallel_probe_plan,
)


FIELDS = (
    "row_id",
    "tier",
    "quadrant",
    "node",
    "node_bucket",
    "hardware_group",
    "resource_kind",
    "resource_state",
    "resident_mix",
    "workload_key",
    "workload_env",
    "missing_profiles",
    "measured_profiles",
    "boundary_closed_profiles",
    "status",
    "runner",
)


def _row(
    row_id: str,
    *,
    hardware_group: str,
    resource_kind: str,
    workload_key: str,
    workload_env: str,
    node: str = "logical",
    tier: str = "T0_core_curve",
    resource_state: str = "empty",
    resident_mix: str = "none",
) -> dict[str, str]:
    return {
        "row_id": row_id,
        "tier": tier,
        "quadrant": "q10" if resource_kind == "cpu" else "q01",
        "node": node,
        "node_bucket": f"{node}:{hardware_group}",
        "hardware_group": hardware_group,
        "resource_kind": resource_kind,
        "resource_state": resource_state,
        "resident_mix": resident_mix,
        "workload_key": workload_key,
        "workload_env": workload_env,
        "missing_profiles": "1",
        "measured_profiles": "",
        "boundary_closed_profiles": "",
        "status": "pending_probe",
        "runner": "unit",
    }


def _write_design(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_default_design_uses_current_schemafixed_pending_inventory(tmp_path):
    report = build_parallel_probe_plan(
        design_csv=DEFAULT_DESIGN,
        max_rows_per_lane=0,
        assignment_dir=tmp_path / "assignments",
    )

    assert report["pending_statuses"] == [
        "partial_pending_probe",
        "pending_probe",
    ]
    assert report["pending_row_count"] == 180
    assert [lane["node_lock"] for lane in report["lanes"]] == [
        "jtl110gpu",
        "jtl110gpu2",
        "jtl311linux",
        "node007",
        "node001",
        "node002",
        "node003",
        "node004",
        "node005",
        "node006",
    ]


def test_maximum_wave_uses_all_ten_physical_lanes_once(tmp_path):
    rows = [
        _row(
            "jtl110-cnn",
            hardware_group="gpu_3080ti_12gb_dual",
            resource_kind="gpu",
            workload_key="gpu_cnn_torch_resnet50",
            workload_env="cnn",
        ),
        _row(
            "jtl110-llm",
            hardware_group="gpu_3080ti_12gb_dual",
            resource_kind="gpu",
            workload_key="gpu_llm_distilgpt2",
            workload_env="llm",
        ),
        _row(
            "jtl311-gpu",
            hardware_group="gpu_rtx2080_8gb_dual_cpu_fast",
            resource_kind="gpu",
            workload_key="gpu_cnn_torch_resnet50",
            workload_env="cnn",
        ),
        _row(
            "jtl311-cpu",
            hardware_group="gpu_rtx2080_8gb_dual_cpu_fast",
            resource_kind="cpu",
            workload_key="light_control_local",
            workload_env="light",
        ),
        _row(
            "node007-gpu",
            hardware_group="gpu_node007_4x11gb",
            resource_kind="gpu",
            workload_key="gpu_llm_distilgpt2",
            workload_env="llm",
        ),
    ]
    cpu_workloads = (
        ("hpc-light", "light_control_local", "light", "empty"),
        ("hpc-heavy", "cpu_heavy_local_bench", "cpu", "empty"),
        ("hpc-freq", "freqduet_cpu_ablation_c17_32", "freqduet", "empty"),
        ("hpc-sumo", "sumo_eval_cpu", "sumo", "empty"),
        ("hpc-heavy-half", "cpu_heavy_local_bench", "cpu", "half_loaded"),
        ("hpc-sumo-full", "sumo_eval_cpu", "sumo", "full_loaded"),
    )
    rows.extend(
        _row(
            row_id,
            hardware_group="cpu_hpc_192c",
            resource_kind="cpu",
            workload_key=workload,
            workload_env=env,
            resource_state=state,
            resident_mix=f"factor-{row_id}",
        )
        for row_id, workload, env, state in cpu_workloads
    )
    design = _write_design(tmp_path / "design.csv", rows)

    report = build_parallel_probe_plan(
        design_csv=design,
        max_rows_per_lane=4,
        assignment_dir=tmp_path / "assignments",
    )

    selected = report["selected_rows"]
    assert report["selected_row_count"] == 10
    assert all(lane["selected_count"] <= 1 for lane in report["lanes"])
    assert len({row["node_lock"] for row in selected}) == 10
    assert len({row["factor_lock_key"] for row in selected}) == 10
    assert {row["node_lock"] for row in selected if row["node_lock"].startswith("jtl110gpu")} == {
        "jtl110gpu",
        "jtl110gpu2",
    }
    assert sum(row["node_lock"] == "jtl311linux" for row in selected) == 1
    routes = {row["runner_route"] for row in selected}
    assert {T0_CPU_RUNNER, T0_GPU_RUNNER} <= routes
    assert all("scheduler.py" not in command for lane in report["lanes"] for command in lane["commands"])


def test_homogeneous_lanes_do_not_duplicate_a_factor_row(tmp_path):
    common = dict(
        resource_kind="cpu",
        workload_key="cpu_heavy_local_bench",
        workload_env="cpu",
        resource_state="full_loaded",
        resident_mix="same_workload_to_capacity_boundary",
    )
    design = _write_design(
        tmp_path / "design.csv",
        [
            _row("representative", hardware_group="cpu_hpc_192c", **common),
            _row("equivalent-copy", hardware_group="cpu_hpc_192c_equiv", **common),
        ],
    )

    report = build_parallel_probe_plan(
        design_csv=design,
        assignment_dir=tmp_path / "assignments",
    )

    assert report["selected_row_count"] == 1
    assert sum(lane["selected_count"] for lane in report["lanes"] if lane["node_lock"].startswith("node00")) == 1


def test_external_load_is_only_an_observed_resident_t2_lane(tmp_path):
    design = _write_design(
        tmp_path / "design.csv",
        [
            _row(
                "empty-row",
                hardware_group="gpu_3080ti_12gb_dual",
                resource_kind="gpu",
                workload_key="gpu_cnn_torch_resnet50",
                workload_env="cnn",
            ),
            _row(
                "resident-row",
                hardware_group="gpu_3080ti_12gb_dual",
                resource_kind="gpu",
                workload_key="gpu_llm_distilgpt2",
                workload_env="llm",
                tier="T2_load_state",
                resource_state="cpu_resident",
                resident_mix="cpu_worker_resident",
            ),
        ],
    )
    audit = tmp_path / "availability.json"
    audit.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "node": "jtl110gpu",
                        "parsed": {
                            "availability_class": "cpu_busy_gpu_idle",
                            "cpu_busy": True,
                            "gpu_busy": False,
                            "external_busy_process_count": 3,
                            "safe_for_gpu_only_probe": True,
                            "safe_for_hybrid_probe": False,
                        },
                    },
                    {
                        "node": "jtl110gpu2",
                        "parsed": {
                            "availability_class": "cpu_busy_gpu_idle",
                            "cpu_busy": True,
                            "gpu_busy": False,
                            "external_busy_process_count": 0,
                            "safe_for_gpu_only_probe": False,
                            "safe_for_hybrid_probe": False,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_parallel_probe_plan(
        design_csv=design,
        availability_json=audit,
        assignment_dir=tmp_path / "assignments",
    )

    assert [row["row_id"] for row in report["selected_rows"]] == ["resident-row"]
    selected = report["selected_rows"][0]
    assert selected["lane_mode"] == "observed-resident"
    assert selected["observed_resident"] is True
    assert selected["observed_resident_state"] == "cpu_resident"
    assert selected["runner_route"] == T2_MARGINAL_RUNNER
    command = next(command for lane in report["lanes"] for command in lane["commands"])
    assert f"-m {T2_MARGINAL_RUNNER}" in command
    assert "--from-design" in command
    assert {row["row_id"] for row in report["availability_blocked_rows"]} == {"empty-row"}
    assert "resident-row" not in {row["row_id"] for row in report["runner_not_ready_rows"]}


def test_all_resident_and_mixed_t2_states_use_the_marginal_harness(tmp_path):
    design = _write_design(
        tmp_path / "design.csv",
        [
            _row(
                "high-vram",
                hardware_group="gpu_3080ti_12gb_dual",
                resource_kind="gpu",
                workload_key="gpu_llm_distilgpt2",
                workload_env="llm",
                tier="T2_load_state",
                resource_state="high_vram_resident",
                resident_mix="llm_or_memory_resident",
            ),
            _row(
                "mixed",
                hardware_group="gpu_rtx2080_8gb_dual_cpu_fast",
                resource_kind="gpu",
                workload_key="gpu_cnn_torch_resnet50",
                workload_env="cnn",
                tier="T2_load_state",
                resource_state="mixed_colocation",
                resident_mix="cnn_plus_llm",
            ),
            _row(
                "cpu-resident",
                hardware_group="gpu_node007_4x11gb",
                resource_kind="gpu",
                workload_key="gpu_cnn_torch_resnet50",
                workload_env="cnn",
                tier="T2_load_state",
                resource_state="cpu_resident",
                resident_mix="cpu_worker_resident",
            ),
        ],
    )

    report = build_parallel_probe_plan(
        design_csv=design,
        assignment_dir=tmp_path / "assignments",
    )

    assert {row["row_id"] for row in report["selected_rows"]} == {
        "high-vram",
        "mixed",
        "cpu-resident",
    }
    assert {row["runner_route"] for row in report["selected_rows"]} == {T2_MARGINAL_RUNNER}
    assert report["runner_not_ready_rows"] == []
    commands = [command for lane in report["lanes"] for command in lane["commands"]]
    assert len(commands) == 3
    assert all("live_marginal_under_load_matrix" in command and "--from-design" in command for command in commands)


def test_orchestrator_resumes_leases_and_never_auto_relaunches(tmp_path):
    design = _write_design(
        tmp_path / "design.csv",
        [
            _row(
                "probe-row",
                hardware_group="gpu_3080ti_12gb_dual",
                resource_kind="gpu",
                workload_key="gpu_cnn_torch_resnet50",
                workload_env="cnn",
            )
        ],
    )
    plan = build_parallel_probe_plan(
        design_csv=design,
        assignment_dir=tmp_path / "assignments",
    )
    state_path = tmp_path / "wave-state.json"

    first = prepare_or_resume_wave(
        plan,
        state_path=state_path,
        owner="worker-a",
        lease_ttl_s=10,
        now=100,
    )
    resumed = prepare_or_resume_wave(
        plan,
        state_path=state_path,
        owner="worker-a",
        lease_ttl_s=10,
        now=101,
    )

    assert first["automatic_launch"] is False
    assert first["leases"][0]["lease_id"] == resumed["leases"][0]["lease_id"]
    with pytest.raises(LeaseConflict):
        prepare_or_resume_wave(
            plan,
            state_path=state_path,
            owner="worker-b",
            lease_ttl_s=10,
            now=101,
        )

    next_plan = build_parallel_probe_plan(
        design_csv=design,
        lease_state_json=state_path,
        now=101,
        assignment_dir=tmp_path / "next-assignments",
    )
    assert next_plan["selected_rows"] == []
    assert next_plan["active_lease_row_ids"] == ["probe-row"]

    calls = []

    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=4321)

    lease_id = first["leases"][0]["lease_id"]
    with pytest.raises(PermissionError):
        launch_leased_probe(
            state_path=state_path,
            lease_id=lease_id,
            owner="worker-a",
            popen_factory=fake_popen,
        )
    running = launch_leased_probe(
        state_path=state_path,
        lease_id=lease_id,
        owner="worker-a",
        allow_controlled_launch=True,
        now=102,
        popen_factory=fake_popen,
    )
    assert running["status"] == "running"
    assert calls[0][0][1:3] == ["-m", T0_GPU_RUNNER]

    recovered = recover_wave(state_path=state_path, now=111)
    assert recovered["status"] == "RECOVERY_REQUIRED"
    assert recovered["leases"][0]["status"] == "recovery_required"
    assert "no_automatic_relaunch" in recovered["leases"][0]["recovery_reason"]


def test_command_boundary_rejects_ordinary_scheduler_launch():
    with pytest.raises(UnsafeLaunchCommand):
        validate_controlled_probe_command("python3 -m skill.scheduler --allow-launch")
    with pytest.raises(UnsafeLaunchCommand):
        validate_controlled_probe_command("python3 skill/scheduler.py launch")
