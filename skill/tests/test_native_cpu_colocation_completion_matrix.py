from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import native_cpu_colocation_completion_matrix as matrix
from algorithm.experiments.file_progress_cpu_benchmark import _write_checkpoint


def _ready_task(
    task_id: str,
    *,
    flow_s: float,
    launch_offset_s: float,
    total_units: int,
    unit_s: float,
    eta_s: float,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "returncode": 0,
        "measurement_valid": True,
        "launch_offset_s": launch_offset_s,
        "flow_time_s": flow_s,
        "completion_model": {
            "completion_model_ready": True,
            "unit": "episode",
            "total_units": total_units,
            "completion_unit_s": unit_s,
            "predicted_total_s": eta_s,
            "total_wall_s": flow_s - launch_offset_s,
        },
    }


def test_manifest_profiles_are_independent_task_counts():
    result = matrix.build_native_cpu_colocation_completion_matrix(
        run_id="manifest_contract",
        nodes={"node001", "node006"},
        profiles=(1, 2, 4),
        kinds={"freqduet", "sumo"},
        tasks_per_node=4,
    )

    assert result["status"] == "MANIFEST"
    assert result["allow_launch"] is False
    assert result["profile_axis"] == "colocation_count"
    assert result["allocation_workers"] == 1
    assert result["selected_count"] == 12
    assert result["selected_task_count"] == 28
    assert result["same_node_profiles_serialized"] is True
    assert result["user_task_control_operations"] == []

    outputs = []
    paired_seeds = {}
    for row in result["rows"]:
        assert row["status"] == "MANIFEST"
        assert row["launched"] is False
        assert row["profile"] == row["colocation_count"]
        assert row["profile_label"] == f"p{row['colocation_count']}"
        assert row["allocation_workers"] == 1
        assert row["workload_key"].endswith("_native")
        assert row["node_bucket"].endswith(":cpu_hpc_192c")
        assert row["hardware_class"] == "cpu_hpc_192c"
        assert row["resource_state"] == (
            "empty" if row["colocation_count"] == 1
            else "controlled_colocation"
        )
        assert row["task_count"] == row["colocation_count"]
        assert len(row["tasks"]) == row["colocation_count"]
        for task in row["tasks"]:
            assert task["allocation_workers"] == 1
            assert task["durable_outer_progress"] is True
            assert task["phase_aware_completion_model_required"] is True
            assert task["total_outer_units"] >= 6
            paired_seeds.setdefault(
                (row["kind"], row["profile"], task["task_index"]),
                set(),
            ).add(task["seed"])
            outputs.append(task["output_dir"])

    assert all(len(values) == 1 for values in paired_seeds.values())
    for row in result["rows"]:
        assert len({task["seed"] for task in row["tasks"]}) == len(row["tasks"])
    assert len(outputs) == len(set(outputs))
    assert result["paired_seed_across_nodes"] is True
    assert result["nested_task_set_across_profiles"] is True
    assert result["min_interval_sample_count"] == 5
    assert result["measurement_protocol"] == matrix.MEASUREMENT_PROTOCOL
    assert len(result["measurement_scope_sha256"]) == 64
    scope = result["measurement_scope"]
    assert scope["nodes"] == ["node001", "node006"]
    assert scope["kinds"] == ["freqduet", "sumo"]
    assert scope["profile_counts"] == [1, 2, 4]
    assert scope["expected_cell_count"] == result["selected_count"]
    assert scope["expected_task_count"] == result["selected_task_count"]
    assert scope["protocol_source_sha256"].keys() == set(
        matrix.PROTOCOL_SOURCE_FILES
    )
    assert all(
        len(value) == 64
        for value in scope["protocol_source_sha256"].values()
    )


def test_default_manifest_covers_six_parallel_nodes():
    result = matrix.build_native_cpu_colocation_completion_matrix(
        run_id="six_node_manifest",
        profiles=(1,),
        kinds={"freqduet"},
        tasks_per_node=1,
    )

    assert result["nodes"] == list(matrix.DEFAULT_NODES)
    assert result["selected_count"] == 6
    assert result["max_parallel"] == 6
    assert result["parallelism_scope"] == "nodes"


def test_controlled_light_and_cpu_manifests_keep_colocation_axis():
    result = matrix.build_native_cpu_colocation_completion_matrix(
        run_id="controlled_cpu_manifest",
        nodes={"node001"},
        profiles=(1, 8, 9, 13),
        kinds={"light", "cpu_bench"},
        tasks_per_node=13,
    )

    assert result["profile_axis"] == "colocation_count"
    assert result["controlled_benchmarks_only"] == ["cpu_bench", "light"]
    assert result["selected_count"] == 8
    by_kind_profile = {
        (row["kind"], row["profile"]): row for row in result["rows"]
    }
    assert by_kind_profile[("light", 13)]["workload_key"] == "light_control_local"
    assert by_kind_profile[("light", 13)]["tasks"][0]["total_outer_units"] == 300
    assert by_kind_profile[("cpu_bench", 9)]["workload_key"] == "cpu_heavy_local_bench"
    assert by_kind_profile[("cpu_bench", 9)]["tasks"][0]["total_outer_units"] == 120
    assert all(row["allocation_workers"] == 1 for row in result["rows"])


def test_controlled_cpu_runtime_uses_durable_progress_file():
    cell = matrix.colocation_cells(
        run_id="controlled_cpu_runtime",
        nodes={"node001"},
        profiles=(8,),
        kinds={"cpu_bench"},
        tasks_per_node=8,
    )[0]
    task = cell.tasks[0]

    runtime = matrix._runtime_task_spec(
        cell=cell,
        task=task,
        group_run_id="controlled_cpu_runtime_node001_cpu_p8",
    )

    assert runtime["argv"][0] == "python3"
    assert "--progress-file-glob" in runtime["argv"]
    unit_index = runtime["argv"].index("--unit")
    assert runtime["argv"][unit_index + 1] == "step"
    assert task.progress_file_glob.endswith("/progress.csv")
    child = runtime["argv"][-1]
    assert "file_progress_cpu_benchmark.py" in child
    assert "--checkpoint-interval" in child
    assert "--checkpoint-bytes" in child

    matrix._validate_remote_group_spec(
        {
            "node": cell.node,
            "kind": cell.kind,
            "profile": cell.profile,
            "colocation_count": cell.colocation_count,
            "allocation_workers": 1,
            "group_dir": f"{matrix.REMOTE_GROUP_ROOT}/controlled-cpu-test",
            "tasks": [
                matrix._runtime_task_spec(
                    cell=cell,
                    task=item,
                    group_run_id="controlled_cpu_runtime_node001_cpu_p8",
                )
                for item in cell.tasks
            ],
        }
    )


def test_checkpoint_cost_writes_real_blocks(tmp_path):
    size = 2 * 1024 * 1024

    elapsed = _write_checkpoint(tmp_path, 1, size)

    path = tmp_path / "step_000001.bin"
    stat = path.stat()
    assert elapsed >= 0.0
    assert stat.st_size == size
    assert stat.st_blocks * 512 >= size


def test_required_dispatch_regime_is_bound_into_measurement_scope():
    cell = matrix.colocation_cells(
        run_id="targeted",
        nodes={"node001"},
        profiles=(1,),
        kinds={"freqduet"},
        tasks_per_node=1,
    )[0]
    cell_id = matrix._calibration_cell_id_for_cell(cell)
    result = matrix.build_native_cpu_colocation_completion_matrix(
        run_id="targeted",
        nodes={"node001"},
        profiles=(1,),
        kinds={"freqduet"},
        tasks_per_node=1,
        required_dispatch_regimes={
            cell_id: "cpu_external_idle",
        },
    )

    assert result["dispatch_regime_preflight_enabled"] is True
    assert result["required_dispatch_regimes"] == {
        cell_id: "cpu_external_idle"
    }
    assert result["measurement_scope"]["required_dispatch_regimes"] == {
        cell_id: "cpu_external_idle"
    }


def test_remote_group_defers_before_launch_on_regime_mismatch(
    monkeypatch,
    tmp_path,
    capsys,
):
    group_dir = tmp_path / "group"
    spec_path = tmp_path / "spec.json"
    monkeypatch.setattr(
        matrix,
        "REMOTE_GROUP_ROOT",
        str(tmp_path),
    )
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "group_id": "defer-test",
                "node": "node001",
                "kind": "freqduet",
                "profile": 1,
                "colocation_count": 1,
                "allocation_workers": 1,
                "group_dir": str(group_dir),
                "required_dispatch_regime": "cpu_external_idle",
                "tasks": [
                    {
                        "task_id": "t1",
                        "task_index": 0,
                        "seed": 1,
                        "output_dir": f"{matrix.FREQDUET_ROOT}/out",
                        "allocation_workers": 1,
                        "argv": [
                            "python3",
                            "-m",
                            matrix.WRAPPER_MODULE,
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload_state = {
        "dispatch_state_ready": True,
        "dispatch_state_observed_before_controlled_group": True,
        "dispatch_sample_count": matrix.DISPATCH_CPU_SAMPLE_COUNT,
        "dispatch_logical_cpu_count": 192,
        "dispatch_external_cpu_regime": "cpu_external_moderate",
        "dispatch_external_cpu_fraction": 0.2,
        "dispatch_external_cpu_fraction_p95": 0.3,
        "dispatch_external_cpu_core_equiv": 38.4,
        "dispatch_external_cpu_core_equiv_p95": 57.6,
    }
    monkeypatch.setattr(
        matrix,
        "_sample_dispatch_cpu_state",
        lambda: payload_state,
    )

    def forbidden_popen(*args, **kwargs):
        raise AssertionError("child process launched despite regime mismatch")

    monkeypatch.setattr(matrix.subprocess, "Popen", forbidden_popen)
    assert matrix._run_remote_group(spec_path) == 0

    output = capsys.readouterr().out
    payload = matrix._parse_prefixed_json(
        output,
        matrix.GROUP_RESULT_PREFIX,
    )
    assert payload["status"] == "DEFERRED_REGIME_MISMATCH"
    assert payload["controlled_tasks_launched"] is False
    assert payload["tasks"] == []
    assert payload["observed_dispatch_regime"] == "cpu_external_moderate"
    assert payload["external_load_telemetry"]["dispatch_only_preflight"] is True


def test_remote_dispatch_state_emits_protocol_bound_preflight(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        matrix,
        "_sample_dispatch_cpu_state",
        lambda: {
            "dispatch_state_ready": True,
            "dispatch_sample_count": matrix.DISPATCH_CPU_SAMPLE_COUNT,
            "dispatch_external_cpu_regime": "cpu_external_idle",
        },
    )

    assert matrix._run_remote_dispatch_state() == 0
    payload = matrix._parse_prefixed_json(
        capsys.readouterr().out,
        matrix.DISPATCH_STATE_PREFIX,
    )
    assert payload["measurement_protocol"] == matrix.MEASUREMENT_PROTOCOL
    assert payload["dispatch_sample_count"] == matrix.DISPATCH_CPU_SAMPLE_COUNT
    assert payload["dispatch_external_cpu_regime"] == "cpu_external_idle"


def test_kind_seed_is_stable_when_other_kind_is_filtered_out():
    both = matrix.colocation_cells(
        run_id="both",
        nodes={"node001"},
        profiles=(2,),
        kinds={"freqduet", "sumo"},
        tasks_per_node=2,
        seed_base=100,
    )
    sumo_only = matrix.colocation_cells(
        run_id="sumo",
        nodes={"node001"},
        profiles=(2,),
        kinds={"sumo"},
        tasks_per_node=2,
        seed_base=100,
    )

    both_sumo = next(cell for cell in both if cell.kind == "sumo")
    assert [task.seed for task in both_sumo.tasks] == [
        task.seed for task in sumo_only[0].tasks
    ]


def test_aggregate_reports_eta_makespan_flow_service_and_telemetry():
    telemetry = {
        "resource_telemetry_ready": True,
        "dispatch_state_ready": True,
        "dispatch_sample_count": matrix.DISPATCH_CPU_SAMPLE_COUNT,
        "dispatch_logical_cpu_count": 192,
        "dispatch_external_cpu_regime": "cpu_external_light",
        "dispatch_external_cpu_fraction": 1.5 / 192.0,
        "dispatch_external_cpu_fraction_p95": 2.0 / 192.0,
        "dispatch_external_cpu_core_equiv": 1.5,
        "dispatch_external_cpu_core_equiv_p95": 2.0,
        "dispatch_external_cpu_bucket": "external_c0_2",
        "run_window_external_cpu_core_equiv": 7.5,
        "run_window_external_cpu_bucket": "external_c3_16",
        "external_cpu_core_equiv": 7.5,
        "external_cpu_bucket": "external_c3_16",
    }
    aggregate = matrix.aggregate_task_results(
        [
            _ready_task(
                "task-1",
                flow_s=10.0,
                launch_offset_s=0.01,
                total_units=6,
                unit_s=1.0,
                eta_s=9.0,
            ),
            _ready_task(
                "task-2",
                flow_s=12.0,
                launch_offset_s=0.05,
                total_units=6,
                unit_s=2.0,
                eta_s=11.0,
            ),
        ],
        expected_count=2,
        external_load_telemetry=telemetry,
    )

    assert aggregate["completed_task_count"] == 2
    assert aggregate["failure_count"] == 0
    assert aggregate["all_tasks_ready"] is True
    assert aggregate["per_task_eta_s"] == {"task-1": 9.0, "task-2": 11.0}
    assert aggregate["makespan_s"] == pytest.approx(12.0)
    assert aggregate["mean_flow_s"] == pytest.approx(11.0)
    assert aggregate["aggregate_service_units"] == pytest.approx(12.0)
    assert aggregate["aggregate_service_unit_per_s"] == pytest.approx(1.0)
    assert aggregate["aggregate_model_service_unit_per_s"] == pytest.approx(1.5)
    assert aggregate["launch_skew_s"] == pytest.approx(0.04)
    assert aggregate["near_synchronous_start"] is True
    assert aggregate["external_cpu_core_equiv"] == pytest.approx(7.5)
    assert aggregate["external_cpu_bucket"] == "external_c3_16"
    assert aggregate["dispatch_state_ready"] is True
    assert aggregate["dispatch_sample_count"] == matrix.DISPATCH_CPU_SAMPLE_COUNT
    assert aggregate["dispatch_logical_cpu_count"] == 192
    assert aggregate["dispatch_external_cpu_regime"] == "cpu_external_light"
    assert aggregate["dispatch_external_cpu_core_equiv"] == pytest.approx(1.5)
    assert aggregate["dispatch_external_cpu_bucket"] == "external_c0_2"
    assert aggregate["run_window_external_cpu_core_equiv"] == pytest.approx(7.5)
    assert aggregate["run_window_external_cpu_bucket"] == "external_c3_16"


def test_failed_tasks_are_excluded_from_all_aggregate_metrics():
    failed = {
        "task_id": "failed",
        "returncode": 9,
        "measurement_valid": False,
        "launch_offset_s": 0.02,
        "flow_time_s": 999.0,
        "eta_s": 999.0,
        "completion_model": {
            "completion_model_ready": False,
            "unit": "episode",
            "total_units": 10_000,
            "completion_unit_s": 0.001,
        },
    }
    aggregate = matrix.aggregate_task_results(
        [
            _ready_task(
                "ready",
                flow_s=10.0,
                launch_offset_s=0.01,
                total_units=4,
                unit_s=2.0,
                eta_s=9.0,
            ),
            failed,
        ],
        expected_count=2,
    )

    assert aggregate["completed_task_count"] == 1
    assert aggregate["failure_count"] == 1
    assert aggregate["all_tasks_ready"] is False
    assert aggregate["failed_task_ids"] == ["failed"]
    assert aggregate["per_task_eta_s"] == {"ready": 9.0}
    assert aggregate["makespan_s"] == pytest.approx(10.0)
    assert aggregate["mean_flow_s"] == pytest.approx(10.0)
    assert aggregate["aggregate_service_units"] == pytest.approx(4.0)
    assert aggregate["aggregate_service_unit_per_s"] == pytest.approx(0.4)
    assert aggregate["aggregate_model_service_unit_per_s"] == pytest.approx(0.5)
    assert aggregate["failed_tasks_excluded_from_aggregates"] is True


def test_return_139_fails_closed_even_if_upstream_marks_row_valid():
    segfault = _ready_task(
        "segfault",
        flow_s=10.0,
        launch_offset_s=0.01,
        total_units=4,
        unit_s=2.0,
        eta_s=9.0,
    )
    segfault["returncode"] = 139
    segfault["measurement_valid"] = True

    aggregate = matrix.aggregate_task_results(
        [segfault],
        expected_count=1,
    )

    assert aggregate["completed_task_count"] == 0
    assert aggregate["failure_count"] == 1
    assert aggregate["all_tasks_ready"] is False
    assert aggregate["failed_task_ids"] == ["segfault"]
    assert aggregate["per_task_eta_s"] == {}


def test_safe_default_never_calls_launch(monkeypatch, tmp_path):
    def forbidden_launch(*args, **kwargs):
        raise AssertionError("launch path was called without --allow-launch")

    monkeypatch.setattr(matrix, "_launch_cells", forbidden_launch)
    output = tmp_path / "manifest.json"
    returncode = matrix.main(
        [
            "--run-id",
            "safe_default",
            "--nodes",
            "node001",
            "--profiles",
            "p1",
            "--kinds",
            "freqduet",
            "--tasks-per-node",
            "1",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert returncode == 0
    assert payload["allow_launch"] is False
    assert payload["status"] == "MANIFEST"
    assert payload["launched_count"] == 0


def test_tasks_per_node_is_a_hard_colocation_safety_limit():
    with pytest.raises(ValueError, match="p4"):
        matrix.build_native_cpu_colocation_completion_matrix(
            nodes={"node001"},
            profiles=(4,),
            kinds={"sumo"},
            tasks_per_node=2,
        )


def test_remote_task_requires_five_outer_intervals(tmp_path):
    log_path = tmp_path / "task.log"
    log_path.write_text(
        matrix.COMPLETION_MODEL_PREFIX
        + json.dumps(
            {
                "completion_model_ready": True,
                "progress_source": "durable_csv",
                "interval_sample_count": 3,
                "total_units": 4,
                "completion_unit_s": 1.0,
                "total_wall_s": 4.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    start_path = tmp_path / "start.json"
    start_path.write_text(
        json.dumps({"started_monotonic_s": 1.0, "started_wall_s": 2.0}),
        encoding="utf-8",
    )
    row = matrix._remote_task_result(
        {
            "task": {
                "task_id": "t1",
                "task_index": 0,
                "seed": 1,
                "output_dir": "/tmp/out",
                "progress_file_glob": "/tmp/out/progress.csv",
            },
            "process": SimpleNamespace(returncode=0),
            "log_path": log_path,
            "start_path": start_path,
            "finished_monotonic_s": 5.0,
        },
        release_monotonic_s=0.0,
    )

    assert row["completion_model_ready"] is True
    assert row["interval_sample_count"] == 3
    assert row["interval_gate_ready"] is False
    assert row["measurement_valid"] is False
