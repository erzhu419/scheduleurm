"""Additional corner-case regression tests for scheduler.py.

Keep these small and side-effect free. The legacy runner imports this module and
passes in its check() function plus the already-loaded scheduler module.
"""

from __future__ import annotations

import os
import tempfile
import time


def _raises_system_exit(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


def _base_task(**overrides):
    task = {
        "signature": "EXT/base",
        "cmd": "python train.py --seed 1 --n_steps 100",
        "cwd": "/tmp/ext",
        "extra_env": {},
        "env_spec": "none",
        "image": "",
        "ckpt_dir": None,
        "ckpt_glob": "*",
        "resume_flag": "",
        "result_dir": None,
        "local_result_dir": None,
        "slurm_partition": "",
        "slurm_account": "",
        "slurm_qos": "",
        "priority": "normal",
        "ram_mb": 1000,
        "est_vram_mb": 1000,
        "cpu_cores": 1,
    }
    task.update(overrides)
    return task


def run(check, sch):
    print("\n[external] scheduler corner cases")
    def _jtl110cpu_cfg(port):
        return {
            "host": "tf290q6n.zjz-service.cn",
            "ssh_user": "erzhu419",
            "ssh_port": port,
            "os": "windows",
            "cpu_cores": 128,
            "ram_mb": 512 * 1024,
            "ram_headroom_frac": 0.10,
            "max_vram_per_task": 0,
            "windows_python": r"F:\python\python.exe",
            "windows_workspace_root": r"F:\\",
            "windows_scheduleurm_dir": r"F:\.scheduleurm",
            "windows_auto_pin": True,
            "windows_skip_ht_pair": True,
            "cpu_labor_node": True,
        }
    local_cfg = {
        "host": None,
        "cpu_cores": 16,
        "ram_mb": 56000,
        "max_vram_per_task": None,
        "ram_headroom_frac": 0.25,
    }

    def with_temp_nodes(fn):
        saved = dict(sch.NODES)
        try:
            sch.NODES["local"] = dict(local_cfg)
            sch.NODES["jtl110cpu"] = _jtl110cpu_cfg(22945)
            sch.NODES["jtl110cpu2"] = _jtl110cpu_cfg(23565)
            return fn()
        finally:
            sch.NODES.clear()
            sch.NODES.update(saved)

    def case_env_value_with_equals():
        return sch._parse_env(["A=x=y=z"]) == {"A": "x=y=z"}

    def case_env_invalid_key_rejected():
        return _raises_system_exit(lambda: sch._parse_env(["1BAD=x"]))

    def case_env_reserved_cuda_rejected():
        return _raises_system_exit(lambda: sch._parse_env(["CUDA_VISIBLE_DEVICES=0"]))

    def case_cmd_flag_equals_value():
        got = sch._cmd_flag_values(["--out=foo.txt"], {"--out"})
        return got == {"--out": ["foo.txt"]}

    def case_cmd_flag_space_value():
        got = sch._cmd_flag_values(["--out", "foo.txt"], {"--out"})
        return got == {"--out": ["foo.txt"]}

    def case_cmd_flag_missing_value_ignored():
        got = sch._cmd_flag_values(["--out"], {"--out"})
        return got == {}

    def case_arg_value_equals_form():
        return sch._arg_value(["--seed=42"], "--seed") == "42"

    def case_arg_value_next_token_form():
        return sch._arg_value(["--seed", "42"], "--seed") == "42"

    def case_cpu_flag_detects_empty_cuda_visible_devices():
        return sch._cmd_explicitly_cpu('export CUDA_VISIBLE_DEVICES="" && python train.py')

    def case_cpu_flag_does_not_treat_cuda_zero_as_cpu():
        return not sch._cmd_explicitly_cpu("CUDA_VISIBLE_DEVICES=0 python train.py")

    def case_wait_for_file_string_ready():
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ready.ckpt")
            with open(p, "w") as f:
                f.write("x")
            return sch._queued_wait_for_file_block_reason({
                "status": "queued", "wait_for_files": p,
            }) is None

    def case_wait_for_file_directory_blocks():
        with tempfile.TemporaryDirectory() as td:
            return sch._queued_wait_for_file_block_reason({
                "status": "queued", "wait_for_files": [td],
            }) is not None

    def case_wait_for_file_empty_blocks():
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "empty.ckpt")
            open(p, "w").close()
            return sch._queued_wait_for_file_block_reason({
                "status": "queued", "wait_for_files": [p],
            }) is not None

    def case_simple_sac_snapshot_false_not_local_pinned():
        cmd = "python h2o+_bus_main.py --use-snapshot-reset=false --seed 1"
        return sch._simple_sac_large_data_reason(cmd, "/tmp/SimpleSAC") is None

    def case_simple_sac_dataset_dir_is_local_pinned():
        cmd = "python h2o+_bus_main.py --dataset-dir /data/bus_h2o --seed 1"
        reason = sch._simple_sac_large_data_reason(cmd, "/tmp/SimpleSAC")
        return bool(reason and "outside cwd" in reason)

    def case_eval_relative_checkpoint_resolves_against_cwd():
        with tempfile.TemporaryDirectory() as td:
            got = sch._simple_sac_eval_prereq_files(
                "python eval.py --ckpt-path ckpts/model.pt", td)
            return got == [os.path.join(td, "ckpts", "model.pt")]

    def case_train_checkpoint_flag_is_not_eval_prereq():
        with tempfile.TemporaryDirectory() as td:
            got = sch._simple_sac_eval_prereq_files(
                "python train.py --checkpoint ckpts/model.pt", td)
            return got == []

    def case_simple_sac_train_ckpt_hyphen_current_time():
        with tempfile.TemporaryDirectory() as td:
            cwd = os.path.join(td, "SimpleSAC")
            os.makedirs(cwd)
            got = sch._simple_sac_train_best_ckpt_from_cmd(
                "python h2o+_bus_main.py --seed 7 --current-time r1", cwd)
            want = os.path.join(td, "experiment_output",
                                "h2oplus_bus_seed7_r1", "checkpoint_best.pt")
            return got == want

    def case_run_identity_ignores_scheduling_knobs():
        a = _base_task(priority="low", ram_mb=1000, est_vram_mb=1000)
        b = _base_task(priority="high", ram_mb=9000, est_vram_mb=4000)
        return sch._task_run_identity(a) == sch._task_run_identity(b)

    def case_run_identity_changes_with_extra_env():
        a = _base_task(extra_env={"A": "1"})
        b = _base_task(extra_env={"A": "2"})
        return sch._task_run_identity(a) != sch._task_run_identity(b)

    def case_runtime_tokens_normalize_seed_values():
        a = sch._runtime_cmd_tokens("python train.py --seed 1 --n_steps 100")
        b = sch._runtime_cmd_tokens("python train.py --seed 99 --n_steps 100")
        return a == b

    def case_runtime_closest_rejects_unrelated_history():
        task = _base_task(project="P", cmd="python train.py --n_steps 100")
        history = {
            "exact:other": {
                "total_s": 100,
                "cmd": "python eval.py --episodes 5",
                "project": "Q",
                "cwd": "/tmp/other",
            }
        }
        rec, key = sch._runtime_history_closest(task, history)
        return rec is None and key is None

    def case_pending_eta_does_not_overwrite_existing_eta():
        task = _base_task(status="queued", eta_seconds=999, runtime_total_s_est=123)
        changed = sch._seed_pending_eta_from_history({"tasks": [task]})
        return changed == 0 and task["eta_seconds"] == 999

    def case_pending_eta_uses_legacy_duration_when_no_runtime_history():
        saved_history_get = sch.history_get
        saved_load_runtime_history = sch.load_runtime_history
        sig = "EXT/duration/%d" % time.time_ns()
        try:
            sch.history_get = lambda s: {"dur_s_ewma": 321} if s == sig else None
            sch.load_runtime_history = lambda: {}
            task = _base_task(status="queued", signature=sig)
            changed = sch._seed_pending_eta_from_history({"tasks": [task]})
            return changed == 1 and task.get("eta_seconds") == 321 and task.get("eta_source") == "duration_ewma"
        finally:
            sch.history_get = saved_history_get
            sch.load_runtime_history = saved_load_runtime_history

    def case_gpu_fits_rejects_per_task_cap():
        task = {"est_vram_mb": 1500}
        gpu = {"total_mb": 12000, "used_mb": 0, "free_mb": 12000, "util_pct": 0}
        return not sch._gpu_fits(task, gpu, {"max_vram_per_task": 1000})

    def case_gpu_fits_rejects_small_idle_past_one_third():
        task = {"est_vram_mb": 500}
        gpu = {"total_mb": 12000, "used_mb": 5000, "free_mb": 7000, "util_pct": 1}
        return not sch._gpu_fits(task, gpu, {"max_vram_per_task": None})

    def case_gpu_fits_rejects_saturated_past_one_third():
        task = {"est_vram_mb": 500}
        gpu = {"total_mb": 12000, "used_mb": 5000, "free_mb": 7000, "util_pct": 100}
        return not sch._gpu_fits(task, gpu, {"max_vram_per_task": None})

    def case_gpu_fits_allows_first_large_task_on_empty_gpu():
        task = {"est_vram_mb": 5000}
        gpu = {"total_mb": 12000, "used_mb": 0, "free_mb": 12000, "util_pct": 0}
        return sch._gpu_fits(task, gpu, {"max_vram_per_task": None})

    def case_gpu_fits_allows_slight_one_third_crossing():
        task = {"est_vram_mb": 200}
        gpu = {"total_mb": 12000, "used_mb": 3990, "free_mb": 8010, "util_pct": 1}
        return sch._gpu_fits(task, gpu, {"max_vram_per_task": None, "gpu_util_saturation_pct": None})

    def case_gpu_fits_rejects_crossing_freeze_line_from_below():
        task = {"est_vram_mb": 200}
        gpu = {"total_mb": 12000, "used_mb": 4550, "free_mb": 7450, "util_pct": 1}
        return not sch._gpu_fits(task, gpu, {"max_vram_per_task": None, "gpu_util_saturation_pct": None})

    def case_gpu_fits_rejects_vram_margin_shortfall():
        task = {"est_vram_mb": 700}
        gpu = {"total_mb": 12000, "used_mb": 0, "free_mb": 1000, "util_pct": 0}
        return not sch._gpu_fits(task, gpu, {"max_vram_per_task": None})

    def case_slurm_bucket_missing_est_defaults_gpu():
        return sch._slurm_pending_bucket_for_task({}) == "gpu"

    def case_format_task_location_slurm_cpu_and_gpu():
        cpu = sch._format_task_location({
            "node": "n1", "slurm_job_id": 12, "slurm_state": "PENDING",
            "est_vram_mb": 0, "gpu_idx": None,
        })
        gpu = sch._format_task_location({
            "node": "n1", "slurm_job_id": 13, "slurm_state": "RUNNING",
            "est_vram_mb": 1000, "gpu_idx": None,
        })
        return cpu == "n1:SLURM-CPU#12:PENDING" and gpu == "n1:SLURM-GPU#13:RUNNING"

    def case_jtl110cpu_is_windows_cpu_node():
        def _inner():
            info = sch.NODES.get("jtl110cpu") or {}
            return (
                sch._node_is_windows("jtl110cpu")
                and info.get("cpu_labor_node") is True
                and int(info.get("cpu_cores") or 0) == 128
                and int(info.get("max_vram_per_task") or 0) == 0
            )
        return with_temp_nodes(_inner)

    def case_jtl110cpu2_is_windows_cpu_node():
        def _inner():
            info = sch.NODES.get("jtl110cpu2") or {}
            return (
                sch._node_is_windows("jtl110cpu2")
                and info.get("cpu_labor_node") is True
                and int(info.get("ssh_port") or 0) == 23565
                and int(info.get("cpu_cores") or 0) == 128
                and int(info.get("max_vram_per_task") or 0) == 0
            )
        return with_temp_nodes(_inner)

    def case_real_scheduler_source_defines_windows_cpu_nodes():
        path = getattr(sch, "__file__", "") or ""
        src = open(path, encoding="utf-8").read()
        return (
            '"jtl110cpu"' in src
            and '"jtl110cpu2"' in src
            and '"ssh_port": 23565' in src
            and '"os": "windows"' in src
            and '"cpu_labor_node": True' in src
        )

    def case_cpu_only_prefers_jtl110cpu():
        def _inner():
            task = _base_task(est_vram_mb=0, cpu_cores=8, ram_mb=2000)
            nodes = [
                {"name": "local", "alive": True, "gpus": [],
                 "free_cpu": 16, "total_cpu": 16, "free_ram_mb": 56000,
                 "total_ram_mb": 56000, "running_count": 0},
                {"name": "jtl110cpu", "alive": True, "gpus": [],
                 "free_cpu": 128, "total_cpu": 128, "free_ram_mb": 500000,
                 "total_ram_mb": 524288, "running_count": 0},
            ]
            return sch.pick_placement(task, nodes) == ("jtl110cpu", None)
        return with_temp_nodes(_inner)

    def case_cpu_only_prefers_less_loaded_windows_cpu_node():
        def _inner():
            task = _base_task(est_vram_mb=0, cpu_cores=8, ram_mb=2000)
            nodes = [
                {"name": "jtl110cpu", "alive": True, "gpus": [],
                 "free_cpu": 16, "total_cpu": 128, "free_ram_mb": 400000,
                 "total_ram_mb": 524288, "running_count": 0},
                {"name": "jtl110cpu2", "alive": True, "gpus": [],
                 "free_cpu": 120, "total_cpu": 128, "free_ram_mb": 500000,
                 "total_ram_mb": 524288, "running_count": 0},
            ]
            return sch.pick_placement(task, nodes) == ("jtl110cpu2", None)
        return with_temp_nodes(_inner)

    def case_gpu_task_never_placed_on_jtl110cpu():
        def _inner():
            task = _base_task(est_vram_mb=500, cpu_cores=1, ram_mb=1000)
            nodes = [
                {"name": "jtl110cpu", "alive": True, "gpus": [],
                 "free_cpu": 128, "total_cpu": 128, "free_ram_mb": 500000,
                 "total_ram_mb": 524288, "running_count": 0},
                {"name": "jtl110cpu2", "alive": True, "gpus": [],
                 "free_cpu": 128, "total_cpu": 128, "free_ram_mb": 500000,
                 "total_ram_mb": 524288, "running_count": 0},
            ]
            return sch.pick_placement(task, nodes) is None
        return with_temp_nodes(_inner)

    def case_cpu_worker_plan_901_on_128_physical():
        p = sch._cpu_worker_plan_for_items(901, 128)
        return (
            p["waves"] == 8
            and p["workers"] == 113
            and p["last_wave_items"] == 110
        )

    def case_cpu_batch_plan_splits_two_128_nodes():
        def _inner():
            plan = sch._cpu_batch_plan(901, ["jtl110cpu", "jtl110cpu2"])
            return (
                len(plan) == 2
                and sum(p["items"] for p in plan) == 901
                and all(p["physical_cores"] == 128 for p in plan)
                and all(p["workers"] == 113 for p in plan)
                and max(p["waves"] for p in plan) == 4
                and {p["node"] for p in plan} == {"jtl110cpu", "jtl110cpu2"}
            )
        return with_temp_nodes(_inner)

    def case_cpu_batch_plan_uses_free_physical_cores():
        def _inner():
            states = {
                "jtl110cpu": {"name": "jtl110cpu", "alive": True, "free_cpu": 64,
                              "total_cpu": 128, "logical_cpu": 256},
                "jtl110cpu2": {"name": "jtl110cpu2", "alive": True, "free_cpu": 128,
                               "total_cpu": 128, "logical_cpu": 256},
            }
            plan = sch._cpu_batch_plan(901, ["jtl110cpu", "jtl110cpu2"], states)
            by_node = {p["node"]: p for p in plan}
            return (
                len(plan) == 2
                and by_node["jtl110cpu"]["items"] == 300
                and by_node["jtl110cpu2"]["items"] == 601
                and by_node["jtl110cpu"]["workers"] == 60
                and by_node["jtl110cpu2"]["workers"] == 121
                and by_node["jtl110cpu"]["physical_cores"] == 64
                and by_node["jtl110cpu"]["total_physical_cores"] == 128
                and max(p["waves"] for p in plan) == 5
            )
        return with_temp_nodes(_inner)

    def case_cpu_batch_plan_skips_full_cpu_node():
        def _inner():
            states = {
                "jtl110cpu": {"name": "jtl110cpu", "alive": True, "free_cpu": 0,
                              "total_cpu": 128, "logical_cpu": 256},
                "jtl110cpu2": {"name": "jtl110cpu2", "alive": True, "free_cpu": 128,
                               "total_cpu": 128, "logical_cpu": 256},
            }
            plan = sch._cpu_batch_plan(901, ["jtl110cpu", "jtl110cpu2"], states)
            return (
                len(plan) == 1
                and plan[0]["node"] == "jtl110cpu2"
                and plan[0]["items"] == 901
                and plan[0]["workers"] == 113
                and plan[0]["waves"] == 8
            )
        return with_temp_nodes(_inner)

    def case_cpu_wave_summary_records_each_small_wave():
        got = sch._cpu_wave_summary(901, 113)
        return (
            got["waves"] == 8
            and got["wave_items"] == [113, 113, 113, 113, 113, 113, 113, 110]
            and got["last_wave_items"] == 110
        )

    def case_cpu_batch_log_payload_has_detailed_node_rows():
        def _inner():
            states = {
                "jtl110cpu": {"name": "jtl110cpu", "alive": True, "free_cpu": 64,
                              "total_cpu": 128, "logical_cpu": 256,
                              "free_ram_mb": 400000, "total_ram_mb": 524288},
                "jtl110cpu2": {"name": "jtl110cpu2", "alive": True, "free_cpu": 128,
                               "total_cpu": 128, "logical_cpu": 256,
                               "free_ram_mb": 500000, "total_ram_mb": 524288},
            }
            plan = sch._cpu_batch_plan(901, ["jtl110cpu", "jtl110cpu2"], states)
            payload = sch._cpu_batch_log_payload(901, plan, states, templates={"cmd": "python eval.py"})
            nodes = {n["node"]: n for n in payload["nodes"]}
            return (
                payload["total_items"] == 901
                and payload["node_count"] == 2
                and nodes["jtl110cpu"]["assigned_items"] == 300
                and nodes["jtl110cpu"]["free_physical_cores_used_for_plan"] == 64
                and nodes["jtl110cpu"]["live_node"]["logical_cpu"] == 256
                and nodes["jtl110cpu2"]["wave_plan"]["waves"] == nodes["jtl110cpu2"]["waves"]
            )
        return with_temp_nodes(_inner)

    def case_cpu_ownership_snapshot_separates_ours_external_and_other():
        saved = sch._ClaimManager.__dict__["scheduler_id"]
        try:
            sch._ClaimManager.scheduler_id = staticmethod(lambda: "this-scheduler")
            state = {"tasks": [
                {"id": "tOurs", "status": "running", "node": "nodeA",
                 "origin": "scheduleurm", "scheduler_id": "this-scheduler", "cpu_cores": 4},
                {"id": "tOther", "status": "running", "node": "nodeA",
                 "origin": "scheduleurm", "scheduler_id": "other-scheduler", "cpu_cores": 3},
                {"id": "tQueued", "status": "queued", "node": "nodeA",
                 "origin": "scheduleurm", "scheduler_id": "this-scheduler", "cpu_cores": 8},
            ]}
            nodes = [{"name": "nodeA", "alive": True, "total_cpu": 16,
                      "free_cpu": 6, "free_ram_mb": 1000, "total_ram_mb": 2000}]
            row = sch._cpu_ownership_snapshot(state, nodes)[0]
            return (
                row["used_cpu_est"] == 10
                and row["scheduleurm_cpu_reserved"] == 4
                and row["external_tracked_cpu_reserved"] == 3
                and row["untracked_or_other_user_cpu_est"] == 3
                and row["scheduleurm_task_ids"] == ["tOurs"]
                and row["external_tracked_task_ids"] == ["tOther"]
            )
        finally:
            sch._ClaimManager.scheduler_id = saved

    def case_dispatch_cycle_payload_includes_cpu_accounting():
        saved = sch._ClaimManager.__dict__["scheduler_id"]
        try:
            sch._ClaimManager.scheduler_id = staticmethod(lambda: "sid")
            state = {"tasks": [
                {"id": "t1", "status": "running", "node": "nodeA",
                 "origin": "scheduleurm", "scheduler_id": "sid", "cpu_cores": 2},
                {"id": "t2", "status": "queued", "cpu_cores": 1, "ram_mb": 500,
                 "est_vram_mb": 0, "last_block_reason": "waiting"},
            ]}
            nodes = [{"name": "nodeA", "alive": True, "total_cpu": 8, "free_cpu": 5}]
            events = [{"type": "no_fit", "task_id": "t2", "task": state["tasks"][1]}]
            payload = sch._dispatch_cycle_log_payload(state, nodes, events, 1)
            return (
                payload["event_counts"]["no_fit"] == 1
                and payload["nodes"][0]["scheduleurm_cpu_reserved"] == 2
                and payload["queued_seen_by_dispatch"] == 1
                and payload["no_fit"][0]["reason"] == "waiting"
            )
        finally:
            sch._ClaimManager.scheduler_id = saved

    def case_cpu_parallel_template_and_auto_worker_flag():
        plan = {
            "node": "jtl110cpu", "start": 0, "end": 451,
            "items": 451, "workers": 113, "waves": 4,
            "physical_cores": 128, "last_wave_items": 112,
            "shard_index": 0, "num_shards": 2,
        }
        cmd = "python eval.py --start {start} --end {end} --workers auto --tag {node}"
        got = sch._rewrite_cpu_parallel_cmd(cmd, plan, total_items=901)
        return (
            "--start 0" in got
            and "--end 451" in got
            and "--workers 113" in got
            and "--tag jtl110cpu" in got
        )

    def case_node_physical_cores_infers_half_logical_when_unconfigured():
        saved = dict(sch.NODES)
        try:
            sch.NODES.clear()
            sch.NODES["win"] = {"os": "windows", "windows_skip_ht_pair": True}
            return sch._node_physical_cores("win", {"logical_cpu": 256}) == 128
        finally:
            sch.NODES.clear()
            sch.NODES.update(saved)

    def case_submit_cpu_batch_cli_exists():
        src = open(getattr(sch, "__file__", ""), encoding="utf-8").read()
        return (
            "submit-cpu-batch" in src
            and "cpu-plan" in src
            and "cmd_submit_cpu_batch" in src
            and "SCHEDULEURM_CPU_WORKERS" in src
        )

    def case_cpu_parallel_env_keeps_zero_start_index():
        env = sch._cpu_parallel_env({
            "cpu_parallel_total_items": 901,
            "cpu_parallel_items": 451,
            "cpu_parallel_start": 0,
            "cpu_parallel_end": 451,
            "cpu_auto_workers": 113,
            "cpu_parallel_waves": 4,
            "cpu_parallel_physical_cores": 128,
            "cpu_parallel_shard_index": 0,
            "cpu_parallel_num_shards": 2,
        })
        return (
            env.get("SCHEDULEURM_CPU_SHARD_START") == "0"
            and env.get("SCHEDULEURM_CPU_SHARD_INDEX") == "0"
            and env.get("SCHEDULEURM_CPU_WORKERS") == "113"
        )

    def case_gpu_servers_use_auto_ram_detection():
        try:
            src = open(sch.__file__, encoding="utf-8").read()
        except Exception:
            return False
        line1 = next((ln for ln in src.splitlines() if '"jtl110gpu":' in ln), "")
        line2 = next((ln for ln in src.splitlines() if '"jtl110gpu2":' in ln), "")
        return '"ram_mb": 0' in line1 and '"ram_mb": 0' in line2

    def case_claim_capacity_uses_probed_ram_when_auto():
        saved = dict(sch.NODES)
        try:
            sch.NODES["auto-ram"] = {"host": "rbox", "cpu_cores": 12, "ram_mb": 0}
            cap = sch._ClaimManager._build_capacity(
                "auto-ram", {"name": "auto-ram", "total_ram_mb": 515000, "gpus": []}
            )
            return cap.get("ram_mb") == 515000
        finally:
            sch.NODES.clear()
            sch.NODES.update(saved)

    def case_hybrid_routes_jtl110cpu_to_windows_backend():
        def _inner():
            b = sch.HybridBackend()
            backend1 = b._backend_for("jtl110cpu", {"est_vram_mb": 0})
            backend2 = b._backend_for("jtl110cpu2", {"est_vram_mb": 0})
            return isinstance(backend1, sch.WindowsBackend) and isinstance(backend2, sch.WindowsBackend)
        return with_temp_nodes(_inner)

    def case_windows_path_mapping_to_f_drive_project_layout():
        def _inner():
            got = sch._windows_path_for_node(
                "jtl110cpu",
                "/home/erzhu419/mine_code/offline-sumo/sub/ckpt.pt",
            )
            return got == r"F:\offline-sumo\sub\ckpt.pt"
        return with_temp_nodes(_inner)

    def case_windows_prepare_command_rewrites_python_and_paths():
        def _inner():
            task = _base_task(
                node="jtl110cpu",
                cmd=("/home/erzhu419/anaconda3/bin/python "
                     "/home/erzhu419/mine_code/proj/eval.py "
                     "--ckpt /home/erzhu419/mine_code/proj/ckpts/a.pt"),
            )
            payload = sch._windows_prepare_command(task)
            argv = payload.get("argv") or []
            return (
                len(argv) >= 5
                and argv[0] == sch.NODES["jtl110cpu"]["windows_python"]
                and "-u" in argv
                and r"F:\proj\eval.py" in argv
                and r"F:\proj\ckpts\a.pt" in argv
            )
        return with_temp_nodes(_inner)

    def case_windows_backend_refuses_gpu_task_without_network():
        backend = sch.WindowsBackend()
        ok1, msg1 = backend.launch({"id": "tX", "node": "jtl110cpu", "est_vram_mb": 1})
        ok2, msg2 = backend.launch({"id": "tY", "node": "jtl110cpu2", "est_vram_mb": 1})
        return (not ok1) and (not ok2) and "CPU-only" in msg1 and "CPU-only" in msg2

    def case_windows_cwd_requires_stage_cache_before_launch():
        def _inner():
            sch._STAGING_CACHE.clear()
            sch._STAGING_CAP_EXCEEDED.clear()
            sch._STAGING_FAILS.clear()
            cwd = "/home/erzhu419/mine_code/proj"
            key = ("local", "jtl110cpu", cwd)
            cold = sch._stage_cwd_check("jtl110cpu", cwd)
            sch._STAGING_CACHE[key] = time.time()
            hot = sch._stage_cwd_check("jtl110cpu", cwd)
            sch._STAGING_CACHE.pop(key, None)
            return cold == "needs_stage" and hot == "ready"
        return with_temp_nodes(_inner)

    def case_windows_stage_helper_uses_tar_over_ssh():
        src = open(getattr(sch, "__file__", ""), encoding="utf-8").read()
        return (
            "def _stage_local_dir_to_windows" in src
            and '"tar", "-C"' in src
            and "_ssh_base_args(target_node)" in src
            and "tar -xf - -C $dest" in src
        )

    def case_windows_explicit_env_spec_rejected_without_network():
        backend = sch.WindowsBackend()
        ok, msg = backend.launch({
            "id": "tEnv", "node": "jtl110cpu", "est_vram_mb": 0,
            "env_spec": "docker:some/image:tag",
        })
        return (not ok) and "does not support explicit env_spec" in msg

    def case_record_staged_resume_location_maps_windows_path():
        def _inner():
            task = {"resume_locations": []}
            sch._record_staged_resume_location(task, "jtl110cpu", {
                "node": "local",
                "path": "/home/erzhu419/mine_code/proj/ckpts/a.pt",
                "mtime": 1,
                "size": 2,
            })
            loc = task.get("resume_locations", [{}])[0]
            return (
                loc.get("node") == "jtl110cpu"
                and loc.get("path") == r"F:\proj\ckpts\a.pt"
                and task.get("resume_from") == r"F:\proj\ckpts\a.pt"
            )
        return with_temp_nodes(_inner)

    def case_ckpt_staging_runs_when_cwd_already_cached():
        saved_nodes = dict(sch.NODES)
        saved_load = sch.load_state
        saved_stage = sch._stage_resume_ckpt_for_launch
        try:
            sch.NODES.clear()
            sch.NODES.update({
                "local": {"host": None},
                "remote": {"host": "rbox"},
            })
            cwd = "/home/erzhu419/mine_code/proj"
            ckpt = cwd + "/ckpts"
            task = {
                "id": "tCkptOnly",
                "status": "queued",
                "cwd": cwd,
                "ckpt_dir": ckpt,
                "ckpt_glob": "*",
                "require_node": "remote",
                "resume_managed_by_cmd": True,
                "resume_locations": [{
                    "node": "local",
                    "path": ckpt + "/checkpoint.pt",
                    "mtime": time.time(),
                    "size": 10,
                }],
            }
            sch.load_state = lambda: {"tasks": [task]}
            sch._STAGING_CACHE.clear()
            sch._STAGING_CACHE[("local", "remote", cwd)] = time.time()
            calls = []
            def fake_stage(t, node, src):
                calls.append((t.get("id"), node, src.get("node")))
                return True, "ok"
            sch._stage_resume_ckpt_for_launch = fake_stage
            sch._stage_launch_candidates_outside_lock()
            return calls == [("tCkptOnly", "remote", "local")]
        finally:
            sch.NODES.clear()
            sch.NODES.update(saved_nodes)
            sch.load_state = saved_load
            sch._stage_resume_ckpt_for_launch = saved_stage
            sch._STAGING_CACHE.clear()

    def case_windows_auto_adopt_linux_scans_are_skipped():
        def _inner():
            return (
                sch._node_processes("jtl110cpu") == []
                and sch._node_cpu_processes("jtl110cpu") == []
                and sch._node_ppid_map("jtl110cpu") == {}
                and sch._node_processes("jtl110cpu2") == []
                and sch._node_cpu_processes("jtl110cpu2") == []
                and sch._node_ppid_map("jtl110cpu2") == {}
            )
        return with_temp_nodes(_inner)

    def case_windows_tail_uses_readwrite_share():
        ps = sch._windows_tail_ps(r"F:\.scheduleurm\logs\t0001.log")
        return "FileShare]::ReadWrite" in ps and "Get-Content" not in ps

    def case_windows_probe_ignores_powershell_clixml_noise():
        def _inner():
            saved = sch._run_windows_ps
            try:
                sch._run_windows_ps = lambda *a, **k: (
                    0,
                    '#< CLIXML\n<Objs>progress noise</Objs>\n500000|524288|256|25\n',
                    '',
                )
                got = sch._probe_windows_node("jtl110cpu2")
                return (
                    got.get("alive") is True
                    and got.get("os") == "windows"
                    and got.get("gpus") == []
                    and got.get("total_cpu") == 128
                    and got.get("free_cpu") == 96
                    and got.get("free_ram_mb") == 500000
                )
            finally:
                sch._run_windows_ps = saved
        return with_temp_nodes(_inner)

    def case_windows_probe_has_process_cpu_delta_fallback():
        import inspect
        src = inspect.getsource(sch._probe_windows_node)
        return (
            "GetProcessesByName" in src
            and "TotalProcessorTime.TotalSeconds" in src
            and "Start-Sleep -Milliseconds 700" in src
        )

    def case_windows_probe_error_hints_ssh_key_auth():
        def _inner():
            msg = sch._windows_probe_error_hint(
                "jtl110cpu",
                "Permission denied (publickey,password,keyboard-interactive).",
            )
            return (
                "SSH key auth failed" in msg
                and "id_ed25519.pub" in msg
                and "Do not store the password" in msg
            )
        return with_temp_nodes(_inner)

    def case_windows_wrapper_logs_cpu_plan_and_resource_progress():
        src = open(getattr(sch, "__file__", ""), encoding="utf-8").read()
        return (
            "scheduleurm cpu-plan" in src
            and "scheduleurm resource-progress" in src
            and "resource_log_interval_s" in src
            and "WINDOWS_WRAPPER_RESOURCE_LOG_INTERVAL_S" in src
        )

    def case_watcher_has_periodic_node_cpu_accounting_log():
        src = open(getattr(sch, "__file__", ""), encoding="utf-8").read()
        return (
            "--resource-log-interval" in src
            and "node_cpu_accounting" in src
            and "_cpu_ownership_snapshot" in src
            and "dispatch_cycle" in src
        )

    cases = [
        ("env value preserves equals", case_env_value_with_equals),
        ("env invalid key rejected", case_env_invalid_key_rejected),
        ("env reserved CUDA rejected", case_env_reserved_cuda_rejected),
        ("cmd flag equals value", case_cmd_flag_equals_value),
        ("cmd flag space value", case_cmd_flag_space_value),
        ("cmd flag missing value ignored", case_cmd_flag_missing_value_ignored),
        ("arg value equals form", case_arg_value_equals_form),
        ("arg value next-token form", case_arg_value_next_token_form),
        ("explicit CPU detects empty CUDA_VISIBLE_DEVICES", case_cpu_flag_detects_empty_cuda_visible_devices),
        ("CUDA_VISIBLE_DEVICES=0 is not CPU", case_cpu_flag_does_not_treat_cuda_zero_as_cpu),
        ("wait-for-file string ready", case_wait_for_file_string_ready),
        ("wait-for-file directory blocks", case_wait_for_file_directory_blocks),
        ("wait-for-file empty file blocks", case_wait_for_file_empty_blocks),
        ("SimpleSAC snapshot false not pinned", case_simple_sac_snapshot_false_not_local_pinned),
        ("SimpleSAC dataset dir pinned", case_simple_sac_dataset_dir_is_local_pinned),
        ("eval relative checkpoint resolves cwd", case_eval_relative_checkpoint_resolves_against_cwd),
        ("train checkpoint flag not eval prereq", case_train_checkpoint_flag_is_not_eval_prereq),
        ("SimpleSAC train ckpt handles --current-time", case_simple_sac_train_ckpt_hyphen_current_time),
        ("run identity ignores scheduling knobs", case_run_identity_ignores_scheduling_knobs),
        ("run identity changes with extra env", case_run_identity_changes_with_extra_env),
        ("runtime tokens normalize seed values", case_runtime_tokens_normalize_seed_values),
        ("closest runtime rejects unrelated history", case_runtime_closest_rejects_unrelated_history),
        ("pending ETA does not overwrite existing ETA", case_pending_eta_does_not_overwrite_existing_eta),
        ("pending ETA uses legacy duration fallback", case_pending_eta_uses_legacy_duration_when_no_runtime_history),
        ("gpu fits rejects per-task cap", case_gpu_fits_rejects_per_task_cap),
        ("gpu fits rejects small idle past one third", case_gpu_fits_rejects_small_idle_past_one_third),
        ("gpu fits rejects saturated past one third", case_gpu_fits_rejects_saturated_past_one_third),
        ("gpu fits allows first large task on empty gpu", case_gpu_fits_allows_first_large_task_on_empty_gpu),
        ("gpu fits allows slight one-third crossing", case_gpu_fits_allows_slight_one_third_crossing),
        ("gpu fits rejects crossing freeze line from below", case_gpu_fits_rejects_crossing_freeze_line_from_below),
        ("gpu fits rejects margin shortfall", case_gpu_fits_rejects_vram_margin_shortfall),
        ("slurm bucket missing estimate defaults gpu", case_slurm_bucket_missing_est_defaults_gpu),
        ("format task location slurm cpu and gpu", case_format_task_location_slurm_cpu_and_gpu),
        ("jtl110cpu configured as Windows CPU node", case_jtl110cpu_is_windows_cpu_node),
        ("jtl110cpu2 configured as Windows CPU node", case_jtl110cpu2_is_windows_cpu_node),
        ("scheduler source defines Windows CPU nodes", case_real_scheduler_source_defines_windows_cpu_nodes),
        ("CPU-only placement prefers jtl110cpu", case_cpu_only_prefers_jtl110cpu),
        ("CPU-only placement can prefer jtl110cpu2 when freer", case_cpu_only_prefers_less_loaded_windows_cpu_node),
        ("GPU placement excludes jtl110cpu", case_gpu_task_never_placed_on_jtl110cpu),
        ("CPU worker plan 901 items on 128 physical cores", case_cpu_worker_plan_901_on_128_physical),
        ("CPU batch plan splits 901 items across two 128-core nodes", case_cpu_batch_plan_splits_two_128_nodes),
        ("CPU batch plan uses free physical cores", case_cpu_batch_plan_uses_free_physical_cores),
        ("CPU batch plan skips full CPU node", case_cpu_batch_plan_skips_full_cpu_node),
        ("CPU wave summary records each small wave", case_cpu_wave_summary_records_each_small_wave),
        ("CPU batch log payload has detailed node rows", case_cpu_batch_log_payload_has_detailed_node_rows),
        ("CPU ownership snapshot separates ours/external/other", case_cpu_ownership_snapshot_separates_ours_external_and_other),
        ("dispatch cycle payload includes CPU accounting", case_dispatch_cycle_payload_includes_cpu_accounting),
        ("CPU parallel template rewrites worker auto flag", case_cpu_parallel_template_and_auto_worker_flag),
        ("Node physical cores infers half of logical cores", case_node_physical_cores_infers_half_logical_when_unconfigured),
        ("submit-cpu-batch CLI exists", case_submit_cpu_batch_cli_exists),
        ("CPU parallel env keeps zero start/index", case_cpu_parallel_env_keeps_zero_start_index),
        ("GPU servers use auto RAM detection", case_gpu_servers_use_auto_ram_detection),
        ("Claim capacity uses probed RAM when auto", case_claim_capacity_uses_probed_ram_when_auto),
        ("hybrid backend routes jtl110cpu to WindowsBackend", case_hybrid_routes_jtl110cpu_to_windows_backend),
        ("Windows path mapping uses F drive project layout", case_windows_path_mapping_to_f_drive_project_layout),
        ("Windows command prep rewrites python and paths", case_windows_prepare_command_rewrites_python_and_paths),
        ("Windows backend refuses GPU tasks before network", case_windows_backend_refuses_gpu_task_without_network),
        ("Windows cwd requires staging cache before launch", case_windows_cwd_requires_stage_cache_before_launch),
        ("Windows staging helper uses tar over SSH", case_windows_stage_helper_uses_tar_over_ssh),
        ("Windows explicit env_spec rejected before network", case_windows_explicit_env_spec_rejected_without_network),
        ("Windows staged resume path maps to F drive", case_record_staged_resume_location_maps_windows_path),
        ("ckpt staging runs even when cwd is already cached", case_ckpt_staging_runs_when_cwd_already_cached),
        ("Windows node skips Linux auto-adopt scans", case_windows_auto_adopt_linux_scans_are_skipped),
        ("Windows log tail uses read-write file share", case_windows_tail_uses_readwrite_share),
        ("Windows probe ignores PowerShell CLIXML noise", case_windows_probe_ignores_powershell_clixml_noise),
        ("Windows probe has process CPU delta fallback", case_windows_probe_has_process_cpu_delta_fallback),
        ("Windows probe SSH auth failure gets key hint", case_windows_probe_error_hints_ssh_key_auth),
        ("Windows wrapper logs CPU plan and resource progress", case_windows_wrapper_logs_cpu_plan_and_resource_progress),
        ("watcher has periodic node CPU accounting log", case_watcher_has_periodic_node_cpu_accounting_log),
    ]

    for idx, (name, fn) in enumerate(cases, 1):
        try:
            ok = bool(fn())
            diag = "" if ok else "case returned false"
        except Exception as e:
            ok = False
            diag = repr(e)
        check(f"external corner {idx:02d}: {name}", ok, diag=diag)
