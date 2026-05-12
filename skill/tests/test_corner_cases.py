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
    ]

    for idx, (name, fn) in enumerate(cases, 1):
        try:
            ok = bool(fn())
            diag = "" if ok else "case returned false"
        except Exception as e:
            ok = False
            diag = repr(e)
        check(f"external corner {idx:02d}: {name}", ok, diag=diag)
