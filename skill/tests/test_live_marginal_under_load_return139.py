import subprocess
from pathlib import Path

from algorithm.experiments import live_marginal_under_load_matrix as marginal


def _cpu_resident_rl_scenario() -> marginal.MarginalScenario:
    return marginal.MarginalScenario(
        scenario_id="unit_jtl110gpu2_cpu_resident_resac",
        node="jtl110gpu2",
        gpu=0,
        workload_key="hybrid_rl_resac_hopper",
        workload_env="hopper",
        resource_kind="hybrid_rl",
        resource_state="cpu_resident",
        profile=1,
        background_kind="cpu_full_resident",
        add_kind="rl_resac_hopper_add_light",
        background_kinds=("cpu_full_resident",),
        warmup_s=10,
        timeout_s=30,
        total_units=20,
        resident_mix="cpu_worker_resident",
    )


def test_cpu_resident_workers_scale_from_target_physical_cores() -> None:
    assert marginal._resident_worker_counts(12) == (6, 11)
    assert marginal._resident_worker_counts(192) == (96, 180)

    half = marginal._benchmark_command("cpu_half_resident", label="half", steps=10)
    full = marginal._benchmark_command("cpu_full_resident", label="full", steps=10)
    assert '"${CPU_HALF_WORKERS}"' in half
    assert '"${CPU_FULL_WORKERS}"' in full
    assert "--workers 96" not in half
    assert "--workers 180" not in full


def test_shell_139_is_structured_sigsegv_before_capacity_classification() -> None:
    diagnostic = marginal._returncode_diagnostic(
        139,
        shell_signal_number=11,
        shell_signal_name="SIGSEGV",
    )
    assert diagnostic["kind"] == "signal"
    assert diagnostic["encoding"] == "shell_128_plus_signal"
    assert diagnostic["signal_number"] == 11
    assert diagnostic["signal_name"] == "SIGSEGV"
    assert diagnostic["shell_report_matches"] is True

    reason = marginal._failure_reason(
        rc=0,
        add_rc=139,
        stable={"ready": False},
        stderr="CUDA error was also printed",
        add_log="",
        bg_ready=True,
        bg_alive_before_add=True,
        bg_alive_after_add=True,
    )
    assert reason == "add_probe_signal_SIGSEGV"


def test_resident_must_outlive_add_probe() -> None:
    reason = marginal._failure_reason(
        rc=0,
        add_rc=0,
        stable={"ready": True},
        stderr="",
        add_log="",
        bg_ready=True,
        bg_alive_before_add=True,
        bg_alive_after_add=False,
    )
    assert reason == "background_ended_during_add_probe"
    assert marginal._background_total("cpu_full_resident") > 240
    assert marginal._background_total("rl_resac_ant_resident_light") > 240


def test_remote_script_captures_resources_and_shell_signal_details() -> None:
    script = marginal._remote_script(
        _cpu_resident_rl_scenario(),
        run_id="unit_return139",
        output_root="/tmp/unit_return139",
    )

    assert "lscpu -p=CORE,SOCKET" in script
    assert "CPU_FULL_WORKERS=$((CPU_CORES - CPU_FULL_RESERVE))" in script
    assert "--workers \"${CPU_FULL_WORKERS}\"" in script
    assert f"--steps {marginal.RESIDENT_BACKGROUND_TOTAL_UNITS}" in script
    assert "ulimit -a" in script
    assert "free -h" in script
    assert "nvidia-smi --query-gpu=" in script
    assert "CONTROLLED_CUDA_PROCESSES=1" in script
    assert "RL_CUDA_MEMORY_BUDGET=0.7000" in script
    assert "SHELL_EXIT role=%s" in script
    assert "__ADD_SIGNAL_NUMBER__" in script
    assert "__BACKGROUND_ALIVE_AFTER_ADD__" in script
    assert "__DIAGNOSTICS_LOG_BEGIN__" in script
    assert "trap cleanup_backgrounds EXIT" in script
    assert "setsid timeout --foreground 160s" in script

    subprocess.run(
        ["bash", "-n", "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_rl_cuda_processes_share_memory_budget_and_one_cpu_thread_each() -> None:
    fraction = marginal._rl_cuda_memory_fraction(
        "rl_resac_hopper_add_light",
        concurrent_cuda_processes=8,
    )
    assert fraction == 0.0875
    assert fraction * 8 <= marginal.RL_CUDA_MEMORY_BUDGET

    command = marginal._wrapped_resac_command(
        "rl_resac_hopper_add_light",
        label="bounded",
        total=20,
        terminate_on_stable=True,
        gpu_prefix="CUDA_VISIBLE_DEVICES=0 ",
        concurrent_cuda_processes=8,
    )
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in command
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.0875" in command
    assert "--xla_cpu_multi_thread_eigen=false" in command
    assert "OMP_NUM_THREADS=1" in command
    assert "TF_NUM_INTRAOP_THREADS=1" in command
    assert "MALLOC_ARENA_MAX=2" in command


def test_run_row_persists_signal_and_resource_diagnostics(monkeypatch, tmp_path: Path) -> None:
    stdout = """\
__ADD_RC__ 139
__ADD_SIGNAL_NUMBER__ 11
__ADD_SIGNAL_NAME__ SIGSEGV
__BACKGROUND_RC__ 143
__BACKGROUND_SIGNAL_NUMBER__ 15
__BACKGROUND_SIGNAL_NAME__ SIGTERM
__BACKGROUND_READY__ 1
__BACKGROUND_ALIVE_BEFORE_ADD__ 1
__BACKGROUND_ALIVE_AFTER_ADD__ 1
__CPU_LOGICAL__ 24
__CPU_CORES__ 12
__CPU_HALF_WORKERS__ 6
__CPU_FULL_WORKERS__ 11
__CONTROLLED_CUDA_PROCESSES__ 1
__ADD_LOG_BEGIN__
Iter 1 then native crash
__ADD_LOG_END__
__BACKGROUND_LOG_BEGIN__
CPU_PARALLEL_PROGRESS Step 10/1000000
__BACKGROUND_LOG_END__
__DIAGNOSTICS_LOG_BEGIN__
RESOURCE_SNAPSHOT label=after_add
SHELL_EXIT role=add_probe returncode=139 encoding=128_plus_signal signal_number=11 signal_name=SIGSEGV
__DIAGNOSTICS_LOG_END__
"""
    monkeypatch.setattr(marginal, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(marginal, "_deploy_progress_wrapper", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        marginal,
        "_run_remote_capture",
        lambda *args, **kwargs: (0, stdout, ""),
    )
    monkeypatch.setattr(
        marginal,
        "_last_stable_rate",
        lambda log: {"ready": False, "reason": "not_enough_samples"},
    )
    monkeypatch.setattr(marginal, "_last_rate", lambda log: (0.04, "iter"))

    row = marginal._run_scenario(
        _cpu_resident_rl_scenario(),
        run_id="unit_return139",
    )

    assert row["reason"] == "add_probe_signal_SIGSEGV"
    assert row["exit_diagnostics"]["add_probe"]["signal_name"] == "SIGSEGV"
    assert row["exit_diagnostics"]["add_probe"]["shell_report_matches"] is True
    assert row["target_cpu_cores"] == 12
    assert row["cpu_full_resident_workers"] == 11
    assert row["controlled_cuda_processes"] == 1
    assert row["rl_cuda_memory_budget_fraction"] == 0.70
    assert row["background_alive_after_add"] is True
    assert row["resource_diagnostics_captured"] is True
    assert Path(row["diagnostics_log_path"]).read_text(encoding="utf-8").startswith(
        "RESOURCE_SNAPSHOT"
    )
