from __future__ import annotations

from algorithm.experiments.corner_case_benchmark import (
    ETA_CACHE_KEY_FIELDS,
    build_corner_case_plan,
)


def test_corner_case_plan_covers_requested_resource_states():
    plan = build_corner_case_plan(run_id="unit")
    ids = {row["id"] for row in plan["scenarios"]}
    required = {
        "gpu_empty_single_eta",
        "gpu_half_loaded_add_one",
        "gpu_full_loaded_add_one",
        "node007_30gb_multigpu_plus_small",
        "cpu_empty_single_eta",
        "cpu_half_loaded_add_small",
        "cpu_full_loaded_add_small",
        "mixed_gpu_cpu_resident_plus_small",
        "queue_backlog_eta_load_balance",
        "eta_cache_identical_signature",
    }
    assert required <= ids
    assert plan["scenario_count"] >= len(required)


def test_eta_cache_contract_has_colocation_key():
    fields = set(ETA_CACHE_KEY_FIELDS)
    for key in {
        "workload_key",
        "normalized_parameters",
        "node",
        "device_model",
        "co_location_profile",
        "resource_occupancy_state",
        "algorithm_mode",
        "hard_rule_mode",
        "python_or_conda_env",
        "cuda_visible_devices",
    }:
        assert key in fields


def test_hard_rule_modes_include_safety_and_clean_bench():
    plan = build_corner_case_plan(run_id="unit")
    modes = {row["mode"] for row in plan["hard_rule_modes"]}
    assert {"safety_on", "clean_bench"} <= modes


def test_sota_policy_semantics_are_not_collapsed_to_one_policy():
    plan = build_corner_case_plan(run_id="unit")
    policies = {row["policy"] for row in plan["sota_policy_semantics"]}
    assert {
        "gavel_finish_time_fairness",
        "pollux_goodput_resize",
        "sia_multi_cluster_goodput",
        "gandiva_pack_time_slice",
        "salus_memory_pack",
        "iadeep_interference_aware",
    } <= policies


def test_benchmark_commands_have_progress_eta_source():
    plan = build_corner_case_plan(run_id="unit")
    progress_commands = []
    for row in plan["scenarios"]:
        for cmd in row.get("probe_commands") or []:
            if "scheduler.py status" in cmd or "pytest " in cmd or "live_trace_dryrun" in cmd:
                continue
            progress_commands.append(cmd)
    assert progress_commands
    assert all("progress_benchmark.py" in cmd or "Progress" not in cmd for cmd in progress_commands)
    assert all("--log-interval" in cmd or "cpu_progress_benchmark.py" in cmd for cmd in progress_commands)


def test_plan_keeps_algorithm_logic_out_of_scheduler_body():
    plan = build_corner_case_plan(run_id="unit")
    surface = set(plan["queue_state_contract"]["scheduler_surface"])
    assert "algorithm/theorem_dispatch/queue_state.py" in surface
    assert "algorithm/theorem_dispatch/policy.py" in surface
    assert "skill/scheduler.py optional A/B hook only" in surface
