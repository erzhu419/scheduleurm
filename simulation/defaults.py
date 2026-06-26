"""Default trace-cache assembly for Scheduleurm replay experiments."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from .fast_forward import ReplayPolicy, WorkloadSpec
from .service_cache import (
    ProfileRecord,
    ServiceRateCache,
    add_protocol_cpu_curve,
    deterministic_makespan_s,
    deterministic_mean_flow_s,
    records_from_summary_file,
)
from .tasksets import empirical_replay_specs


RUN_ROOT = Path("/home/erzhu419/.claude/scheduler/experiments/runs")
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_default_cache() -> ServiceRateCache:
    cache = ServiceRateCache()
    _add_summary_dir(
        cache,
        RUN_ROOT / "module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_002" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module12_resac_ant_real_dense_jtl110gpu2_gpu1_20260603_003" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module42_q11_live_sanity_jtl110gpu2_gpu1_profile10_20260608_001" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module43_q11_live_sanity_jtl110gpu2_gpu1_profile9_20260608_001" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module44_q11_live_sanity_jtl110gpu2_gpu1_profile1_20260608_001" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module45_q11_live_sanity_jtl110gpu2_gpu1_profile8_20260608_001" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module46_q11_live_sanity_jtl110gpu2_gpu1_profiles2_3_20260608_001" / "reports",
        workload_key="hybrid_rl_resac_ant",
        command_fingerprint="resac_ant_real_v1",
        resource_kind="hybrid_rl",
        total_units=80,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module6_service_curve_jtl110gpu2_size8192_20260603_001" / "reports",
        workload_key="gpu_heavy_jax_matmul",
        command_fingerprint="jax_matmul_size8192_v1",
        resource_kind="gpu_heavy",
        total_units=2400,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module22_q01_gpu_heavy_jax8192_profiles4_8_20260604_001" / "reports",
        workload_key="gpu_heavy_jax_matmul",
        command_fingerprint="jax_matmul_size8192_v1",
        resource_kind="gpu_heavy",
        total_units=2400,
        node_bucket="jtl110gpu2:12gb",
    )
    _add_summary_file(
        cache,
        ARTIFACT_ROOT / "node007_q01_pareto_p1_20260613_profile_1_per_gpu_summary.json",
        workload_key="gpu_heavy_jax_matmul",
        command_fingerprint="jax_matmul_size8192_v1",
        resource_kind="gpu_heavy",
        total_units=2400,
        node_bucket="node007:11gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "cnn_jax_convstack_jtl110gpu_profile1_20260612_002" / "reports",
        workload_key="gpu_cnn_jax_convstack",
        command_fingerprint="jax_cnn_convstack_b16_i128_d4_c64_v1",
        resource_kind="gpu_cnn",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "cnn_jax_convstack_jtl110gpu_profiles2_3_20260612_001" / "reports",
        workload_key="gpu_cnn_jax_convstack",
        command_fingerprint="jax_cnn_convstack_b16_i128_d4_c64_v1",
        resource_kind="gpu_cnn",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "cnn_torch_resnet50_jtl110gpu_profile1_20260612_002" / "reports",
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="torch_resnet50_train_b16_i224_fp32_v1",
        resource_kind="gpu_cnn",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "cnn_torch_resnet50_jtl110gpu_profiles2_3_20260612_001" / "reports",
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="torch_resnet50_train_b16_i224_fp32_v1",
        resource_kind="gpu_cnn",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "cnn_torch_resnet50_jtl110gpu_profiles4_5_20260612_001" / "reports",
        workload_key="gpu_cnn_torch_resnet50",
        command_fingerprint="torch_resnet50_train_b16_i224_fp32_v1",
        resource_kind="gpu_cnn",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "llm_torch_distilgpt2_jtl110gpu_profile1_20260612_003" / "reports",
        workload_key="gpu_llm_distilgpt2",
        command_fingerprint="torch_distilgpt2_forward_b2_s128_fp16_v1",
        resource_kind="gpu_llm",
        total_units=80,
        node_bucket="jtl110gpu:12gb",
    )
    for run_name in (
        "llm_torch_distilgpt2_jtl110gpu_profiles2_3_20260612_001",
        "llm_torch_distilgpt2_jtl110gpu_profiles4_5_20260612_001",
        "llm_torch_distilgpt2_jtl110gpu_profiles6_8_20260612_001",
        "llm_torch_distilgpt2_jtl110gpu_profiles10_12_20260612_001",
        "llm_torch_distilgpt2_jtl110gpu_profile11_20260612_001",
    ):
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key="gpu_llm_distilgpt2",
            command_fingerprint="torch_distilgpt2_forward_b2_s128_fp16_v1",
            resource_kind="gpu_llm",
            total_units=80,
            node_bucket="jtl110gpu:12gb",
        )
    for run_name in (
        "node007_eta_matrix_cnn_p1_retry_20260613_cnn_torch_gpu_single_gpu_single_task",
        "node007_eta_matrix_cnn_v2_20260613_cnn_torch_gpu_single_gpu_multi_task",
        "node007_eta_matrix_cnn_v2_20260613_cnn_torch_gpu_four_gpu_single_task_each",
        "node007_eta_matrix_cnn_v2_20260613_cnn_torch_gpu_four_gpu_two_tasks_each",
    ):
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key="gpu_cnn_torch_progress_stack",
            command_fingerprint="torch_cnn_progress_stack_bench_v1",
            resource_kind="gpu_cnn",
            total_units=80,
            node_bucket="node007-direct:4x12gb",
        )
    for run_name in (
        "node007_eta_matrix_llm_20260613_llm_torch_transformer_single_gpu_single_task",
        "node007_eta_matrix_llm_20260613_llm_torch_transformer_single_gpu_multi_task",
        "node007_eta_matrix_llm_20260613_llm_torch_transformer_four_gpu_single_task_each",
        "node007_eta_matrix_llm_20260613_llm_torch_transformer_four_gpu_two_tasks_each",
    ):
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key="gpu_llm_torch_decoder_stack",
            command_fingerprint="torch_decoder_stack_bench_v1",
            resource_kind="gpu_llm",
            total_units=80,
            node_bucket="node007-direct:4x12gb",
        )
    for run_name in (
        "node007_eta_matrix_rl_20260613_rl_resac_ant_single_gpu_single_task",
        "node007_eta_matrix_rl_20260613_rl_resac_ant_single_gpu_multi_task",
        "node007_eta_matrix_rl_20260613_rl_resac_ant_four_gpu_two_tasks_each",
        "node007_eta_matrix_rl_fourgpu_p1_retry_20260613_rl_resac_ant_four_gpu_single_task_each",
    ):
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key="hybrid_rl_resac_ant",
            command_fingerprint="resac_ant_real_tqdm_stable_v1",
            resource_kind="hybrid_rl",
            total_units=80,
            node_bucket="node007-direct:4x12gb",
        )
    for run_name in (
        "node007_eta_matrix_rl_20260613_rl_resac_ant_single_gpu_single_task",
        "node007_eta_matrix_rl_20260613_rl_resac_ant_single_gpu_multi_task",
        "node007_eta_matrix_rl_20260613_rl_resac_ant_four_gpu_two_tasks_each",
        "node007_eta_matrix_rl_fourgpu_p1_retry_20260613_rl_resac_ant_four_gpu_single_task_each",
    ):
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key="hybrid_rl_resac_ant_node007_tqdm",
            command_fingerprint="resac_ant_node007_tqdm_stable_v1",
            resource_kind="hybrid_rl",
            total_units=80,
            node_bucket="node007-direct:4x12gb",
        )
    _add_fresh_node007_eta_matrix(cache)
    _add_summary_dir(
        cache,
        RUN_ROOT / "module23_q00_light_control_local_profiles1_16_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module23_q00_light_control_local_profile16_boundary_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module24_q00_light_control_local_profiles3_7_20260604_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module38_q00_light_control_local_profiles9_15_exact_20260605_001" / "reports",
        workload_key="light_control_local",
        command_fingerprint="cpu_light_sleep_20ms_v1",
        resource_kind="light_control",
        total_units=10000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module25_q10_cpu_heavy_local_profiles1_8_20260604_001" / "reports",
        workload_key="cpu_heavy_local_bench",
        command_fingerprint="cpu_heavy_local_progress_v1",
        resource_kind="cpu_heavy",
        total_units=1000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        RUN_ROOT / "module33_q10_cpu_heavy_local_profiles9_32_20260605_001" / "reports",
        workload_key="cpu_heavy_local_bench",
        command_fingerprint="cpu_heavy_local_progress_v1",
        resource_kind="cpu_heavy",
        total_units=1000,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module56_freqduet_cpu_c17_32_jtl110cpu2_curve_p124_reports",
        workload_key="freqduet_cpu_ablation_c17_32",
        command_fingerprint="freqduet_cpu_ablation_terminal_hiro_c17_32_v1",
        resource_kind="cpu_sumo_transit",
        total_units=72,
        node_bucket="jtl110cpu2:cpu128",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module56_freqduet_cpu_c17_32_jtl110cpu2_boundary_p8_reports",
        workload_key="freqduet_cpu_ablation_c17_32",
        command_fingerprint="freqduet_cpu_ablation_terminal_hiro_c17_32_v1",
        resource_kind="cpu_sumo_transit",
        total_units=72,
        node_bucket="jtl110cpu2:cpu128",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module74_freqduet_runner_v3_c17_32_completed_history_reports",
        workload_key="freqduet_runner_v3_c17_32_completed_history",
        command_fingerprint="freqduet_runner_v3_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module74_freqduet_paper_longtrain_c17_32_completed_history_reports",
        workload_key="freqduet_paper_longtrain_c17_32_completed_history",
        command_fingerprint="freqduet_paper_longtrain_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module74_transit_native_promotion_c17_32_residual_completed_history_reports",
        workload_key="transit_native_promotion_c17_32_residual_completed_history",
        command_fingerprint="transit_native_promotion_c17_32_residual_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module75_bamor_train_compare_c9_16_completed_history_reports",
        workload_key="bamor_train_compare_c9_16_completed_history",
        command_fingerprint="bamor_train_compare_c9_16_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module75_bamor_mujoco_c9_16_completed_history_reports",
        workload_key="bamor_mujoco_c9_16_completed_history",
        command_fingerprint="bamor_mujoco_c9_16_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module75_bamor_diagnostic_shard_c9_16_completed_history_reports",
        workload_key="bamor_diagnostic_shard_c9_16_completed_history",
        command_fingerprint="bamor_diagnostic_shard_c9_16_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module76_freqduet_ablation_c_le2_completed_history_reports",
        workload_key="freqduet_cpu_ablation_c_le2_completed_history",
        command_fingerprint="freqduet_cpu_ablation_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module76_freqduet_baseline_rule_c_le2_completed_history_reports",
        workload_key="freqduet_baseline_rule_c_le2_completed_history",
        command_fingerprint="freqduet_baseline_rule_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module76_freqduet_preflight_c_le2_completed_history_reports",
        workload_key="freqduet_preflight_c_le2_completed_history",
        command_fingerprint="freqduet_preflight_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module76_transit_freqhrl_analysis_matrix_c_le2_completed_history_reports",
        workload_key="transit_freqhrl_analysis_matrix_c_le2_completed_history",
        command_fingerprint="transit_freqhrl_analysis_matrix_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module76_transit_freqhrl_merge_c_le2_completed_history_reports",
        workload_key="transit_freqhrl_merge_c_le2_completed_history",
        command_fingerprint="transit_freqhrl_merge_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module77_bamor_train_compare_c_le2_completed_history_reports",
        workload_key="bamor_train_compare_c_le2_completed_history",
        command_fingerprint="bamor_train_compare_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module77_bamor_mujoco_c_le2_completed_history_reports",
        workload_key="bamor_mujoco_c_le2_completed_history",
        command_fingerprint="bamor_mujoco_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module77_bamor_diagnostic_shard_c_le2_completed_history_reports",
        workload_key="bamor_diagnostic_shard_c_le2_completed_history",
        command_fingerprint="bamor_diagnostic_shard_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_offline_sumo_eval_c_le2_completed_history_reports",
        workload_key="offline_sumo_eval_c_le2_completed_history",
        command_fingerprint="offline_sumo_eval_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module86_offline_sumo_eval_c33_64_completed_history_reports",
        workload_key="offline_sumo_eval_c33_64_completed_history",
        command_fingerprint="offline_sumo_eval_c33_64_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_h2oplus_shell_eval_c_le2_completed_history_reports",
        workload_key="h2oplus_shell_eval_c_le2_completed_history",
        command_fingerprint="h2oplus_shell_eval_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_zsw_metrics_parser_c_le2_completed_history_reports",
        workload_key="zsw_metrics_parser_c_le2_completed_history",
        command_fingerprint="zsw_metrics_parser_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_resco_config_eval_c_le2_completed_history_reports",
        workload_key="resco_config_eval_c_le2_completed_history",
        command_fingerprint="resco_config_eval_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_nature_emissions_extract_c_le2_completed_history_reports",
        workload_key="nature_emissions_extract_c_le2_completed_history",
        command_fingerprint="nature_emissions_extract_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module78_nature_emissions_sumo_c_le2_completed_history_reports",
        workload_key="nature_emissions_sumo_c_le2_completed_history",
        command_fingerprint="nature_emissions_sumo_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module79_transit_native_promotion_c3_8_persistent_stress_completed_history_reports",
        workload_key="transit_native_promotion_c3_8_persistent_stress_completed_history",
        command_fingerprint="transit_native_promotion_c3_8_persistent_stress_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module79_transit_native_real_demand_batch_c3_8_completed_history_reports",
        workload_key="transit_native_real_demand_batch_c3_8_completed_history",
        command_fingerprint="transit_native_real_demand_batch_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module79_transit_native_real_demand_alighting_c3_8_completed_history_reports",
        workload_key="transit_native_real_demand_alighting_c3_8_completed_history",
        command_fingerprint="transit_native_real_demand_alighting_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module80_bamor_mujoco_c17_32_completed_history_reports",
        workload_key="bamor_mujoco_c17_32_completed_history",
        command_fingerprint="bamor_mujoco_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module80_bamor_diagnostic_shard_c17_32_completed_history_reports",
        workload_key="bamor_diagnostic_shard_c17_32_completed_history",
        command_fingerprint="bamor_diagnostic_shard_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module81_freqduet_runner_v3_c33_64_completed_history_reports",
        workload_key="freqduet_runner_v3_c33_64_completed_history",
        command_fingerprint="freqduet_runner_v3_c33_64_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module82_cfcmt_snapshot_generation_c3_8_completed_history_reports",
        workload_key="cfcmt_snapshot_generation_c3_8_completed_history",
        command_fingerprint="cfcmt_snapshot_generation_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module82_cfcmt_pytest_sumo_c3_8_completed_history_reports",
        workload_key="cfcmt_pytest_sumo_c3_8_completed_history",
        command_fingerprint="cfcmt_pytest_sumo_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module82_cfcmt_traffic_signal_phase1_c3_8_completed_history_reports",
        workload_key="cfcmt_traffic_signal_phase1_c3_8_completed_history",
        command_fingerprint="cfcmt_traffic_signal_phase1_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module82_zsw_m21_sumo_eval_c3_8_completed_history_reports",
        workload_key="zsw_m21_sumo_eval_c3_8_completed_history",
        command_fingerprint="zsw_m21_sumo_eval_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_trading_public_csv_c3_8_completed_history_reports",
        workload_key="transit_trading_public_csv_c3_8_completed_history",
        command_fingerprint="transit_trading_public_csv_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_trading_pressure_merge_c3_8_completed_history_reports",
        workload_key="transit_trading_pressure_merge_c3_8_completed_history",
        command_fingerprint="transit_trading_pressure_merge_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_trading_policy_c3_8_completed_history_reports",
        workload_key="transit_trading_policy_c3_8_completed_history",
        command_fingerprint="transit_trading_policy_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_surrogate_c3_8_completed_history_reports",
        workload_key="transit_surrogate_c3_8_completed_history",
        command_fingerprint="transit_surrogate_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_freqhrl_tests_c3_8_completed_history_reports",
        workload_key="transit_freqhrl_tests_c3_8_completed_history",
        command_fingerprint="transit_freqhrl_tests_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module83_transit_native_merge_c3_8_completed_history_reports",
        workload_key="transit_native_merge_c3_8_completed_history",
        command_fingerprint="transit_native_merge_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module84_transit_trading_pressure_matrix_c17_32_completed_history_reports",
        workload_key="transit_trading_pressure_matrix_c17_32_completed_history",
        command_fingerprint="transit_trading_pressure_matrix_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module84_transit_trading_promotion_recovery_c17_32_completed_history_reports",
        workload_key="transit_trading_promotion_recovery_c17_32_completed_history",
        command_fingerprint="transit_trading_promotion_recovery_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module84_transit_demand_estimator_c17_32_completed_history_reports",
        workload_key="transit_demand_estimator_c17_32_completed_history",
        command_fingerprint="transit_demand_estimator_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module84_transit_gap_closure_c17_32_completed_history_reports",
        workload_key="transit_gap_closure_c17_32_completed_history",
        command_fingerprint="transit_gap_closure_c17_32_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module88_transit_trading_policy_c33_64_completed_history_reports",
        workload_key="transit_trading_policy_c33_64_completed_history",
        command_fingerprint="transit_trading_policy_c33_64_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module88_transit_trading_pressure_matrix_c33_64_completed_history_reports",
        workload_key="transit_trading_pressure_matrix_c33_64_completed_history",
        command_fingerprint="transit_trading_pressure_matrix_c33_64_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module89_transit_trading_pressure_matrix_c9_16_completed_history_reports",
        workload_key="transit_trading_pressure_matrix_c9_16_completed_history",
        command_fingerprint="transit_trading_pressure_matrix_c9_16_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module58_freqduet_ablation_c9_16_jtl110cpu2_curve_p1248_reports",
        workload_key="freqduet_cpu_ablation_c9_16",
        command_fingerprint="freqduet_cpu_ablation_main_hiro_c9_16_v1",
        resource_kind="cpu_sumo_transit",
        total_units=2,
        node_bucket="jtl110cpu2:cpu128",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module60_freqduet_ablation_c3_8_completed_history_reports",
        workload_key="freqduet_cpu_ablation_c3_8_completed_history",
        command_fingerprint="freqduet_cpu_ablation_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module93_freqduet_spacectx_ep100_c3_8_completed_history_reports",
        workload_key="freqduet_spacectx_ep100_c3_8_completed_history",
        command_fingerprint="freqduet_spacectx_ep100_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module61_freqduet_ablation_c33_64_completed_history_reports",
        workload_key="freqduet_cpu_ablation_c33_64_completed_history",
        command_fingerprint="freqduet_cpu_ablation_c33_64_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module69_freqduet_ablation_c65p_completed_history_reports",
        workload_key="freqduet_cpu_ablation_c65p_completed_history",
        command_fingerprint="freqduet_cpu_ablation_c65p_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module69_freqduet_promoted_ep100_c65p_completed_history_reports",
        workload_key="freqduet_promoted_ep100_c65p_completed_history",
        command_fingerprint="freqduet_promoted_ep100_c65p_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module62_freqduet_runner_v3_c_le2_completed_history_reports",
        workload_key="freqduet_runner_v3_c_le2_completed_history",
        command_fingerprint="freqduet_runner_v3_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module69_transit_native_promotion_c65p_completed_history_reports",
        workload_key="transit_native_promotion_c65p_completed_history",
        command_fingerprint="transit_native_promotion_c65p_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_trading_sweep_c_le2_completed_history_reports",
        workload_key="transit_trading_sweep_c_le2_completed_history",
        command_fingerprint="transit_trading_sweep_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_trading_policy_c_le2_completed_history_reports",
        workload_key="transit_trading_policy_c_le2_completed_history",
        command_fingerprint="transit_trading_policy_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_surrogate_validation_c_le2_completed_history_reports",
        workload_key="transit_surrogate_validation_c_le2_completed_history",
        command_fingerprint="transit_surrogate_validation_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_native_promotion_c_le2_completed_history_reports",
        workload_key="transit_native_promotion_c_le2_completed_history",
        command_fingerprint="transit_native_promotion_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_native_control_c_le2_completed_history_reports",
        workload_key="transit_native_control_c_le2_completed_history",
        command_fingerprint="transit_native_control_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module70_transit_freqhrl_import_smoke_c_le2_completed_history_reports",
        workload_key="transit_freqhrl_import_smoke_c_le2_completed_history",
        command_fingerprint="transit_freqhrl_import_smoke_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module90_transit_freqhrl_tests_c_le2_completed_history_reports",
        workload_key="transit_freqhrl_tests_c_le2_completed_history",
        command_fingerprint="transit_freqhrl_tests_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module71_transit_native_promotion_c9_16_bounded_wait_completed_history_reports",
        workload_key="transit_native_promotion_c9_16_bounded_wait_completed_history",
        command_fingerprint="transit_native_promotion_c9_16_bounded_wait_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module71_transit_native_promotion_c9_16_residual_completed_history_reports",
        workload_key="transit_native_promotion_c9_16_residual_completed_history",
        command_fingerprint="transit_native_promotion_c9_16_residual_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module85_transit_native_promotion_c9_16_wait_credit_shell_completed_history_reports",
        workload_key="transit_native_promotion_c9_16_wait_credit_shell_completed_history",
        command_fingerprint="transit_native_promotion_c9_16_wait_credit_shell_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module71_freqduet_runner_v3_c9_16_residual_completed_history_reports",
        workload_key="freqduet_runner_v3_c9_16_residual_completed_history",
        command_fingerprint="freqduet_runner_v3_c9_16_residual_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_feed_conversion_c_le2_completed_history_reports",
        workload_key="cfcmt_feed_conversion_c_le2_completed_history",
        command_fingerprint="cfcmt_feed_conversion_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_env_validation_c_le2_completed_history_reports",
        workload_key="cfcmt_env_validation_c_le2_completed_history",
        command_fingerprint="cfcmt_env_validation_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_sumo_generation_c_le2_completed_history_reports",
        workload_key="cfcmt_sumo_generation_c_le2_completed_history",
        command_fingerprint="cfcmt_sumo_generation_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_snapshot_generation_c_le2_completed_history_reports",
        workload_key="cfcmt_snapshot_generation_c_le2_completed_history",
        command_fingerprint="cfcmt_snapshot_generation_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_policy_rollout_c_le2_completed_history_reports",
        workload_key="cfcmt_policy_rollout_c_le2_completed_history",
        command_fingerprint="cfcmt_policy_rollout_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module72_cfcmt_traffic_signal_phase2_c_le2_completed_history_reports",
        workload_key="cfcmt_traffic_signal_phase2_c_le2_completed_history",
        command_fingerprint="cfcmt_traffic_signal_phase2_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module66_freqduet_runner_v3_c3_8_completed_history_reports",
        workload_key="freqduet_runner_v3_c3_8_completed_history",
        command_fingerprint="freqduet_runner_v3_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module63_transit_native_promotion_c17_32_seedrange_completed_history_reports",
        workload_key="transit_native_promotion_c17_32_seedrange_completed_history",
        command_fingerprint="transit_native_promotion_c17_32_seedrange_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module68_transit_native_promotion_c33_64_batch_completed_history_reports",
        workload_key="transit_native_promotion_c33_64_batch_completed_history",
        command_fingerprint="transit_native_promotion_c33_64_batch_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module68_transit_native_promotion_c33_64_single_seed_completed_history_reports",
        workload_key="transit_native_promotion_c33_64_single_seed_completed_history",
        command_fingerprint="transit_native_promotion_c33_64_single_seed_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module64_bamor_cpu_training_c3_8_completed_history_reports",
        workload_key="bamor_cpu_training_c3_8_completed_history",
        command_fingerprint="bamor_cpu_training_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module67_bamor_train_compare_c3_8_completed_history_reports",
        workload_key="bamor_train_compare_c3_8_completed_history",
        command_fingerprint="bamor_train_compare_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module67_bamor_mujoco_c3_8_completed_history_reports",
        workload_key="bamor_mujoco_c3_8_completed_history",
        command_fingerprint="bamor_mujoco_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module67_bamor_diagnostic_shard_c3_8_completed_history_reports",
        workload_key="bamor_diagnostic_shard_c3_8_completed_history",
        command_fingerprint="bamor_diagnostic_shard_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module91_bamor_mujoco_policy_union_c3_8_completed_history_reports",
        workload_key="bamor_mujoco_policy_union_c3_8_completed_history",
        command_fingerprint="bamor_mujoco_policy_union_c3_8_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module92_assumption_agent_unittest_completed_history_reports",
        workload_key="assumption_agent_unittest_completed_history",
        command_fingerprint="assumption_agent_unittest_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module92_assumption_agent_meta_qa_evolution_completed_history_reports",
        workload_key="assumption_agent_meta_qa_evolution_completed_history",
        command_fingerprint="assumption_agent_meta_qa_evolution_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module92_assumption_agent_phase2_v20_framework_completed_history_reports",
        workload_key="assumption_agent_phase2_v20_framework_completed_history",
        command_fingerprint="assumption_agent_phase2_v20_framework_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    for artifact_name, workload_key, resource_kind, node_bucket in (
        (
            "module94_assumption_agent_performance_validation_completed_history",
            "assumption_agent_performance_validation_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_assumption_agent_live_benchmark_completed_history",
            "assumption_agent_live_benchmark_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_cfcmt_cpu_eval_completed_history",
            "cfcmt_cpu_eval_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_cfcmt_pytest_cpu_completed_history",
            "cfcmt_pytest_cpu_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_sensing_voltage_cache_completed_history",
            "sensing_voltage_cache_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_sensing_pems_cache_completed_history",
            "sensing_pems_cache_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_nature_emissions_routeguard_analysis_completed_history",
            "nature_emissions_routeguard_analysis_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_scheduleurm_control_plane_completed_history",
            "scheduleurm_control_plane_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_scheduleurm_hpc_relay_smoke_completed_history",
            "scheduleurm_hpc_relay_smoke_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_bapr_id_ood_merge_cpu_eval_completed_history",
            "bapr_id_ood_merge_cpu_eval_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_resac_bus_seed_extension_cpu_eval_completed_history",
            "resac_bus_seed_extension_cpu_eval_completed_history",
            "cpu_sumo_transit",
            "production-history:cpu",
        ),
        (
            "module94_h2oplus_snapshot_gpu_completed_history",
            "h2oplus_snapshot_gpu_completed_history",
            "hybrid_rl",
            "production-history:gpu",
        ),
    ):
        _add_summary_dir(
            cache,
            ARTIFACT_ROOT / f"{artifact_name}_reports",
            workload_key=workload_key,
            command_fingerprint=f"{workload_key}_v1",
            resource_kind=resource_kind,
            total_units=1,
            node_bucket=node_bucket,
        )
    for artifact_name, workload_key in (
        (
            "module95_transit_native_real_demand_batch_c9_16_completed_history",
            "transit_native_real_demand_batch_c9_16_completed_history",
        ),
        (
            "module95_resac_conda_pack_completed_history",
            "resac_conda_pack_completed_history",
        ),
    ):
        _add_summary_dir(
            cache,
            ARTIFACT_ROOT / f"{artifact_name}_reports",
            workload_key=workload_key,
            command_fingerprint=f"{workload_key}_v1",
            resource_kind="cpu_sumo_transit",
            total_units=1,
            node_bucket="production-history:cpu",
        )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module96_resac_review5_jax_train_fabric_completed_history_reports",
        workload_key="resac_review5_jax_train_fabric_completed_history",
        command_fingerprint="resac_review5_jax_train_fabric_completed_history_v1",
        resource_kind="hybrid_rl",
        total_units=1,
        node_bucket="production-history:gpu-fabric",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module97_cpu_heavy_local_fabric_completed_history_reports",
        workload_key="cpu_heavy_local_fabric_completed_history",
        command_fingerprint="cpu_heavy_local_fabric_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu-fabric",
    )
    for artifact_name, workload_key in (
        (
            "module98_hybrid_rl_resac_jmlr_project_fabric_completed_history",
            "hybrid_rl_resac_jmlr_project_fabric_completed_history",
        ),
        (
            "module98_hybrid_rl_resac_project_fabric_completed_history",
            "hybrid_rl_resac_project_fabric_completed_history",
        ),
        (
            "module98_hybrid_rl_bapr_project_fabric_completed_history",
            "hybrid_rl_bapr_project_fabric_completed_history",
        ),
        (
            "module98_hybrid_rl_bapr_v15_project_fabric_completed_history",
            "hybrid_rl_bapr_v15_project_fabric_completed_history",
        ),
        (
            "module98_hybrid_rl_cs_bapr_project_fabric_completed_history",
            "hybrid_rl_cs_bapr_project_fabric_completed_history",
        ),
        (
            "module98_hybrid_rl_sensing_v10k_project_fabric_completed_history",
            "hybrid_rl_sensing_v10k_project_fabric_completed_history",
        ),
    ):
        _add_summary_dir(
            cache,
            ARTIFACT_ROOT / f"{artifact_name}_reports",
            workload_key=workload_key,
            command_fingerprint=f"{workload_key}_v1",
            resource_kind="hybrid_rl",
            total_units=1,
            node_bucket="production-history:gpu-fabric",
        )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module99_transit_real_demand_c9_16_profile_extension_reports",
        workload_key="transit_native_real_demand_safe_wait_c9_16_profile_extension",
        command_fingerprint="transit_native_real_demand_safe_wait_c9_16_profile_extension_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu-fabric",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module65_zsw_tsp_sumo_eval_c_le2_completed_history_reports",
        workload_key="zsw_tsp_sumo_eval_c_le2_completed_history",
        command_fingerprint="zsw_tsp_sumo_eval_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module59_simple_sac_sumo_eval_cle2_completed_history_reports",
        workload_key="sumo_eval_simple_sac_c_le2",
        command_fingerprint="simple_sac_run_multiseed_eval_c_le2_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="local:cpu",
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module57_freqduet_runner_v3_c9_16_jtl110cpu2_curve_p1248_reports",
        workload_key="freqduet_runner_v3_allfreq_alllayers_c9_16",
        command_fingerprint="freqduet_runner_v3_allfreq_alllayers_c9_16_v1",
        resource_kind="cpu_sumo_transit",
        total_units=3,
        node_bucket="jtl110cpu2:cpu128",
    )
    add_protocol_cpu_curve(
        cache,
        workload_key="cpu_heavy_protocol",
        total_units=3600,
        max_workers=32,
        per_worker_rate=1.0,
        saturation_workers=16,
    )
    _add_summary_dir(
        cache,
        ARTIFACT_ROOT / "module102_freqduet_snapshot_counterfactual_c9_16_completed_history_reports",
        workload_key="freqduet_snapshot_counterfactual_c9_16_completed_history",
        command_fingerprint="freqduet_snapshot_counterfactual_c9_16_completed_history_v1",
        resource_kind="cpu_sumo_transit",
        total_units=1,
        node_bucket="production-history:cpu",
    )
    return cache


def default_workload_specs() -> list[WorkloadSpec]:
    return empirical_replay_specs()


def legacy_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="legacy_fixed_caps",
        fixed_profiles={
            "hybrid_rl_resac_ant": 5,
            "gpu_heavy_jax_matmul": 3,
            "gpu_cnn_jax_convstack": 3,
            "gpu_cnn_torch_resnet50": 3,
            "gpu_cnn_torch_progress_stack": 3,
            "gpu_llm_distilgpt2": 3,
            "gpu_llm_torch_decoder_stack": 3,
            "hybrid_rl_resac_ant_node007_tqdm": 3,
            "light_control_local": 1,
            "cpu_heavy_protocol": 32,
            "cpu_heavy_local_bench": 9,
            "freqduet_cpu_ablation_c17_32": 1,
            "freqduet_cpu_ablation_c3_8_completed_history": 1,
            "freqduet_spacectx_ep100_c3_8_completed_history": 1,
            "freqduet_cpu_ablation_c33_64_completed_history": 1,
            "freqduet_cpu_ablation_c65p_completed_history": 1,
            "freqduet_cpu_ablation_c9_16": 1,
            "freqduet_promoted_ep100_c65p_completed_history": 1,
            "freqduet_runner_v3_c_le2_completed_history": 1,
            "freqduet_runner_v3_c3_8_completed_history": 1,
            "freqduet_runner_v3_c17_32_completed_history": 1,
            "freqduet_paper_longtrain_c17_32_completed_history": 1,
            "sumo_eval_simple_sac_c_le2": 1,
            "transit_native_promotion_c17_32_seedrange_completed_history": 1,
            "transit_native_promotion_c17_32_residual_completed_history": 1,
            "transit_native_promotion_c33_64_batch_completed_history": 1,
            "transit_native_promotion_c33_64_single_seed_completed_history": 1,
            "transit_native_promotion_c65p_completed_history": 1,
            "transit_trading_sweep_c_le2_completed_history": 1,
            "transit_trading_policy_c_le2_completed_history": 1,
            "transit_surrogate_validation_c_le2_completed_history": 1,
            "transit_native_promotion_c_le2_completed_history": 1,
            "transit_native_control_c_le2_completed_history": 1,
            "transit_freqhrl_import_smoke_c_le2_completed_history": 1,
            "transit_freqhrl_tests_c_le2_completed_history": 1,
            "transit_native_promotion_c9_16_bounded_wait_completed_history": 1,
            "transit_native_promotion_c9_16_residual_completed_history": 1,
            "transit_native_promotion_c9_16_wait_credit_shell_completed_history": 1,
            "freqduet_runner_v3_c9_16_residual_completed_history": 1,
            "cfcmt_feed_conversion_c_le2_completed_history": 1,
            "cfcmt_env_validation_c_le2_completed_history": 1,
            "cfcmt_sumo_generation_c_le2_completed_history": 1,
            "cfcmt_snapshot_generation_c_le2_completed_history": 1,
            "cfcmt_snapshot_generation_c3_8_completed_history": 1,
            "cfcmt_pytest_sumo_c3_8_completed_history": 1,
            "cfcmt_traffic_signal_phase1_c3_8_completed_history": 1,
            "cfcmt_policy_rollout_c_le2_completed_history": 1,
            "cfcmt_traffic_signal_phase2_c_le2_completed_history": 1,
            "transit_trading_public_csv_c3_8_completed_history": 1,
            "transit_trading_pressure_merge_c3_8_completed_history": 1,
            "transit_trading_policy_c3_8_completed_history": 1,
            "transit_surrogate_c3_8_completed_history": 1,
            "transit_freqhrl_tests_c3_8_completed_history": 1,
            "transit_native_merge_c3_8_completed_history": 1,
            "transit_trading_pressure_matrix_c17_32_completed_history": 1,
            "transit_trading_promotion_recovery_c17_32_completed_history": 1,
            "transit_demand_estimator_c17_32_completed_history": 1,
            "transit_gap_closure_c17_32_completed_history": 1,
            "transit_trading_policy_c33_64_completed_history": 1,
            "transit_trading_pressure_matrix_c33_64_completed_history": 1,
            "transit_trading_pressure_matrix_c9_16_completed_history": 1,
            "bamor_train_compare_c9_16_completed_history": 1,
            "bamor_mujoco_c9_16_completed_history": 1,
            "bamor_diagnostic_shard_c9_16_completed_history": 1,
            "freqduet_cpu_ablation_c_le2_completed_history": 1,
            "freqduet_baseline_rule_c_le2_completed_history": 1,
            "freqduet_preflight_c_le2_completed_history": 1,
            "transit_freqhrl_analysis_matrix_c_le2_completed_history": 1,
            "transit_freqhrl_merge_c_le2_completed_history": 1,
            "bamor_train_compare_c_le2_completed_history": 1,
            "bamor_mujoco_c_le2_completed_history": 1,
            "bamor_diagnostic_shard_c_le2_completed_history": 1,
            "offline_sumo_eval_c_le2_completed_history": 1,
            "offline_sumo_eval_c33_64_completed_history": 1,
            "h2oplus_shell_eval_c_le2_completed_history": 1,
            "zsw_metrics_parser_c_le2_completed_history": 1,
            "resco_config_eval_c_le2_completed_history": 1,
            "nature_emissions_extract_c_le2_completed_history": 1,
            "nature_emissions_sumo_c_le2_completed_history": 1,
            "transit_native_promotion_c3_8_persistent_stress_completed_history": 1,
            "transit_native_real_demand_batch_c3_8_completed_history": 1,
            "transit_native_real_demand_alighting_c3_8_completed_history": 1,
            "bamor_mujoco_c17_32_completed_history": 1,
            "bamor_diagnostic_shard_c17_32_completed_history": 1,
            "freqduet_runner_v3_c33_64_completed_history": 1,
            "bamor_cpu_training_c3_8_completed_history": 1,
            "bamor_train_compare_c3_8_completed_history": 1,
            "bamor_mujoco_c3_8_completed_history": 1,
            "bamor_diagnostic_shard_c3_8_completed_history": 1,
            "bamor_mujoco_policy_union_c3_8_completed_history": 1,
            "assumption_agent_unittest_completed_history": 1,
            "assumption_agent_meta_qa_evolution_completed_history": 1,
            "assumption_agent_phase2_v20_framework_completed_history": 1,
            "assumption_agent_performance_validation_completed_history": 1,
            "assumption_agent_live_benchmark_completed_history": 1,
            "cfcmt_cpu_eval_completed_history": 1,
            "cfcmt_pytest_cpu_completed_history": 1,
            "sensing_voltage_cache_completed_history": 1,
            "sensing_pems_cache_completed_history": 1,
            "nature_emissions_routeguard_analysis_completed_history": 1,
            "scheduleurm_control_plane_completed_history": 1,
            "scheduleurm_hpc_relay_smoke_completed_history": 1,
            "bapr_id_ood_merge_cpu_eval_completed_history": 1,
            "resac_bus_seed_extension_cpu_eval_completed_history": 1,
            "h2oplus_snapshot_gpu_completed_history": 1,
            "transit_native_real_demand_batch_c9_16_completed_history": 1,
            "resac_conda_pack_completed_history": 1,
            "resac_review5_jax_train_fabric_completed_history": 1,
            "cpu_heavy_local_fabric_completed_history": 1,
            "hybrid_rl_resac_jmlr_project_fabric_completed_history": 1,
            "hybrid_rl_resac_project_fabric_completed_history": 1,
            "hybrid_rl_bapr_project_fabric_completed_history": 1,
            "hybrid_rl_bapr_v15_project_fabric_completed_history": 1,
            "hybrid_rl_cs_bapr_project_fabric_completed_history": 1,
            "hybrid_rl_sensing_v10k_project_fabric_completed_history": 1,
            "transit_native_real_demand_safe_wait_c9_16_profile_extension": 1,
            "zsw_tsp_sumo_eval_c_le2_completed_history": 1,
            "zsw_m21_sumo_eval_c3_8_completed_history": 1,
            "freqduet_runner_v3_allfreq_alllayers_c9_16": 1,
            "freqduet_snapshot_counterfactual_c9_16_completed_history": 1,
        },
    )


def calibrated_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_guarded_knee",
        calibrated=True,
        calibrated_objective="guarded_mean_flow",
        max_makespan_regret=0.02,
    )


def calibrated_delay_statewise_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_delay_statewise",
        calibrated=True,
        calibrated_objective="makespan",
        max_makespan_regret=0.02,
        statewise_regret_slack=0.6,
        statewise=True,
        guarded_resource_kinds=("gpu_heavy",),
        statewise_resource_kinds=("gpu_heavy",),
    )


def calibrated_backlog_aware_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_backlog_aware_guarded",
        calibrated=True,
        calibrated_objective="guarded_mean_flow",
        max_makespan_regret=0.05,
        min_makespan_regret=0.0,
        statewise_regret_slack=0.6,
        statewise=True,
        backlog_aware_guard=True,
        backlog_reference_tasks=64,
        statewise_service_dominance_guard=True,
        guarded_resource_kinds=("gpu_heavy", "gpu_cnn", "gpu_llm", "hybrid_rl", "cpu_heavy"),
        statewise_resource_kinds=("gpu_heavy",),
        statewise_workload_keys=(
            "gpu_cnn_torch_progress_stack",
            "hybrid_rl_resac_ant_node007_tqdm",
        ),
    )


def calibrated_makespan_policy() -> ReplayPolicy:
    return ReplayPolicy(
        name="calibrated_fast_forward_makespan",
        calibrated=True,
        calibrated_objective="makespan",
    )


def calibrated_candidate_policy(cache: ServiceRateCache, specs: list[WorkloadSpec]) -> ReplayPolicy:
    return calibrated_backlog_aware_policy()


def calibrated_scalar_candidate_policy(cache: ServiceRateCache, specs: list[WorkloadSpec]) -> ReplayPolicy:
    if len(specs) <= 1:
        return calibrated_policy()
    return calibrated_global_guarded_policy(cache, specs)


def calibrated_global_guarded_policy(
    cache: ServiceRateCache,
    specs: list[WorkloadSpec],
    *,
    max_global_makespan_regret: float = 0.02,
) -> ReplayPolicy:
    """Choose fixed profiles by global makespan guard plus weighted flow.

    This is the multi-workload replay analogue of the robust MaxWeight guard:
    first preserve the portfolio support/makespan objective, then spend the
    available slack on finite-batch delay.
    """

    rows_by_key = [_profile_rows(cache, spec) for spec in specs]
    if not rows_by_key:
        return calibrated_policy()
    best_global = min(max(row["makespan_s"] for row in combo) for combo in product(*rows_by_key))
    guard = best_global * (1.0 + max(0.0, float(max_global_makespan_regret)))
    admissible = [
        combo for combo in product(*rows_by_key)
        if max(row["makespan_s"] for row in combo) <= guard
    ]
    if not admissible:
        admissible = list(product(*rows_by_key))
    total_tasks = max(1, sum(max(0, int(spec.task_count)) for spec in specs))

    def combo_key(combo: tuple[dict, ...]) -> tuple[float, float, tuple[int, ...]]:
        weighted_flow = sum(
            row["mean_flow_s"] * max(0, int(row["task_count"]))
            for row in combo
        ) / float(total_tasks)
        global_makespan = max(row["makespan_s"] for row in combo)
        profiles = tuple(int(row["profile"]) for row in combo)
        return (global_makespan, weighted_flow, profiles)

    chosen = min(admissible, key=combo_key)
    return ReplayPolicy(
        name="calibrated_global_guarded",
        fixed_profiles={str(row["workload_key"]): int(row["profile"]) for row in chosen},
    )


def _profile_rows(cache: ServiceRateCache, spec: WorkloadSpec) -> list[dict]:
    records = cache.profiles(spec.workload_key)
    if not records:
        raise KeyError(f"no service cache entries for workload {spec.workload_key!r}")
    return [_profile_row(record, spec) for record in records]


def _profile_row(record: ProfileRecord, spec: WorkloadSpec) -> dict:
    return {
        "workload_key": spec.workload_key,
        "task_count": spec.task_count,
        "profile": record.profile,
        "makespan_s": deterministic_makespan_s(
            task_count=spec.task_count,
            total_units=spec.total_units,
            resource_count=spec.resource_count,
            profile=record.profile,
            aggregate_rate=record.aggregate_rate,
        ),
        "mean_flow_s": deterministic_mean_flow_s(
            task_count=spec.task_count,
            total_units=spec.total_units,
            resource_count=spec.resource_count,
            profile=record.profile,
            aggregate_rate=record.aggregate_rate,
        ),
    }


def _add_summary_dir(
    cache: ServiceRateCache,
    reports_dir: Path,
    *,
    workload_key: str,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str,
    force_replace: bool = False,
) -> None:
    paths = sorted(reports_dir.glob("profile_*_per_gpu_summary.json"))
    paths += sorted(reports_dir.glob("profile_*_per_resource_summary.json"))
    for path in paths:
        _add_summary_file(
            cache,
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
            force_replace=force_replace,
        )


def _add_summary_file(
    cache: ServiceRateCache,
    path: Path,
    *,
    workload_key: str,
    command_fingerprint: str,
    resource_kind: str,
    total_units: float,
    node_bucket: str,
    force_replace: bool = False,
) -> None:
    if not path.exists():
        return
    try:
        for record in records_from_summary_file(
            path,
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=total_units,
            node_bucket=node_bucket,
        ):
            cache.add(record, force_replace=force_replace)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _add_fresh_node007_eta_matrix(cache: ServiceRateCache) -> None:
    """Prefer task-native tqdm ETA probes collected on idle node007 GPUs.

    These rows intentionally override older node007 replay rows for the same
    workload/profile.  The old rows remain on disk for audit, but current replay
    should use the fresh co-location service rates because they are measured from
    each workload's own progress bar rather than legacy history ETA.
    """

    for workload_key in (
        "gpu_cnn_torch_resnet50",
        "gpu_llm_distilgpt2",
    ):
        cache.clear_capacity_boundaries(workload_key)

    fresh_runs = (
        (
            "node007_eta_matrix_fresh_20260626_cnn_gpus1_3_v2",
            "gpu_cnn_torch_progress_stack",
            "torch_cnn_progress_stack_bench_v1",
            "gpu_cnn",
            "node007-direct:3x12gb-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_cnn_gpus1_3_v2",
            "gpu_cnn_torch_resnet50",
            "torch_resnet50_train_b32_i224_amp_fresh_tqdm_v2",
            "gpu_cnn",
            "node007-direct:3x12gb-resnet50-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_cnn_gpus0_3_p5_8",
            "gpu_cnn_torch_progress_stack",
            "torch_cnn_progress_stack_bench_v1",
            "gpu_cnn",
            "node007-direct:4x12gb-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_cnn_gpus0_3_p5_8",
            "gpu_cnn_torch_resnet50",
            "torch_resnet50_train_b32_i224_amp_fresh_tqdm_v2",
            "gpu_cnn",
            "node007-direct:4x12gb-resnet50-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_llm_gpus1_3",
            "gpu_llm_torch_decoder_stack",
            "torch_decoder_stack_bench_v1",
            "gpu_llm",
            "node007-direct:3x12gb-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_llm_gpus0_3_p5_8",
            "gpu_llm_torch_decoder_stack",
            "torch_decoder_stack_bench_v1",
            "gpu_llm",
            "node007-direct:4x12gb-fresh-tqdm-20260626",
        ),
        (
            "jtl110gpu_fresh_20260626_llm_distilgpt2_direct_p1_11",
            "gpu_llm_distilgpt2",
            "torch_distilgpt2_forward_b2_s128_fp16_fresh_direct_tqdm_v2",
            "gpu_llm",
            "jtl110gpu:2x12gb-distilgpt2-fresh-direct-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_rl_gpus1_3",
            "hybrid_rl_resac_ant",
            "resac_ant_real_tqdm_stable_v1",
            "hybrid_rl",
            "node007-direct:3x12gb-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_rl_gpus0_3_p6_8",
            "hybrid_rl_resac_ant",
            "resac_ant_real_tqdm_stable_v1",
            "hybrid_rl",
            "node007-direct:4x12gb-fresh-tqdm-20260626-boundary",
        ),
        (
            "node007_eta_matrix_fresh_20260626_rl_gpus1_3",
            "hybrid_rl_resac_ant_node007_tqdm",
            "resac_ant_node007_tqdm_stable_v1",
            "hybrid_rl",
            "node007-direct:3x12gb-fresh-tqdm-20260626",
        ),
        (
            "node007_eta_matrix_fresh_20260626_rl_gpus0_3_p6_8",
            "hybrid_rl_resac_ant_node007_tqdm",
            "resac_ant_node007_tqdm_stable_v1",
            "hybrid_rl",
            "node007-direct:4x12gb-fresh-tqdm-20260626-boundary",
        ),
    )
    for run_name, workload_key, command_fingerprint, resource_kind, node_bucket in fresh_runs:
        _add_summary_dir(
            cache,
            RUN_ROOT / run_name / "reports",
            workload_key=workload_key,
            command_fingerprint=command_fingerprint,
            resource_kind=resource_kind,
            total_units=80,
            node_bucket=node_bucket,
            force_replace=True,
        )
