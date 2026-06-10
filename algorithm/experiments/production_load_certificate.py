"""Empirical production-load capacity certificate.

This module estimates arrival rates from Scheduleurm task records and checks
them against the measured service-action slice.  It separates two questions:

* measured-bucket capacity: can the mapped workload classes be supported?
* global production coverage: did every production task map to a measured
  service bucket?

The second question is intentionally strict.  A positive capacity LP on mapped
tasks is not a global theorem certificate when a large fraction of production
tasks are unmapped or only representative-mapped.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache
from simulation.tasksets import TaskSetMember, taskset_by_name

from .capacity_lp import solve_capacity_slack
from .empirical_slack_certificate import (
    _action_features,
    _action_id,
    _class_service,
    _finite_support_second_moment_bound,
    _profile_domain,
    _product_actions,
)


DEFAULT_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "production_freqduet_cpu_c17_32",
    "production_freqduet_cpu_ablation_c3_8_completed_history",
    "production_freqduet_runner_v3_c3_8_completed_history",
    "production_freqduet_cpu_ablation_c33_64_completed_history",
    "production_freqduet_runner_v3_c33_64_completed_history",
    "production_freqduet_cpu_ablation_c65p_completed_history",
    "production_freqduet_promoted_ep100_c65p_completed_history",
    "production_transit_native_promotion_c33_64_batch_completed_history",
    "production_transit_native_promotion_c33_64_single_seed_completed_history",
    "production_transit_native_promotion_c65p_completed_history",
    "production_transit_trading_sweep_c_le2_completed_history",
    "production_transit_trading_policy_c_le2_completed_history",
    "production_transit_surrogate_validation_c_le2_completed_history",
    "production_transit_native_promotion_c_le2_completed_history",
    "production_transit_native_control_c_le2_completed_history",
    "production_transit_freqhrl_import_smoke_c_le2_completed_history",
    "production_transit_native_promotion_c9_16_bounded_wait_completed_history",
    "production_transit_native_promotion_c9_16_residual_completed_history",
    "production_transit_native_promotion_c9_16_wait_credit_shell_completed_history",
    "production_freqduet_runner_v3_c9_16_residual_completed_history",
    "production_cfcmt_feed_conversion_c_le2_completed_history",
    "production_cfcmt_env_validation_c_le2_completed_history",
    "production_cfcmt_sumo_generation_c_le2_completed_history",
    "production_cfcmt_snapshot_generation_c_le2_completed_history",
    "production_cfcmt_snapshot_generation_c3_8_completed_history",
    "production_cfcmt_pytest_sumo_c3_8_completed_history",
    "production_cfcmt_traffic_signal_phase1_c3_8_completed_history",
    "production_cfcmt_policy_rollout_c_le2_completed_history",
    "production_cfcmt_traffic_signal_phase2_c_le2_completed_history",
    "production_bamor_train_compare_c9_16_completed_history",
    "production_bamor_mujoco_c9_16_completed_history",
    "production_bamor_diagnostic_shard_c9_16_completed_history",
    "production_freqduet_cpu_ablation_c_le2_completed_history",
    "production_freqduet_baseline_rule_c_le2_completed_history",
    "production_freqduet_preflight_c_le2_completed_history",
    "production_transit_freqhrl_analysis_matrix_c_le2_completed_history",
    "production_transit_freqhrl_merge_c_le2_completed_history",
    "production_bamor_train_compare_c_le2_completed_history",
    "production_bamor_mujoco_c_le2_completed_history",
    "production_bamor_diagnostic_shard_c_le2_completed_history",
    "production_offline_sumo_eval_c_le2_completed_history",
    "production_offline_sumo_eval_c33_64_completed_history",
    "production_h2oplus_shell_eval_c_le2_completed_history",
    "production_zsw_metrics_parser_c_le2_completed_history",
    "production_resco_config_eval_c_le2_completed_history",
    "production_nature_emissions_extract_c_le2_completed_history",
    "production_nature_emissions_sumo_c_le2_completed_history",
    "production_transit_native_promotion_c3_8_persistent_stress_completed_history",
    "production_transit_native_real_demand_batch_c3_8_completed_history",
    "production_transit_native_real_demand_alighting_c3_8_completed_history",
    "production_transit_trading_public_csv_c3_8_completed_history",
    "production_transit_trading_pressure_merge_c3_8_completed_history",
    "production_transit_trading_policy_c3_8_completed_history",
    "production_transit_surrogate_c3_8_completed_history",
    "production_transit_freqhrl_tests_c3_8_completed_history",
    "production_transit_native_merge_c3_8_completed_history",
    "production_transit_trading_pressure_matrix_c17_32_completed_history",
    "production_transit_trading_promotion_recovery_c17_32_completed_history",
    "production_transit_demand_estimator_c17_32_completed_history",
    "production_transit_gap_closure_c17_32_completed_history",
    "production_bamor_mujoco_c17_32_completed_history",
    "production_bamor_diagnostic_shard_c17_32_completed_history",
    "production_freqduet_cpu_ablation_c9_16",
    "production_freqduet_runner_v3_c_le2_completed_history",
    "production_bamor_train_compare_c3_8_completed_history",
    "production_bamor_mujoco_c3_8_completed_history",
    "production_bamor_diagnostic_shard_c3_8_completed_history",
    "production_zsw_tsp_sumo_eval_c_le2_completed_history",
    "production_zsw_m21_sumo_eval_c3_8_completed_history",
    "production_sumo_eval_simple_sac_c_le2",
    "production_transit_native_promotion_c17_32_seedrange_completed_history",
    "production_freqduet_runner_v3_c17_32_completed_history",
    "production_freqduet_paper_longtrain_c17_32_completed_history",
    "production_transit_native_promotion_c17_32_residual_completed_history",
    "production_freqduet_runner_v3_allfreq_alllayers_c9_16",
    "q11_cpu_gpu_coupled",
)


def build_production_load_certificate(
    *,
    records: Iterable[Mapping[str, Any]],
    window_days: float = 30.0,
    include_representative: bool = False,
    taskset_names: Iterable[str] = DEFAULT_TASKSETS,
    now_ts: float | None = None,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    window_records = [
        row for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
    ]
    members = _members_for_tasksets(taskset_names)
    member_by_key = {member.workload_key: member for member in members}
    classification_rows = []
    mapped_counts: Counter[str] = Counter()
    mapped_units: Counter[str] = Counter()
    representative_counts: Counter[str] = Counter()
    unmapped_reasons: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()

    for row in window_records:
        status_counts[str(row.get("status") or "")] += 1
        project_counts[str(row.get("project") or "")] += 1
        cls = classify_record(row, include_representative=include_representative)
        out = {
            "task_id": row.get("id") or row.get("task_id"),
            "status": row.get("status"),
            "project": row.get("project"),
            "signature": row.get("signature"),
            "submitted_at": _submitted_at(row),
            "classification": cls,
        }
        classification_rows.append(out)
        key = cls.get("workload_key")
        if key in member_by_key:
            units = _classified_units(cls, fallback=member_by_key[key].total_units)
            mapped_counts[key] += 1
            mapped_units[key] += units
            if cls.get("mapping_mode") != "strict_measured":
                representative_counts[key] += 1
        else:
            unmapped_reasons[str(cls.get("reason") or "unmapped")] += 1

    lam = {
        key: float(units) / window_s
        for key, units in sorted(mapped_units.items())
        if units > 0
    }
    cache = build_default_cache()
    domains = {member.workload_key: _profile_domain(cache, member) for member in members}
    full_action_count_estimate = _product_action_count(domains, members)
    if full_action_count_estimate > 100_000:
        actions = _dominating_product_actions(cache, tuple(members), domains)
        action_generation = "dominating_product_action_certificate"
    else:
        actions = _product_actions(cache, tuple(members), domains)
        action_generation = "full_product_actions"
    capacity = solve_capacity_slack(actions, lam)
    moment = {
        "B": _finite_support_second_moment_bound(actions, lam),
        "status": "finite_support_bound_from_measured_action_slice_and_empirical_load",
        "usable_for_theorem": bool(actions),
    }
    mapped_task_count = sum(mapped_counts.values())
    representative_task_count = sum(representative_counts.values())
    unmapped_task_count = len(window_records) - mapped_task_count
    mapped_capacity_usable = bool(capacity.get("usable_for_theorem"))
    global_coverage_usable = unmapped_task_count == 0 and representative_task_count == 0
    return {
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "window_days": float(window_days),
            "window_s": window_s,
        },
        "taskset_names": list(taskset_names),
        "include_representative": bool(include_representative),
        "record_count_total": len(materialized),
        "record_count_window": len(window_records),
        "mapped_task_count": mapped_task_count,
        "representative_mapped_task_count": representative_task_count,
        "unmapped_task_count": unmapped_task_count,
        "mapped_fraction": mapped_task_count / len(window_records) if window_records else 0.0,
        "strict_mapped_fraction": (
            (mapped_task_count - representative_task_count) / len(window_records)
            if window_records else 0.0
        ),
        "status_counts": dict(status_counts),
        "top_projects": dict(project_counts.most_common(20)),
        "mapped_counts": dict(mapped_counts),
        "mapped_units": dict(mapped_units),
        "lambda": lam,
        "capacity": capacity,
        "moment": moment,
        "action_profile_domains": domains,
        "full_action_count": full_action_count_estimate,
        "action_count_evaluated": len(actions),
        "action_generation": action_generation,
        "mapped_capacity_usable_for_theorem": mapped_capacity_usable,
        "global_coverage_usable_for_theorem": global_coverage_usable,
        "usable_for_global_theorem": mapped_capacity_usable and global_coverage_usable,
        "unmapped_reasons": dict(unmapped_reasons),
        "classification_sample": classification_rows[:200],
        "interpretation": (
            "Positive capacity on mapped measured buckets is a load certificate only "
            "for those buckets. Global production stability remains open unless "
            "unmapped and representative-mapped tasks are eliminated or separately "
            "certified by service measurements."
        ),
    }


def classify_record(
    row: Mapping[str, Any],
    *,
    include_representative: bool = False,
) -> dict[str, Any]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()
    est_vram = _as_float(row.get("est_vram_mb", row.get("vram_mb", 0.0)))
    cpu = _as_float(row.get("cpu_cores", 0.0))

    if "scheduleurmbench" in text:
        if any(token in text for token in ("q00", "light_control", "light-control")):
            return _mapped("light_control_local", "strict_measured", "scheduleurmbench_q00")
        if any(token in text for token in ("q01", "jax_matmul", "matmul", "gpu_heavy")):
            return _mapped("gpu_heavy_jax_matmul", "strict_measured", "scheduleurmbench_q01")
        if any(token in text for token in ("q10", "cpu_heavy", "cpu-heavy")):
            return _mapped("cpu_heavy_local_bench", "strict_measured", "scheduleurmbench_q10")
        if any(token in text for token in (
            "q11", "resac_ant", "resac_walker", "bapr_ant", "hybrid_rl",
        )):
            return _mapped("hybrid_rl_resac_ant", "strict_measured", "scheduleurmbench_q11")
        return {"workload_key": None, "mapping_mode": "unmapped", "reason": "scheduleurmbench_unknown"}

    if _is_freqduet_cpu_ablation_c17_32(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "freqduet_cpu_ablation_c17_32",
            "strict_measured",
            "module56_freqduet_cpu_ablation_c17_32",
        )

    c65p_units = _freqduet_ablation_c65p_units(row=row, est_vram=est_vram, cpu=cpu)
    if c65p_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c65p_completed_history",
            "strict_measured",
            "module69_freqduet_ablation_c65p_completed_history",
            units=c65p_units,
        )

    promoted_c65p_units = _freqduet_promoted_ep100_c65p_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if promoted_c65p_units is not None:
        return _mapped(
            "freqduet_promoted_ep100_c65p_completed_history",
            "strict_measured",
            "module69_freqduet_promoted_ep100_c65p_completed_history",
            units=promoted_c65p_units,
        )

    native_c65p_units = _native_promotion_c65p_seed_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if native_c65p_units is not None:
        return _mapped(
            "transit_native_promotion_c65p_completed_history",
            "strict_measured",
            "module69_transit_native_promotion_c65p_completed_history",
            units=native_c65p_units,
        )

    c33_64_units = _freqduet_ablation_c33_64_units(row=row, est_vram=est_vram, cpu=cpu)
    if c33_64_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c33_64_completed_history",
            "strict_measured",
            "module61_freqduet_cpu_ablation_c33_64_completed_history",
            units=c33_64_units,
        )

    native_c33_64_units = _native_promotion_c33_64_seed_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if native_c33_64_units is not None:
        if native_c33_64_units <= 1.0:
            return _mapped(
                "transit_native_promotion_c33_64_single_seed_completed_history",
                "strict_measured",
                "module68_transit_native_promotion_c33_64_single_seed_completed_history",
                units=native_c33_64_units,
            )
        return _mapped(
            "transit_native_promotion_c33_64_batch_completed_history",
            "strict_measured",
            "module68_transit_native_promotion_c33_64_batch_completed_history",
            units=native_c33_64_units,
        )

    runner_c33_64_units = _freqduet_runner_v3_c33_64_units(row=row, est_vram=est_vram, cpu=cpu)
    if runner_c33_64_units is not None:
        return _mapped(
            "freqduet_runner_v3_c33_64_completed_history",
            "strict_measured",
            "module81_freqduet_runner_v3_c33_64_completed_history",
            units=runner_c33_64_units,
        )

    offline_sumo_c33_64 = _offline_sumo_eval_c33_64_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if offline_sumo_c33_64 is not None:
        return _mapped(
            "offline_sumo_eval_c33_64_completed_history",
            "strict_measured",
            "module86_offline_sumo_eval_c33_64_completed_history",
            units=offline_sumo_c33_64,
        )

    native_c17_32_units = _native_promotion_c17_32_seedrange_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if native_c17_32_units is not None:
        return _mapped(
            "transit_native_promotion_c17_32_seedrange_completed_history",
            "strict_measured",
            "module63_transit_native_promotion_c17_32_seedrange_completed_history",
            units=native_c17_32_units,
        )

    runner_c17_32_units = _freqduet_runner_v3_c17_32_units(row=row, est_vram=est_vram, cpu=cpu)
    if runner_c17_32_units is not None:
        return _mapped(
            "freqduet_runner_v3_c17_32_completed_history",
            "strict_measured",
            "module74_freqduet_runner_v3_c17_32_completed_history",
            units=runner_c17_32_units,
        )

    longtrain_c17_32_units = _freqduet_paper_longtrain_c17_32_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if longtrain_c17_32_units is not None:
        return _mapped(
            "freqduet_paper_longtrain_c17_32_completed_history",
            "strict_measured",
            "module74_freqduet_paper_longtrain_c17_32_completed_history",
            units=longtrain_c17_32_units,
        )

    native_c17_32_residual_units = _native_promotion_c17_32_residual_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if native_c17_32_residual_units is not None:
        return _mapped(
            "transit_native_promotion_c17_32_residual_completed_history",
            "strict_measured",
            "module74_transit_native_promotion_c17_32_residual_completed_history",
            units=native_c17_32_residual_units,
        )

    transit_c17_32 = _transit_freqhrl_c17_32_residual_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if transit_c17_32 is not None:
        workload_key, units, reason = transit_c17_32
        return _mapped(workload_key, "strict_measured", reason, units=units)

    native_c9_16 = _native_promotion_c9_16_seed_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if native_c9_16 is not None:
        workload_key, units, reason = native_c9_16
        return _mapped(
            workload_key,
            "strict_measured",
            reason,
            units=units,
        )

    bamor_c_le2_script_units = _bamor_cpu_training_c_le2_script_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if bamor_c_le2_script_units is not None:
        script, units = bamor_c_le2_script_units
        if script == "train_compare_baselines.py":
            return _mapped(
                "bamor_train_compare_c_le2_completed_history",
                "strict_measured",
                "module77_bamor_train_compare_c_le2_completed_history",
                units=units,
            )
        if script == "train_bamor_mujoco.py":
            return _mapped(
                "bamor_mujoco_c_le2_completed_history",
                "strict_measured",
                "module77_bamor_mujoco_c_le2_completed_history",
                units=units,
            )
        if script == "run_bamor_diagnostic_shard.py":
            return _mapped(
                "bamor_diagnostic_shard_c_le2_completed_history",
                "strict_measured",
                "module77_bamor_diagnostic_shard_c_le2_completed_history",
                units=units,
            )

    bamor_script_units = _bamor_cpu_training_c3_8_script_units(row=row, est_vram=est_vram, cpu=cpu)
    if bamor_script_units is not None:
        script, units = bamor_script_units
        if script == "train_compare_baselines.py":
            return _mapped(
                "bamor_train_compare_c3_8_completed_history",
                "strict_measured",
                "module67_bamor_train_compare_c3_8_completed_history",
                units=units,
            )
        if script == "train_bamor_mujoco.py":
            return _mapped(
                "bamor_mujoco_c3_8_completed_history",
                "strict_measured",
                "module67_bamor_mujoco_c3_8_completed_history",
                units=units,
            )
        if script == "run_bamor_diagnostic_shard.py":
            return _mapped(
                "bamor_diagnostic_shard_c3_8_completed_history",
                "strict_measured",
                "module67_bamor_diagnostic_shard_c3_8_completed_history",
                units=units,
            )

    bamor_c9_16_script_units = _bamor_cpu_training_c9_16_script_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if bamor_c9_16_script_units is not None:
        script, units = bamor_c9_16_script_units
        if script == "train_compare_baselines.py":
            return _mapped(
                "bamor_train_compare_c9_16_completed_history",
                "strict_measured",
                "module75_bamor_train_compare_c9_16_completed_history",
                units=units,
            )
        if script == "train_bamor_mujoco.py":
            return _mapped(
                "bamor_mujoco_c9_16_completed_history",
                "strict_measured",
                "module75_bamor_mujoco_c9_16_completed_history",
                units=units,
            )
        if script == "run_bamor_diagnostic_shard.py":
            return _mapped(
                "bamor_diagnostic_shard_c9_16_completed_history",
                "strict_measured",
                "module75_bamor_diagnostic_shard_c9_16_completed_history",
                units=units,
            )

    bamor_c17_32_script_units = _bamor_cpu_training_c17_32_script_units(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
    )
    if bamor_c17_32_script_units is not None:
        script, units = bamor_c17_32_script_units
        if script == "train_bamor_mujoco.py":
            return _mapped(
                "bamor_mujoco_c17_32_completed_history",
                "strict_measured",
                "module80_bamor_mujoco_c17_32_completed_history",
                units=units,
            )
        if script == "run_bamor_diagnostic_shard.py":
            return _mapped(
                "bamor_diagnostic_shard_c17_32_completed_history",
                "strict_measured",
                "module80_bamor_diagnostic_shard_c17_32_completed_history",
                units=units,
            )

    bamor_c3_8_units = _bamor_cpu_training_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if bamor_c3_8_units is not None:
        return _mapped(
            "bamor_cpu_training_c3_8_completed_history",
            "strict_measured",
            "module64_bamor_cpu_training_c3_8_completed_history",
            units=bamor_c3_8_units,
        )

    c3_8_units = _freqduet_ablation_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if c3_8_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c3_8_completed_history",
            "strict_measured",
            "module60_freqduet_cpu_ablation_c3_8_completed_history",
            units=c3_8_units,
        )

    runner_c3_8_units = _freqduet_runner_v3_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if runner_c3_8_units is not None:
        return _mapped(
            "freqduet_runner_v3_c3_8_completed_history",
            "strict_measured",
            "module66_freqduet_runner_v3_c3_8_completed_history",
            units=runner_c3_8_units,
        )

    zsw_m21_c3_8_units = _zsw_m21_sumo_eval_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if zsw_m21_c3_8_units is not None:
        return _mapped(
            "zsw_m21_sumo_eval_c3_8_completed_history",
            "strict_measured",
            "module82_zsw_m21_sumo_eval_c3_8_completed_history",
            units=zsw_m21_c3_8_units,
        )

    cfcmt_c3_8 = _cfcmt_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if cfcmt_c3_8 is not None:
        workload_key, units, reason = cfcmt_c3_8
        return _mapped(workload_key, "strict_measured", reason, units=units)

    transit_c3_8 = _transit_freqhrl_c3_8_units(row=row, est_vram=est_vram, cpu=cpu)
    if transit_c3_8 is not None:
        workload_key, units, reason = transit_c3_8
        return _mapped(workload_key, "strict_measured", reason, units=units)

    c9_16_units = _freqduet_ablation_c9_16_units(row=row, est_vram=est_vram, cpu=cpu)
    if c9_16_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c9_16",
            "strict_measured",
            "module58_freqduet_cpu_ablation_c9_16",
            units=c9_16_units,
        )

    runner_c9_16_units = _freqduet_runner_v3_c9_16_residual_units(row=row, est_vram=est_vram, cpu=cpu)
    if runner_c9_16_units is not None:
        return _mapped(
            "freqduet_runner_v3_c9_16_residual_completed_history",
            "strict_measured",
            "module71_freqduet_runner_v3_c9_16_residual_completed_history",
            units=runner_c9_16_units,
        )

    c_le2_units = _freqduet_ablation_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if c_le2_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c_le2_completed_history",
            "strict_measured",
            "module76_freqduet_cpu_ablation_c_le2_completed_history",
            units=c_le2_units,
        )

    baseline_c_le2_units = _freqduet_baseline_rule_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if baseline_c_le2_units is not None:
        return _mapped(
            "freqduet_baseline_rule_c_le2_completed_history",
            "strict_measured",
            "module76_freqduet_baseline_rule_c_le2_completed_history",
            units=baseline_c_le2_units,
        )

    preflight_c_le2_units = _freqduet_preflight_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if preflight_c_le2_units is not None:
        return _mapped(
            "freqduet_preflight_c_le2_completed_history",
            "strict_measured",
            "module76_freqduet_preflight_c_le2_completed_history",
            units=preflight_c_le2_units,
        )

    runner_c_le2_units = _freqduet_runner_v3_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if runner_c_le2_units is not None:
        return _mapped(
            "freqduet_runner_v3_c_le2_completed_history",
            "strict_measured",
            "module62_freqduet_runner_v3_c_le2_completed_history",
            units=runner_c_le2_units,
        )

    zsw_sumo_c_le2_units = _zsw_tsp_sumo_eval_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if zsw_sumo_c_le2_units is not None:
        return _mapped(
            "zsw_tsp_sumo_eval_c_le2_completed_history",
            "strict_measured",
            "module65_zsw_tsp_sumo_eval_c_le2_completed_history",
            units=zsw_sumo_c_le2_units,
        )

    cfcmt_c_le2 = _cfcmt_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if cfcmt_c_le2 is not None:
        workload_key, units, reason = cfcmt_c_le2
        return _mapped(workload_key, "strict_measured", reason, units=units)

    transit_c_le2 = _transit_freqhrl_c_le2_units(row=row, est_vram=est_vram, cpu=cpu)
    if transit_c_le2 is not None:
        workload_key, units, reason = transit_c_le2
        return _mapped(workload_key, "strict_measured", reason, units=units)

    if _is_simple_sac_sumo_eval_c_le2(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "sumo_eval_simple_sac_c_le2",
            "strict_measured",
            "module59_simple_sac_sumo_eval_c_le2_completed_history",
            units=1.0,
        )

    sumo_c_le2 = _sumo_eval_c_le2_completed_history_units(row=row, est_vram=est_vram, cpu=cpu)
    if sumo_c_le2 is not None:
        workload_key, units, reason = sumo_c_le2
        return _mapped(workload_key, "strict_measured", reason, units=units)

    if _is_freqduet_runner_v3_allfreq_alllayers_c9_16(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "freqduet_runner_v3_allfreq_alllayers_c9_16",
            "strict_measured",
            "module57_freqduet_runner_v3_allfreq_alllayers_c9_16",
        )

    if include_representative:
        if est_vram > 0 and any(
            token in text for token in (
                "re-sac", "resac", "bapr", "jax_experiments.train",
                "sac_", "halfcheetah", "walker2d", "hopper", "ant-v2", "humanoid",
            )
        ):
            return _mapped("hybrid_rl_resac_ant", "representative", "gpu_rl_representative")
        if est_vram <= 0 and cpu >= 4 and any(
            token in text for token in (
                "analysis", "audit", "cpu_eval", "stage", "sweep", "preprocess",
            )
        ):
            return _mapped("cpu_heavy_local_bench", "representative", "cpu_heavy_representative")

    if est_vram > 0:
        return {"workload_key": None, "mapping_mode": "unmapped", "reason": "unmapped_gpu"}
    return {"workload_key": None, "mapping_mode": "unmapped", "reason": "unmapped_cpu"}


def load_scheduler_records(
    *,
    queue_path: str | Path | None = None,
    archive_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = Path(queue_path).expanduser() if queue_path else state_dir / "queue.json"
    archive = Path(archive_path).expanduser() if archive_path else state_dir / "queue_archive.jsonl"
    rows: list[dict[str, Any]] = []
    if archive.exists():
        with archive.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if queue.exists():
        payload = json.loads(queue.read_text(encoding="utf-8"))
        tasks = payload.get("tasks", payload if isinstance(payload, list) else [])
        rows.extend(dict(row) for row in tasks)
    return _dedupe_records(rows)


def _members_for_tasksets(taskset_names: Iterable[str]) -> tuple[TaskSetMember, ...]:
    seen: set[str] = set()
    members: list[TaskSetMember] = []
    for name in taskset_names:
        for member in taskset_by_name(str(name)).members:
            if member.workload_key in seen:
                continue
            seen.add(member.workload_key)
            members.append(member)
    return tuple(members)


def _mapped(workload_key: str, mode: str, reason: str, *, units: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"workload_key": workload_key, "mapping_mode": mode, "reason": reason}
    if units is not None and math.isfinite(float(units)) and float(units) > 0:
        out["units"] = float(units)
    return out


def _classified_units(cls: Mapping[str, Any], *, fallback: float) -> float:
    raw = cls.get("units")
    try:
        units = float(raw)
        if math.isfinite(units) and units > 0:
            return units
    except (TypeError, ValueError):
        pass
    return float(fallback)


def _product_action_count(
    domains: Mapping[str, list[int]],
    members: Iterable[TaskSetMember],
) -> int:
    out = 1
    for member in members:
        out *= max(1, len(domains.get(member.workload_key, [])))
    return int(out)


def _dominating_product_actions(
    cache: Any,
    members: tuple[TaskSetMember, ...],
    domains: Mapping[str, list[int]],
) -> list[dict[str, Any]]:
    profiles: dict[str, int] = {}
    service: dict[str, float] = {}
    for member in members:
        choices = sorted(int(profile) for profile in domains[member.workload_key])
        best_profile = max(
            choices,
            key=lambda profile: _class_service(cache, member, profile),
        )
        profiles[member.workload_key] = best_profile
        service[member.workload_key] = _class_service(cache, member, best_profile)
    action_id = _action_id(profiles)
    return [
        {
            "action_id": action_id,
            "profiles": profiles,
            "features": _action_features(action_id, profiles, members),
            "service_vector": service,
            "lower_service": service,
            "penalty_units": 0.0,
        }
    ]


def _is_freqduet_cpu_ablation_c17_32(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> bool:
    if est_vram > 0:
        return False
    if not (16.0 < float(cpu) <= 32.0):
        return False
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "").lower()
    if (
        project == "bamor"
        or "/bamor" in cwd
        or "run_bamor_diagnostic_shard.py" in cmd
        or "train_compare_baselines.py" in cmd
    ):
        return False
    return "run_freqduet_ablation.py" in cmd


def _freqduet_ablation_c33_64_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (32.0 < float(cpu) <= 64.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_ablation_c65p_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) <= 64.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    worker_threads = _shell_option(cmd, "--worker-threads")
    if worker_threads is not None and worker_threads != "1":
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_promoted_ep100_c65p_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) <= 64.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_promoted_ep100_hpc_batch.sh" not in cmd_lower:
        return None
    if "freqduet" not in " ".join((project, cwd, signature, cmd_lower)):
        return None
    units = _parse_freqduet_promoted_ep100_units(cmd)
    return units if units is not None and units > 0 else None


def _parse_freqduet_promoted_ep100_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    if not any(str(token).endswith("run_freqduet_promoted_ep100_hpc_batch.sh") for token in tokens):
        return None
    episodes = 100
    for token in tokens:
        value = str(token)
        if value.startswith("EPISODES="):
            try:
                episodes = int(value.split("=", 1)[1])
            except ValueError:
                return None
    if episodes != 100:
        return None

    start_raw = _shell_option_from_tokens(tokens, "--job-start")
    end_raw = _shell_option_from_tokens(tokens, "--job-end")
    if start_raw is None or end_raw is None:
        return None
    try:
        job_start = int(start_raw)
        job_end = int(end_raw)
    except ValueError:
        return None
    job_count = max(0, job_end - job_start)
    return float(job_count * episodes) if job_count > 0 else None


def _native_promotion_c17_32_seedrange_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (16.0 < float(cpu) <= 32.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "native_promotion_replan_validation" not in cmd_lower:
        return None
    units = _parse_native_promotion_seedrange_units(cmd)
    return units if units is not None and units > 0 else None


def _native_promotion_c17_32_residual_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (16.0 < float(cpu) <= 32.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "native_promotion_replan_validation" not in cmd_lower:
        return None
    if _parse_native_promotion_seedrange_units(cmd) is not None:
        return None
    units = _parse_native_promotion_seed_units(cmd)
    return units if units is not None and units > 0 else None


def _transit_freqhrl_c17_32_residual_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if not (16.0 < float(cpu) <= 32.0):
        return None
    text = " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()
    if "bamor" in text:
        return None
    if not any(token in text for token in ("transitduet", "freq_hrl", "freqhrl", "transit_hrl")):
        return None
    cmd = str(row.get("cmd") or "")
    tokens = _shlex_tokens(cmd)
    module = _python_module_from_tokens(tokens)
    if not module:
        return None

    if module.endswith("trading.pressure_test_matrix"):
        scenario_count = _transit_count_option(tokens, "--scenarios", 6)
        baseline_count = _transit_count_option(tokens, "--baselines", 12)
        units = _transit_seed_step_asset_units(tokens) * scenario_count * baseline_count
        return (
            "transit_trading_pressure_matrix_c17_32_completed_history",
            float(units),
            "module84_transit_trading_pressure_matrix_c17_32_completed_history",
        )
    if module.endswith("trading.promotion_recovery_validation"):
        return (
            "transit_trading_promotion_recovery_c17_32_completed_history",
            1.0,
            "module84_transit_trading_promotion_recovery_c17_32_completed_history",
        )
    if module.endswith("transit.demand_estimator_validation"):
        seeds = _transit_seed_count(tokens, "--seeds", 1)
        steps = _transit_int_option(tokens, "--steps", 720)
        units = float(seeds * steps)
        return (
            "transit_demand_estimator_c17_32_completed_history",
            units,
            "module84_transit_demand_estimator_c17_32_completed_history",
        )
    if module.endswith("transit.gap_closure_validation"):
        units = 4.0 * _transit_surrogate_train_eval_units(tokens, default_iterations=5)
        return (
            "transit_gap_closure_c17_32_completed_history",
            float(units),
            "module84_transit_gap_closure_c17_32_completed_history",
        )
    return None


def _native_promotion_c9_16_seed_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if not (8.0 < float(cpu) <= 16.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "native_promotion_replan_validation" not in cmd_lower:
        return None
    shell_units = _parse_native_promotion_shell_seedrange_units(cmd)
    if shell_units is not None and shell_units > 0 and "wait_credit_aligned_v39" in cmd_lower:
        return (
            "transit_native_promotion_c9_16_wait_credit_shell_completed_history",
            float(shell_units),
            "module85_transit_native_promotion_c9_16_wait_credit_shell_completed_history",
        )
    units = _parse_native_promotion_seed_units(cmd)
    if units is None or units <= 0:
        return None
    if "bounded_wait_nofinal_v14" in cmd_lower:
        return (
            "transit_native_promotion_c9_16_bounded_wait_completed_history",
            float(units),
            "module71_transit_native_promotion_c9_16_bounded_wait_completed_history",
        )
    return (
        "transit_native_promotion_c9_16_residual_completed_history",
        float(units),
        "module71_transit_native_promotion_c9_16_residual_completed_history",
    )


def _parse_native_promotion_seedrange_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    if not any("native_promotion_replan_validation" in str(token) for token in tokens):
        return None

    def opt(name: str) -> str | None:
        if name not in tokens:
            return None
        idx = tokens.index(name)
        return tokens[idx + 1] if idx + 1 < len(tokens) else None

    start_raw = opt("--seed-index-start")
    end_raw = opt("--seed-index-end")
    if start_raw is None or end_raw is None:
        return None
    episodes_raw = opt("--episodes") or "1"
    try:
        seed_start = int(start_raw)
        seed_end = int(end_raw)
        episodes = float(episodes_raw)
    except ValueError:
        return None
    seed_count = max(0, seed_end - seed_start)
    if seed_count <= 0 or not math.isfinite(episodes) or episodes <= 0:
        return None
    return float(seed_count) * float(episodes)


def _parse_native_promotion_shell_seedrange_units(cmd: str) -> float | None:
    import re

    if "native_promotion_replan_validation" not in str(cmd):
        return None
    assignments: dict[str, int] = {}
    for name, left, right in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)=\$\(\(\s*(-?\d+)\s*\+\s*(-?\d+)\s*\)\)",
        str(cmd),
    ):
        assignments[str(name)] = int(left) + int(right)
    if not assignments:
        return None
    start_match = re.search(
        r"--seed-index-start\s+[\"']?\$([A-Za-z_][A-Za-z0-9_]*)[\"']?",
        str(cmd),
    )
    end_match = re.search(
        r"--seed-index-end\s+[\"']?\$([A-Za-z_][A-Za-z0-9_]*)[\"']?",
        str(cmd),
    )
    if start_match is None or end_match is None:
        return None
    start = assignments.get(start_match.group(1))
    end = assignments.get(end_match.group(1))
    if start is None or end is None:
        return None
    episode_match = re.search(r"--episodes\s+([0-9]+(?:\.[0-9]+)?)", str(cmd))
    episodes = 1.0
    if episode_match is not None:
        episodes = float(episode_match.group(1))
    seed_count = max(0, int(end) - int(start))
    if seed_count <= 0 or not math.isfinite(episodes) or episodes <= 0:
        return None
    return float(seed_count) * float(episodes)


def _native_promotion_c33_64_seed_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (32.0 < float(cpu) <= 64.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "native_promotion_replan_validation" not in cmd_lower:
        return None
    units = _parse_native_promotion_seed_units(cmd)
    return units if units is not None and units > 0 else None


def _native_promotion_c65p_seed_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) <= 64.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "native_promotion_replan_validation" not in cmd_lower:
        return None
    units = _parse_native_promotion_seed_units(cmd)
    return units if units is not None and units > 0 else None


def _parse_native_promotion_seed_units(cmd: str) -> float | None:
    units = _parse_native_promotion_seedrange_units(cmd)
    if units is not None and units > 0:
        return units
    units = _parse_native_promotion_cli_seedlist_units(cmd)
    if units is not None and units > 0:
        return units
    for snippet in _python_c_snippets(cmd):
        units = _parse_native_promotion_python_seed_units(snippet)
        if units is not None and units > 0:
            return units
        seed_count = _parse_native_promotion_python_seed_count(snippet)
        if seed_count is not None and seed_count > 0:
            episodes = _native_promotion_cli_episodes(cmd)
            if episodes is None:
                return None
            return float(seed_count) * float(episodes)
    return None


def _parse_native_promotion_cli_seedlist_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    if not any("native_promotion_replan_validation" in str(token) for token in tokens):
        return None
    if "--seeds" not in tokens:
        return None
    idx = tokens.index("--seeds") + 1
    seeds = []
    while idx < len(tokens) and not str(tokens[idx]).startswith("--"):
        seeds.append(str(tokens[idx]))
        idx += 1
    if not seeds:
        return None
    if any("$" in seed for seed in seeds):
        return None
    episodes = 1.0
    if "--episodes" in tokens:
        ep_idx = tokens.index("--episodes")
        if ep_idx + 1 >= len(tokens):
            return None
        try:
            episodes = float(tokens[ep_idx + 1])
        except ValueError:
            return None
    if not math.isfinite(episodes) or episodes <= 0:
        return None
    return float(len(seeds)) * float(episodes)


def _native_promotion_cli_episodes(cmd: str) -> float | None:
    raw = _shell_option(cmd, "--episodes")
    if raw is None:
        return 1.0
    try:
        episodes = float(raw)
    except ValueError:
        return None
    return episodes if math.isfinite(episodes) and episodes > 0 else None


def _shell_option(cmd: str, name: str) -> str | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    return _shell_option_from_tokens(tokens, name)


def _shell_option_from_tokens(tokens: list[str], name: str) -> str | None:
    if name not in tokens:
        return None
    idx = tokens.index(name)
    return str(tokens[idx + 1]) if idx + 1 < len(tokens) else None


def _python_c_snippets(cmd: str) -> list[str]:
    import shlex

    out: list[str] = []

    def visit(text: str, depth: int) -> None:
        if depth > 2:
            return
        try:
            tokens = shlex.split(str(text))
        except ValueError:
            tokens = str(text).split()
        for idx, token in enumerate(tokens):
            value = str(token)
            if value == "-c" and idx + 1 < len(tokens):
                out.append(str(tokens[idx + 1]))
            if value in {"-lc", "-ic"} and idx + 1 < len(tokens):
                visit(str(tokens[idx + 1]), depth + 1)

    visit(str(cmd), 0)
    return out


def _parse_native_promotion_python_seed_units(snippet: str) -> float | None:
    import ast

    tree = _parse_python_ast_relaxed(snippet)
    if tree is None:
        return None
    seed_counts: dict[str, int] = {}
    numeric_values: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = str(node.targets[0].id)
        count = _ast_seed_count(node.value)
        if count is not None:
            seed_counts[name] = count
            continue
        number = _ast_constant_float(node.value)
        if number is not None:
            numeric_values[name] = number

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "run_validation":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        seed_node = keywords.get("seeds")
        if seed_node is None:
            return None
        seed_count = _ast_seed_count(seed_node, seed_counts)
        if seed_count is None and isinstance(seed_node, ast.Name):
            seed_count = seed_counts.get(str(seed_node.id))
        episodes = 1.0
        episode_node = keywords.get("episodes")
        if episode_node is not None:
            episodes = _ast_constant_float(episode_node)
            if episodes is None and isinstance(episode_node, ast.Name):
                episodes = numeric_values.get(str(episode_node.id))
        if seed_count is None or episodes is None:
            return None
        if seed_count <= 0 or not math.isfinite(float(episodes)) or float(episodes) <= 0:
            return None
        return float(seed_count) * float(episodes)
    return None


def _parse_native_promotion_python_seed_count(snippet: str) -> int | None:
    import ast

    tree = _parse_python_ast_relaxed(snippet)
    if tree is None:
        return None
    seed_counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        count = _ast_seed_count(node.value)
        if count is not None and count > 0:
            seed_counts[str(node.targets[0].id)] = count
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            continue
        count = _ast_seed_count(node, seed_counts)
        if count is not None and count > 0:
            return count
    for node in ast.walk(tree):
        count = _ast_seed_count(node, seed_counts)
        if count is not None and count > 0:
            return count
    return None


def _parse_python_ast_relaxed(snippet: str) -> Any | None:
    import ast

    candidate = str(snippet).strip()
    for _ in range(8):
        try:
            return ast.parse(candidate)
        except SyntaxError:
            stripped = candidate.rstrip().rstrip(";").rstrip()
            if stripped.endswith(")") and stripped.count(")") > stripped.count("("):
                candidate = stripped[:-1].rstrip()
                continue
            return None
    return None


def _ast_seed_count(node: Any, sequence_counts: Mapping[str, int] | None = None) -> int | None:
    import ast

    counts = dict(sequence_counts or {})
    if isinstance(node, ast.Name):
        return counts.get(str(node.id))
    if isinstance(node, (ast.List, ast.Tuple)):
        return len(node.elts)
    if isinstance(node, ast.Subscript):
        total = None
        if isinstance(node.value, ast.Name):
            total = counts.get(str(node.value.id))
        if isinstance(node.slice, ast.Slice):
            return _ast_slice_length(node.slice, total=total)
        return 1 if total is not None else None
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)) and len(node.generators) == 1:
        return _ast_range_length(node.generators[0].iter)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
    ):
        return _ast_range_length(node.args[0]) or _ast_seed_count(node.args[0], counts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "map"
        and len(node.args) >= 2
    ):
        return _ast_seed_count(node.args[1], counts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        if len(node.args) == 1:
            return _ast_seed_count(node.args[0], counts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
        if len(node.args) == 1:
            return _ast_seed_count(node.args[0], counts)
    return None


def _ast_slice_length(node: Any, *, total: int | None) -> int | None:
    import ast

    if not isinstance(node, ast.Slice):
        return None

    def bound(value: Any, default: int | None) -> int | None:
        if value is None:
            return default
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return int(value.value)
        if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            inner = bound(value.operand, None)
            return -inner if inner is not None else None
        return None

    start = bound(node.lower, 0)
    stop = bound(node.upper, total)
    step = bound(node.step, 1)
    if start is None or stop is None or step is None or step == 0:
        return None
    return max(0, len(range(start, stop, step)))


def _ast_range_length(node: Any) -> int | None:
    import ast

    if not isinstance(node, ast.Call):
        return None
    if not isinstance(node.func, ast.Name) or node.func.id != "range":
        return None
    raw = [_ast_constant_int(arg) for arg in node.args]
    if any(value is None for value in raw) or len(raw) not in (1, 2, 3):
        return None
    values = [int(value) for value in raw if value is not None]
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    elif len(values) == 2:
        start, stop, step = values[0], values[1], 1
    else:
        start, stop, step = values[0], values[1], values[2]
    if step == 0:
        return None
    return len(range(start, stop, step))


def _ast_constant_int(node: Any) -> int | None:
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_constant_int(node.operand)
        return -value if value is not None else None
    return None


def _ast_constant_float(node: Any) -> float | None:
    import ast

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_constant_float(node.operand)
        return -value if value is not None else None
    return None


def _bamor_cpu_training_c3_8_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    parsed = _bamor_cpu_training_c3_8_script_units(row=row, est_vram=est_vram, cpu=cpu)
    if parsed is None:
        return None
    _script, units = parsed
    return units if units > 0 else None


def _bamor_cpu_training_c3_8_script_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float] | None:
    return _bamor_cpu_training_script_units_in_cpu_range(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
        lower_exclusive=2.0,
        upper_inclusive=8.0,
    )


def _bamor_cpu_training_c_le2_script_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float] | None:
    return _bamor_cpu_training_script_units_in_cpu_range(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
        lower_exclusive=-1.0,
        upper_inclusive=2.0,
    )


def _bamor_cpu_training_c9_16_script_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float] | None:
    return _bamor_cpu_training_script_units_in_cpu_range(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
        lower_exclusive=8.0,
        upper_inclusive=16.0,
    )


def _bamor_cpu_training_c17_32_script_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float] | None:
    return _bamor_cpu_training_script_units_in_cpu_range(
        row=row,
        est_vram=est_vram,
        cpu=cpu,
        lower_exclusive=16.0,
        upper_inclusive=32.0,
    )


def _bamor_cpu_training_script_units_in_cpu_range(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
    lower_exclusive: float,
    upper_inclusive: float,
) -> tuple[str, float] | None:
    if est_vram > 0:
        return None
    if not (float(lower_exclusive) < float(cpu) <= float(upper_inclusive)):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    if project != "bamor" and "/bamor" not in cwd and not signature.startswith("bamor/"):
        return None
    script = _python_script_basename(cmd)
    if script not in {
        "train_compare_baselines.py",
        "train_bamor_mujoco.py",
        "run_bamor_diagnostic_shard.py",
    }:
        return None
    units = _parse_bamor_training_step_units(cmd)
    if units is None or units <= 0:
        return None
    return script, float(units)


def _python_script_basename(cmd: str) -> str:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    for token in tokens:
        value = str(token)
        if value.endswith(".py"):
            return value.rsplit("/", 1)[-1]
    return ""


def _parse_bamor_training_step_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    script = next((str(token) for token in tokens if str(token).endswith(".py")), "")
    if not script:
        return None

    def opt(*names: str) -> str | None:
        for name in names:
            if name in tokens:
                idx = tokens.index(name)
                return tokens[idx + 1] if idx + 1 < len(tokens) else None
        return None

    def int_at_least(raw: str | None, minimum: int) -> int | None:
        try:
            value = int(str(raw))
        except (TypeError, ValueError):
            return None
        return value if value >= minimum else None

    def explicit_method_count(raw: str | None) -> int | None:
        if raw is None:
            return None
        methods = [part for part in str(raw).split() if part.strip()]
        if not methods or any(part == "all" for part in methods):
            return None
        return len(methods)

    if script.endswith("train_compare_baselines.py"):
        method_count = explicit_method_count(opt("--method"))
        total_steps = int_at_least(opt("--total_steps", "--total-steps"), 1)
        seeds = int_at_least(opt("--seeds"), 1)
        if method_count is None or total_steps is None or seeds is None:
            return None
        return float(method_count * seeds * total_steps)

    if script.endswith("train_bamor_mujoco.py"):
        method_count = explicit_method_count(opt("--method"))
        total_steps = int_at_least(opt("--total_steps", "--total-steps"), 1)
        num_seeds = int_at_least(opt("--num_seeds", "--num-seeds") or "1", 1)
        if method_count is None or total_steps is None or num_seeds is None:
            return None
        return float(method_count * num_seeds * total_steps)

    if script.endswith("run_bamor_diagnostic_shard.py"):
        start = int_at_least(opt("--start"), 0)
        end = int_at_least(opt("--end"), 0)
        item_offset = int_at_least(opt("--item-offset") or "0", 0)
        methods_raw = opt("--methods")
        method_count = explicit_method_count(methods_raw)
        seeds = int_at_least(opt("--seeds"), 1)
        total_steps = int_at_least(opt("--total-steps", "--total_steps"), 1)
        if (
            start is None
            or end is None
            or item_offset is None
            or method_count is None
            or seeds is None
            or total_steps is None
            or end <= start
        ):
            return None
        global_start = item_offset + start
        global_end = item_offset + end
        if global_start < 0 or global_end > method_count * seeds:
            return None
        return float((end - start) * total_steps)

    return None


def _freqduet_ablation_c3_8_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (2.0 < float(cpu) <= 8.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_ablation_c9_16_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (8.0 < float(cpu) <= 16.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    if "--worker-threads" in cmd_lower and "--worker-threads 1" not in cmd_lower:
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_ablation_c_le2_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_baseline_rule_c_le2_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, cmd_lower))
    if "bamor" in text:
        return None
    if "run_baseline_rule.py" not in cmd_lower:
        return None
    if "freqduet" not in text:
        return None
    episodes = _positive_float_option(_shlex_tokens(cmd), "--episodes", default=0.0)
    return episodes if episodes > 0 else None


def _freqduet_preflight_c_le2_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if "bamor" in text:
        return None
    if "freqduet" not in text:
        return None
    if "freqduet_autoadopt_spin.py" in cmd_lower:
        return None
    preflight_tokens = (
        "preflight",
        "env-smoke",
        "hpc-cpu-probe",
        "py_compile",
        "import_check",
        "import sys",
        "deps ok",
    )
    return 1.0 if any(token in text for token in preflight_tokens) else None


def _parse_freqduet_ablation_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()

    def opt(name: str) -> str | None:
        if name not in tokens:
            return None
        idx = tokens.index(name)
        return tokens[idx + 1] if idx + 1 < len(tokens) else None

    episodes_raw = opt("--episodes")
    try:
        episodes = int(episodes_raw) if episodes_raw is not None else 0
    except ValueError:
        episodes = 0
    if episodes <= 0:
        return None

    job_start_raw = opt("--job-start")
    job_end_raw = opt("--job-end")
    if job_start_raw is not None or job_end_raw is not None:
        try:
            job_start = int(job_start_raw) if job_start_raw is not None else 0
            job_end = int(job_end_raw) if job_end_raw is not None else 0
        except ValueError:
            return None
        jobs = max(0, job_end - job_start)
        return float(jobs * episodes) if jobs > 0 else None

    configs_raw = opt("--configs") or ""
    seeds_raw = opt("--seeds") or ""
    if "$" in configs_raw or "$" in seeds_raw:
        return None
    configs = [part for part in configs_raw.split(",") if part.strip()]
    seeds = [part for part in seeds_raw.split(",") if part.strip()]
    jobs = len(configs) * len(seeds)
    return float(jobs * episodes) if jobs > 0 else None


def _freqduet_runner_v3_c_le2_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "runner_v3.py" not in cmd_lower:
        return None
    return _parse_runner_v3_episode_units(cmd)


def _freqduet_runner_v3_c3_8_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (2.0 < float(cpu) <= 8.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "runner_v3.py" not in cmd_lower:
        return None
    if "run_freqduet_ablation.py" in cmd_lower:
        return None
    if "freqduet" not in text and "/transitduet/freqduet/" not in cwd:
        return None
    units = _parse_runner_v3_episode_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_runner_v3_c17_32_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (16.0 < float(cpu) <= 32.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "runner_v3.py" not in cmd_lower:
        return None
    if "run_freqduet_ablation.py" in cmd_lower:
        return None
    if "freqduet" not in text and "/transitduet/freqduet/" not in cwd:
        return None
    units = _parse_runner_v3_episode_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_runner_v3_c33_64_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (32.0 < float(cpu) <= 64.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "runner_v3.py" not in cmd_lower:
        return None
    if "run_freqduet_ablation.py" in cmd_lower:
        return None
    if "freqduet" not in text and "/transitduet/freqduet/" not in cwd:
        return None
    units = _parse_runner_v3_episode_units(cmd)
    return units if units is not None and units > 0 else None


def _freqduet_paper_longtrain_c17_32_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (16.0 < float(cpu) <= 32.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_paper_longtrain_matrix.sh" not in cmd_lower:
        return None
    if "freqduet" not in text and "/transitduet/freqduet/" not in cwd:
        return None
    # The production command used --skip-existing, so job-count times episodes
    # would overstate completed work when prior results are reused.
    return 1.0


def _freqduet_runner_v3_c9_16_residual_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (8.0 < float(cpu) <= 16.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    description = str(row.get("description") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, description, cmd_lower))
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "runner_v3.py" not in cmd_lower:
        return None
    if "configs_freqduet/f_allfreq_alllayers_hiro.yaml" in cmd_lower:
        return None
    if "freqduet" not in text and "/transitduet/freqduet/" not in cwd:
        return None
    units = _parse_runner_v3_episode_units(cmd)
    return units if units is not None and units > 0 else None


def _parse_runner_v3_episode_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    if not any(str(token).endswith("runner_v3.py") for token in tokens):
        return None
    if "--episodes" not in tokens:
        return None
    idx = tokens.index("--episodes")
    if idx + 1 >= len(tokens):
        return None
    try:
        episodes = float(tokens[idx + 1])
    except ValueError:
        return None
    return episodes if math.isfinite(episodes) and episodes > 0 else None


def _cfcmt_c_le2_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project != "cfcmt" and "/cfcmt" not in cwd and not signature.startswith("cfcmt/"):
        return None
    parsed = _parse_cfcmt_c_le2_units(cmd)
    if parsed is None:
        return None
    workload_key, units = parsed
    return workload_key, units, f"module72_{workload_key}"


def _cfcmt_c3_8_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if not (2.0 < float(cpu) <= 8.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project != "cfcmt" and "/cfcmt" not in cwd and not signature.startswith("cfcmt/"):
        return None

    if "pytest" in cmd_lower and any(
        token in cmd_lower
        for token in (
            "test_sumo_apc_avl_sumo_generation.py",
            "test_sumo_apc_avl_snapshot_generation.py",
            "test_sumo_policy_rollout_validation.py",
            "test_traffic_signal_sumo_phase1.py",
            "test_traffic_signal_sumo_phase2.py",
            "test_traffic_signal_resco_probe.py",
            "test_traffic_signal_resco_phase_benchmark.py",
            "test_traffic_signal_resco_cfcmt_benchmark.py",
            "test_traffic_signal_transfer_feasibility.py",
        )
    ):
        return (
            "cfcmt_pytest_sumo_c3_8_completed_history",
            1.0,
            "module82_cfcmt_pytest_sumo_c3_8_completed_history",
        )

    if (
        "cf_h2o.eval.sumo_apc_avl_snapshot_generation" in cmd_lower
        or "sumo_apc_avl_snapshot_generation.py" in cmd_lower
    ):
        parsed = _parse_cfcmt_c_le2_units(cmd)
        if parsed is not None:
            workload_key, units = parsed
            if workload_key == "cfcmt_snapshot_generation_c_le2_completed_history":
                return (
                    "cfcmt_snapshot_generation_c3_8_completed_history",
                    units,
                    "module82_cfcmt_snapshot_generation_c3_8_completed_history",
                )

    if "traffic_signal_sumo_phase1.py" in cmd_lower:
        return (
            "cfcmt_traffic_signal_phase1_c3_8_completed_history",
            1.0,
            "module82_cfcmt_traffic_signal_phase1_c3_8_completed_history",
        )

    return None


def _parse_cfcmt_c_le2_units(cmd: str) -> tuple[str, float] | None:
    tokens = _shlex_tokens(cmd)
    cmd_lower = str(cmd).lower()
    if "gtfs_to_h2o_xlsx.py" in cmd_lower or "lta_to_h2o_xlsx.py" in cmd_lower:
        return "cfcmt_feed_conversion_c_le2_completed_history", 1.0
    if "validate_h2o_city_env.py" in cmd_lower:
        max_steps = _positive_float_option(tokens, "--max-steps", default=1.0)
        return "cfcmt_env_validation_c_le2_completed_history", max_steps

    # These command lines carry stage2-report paths containing
    # "sumo_generation"; keep rollout and snapshot checks before generation.
    if "sumo_policy_rollout_validation" in cmd_lower:
        policies = _csv_value_count(_shell_option_from_tokens(tokens, "--policies"), default_count=1)
        events = _positive_float_option(tokens, "--max-events-per-city-policy", default=0.0)
        if events <= 0.0:
            events = _positive_float_option(tokens, "--max-steps", default=1.0)
        return "cfcmt_policy_rollout_c_le2_completed_history", float(policies) * events
    if "sumo_apc_avl_snapshot_generation" in cmd_lower:
        period = _positive_float_option(tokens, "--snapshot-period", default=60.0)
        stage2 = str(_shell_option_from_tokens(tokens, "--stage2-report") or "").lower()
        duration = _cfcmt_stage2_report_duration(stage2)
        return "cfcmt_snapshot_generation_c_le2_completed_history", max(1.0, duration / period)
    if "sumo_apc_avl_sumo_generation" in cmd_lower:
        duration = _positive_float_option(tokens, "--duration-sec", default=0.0)
        if duration <= 0.0:
            return None
        return "cfcmt_sumo_generation_c_le2_completed_history", duration
    if "traffic_signal_sumo_phase2.py" in cmd_lower:
        return "cfcmt_traffic_signal_phase2_c_le2_completed_history", 1.0
    return None


def _positive_float_option(tokens: list[str], name: str, *, default: float) -> float:
    raw = _shell_option_from_tokens(tokens, name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if math.isfinite(value) and value > 0.0 else float(default)


def _csv_value_count(value: str | None, *, default_count: int) -> int:
    if value is None:
        return max(1, int(default_count))
    count = len([part for part in str(value).split(",") if part.strip()])
    return count if count > 0 else max(1, int(default_count))


def _cfcmt_stage2_report_duration(stage2_report: str) -> float:
    value = str(stage2_report).lower()
    if "full_day" in value:
        return 86400.0
    if "4h" in value:
        return 14400.0
    return 1800.0


def _zsw_tsp_sumo_eval_c_le2_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    if (
        project not in {"zsw_platform", "zsw_tsp_m0_gpu1"}
        and "zsw_platform" not in cwd
        and "zsw_tsp_m0_gpu1" not in cwd
        and not signature.startswith(("zsw_platform/", "zsw_tsp_m0_gpu1/"))
    ):
        return None
    units = _parse_zsw_tsp_sumo_duration_units(cmd)
    return units if units is not None and units > 0 else None


def _zsw_m21_sumo_eval_c3_8_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (2.0 < float(cpu) <= 8.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if (
        project not in {"zsw_platform", "zsw_tsp_m0_gpu1"}
        and "zsw_platform" not in cwd
        and "zsw_tsp_m0_gpu1" not in cwd
        and not signature.startswith(("zsw_platform/", "zsw_tsp_m0_gpu1/"))
    ):
        return None
    if "m21_cycle_conserving_tsp_runner.py" not in cmd_lower:
        return None
    units = _parse_zsw_tsp_sumo_duration_units(cmd)
    return units if units is not None and units > 0 else None


def _parse_zsw_tsp_sumo_duration_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    accepted_scripts = {
        "baseline_runner.py",
        "m21_cycle_conserving_tsp_runner.py",
        "oracle_tsp_runner.py",
        "m2_scored_oracle_tsp_runner.py",
    }
    script = next(
        (
            str(token).rsplit("/", 1)[-1]
            for token in tokens
            if str(token).rsplit("/", 1)[-1] in accepted_scripts
        ),
        "",
    )
    if not script:
        return None
    if "--duration" not in tokens:
        return None
    idx = tokens.index("--duration")
    if idx + 1 >= len(tokens):
        return None
    try:
        duration = float(tokens[idx + 1])
    except ValueError:
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def _is_simple_sac_sumo_eval_c_le2(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> bool:
    if est_vram > 0:
        return False
    if float(cpu) > 2.0:
        return False
    project = str(row.get("project") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project != "simplesac":
        return False
    if "run_multiseed_eval.sh" not in cmd_lower:
        return False
    if "bash -lc" in cmd_lower or "$" in cmd:
        return False
    return _parse_simple_sac_multiseed_eval_identity(cmd) is not None


def _sumo_eval_c_le2_completed_history_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    text = " ".join((project, cwd, signature, str(row.get("description") or "").lower(), cmd_lower))

    if project == "offline-sumo" or "/offline-sumo" in cwd or signature.startswith("offline-sumo/"):
        accepted = (
            "eval_checkpoints_parallel.py",
            "eval_operational.py",
            "eval_e10.py",
            "eval_passenger.py",
            "eval_nobc.py",
            "eval_arbitrary_ckpts.py",
        )
        if any(script in cmd_lower for script in accepted):
            return (
                "offline_sumo_eval_c_le2_completed_history",
                1.0,
                "module78_offline_sumo_eval_c_le2_completed_history",
            )

    if (
        project in {"h2oplus", "simplesac"}
        or "/sumo-rl/h2oplus" in cwd
        or signature.startswith("h2oplus/")
    ):
        if (
            "bash -lc" in cmd_lower
            or "run_p4_hold_calibration.sh" in cmd_lower
            or "checkpoint_epoch80" in cmd_lower
        ) and (
            "run_multiseed_eval.sh" in cmd_lower
            or "run_p4_hold_calibration.sh" in cmd_lower
            or "checkpoint_epoch80" in cmd_lower
        ):
            return (
                "h2oplus_shell_eval_c_le2_completed_history",
                1.0,
                "module78_h2oplus_shell_eval_c_le2_completed_history",
            )

    if (
        "zsw_platform" in text
        or "zsw_tsp_m0_gpu1" in text
    ) and "m1_metrics_parser.py" in cmd_lower:
        return (
            "zsw_metrics_parser_c_le2_completed_history",
            1.0,
            "module78_zsw_metrics_parser_c_le2_completed_history",
        )

    if (
        project == "config"
        and "resco_benchmark/config" in cwd
        and "main.py" in cmd_lower
        and "episodes:" in cmd_lower
    ):
        return (
            "resco_config_eval_c_le2_completed_history",
            1.0,
            "module78_resco_config_eval_c_le2_completed_history",
        )

    if project.startswith("nature_emissions") or signature.startswith("nature_emissions"):
        if "extract_open_berlin_corridor_actual_demand.py" in cmd_lower:
            return (
                "nature_emissions_extract_c_le2_completed_history",
                1.0,
                "module78_nature_emissions_extract_c_le2_completed_history",
            )
        if "sumo " in f" {cmd_lower} " and ".sumocfg" in cmd_lower:
            return (
                "nature_emissions_sumo_c_le2_completed_history",
                1.0,
                "module78_nature_emissions_sumo_c_le2_completed_history",
            )

    return None


def _offline_sumo_eval_c33_64_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> float | None:
    if est_vram > 0:
        return None
    if not (32.0 < float(cpu) <= 64.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    signature = str(row.get("signature") or "").lower()
    cmd_lower = str(row.get("cmd") or "").lower()
    if not (project == "offline-sumo" or "/offline-sumo" in cwd or signature.startswith("offline-sumo/")):
        return None
    accepted = (
        "eval_checkpoints_parallel.py",
        "eval_operational.py",
        "eval_e10.py",
        "eval_passenger.py",
        "eval_nobc.py",
        "eval_arbitrary_ckpts.py",
    )
    if not any(script in cmd_lower for script in accepted):
        return None
    return 1.0


def _parse_simple_sac_multiseed_eval_identity(cmd: str) -> tuple[str, int, float] | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    idx = next((i for i, token in enumerate(tokens) if token.endswith("run_multiseed_eval.sh")), None)
    if idx is None or idx + 3 >= len(tokens):
        return None
    method = str(tokens[idx + 1])
    if not method or method.startswith("-"):
        return None
    try:
        seed = int(tokens[idx + 2])
        od_scale = float(tokens[idx + 3])
    except ValueError:
        return None
    return (method, seed, od_scale)


def _is_freqduet_runner_v3_allfreq_alllayers_c9_16(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> bool:
    if est_vram > 0:
        return False
    if not (8.0 < float(cpu) <= 16.0):
        return False
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "").lower()
    text = " ".join(str(row.get(key) or "") for key in ("project", "signature", "description", "cmd", "cwd")).lower()
    if project == "bamor" or "/bamor" in cwd:
        return False
    if "runner_v3.py" not in cmd:
        return False
    if "configs_freqduet/f_allfreq_alllayers_hiro.yaml" not in cmd:
        return False
    return "freqduet" in text or "/transitduet/freqduet/" in cwd


def _transit_freqhrl_c_le2_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if float(cpu) > 2.0:
        return None
    text = " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()
    if "bamor" in text:
        return None
    if not any(token in text for token in ("transitduet", "freq_hrl", "transit_hrl")):
        return None
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if "native_promotion_replan_validation" in cmd_lower and "print('import_ok')" in cmd_lower:
        return (
            "transit_freqhrl_import_smoke_c_le2_completed_history",
            1.0,
            "module70_transit_freqhrl_import_smoke_c_le2_completed_history",
        )

    tokens = _shlex_tokens(cmd)
    module = _python_module_from_tokens(tokens)
    if not module:
        return None

    units: float | None = None
    workload_key = ""
    reason = ""
    if module.endswith("trading.promotion_sweep"):
        units = _transit_trading_promotion_sweep_units(tokens)
        workload_key = "transit_trading_sweep_c_le2_completed_history"
        reason = "module70_transit_trading_sweep_c_le2_completed_history"
    elif module.endswith("trading.performance_validation"):
        units = _transit_seed_step_asset_units(tokens) * 11.0
        workload_key = "transit_trading_sweep_c_le2_completed_history"
        reason = "module70_transit_trading_sweep_c_le2_completed_history"
    elif module.endswith("trading.pressure_test_matrix"):
        scenario_count = _transit_count_option(tokens, "--scenarios", 6)
        baseline_count = _transit_count_option(tokens, "--baselines", 12)
        units = _transit_seed_step_asset_units(tokens) * scenario_count * baseline_count
        workload_key = "transit_trading_sweep_c_le2_completed_history"
        reason = "module70_transit_trading_sweep_c_le2_completed_history"
    elif module.endswith("trading.encoder_ablation"):
        method_count = _transit_count_option(tokens, "--methods", 6)
        units = _transit_seed_step_asset_units(tokens) * method_count
        workload_key = "transit_trading_sweep_c_le2_completed_history"
        reason = "module70_transit_trading_sweep_c_le2_completed_history"
    elif module.endswith("trading.policy_entry"):
        units = _transit_trading_policy_entry_units(tokens)
        workload_key = "transit_trading_policy_c_le2_completed_history"
        reason = "module70_transit_trading_policy_c_le2_completed_history"
    elif module.endswith("trading.ppo_actor_critic"):
        units = _transit_policy_train_eval_units(tokens, default_iterations=8)
        workload_key = "transit_trading_policy_c_le2_completed_history"
        reason = "module70_transit_trading_policy_c_le2_completed_history"
    elif module.endswith("transit.gap_closure_validation"):
        units = 4.0 * _transit_surrogate_train_eval_units(tokens, default_iterations=5)
        workload_key = "transit_surrogate_validation_c_le2_completed_history"
        reason = "module70_transit_surrogate_validation_c_le2_completed_history"
    elif module.endswith("transit.ppo_surrogate"):
        units = _transit_surrogate_train_eval_units(tokens, default_iterations=8)
        workload_key = "transit_surrogate_validation_c_le2_completed_history"
        reason = "module70_transit_surrogate_validation_c_le2_completed_history"
    elif module.endswith("transit.native_promotion_replan_validation"):
        variants = _transit_count_option(tokens, "--variants", 4)
        seed_episode_units = _parse_native_promotion_seed_units(cmd)
        if seed_episode_units is None:
            seed_episode_units = _transit_seed_episode_units(tokens, seed_default=8)
        units = variants * seed_episode_units
        workload_key = "transit_native_promotion_c_le2_completed_history"
        reason = "module70_transit_native_promotion_c_le2_completed_history"
    elif module.endswith("transit.native_wait_credit_validation"):
        units = 2.0 * _transit_seed_episode_units(tokens, seed_default=5)
        workload_key = "transit_native_control_c_le2_completed_history"
        reason = "module70_transit_native_control_c_le2_completed_history"
    elif module.endswith("transit.native_real_demand_control_validation"):
        source_count = _transit_count_option(tokens, "--sources", 2)
        units = 2.0 * source_count * _transit_seed_episode_units(tokens, seed_default=3)
        workload_key = "transit_native_control_c_le2_completed_history"
        reason = "module70_transit_native_control_c_le2_completed_history"
    elif module.endswith("transit.merge_native_real_demand_shards") or module.endswith(
        "transit.merge_native_promotion_shards"
    ):
        units = 1.0
        workload_key = "transit_freqhrl_merge_c_le2_completed_history"
        reason = "module76_transit_freqhrl_merge_c_le2_completed_history"
    elif module in {
        "freq_hrl.experiments.trading.order_book_large_replay_manifest_validation",
        "freq_hrl.experiments.encoder_cross_domain_matrix",
        "freq_hrl.experiments.leakage_no_tradeoff_matrix",
        "freq_hrl.experiments.theory_appendix",
        "freq_hrl.experiments.top_journal_unified_matrix",
    }:
        units = 1.0
        workload_key = "transit_freqhrl_analysis_matrix_c_le2_completed_history"
        reason = "module76_transit_freqhrl_analysis_matrix_c_le2_completed_history"

    if units is None or not math.isfinite(float(units)) or float(units) <= 0:
        return None
    return workload_key, float(units), reason


def _transit_freqhrl_c3_8_units(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> tuple[str, float, str] | None:
    if est_vram > 0:
        return None
    if not (2.0 < float(cpu) <= 8.0):
        return None
    text = " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()
    if not any(token in text for token in ("freq_hrl", "freqhrl", "transitduet", "native_real_demand")):
        return None
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    tokens = _shlex_tokens(cmd)
    module = _python_module_from_tokens(tokens)
    if module in {"unittest", "pytest"} and any(
        token in cmd_lower
        for token in (
            "transit_hrl/tests",
            "transit_hrl.tests",
            "test_native_transit_ppo_bridge.py",
            "test_native_promotion_replan_validation.py",
        )
    ):
        return (
            "transit_freqhrl_tests_c3_8_completed_history",
            1.0,
            "module83_transit_freqhrl_tests_c3_8_completed_history",
        )
    if "native_promotion_replan_validation" in cmd_lower:
        units = _parse_native_promotion_seed_units(cmd)
        if units is None or units <= 0:
            return None
        return (
            "transit_native_promotion_c3_8_persistent_stress_completed_history",
            float(units),
            "module79_transit_native_promotion_c3_8_persistent_stress_completed_history",
        )
    if "native_real_demand_control_validation" in cmd_lower:
        tokens = _shlex_tokens(cmd)
        source_count = _transit_count_option(tokens, "--sources", 2)
        units = 2.0 * source_count * _transit_seed_episode_units(tokens, seed_default=3)
        if units <= 0:
            return None
        if "alighting_safe" in text or "alighting_rescue" in text:
            return (
                "transit_native_real_demand_alighting_c3_8_completed_history",
                float(units),
                "module79_transit_native_real_demand_alighting_c3_8_completed_history",
            )
        return (
            "transit_native_real_demand_batch_c3_8_completed_history",
            float(units),
            "module79_transit_native_real_demand_batch_c3_8_completed_history",
        )
    if module.endswith("trading.public_market_data"):
        steps = _transit_int_option(tokens, "--steps", 1500)
        csv_count = _transit_count_option(tokens, "--csv-files", 1)
        units = float(steps * csv_count)
        return (
            "transit_trading_public_csv_c3_8_completed_history",
            units,
            "module83_transit_trading_public_csv_c3_8_completed_history",
        )
    if module.endswith("trading.merge_pressure_matrix"):
        return (
            "transit_trading_pressure_merge_c3_8_completed_history",
            1.0,
            "module83_transit_trading_pressure_merge_c3_8_completed_history",
        )
    if module.endswith("trading.policy_entry"):
        units = _transit_trading_policy_entry_units(tokens)
        return (
            "transit_trading_policy_c3_8_completed_history",
            float(units),
            "module83_transit_trading_policy_c3_8_completed_history",
        )
    if module.endswith("transit.ppo_surrogate"):
        units = _transit_surrogate_train_eval_units(tokens, default_iterations=8)
        return (
            "transit_surrogate_c3_8_completed_history",
            float(units),
            "module83_transit_surrogate_c3_8_completed_history",
        )
    if module.endswith("transit.merge_native_promotion_shards"):
        return (
            "transit_native_merge_c3_8_completed_history",
            1.0,
            "module83_transit_native_merge_c3_8_completed_history",
        )
    return None


def _shlex_tokens(cmd: str) -> list[str]:
    import shlex

    try:
        return shlex.split(str(cmd))
    except ValueError:
        return str(cmd).split()


def _python_module_from_tokens(tokens: list[str]) -> str:
    for idx, token in enumerate(tokens):
        if token == "-m" and idx + 1 < len(tokens):
            return str(tokens[idx + 1])
    return ""


def _transit_option(tokens: list[str], name: str, default: str = "") -> str:
    value = _shell_option_from_tokens(tokens, name)
    return str(value) if value is not None else str(default)


def _transit_int_option(tokens: list[str], name: str, default: int) -> int:
    try:
        value = int(float(_transit_option(tokens, name, str(default))))
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)


def _transit_values(tokens: list[str], name: str) -> list[str]:
    if name not in tokens:
        return []
    idx = tokens.index(name) + 1
    values: list[str] = []
    while idx < len(tokens) and not str(tokens[idx]).startswith("--"):
        for part in str(tokens[idx]).split(","):
            if part.strip():
                values.append(part.strip())
        idx += 1
    return values


def _transit_count_option(tokens: list[str], name: str, default_count: int) -> int:
    values = _transit_values(tokens, name)
    if values:
        return len(values)
    return max(1, int(default_count))


def _transit_seed_count(tokens: list[str], name: str, default_count: int) -> int:
    return _transit_count_option(tokens, name, default_count)


def _transit_seed_step_asset_units(tokens: list[str]) -> float:
    seeds = _transit_seed_count(tokens, "--seeds", 5)
    steps = _transit_int_option(tokens, "--steps", 720)
    assets = _transit_int_option(tokens, "--assets", 3)
    return float(seeds * steps * assets)


def _transit_trading_promotion_sweep_units(tokens: list[str]) -> float:
    grid = (
        _transit_count_option(tokens, "--thresholds", 4)
        * _transit_count_option(tokens, "--ratios", 4)
        * _transit_count_option(tokens, "--regime-thresholds", 4)
        * _transit_count_option(tokens, "--min-age-s", 1)
        * _transit_count_option(tokens, "--activation-strength-thresholds", 1)
        * _transit_count_option(tokens, "--startup-strength-age-s", 1)
        * _transit_count_option(tokens, "--startup-strength-thresholds", 1)
        * _transit_count_option(tokens, "--mid-gains", 3)
        * _transit_count_option(tokens, "--adapt-gains", 5)
        * 2
    )
    return _transit_seed_step_asset_units(tokens) * float(grid)


def _transit_trading_policy_entry_units(tokens: list[str]) -> float:
    mode = _transit_option(tokens, "--mode", "eval")
    policy = _transit_option(tokens, "--policy", "heuristic")
    if mode != "train":
        eval_seeds = _transit_seed_count(tokens, "--eval-seeds", _transit_seed_count(tokens, "--seeds", 1))
        steps = _transit_int_option(tokens, "--steps", 360)
        assets = _transit_int_option(tokens, "--assets", 3)
        return float(eval_seeds * steps * assets)
    if policy == "pg_linear":
        iterations = _transit_int_option(tokens, "--pg-iterations", 12)
    elif policy == "ac_linear":
        iterations = _transit_int_option(tokens, "--ac-iterations", 20)
    elif policy == "linear":
        iterations = (
            _transit_int_option(tokens, "--generations", 8)
            * _transit_int_option(tokens, "--population", 12)
        )
    else:
        iterations = 1
    return _transit_policy_train_eval_units(tokens, default_iterations=iterations)


def _transit_policy_train_eval_units(tokens: list[str], *, default_iterations: int) -> float:
    train = _transit_seed_count(tokens, "--train-seeds", 3)
    fallback_eval = train if not _transit_values(tokens, "--eval-seeds") else 3
    evals = _transit_seed_count(tokens, "--eval-seeds", fallback_eval)
    steps = _transit_int_option(tokens, "--steps", 360)
    assets = _transit_int_option(tokens, "--assets", 3)
    iterations = _transit_int_option(tokens, "--iterations", int(default_iterations))
    return float(train * steps * assets * iterations + evals * steps * assets)


def _transit_surrogate_train_eval_units(tokens: list[str], *, default_iterations: int) -> float:
    train = _transit_seed_count(tokens, "--train-seeds", 3)
    fallback_eval = train if not _transit_values(tokens, "--eval-seeds") else 3
    evals = _transit_seed_count(tokens, "--eval-seeds", fallback_eval)
    steps = _transit_int_option(tokens, "--steps", 240)
    corridors = _transit_int_option(tokens, "--corridors", 2)
    iterations = _transit_int_option(tokens, "--iterations", int(default_iterations))
    return float(train * steps * corridors * iterations + evals * steps * corridors)


def _transit_seed_episode_units(tokens: list[str], *, seed_default: int) -> float:
    seeds = _transit_seed_count(tokens, "--seeds", seed_default)
    episodes = _transit_int_option(tokens, "--episodes", 1)
    return float(seeds * episodes)


def _dedupe_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    no_id = []
    for row in records:
        rec = dict(row)
        key = str(rec.get("id") or rec.get("task_id") or "")
        if not key:
            no_id.append(rec)
            continue
        old = by_id.get(key)
        if old is None or _record_timestamp(rec) >= _record_timestamp(old):
            by_id[key] = rec
    return list(by_id.values()) + no_id


def _window_end(records: list[Mapping[str, Any]], *, now_ts: float | None) -> float:
    if now_ts is not None and math.isfinite(float(now_ts)):
        return float(now_ts)
    timestamps = [_record_timestamp(row) for row in records if _record_timestamp(row) > 0]
    return max(timestamps, default=time.time())


def _record_timestamp(row: Mapping[str, Any]) -> float:
    return max(
        _as_float(row.get("finished_at")),
        _as_float(row.get("started_at")),
        _as_float(row.get("submitted_at")),
        _as_float(row.get("created_at")),
    )


def _submitted_at(row: Mapping[str, Any]) -> float:
    return _as_float(row.get("submitted_at"), _record_timestamp(row))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    cap = report.get("capacity") or {}
    rows = [
        "# Production Load Capacity Certificate",
        "",
        "```text",
        f"window_days = {report.get('window', {}).get('window_days')}",
        f"include_representative = {report.get('include_representative')}",
        f"record_count_window = {report.get('record_count_window')}",
        f"mapped_task_count = {report.get('mapped_task_count')}",
        f"representative_mapped_task_count = {report.get('representative_mapped_task_count')}",
        f"unmapped_task_count = {report.get('unmapped_task_count')}",
        f"mapped_fraction = {report.get('mapped_fraction')}",
        f"action_generation = {report.get('action_generation')}",
        f"full_action_count = {report.get('full_action_count')}",
        f"action_count_evaluated = {report.get('action_count_evaluated')}",
        "```",
        "",
        "| Workload | Count | Lambda |",
        "|---|---:|---:|",
    ]
    counts = report.get("mapped_counts") or {}
    lam = report.get("lambda") or {}
    for key in sorted(set(counts) | set(lam)):
        rows.append(f"| `{key}` | {int(counts.get(key, 0))} | {float(lam.get(key, 0.0)):.9f} |")
    rows.extend(
        [
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| `delta` | {float(cap.get('delta') or 0.0):.9f} |",
            f"| `mapped_capacity_usable_for_theorem` | {str(bool(report.get('mapped_capacity_usable_for_theorem'))).lower()} |",
            f"| `global_coverage_usable_for_theorem` | {str(bool(report.get('global_coverage_usable_for_theorem'))).lower()} |",
            f"| `usable_for_global_theorem` | {str(bool(report.get('usable_for_global_theorem'))).lower()} |",
            "",
            "## Unmapped Reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted((report.get("unmapped_reasons") or {}).items()):
        rows.append(f"| `{reason}` | {int(count)} |")
    rows.extend(["", str(report.get("interpretation") or ""), ""])
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.production_load_certificate")
    p.add_argument("--queue-path", default="")
    p.add_argument("--archive-path", default="")
    p.add_argument("--window-days", type=float, default=30.0)
    p.add_argument("--include-representative", action="store_true")
    p.add_argument("--tasksets", default=",".join(DEFAULT_TASKSETS))
    p.add_argument("--output", required=True)
    p.add_argument("--markdown-output", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_production_load_certificate(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        ),
        window_days=args.window_days,
        include_representative=args.include_representative,
        taskset_names=[name.strip() for name in args.tasksets.split(",") if name.strip()],
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_markdown(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("mapped_capacity_usable_for_theorem") else 2


if __name__ == "__main__":
    raise SystemExit(main())
