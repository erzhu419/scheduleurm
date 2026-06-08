import math

from algorithm.experiments.capacity_lp import drift_margin_certificate, solve_capacity_slack
from algorithm.experiments.action_model import (
    build_action_family_report,
    candidate_families_by_slot,
    chosen_action_for_slot,
    validate_action_row,
)
from algorithm.experiments.fabric_metric import (
    audit_cover,
    calibrate_cover_and_lipschitz,
    calibrate_lipschitz,
)
from algorithm.experiments.empirical_slack_certificate import (
    build_measured_finite_slice_certificate,
)
from algorithm.experiments.oracle_audit import audit_slots
from algorithm.experiments.oracle_trace_enrichment import enrich_trace_slots
from algorithm.experiments.penalty_fit import fit_penalty_envelope
from algorithm.experiments.live_validation import compare_replay_to_live
from algorithm.experiments.portfolio_live_proxy import build_composed_live_report
from algorithm.experiments.production_cpu_workload_curve import (
    build_submission_plan,
    render_cpu_workload_command,
)
from algorithm.experiments.production_load_certificate import (
    build_production_load_certificate,
    classify_record,
)
from algorithm.experiments.production_coverage_drilldown import (
    build_coverage_drilldown,
    bucket_obligation,
    population_label,
)
from algorithm.experiments.production_bucket_probe_manifest import build_probe_manifest
from algorithm.experiments.report import summarize_certificate
from algorithm.experiments.service_model import (
    calibrate_service_lower_bounds,
    estimate_epsilon_est,
)
from algorithm.experiments.slack_accounting import (
    build_slack_certificate,
    markdown_slack_table,
)
from algorithm.experiments.slot_builder import progress_service_units, queue_step
from algorithm.experiments.theorem_oracle_trace_bridge import (
    build_theorem_oracle_audit_from_trace,
)
from simulation.defaults import build_default_cache


def _features(bucket="b0", count="n1", mem="pm2", util="u1"):
    return {
        "finite": {
            "class_key": "class:i",
            "regime_key": "regime:z",
            "post_count_bucket": count,
            "post_vram_bucket": mem,
            "util_bucket": util,
            "resource_bucket": "v2.r2.c1",
            "task_kind": "gpu_train",
        },
        "candidate_bucket": bucket,
        "post_vram_frac": 0.25 if mem == "pm2" else 0.50,
        "used_vram_frac": 0.10,
        "util_pct": 20 if util == "u1" else 80,
        "running_task_count": 1 if count == "n1" else 3,
        "post_task_count": 2 if count == "n1" else 4,
        "legacy_runtime_s": 120.0,
    }


def test_fabric_cover_and_lipschitz_calibration(check, sch):
    full = [
        {"action_id": "a0", "features": _features("b0"), "service_vector": {"i": 10.0}},
        {"action_id": "a1", "features": _features("b1", count="n3"), "service_vector": {"i": 7.0}},
    ]
    cand = [
        {"action_id": "c0", "features": _features("b0"), "service_vector": {"i": 10.0}},
        {"action_id": "c1", "features": _features("b1", count="n3"), "service_vector": {"i": 7.0}},
    ]
    cover = audit_cover(full, cand, rho=0.0, cover_domain="all_feasible")
    check("fabric cover audit computes max min-distance",
          cover["rho_sample"] == 0.0 and cover["usable_for_theorem"],
          diag=str(cover))

    calibrated = calibrate_cover_and_lipschitz(full, cand, rho=0.0, cover_domain="all_feasible")
    check("fabric calibration emits L rho Lrho certificate",
          calibrated["usable_for_theorem"]
          and calibrated["rho"] == 0.0
          and calibrated["Lrho"] == 0.0
          and "cover" in calibrated
          and "lipschitz" in calibrated,
          diag=str(calibrated))

    lip = calibrate_lipschitz(full, d_min=1e-6)
    check("fabric Lipschitz calibration returns theorem L",
          lip["pair_count"] > 0 and lip["L"] >= 0.0 and lip["usable_for_theorem"],
          diag=str(lip))

    bad = [dict(full[0]), dict(full[0])]
    bad[1]["action_id"] = "a_bad"
    bad[1]["service_vector"] = {"i": 11.0}
    bad_lip = calibrate_lipschitz(bad, d_min=1e-6)
    check("fabric calibration catches zero-metric service disagreement",
          bad_lip["zero_metric_violation_count"] > 0 and not bad_lip["usable_for_theorem"],
          diag=str(bad_lip))


def test_slot_builder_queue_recurrence_and_censoring(check, sch):
    q1 = queue_step(
        {"i": 5.0, "j": 1.0},
        {"i": 1.0, "j": 2.0},
        {"i": 3.0, "j": 5.0},
    )
    check("slot builder applies Q+=max(Q-S,0)+A",
          q1 == {"i": 3.0, "j": 2.0},
          diag=str(q1))
    service = progress_service_units(100.0, 125.0)
    check("progress increase becomes nonnegative service",
          service["service_units"] == 25.0 and not service["service_censored"],
          diag=str(service))
    reset = progress_service_units(125.0, 10.0)
    check("progress reset is censored by default",
          reset["service_units"] == 0.0 and reset["service_censored"],
          diag=str(reset))


def test_experiment_action_family_reconstruction(check, sch):
    row = {
        "slot_id": "s0",
        "action_id": "a0",
        "chosen": True,
        "candidate_source": "exact_full",
        "assignments": [{
            "task_id": "t0",
            "node": "n0",
            "gpu_idx": 0,
            "class_key": "i",
            "candidate_bucket": "b0",
        }],
    }
    validation = validate_action_row(row)
    families = candidate_families_by_slot([row])
    chosen = chosen_action_for_slot([row], "s0")
    check("experiment action row validates complete global action",
          validation["valid"],
          diag=str(validation))
    check("experiment action family reconstructs statewise full/candidate sets",
          len(families["s0"]["full_actions"]) == 1
          and len(families["s0"]["candidate_actions"]) == 1
          and families["s0"]["chosen_action_id"] == "a0",
          diag=str(families))
    check("experiment chosen action reconstruction is unique",
          chosen["action_id"] == "a0",
          diag=str(chosen))

    statewise = dict(row)
    statewise["slot_id"] = "s1"
    statewise["candidate_source"] = "all_feasible"
    statewise["chosen"] = False
    statewise_family = candidate_families_by_slot([statewise])
    check("experiment action model treats all_feasible as full family",
          len(statewise_family["s1"]["full_actions"]) == 1,
          diag=str(statewise_family))

    report = build_action_family_report([row])
    check("experiment action family report is theorem-usable on complete slot",
          report["usable_for_theorem"] and report["slot_count"] == 1,
          diag=str(report))
    missing_chosen = dict(row)
    missing_chosen["chosen"] = False
    bad_report = build_action_family_report([missing_chosen])
    check("experiment action family report refuses missing chosen action",
          bad_report["slot_error_count"] == 1 and not bad_report["usable_for_theorem"],
          diag=str(bad_report))


def test_service_lower_bound_and_epsilon_est(check, sch):
    samples = [
        {
            "class_key": "i",
            "regime_key": "z",
            "feature_bucket": "b",
            "service_units_per_delta_ref": 10.0,
        }
        for _ in range(5)
    ]
    lower = calibrate_service_lower_bounds(samples, min_samples=5)
    check("service model emits usable LCB row",
          len(lower) == 1 and lower[0]["usable_for_theorem"]
          and math.isclose(lower[0]["lcb_service_per_delta_ref"], 10.0),
          diag=str(lower))
    eps = estimate_epsilon_est([
        {
            "class_key": "i",
            "regime_key": "z",
            "feature_bucket": "b",
            "service_units_per_delta_ref": 11.0,
        }
    ], lower)
    check("epsilon_est is positive validation residual over lower service",
          math.isclose(eps["epsilon_est"], 1.0) and eps["usable_for_theorem"],
          diag=str(eps))

    weak_lower = calibrate_service_lower_bounds(samples[:2], min_samples=5)
    weak_eps = estimate_epsilon_est([
        {
            "class_key": "i",
            "regime_key": "z",
            "feature_bucket": "b",
            "service_units_per_delta_ref": 10.0,
        }
    ], weak_lower)
    check("epsilon_est refuses unusable service lower bounds",
          weak_eps["unusable_lower_bound_count"] == 1 and not weak_eps["usable_for_theorem"],
          diag=str(weak_eps))


def test_penalty_and_oracle_envelopes(check, sch):
    penalty = fit_penalty_envelope([
        {"q_norm": 0.0, "penalty_units": 2.0},
        {"q_norm": 10.0, "penalty_units": 5.0},
    ])
    check("penalty fit charges fixed and queue-scaled constants",
          math.isclose(penalty["P0"], 2.0)
          and math.isclose(penalty["beta"], 0.3)
          and penalty["usable_for_theorem"],
          diag=str(penalty))

    audit = audit_slots([
        {
            "slot_id": "s0",
            "queue_vector": {"i": 10.0},
            "chosen_action_id": "a",
            "candidate_actions": [
                {"action_id": "a", "lower_service": {"i": 0.5}, "penalty_units": 0.0},
                {"action_id": "b", "lower_service": {"i": 0.8}, "penalty_units": 0.0},
            ],
        }
    ])
    check("oracle audit computes gap and alpha1 envelope",
          math.isclose(audit["rows"][0]["oracle_gap"], 3.0)
          and math.isclose(audit["alpha1"], 0.3)
          and audit["usable_for_theorem"],
          diag=str(audit))


def test_theorem_oracle_trace_bridge_refuses_sort_key_and_accepts_lower_service(check, sch):
    rejected = build_theorem_oracle_audit_from_trace([
        {
            "slot_id": "sort-only",
            "score_semantics": "scheduler_sort_key_minimization",
            "queue_vector": {"i": 10.0},
            "candidates": [
                {"action_id": "a", "selected": True, "primary_numeric_score": 1.0},
                {"action_id": "b", "selected": False, "primary_numeric_score": 2.0},
            ],
        }
    ])
    check("theorem oracle bridge refuses scheduler-sort-key-only trace",
          rejected["status"] == "NOT_THEOREM_TRACE"
          and not rejected["usable_for_theorem"]
          and rejected["blockers"][0]["reason"] == "score_semantics_not_robust_maxweight_lower_service",
          diag=str(rejected))

    accepted = build_theorem_oracle_audit_from_trace([
        {
            "slot_id": "mw",
            "score_semantics": "robust_maxweight_lower_service",
            "queue_vector": {"i": 10.0},
            "candidates": [
                {"action_id": "a", "selected": True, "lower_service": {"i": 0.5}, "penalty_units": 0.0},
                {"action_id": "b", "selected": False, "lower_service": {"i": 0.8}, "penalty_units": 0.0},
            ],
        }
    ])
    check("theorem oracle bridge computes alpha envelope from lower-service trace",
          accepted["status"] == "THEOREM_ORACLE_PASS"
          and math.isclose(accepted["rows"][0]["oracle_gap"], 3.0)
          and math.isclose(accepted["alpha1"], 0.3)
          and accepted["usable_for_theorem"],
          diag=str(accepted))


def test_oracle_trace_enrichment_attaches_lower_service_before_theorem_bridge(check, sch):
    slots = [
        {
            "slot_id": "live-slot",
            "score_semantics": "scheduler_sort_key_minimization",
            "queue_vector": {},
            "candidates": [
                {"action_id": "a", "candidate_bucket": "b0", "selected": True, "primary_numeric_score": 0.0},
                {"action_id": "b", "candidate_bucket": "b1", "selected": False, "primary_numeric_score": 1.0},
            ],
        }
    ]
    no_queue = enrich_trace_slots(
        slots,
        service_rows=[{"candidate_bucket": "b0", "lower_service": {"i": 0.5}}],
        queue_vector={},
    )
    check("oracle trace enrichment refuses missing queue vector before theorem bridge",
          no_queue["status"] == "ENRICHMENT_BLOCKED"
          and no_queue["blockers"][0]["reason"] == "missing_queue_vector",
          diag=str(no_queue))

    blocked = enrich_trace_slots(
        slots,
        service_rows=[{"candidate_bucket": "b0", "lower_service": {"i": 0.5}}],
        queue_vector={"i": 10.0},
    )
    check("oracle trace enrichment refuses partially covered candidate families",
          blocked["status"] == "ENRICHMENT_BLOCKED"
          and blocked["blockers"][0]["reason"] == "candidate_lower_service_not_found",
          diag=str(blocked))

    enriched = enrich_trace_slots(
        slots,
        service_rows=[
            {"candidate_bucket": "b0", "lower_service": {"i": 0.5}, "penalty_units": 0.0},
            {"candidate_bucket": "b1", "lower_service": {"i": 0.8}, "penalty_units": 0.0},
        ],
        queue_vector={"i": 10.0},
    )
    audit = enriched["oracle_audit"]
    check("oracle trace enrichment produces theorem oracle audit from bucket lookup",
          enriched["status"] == "ENRICHED_THEOREM_PASS"
          and audit["status"] == "THEOREM_ORACLE_PASS"
          and math.isclose(audit["rows"][0]["oracle_gap"], 3.0)
          and math.isclose(audit["alpha1"], 0.3),
          diag=str(enriched))


def test_slack_accounting_certificate_combines_theorem_constants(check, sch):
    cert = build_slack_certificate(
        fabric={"L": 0.5, "rho": 0.2, "usable_for_theorem": True},
        service={"epsilon_est": 0.05, "usable_for_theorem": True},
        penalty={"P0": 2.0, "beta": 0.10, "usable_for_theorem": True},
        oracle={"alpha0": 1.0, "alpha1": 0.05, "usable_for_theorem": True},
        capacity={"delta": 0.40, "usable_for_theorem": True},
        moment={"B": 3.0, "usable_for_theorem": True},
        alpha=1.0,
        N=25,
    )
    check("slack accounting computes positive eta and finite-set threshold",
          math.isclose(cert["Lrho"], 0.10)
          and math.isclose(cert["slack_consumed"], 0.30)
          and math.isclose(cert["eta"], 0.10)
          and cert["finite_set_threshold_N"] == 70
          and not cert["usable_for_theorem"],
          diag=str(cert))
    cert_pass = build_slack_certificate(
        fabric={"L": 0.5, "rho": 0.2, "usable_for_theorem": True},
        service={"epsilon_est": 0.05, "usable_for_theorem": True},
        penalty={"P0": 2.0, "beta": 0.10, "usable_for_theorem": True},
        oracle={"alpha0": 1.0, "alpha1": 0.05, "usable_for_theorem": True},
        capacity={"delta": 0.40, "usable_for_theorem": True},
        moment={"B": 3.0, "usable_for_theorem": True},
        alpha=1.0,
        N=70,
    )
    check("slack accounting passes only when N covers additive drift constant",
          cert_pass["usable_for_theorem"] and cert_pass["all_components_theorem_usable"],
          diag=str(cert_pass))
    table = markdown_slack_table(cert_pass)
    check("slack accounting emits reviewer-facing markdown table",
          "| `eta` | 0.100000000 |" in table
          and "| `fabric` | true |" in table,
          diag=table)

    neg = build_slack_certificate(
        fabric={"L": 1.0, "rho": 0.4, "usable_for_theorem": True},
        service={"epsilon_est": 0.1, "usable_for_theorem": True},
        penalty={"beta": 0.1, "usable_for_theorem": True},
        oracle={"alpha1": 0.1, "usable_for_theorem": True},
        capacity={"delta": 0.3, "usable_for_theorem": True},
        moment={"B": 1.0, "usable_for_theorem": True},
    )
    check("slack accounting refuses nonpositive drift margin",
          neg["eta"] < 0.0 and not neg["usable_for_theorem"],
          diag=str(neg))


def test_measured_finite_slice_slack_certificate_is_positive(check, sch):
    cert = build_measured_finite_slice_certificate(
        taskset_name="hybrid_research_portfolio",
        load_fraction=0.80,
    )
    check("measured portfolio finite-slice certificate closes positive eta condition",
          cert["usable_for_theorem"]
          and cert["eta"] > 0.0
          and cert["delta"] > cert["slack_consumed"]
          and cert["selected_profiles"]["hybrid_rl_resac_ant"] == 3,
          diag=str({
              "delta": cert.get("delta"),
              "eta": cert.get("eta"),
              "selected_profiles": cert.get("selected_profiles"),
              "component_status": cert.get("component_status"),
          }))


def test_production_load_certificate_separates_capacity_from_global_coverage(check, sch):
    records = [
        {
            "id": "m0",
            "project": "ScheduleurmBench",
            "signature": "ScheduleurmBench/q01_gpu_heavy_jax_matmul/profile_1",
            "submitted_at": 900.0,
            "status": "done",
            "est_vram_mb": 1000,
        },
        {
            "id": "u0",
            "project": "UnknownProject",
            "signature": "UnknownProject/gpu",
            "submitted_at": 950.0,
            "status": "done",
            "est_vram_mb": 1000,
        },
    ]
    report = build_production_load_certificate(
        records=records,
        window_days=1.0,
        now_ts=1000.0,
    )
    check("production load certificate can pass mapped capacity while refusing global coverage",
          report["mapped_capacity_usable_for_theorem"]
          and not report["global_coverage_usable_for_theorem"]
          and not report["usable_for_global_theorem"]
          and report["unmapped_task_count"] == 1,
          diag=str(report))

    all_mapped = build_production_load_certificate(
        records=[records[0]],
        window_days=1.0,
        now_ts=1000.0,
    )
    check("production load certificate permits global theorem only with full strict coverage",
          all_mapped["mapped_capacity_usable_for_theorem"]
          and all_mapped["global_coverage_usable_for_theorem"]
          and all_mapped["usable_for_global_theorem"],
          diag=str(all_mapped))


def test_module56_freqduet_cpu_subbucket_is_strictly_measured(check, sch):
    row = {
        "id": "freq-c17",
        "project": "FreqDuet",
        "signature": "FreqDuet/ablation/c17_32",
        "description": "FreqDuet terminal ablation control",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 24,
        "cmd": "python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro --seeds 1,2",
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c17_32 CPU records map only after module56 strict measurement",
          cls["workload_key"] == "freqduet_cpu_ablation_c17_32"
          and cls["mapping_mode"] == "strict_measured",
          diag=str(cls))
    bamor = classify_record(
        {
            "id": "bamor-c17",
            "project": "BAMOR",
            "signature": "BAMOR/diagnostic-shard/node001",
            "description": "BAMOR diagnostic shard",
            "submitted_at": 901.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 24,
            "cwd": "/home/erzhu419/mine_code/BAMOR",
            "cmd": "/conda_envs/freqduet-cpu-py310/bin/python run_bamor_diagnostic_shard.py --workers 24",
        },
        include_representative=False,
    )
    check("FreqDuet c17_32 classifier ignores freqduet only in environment paths",
          bamor["workload_key"] is None and bamor["reason"] == "unmapped_cpu",
          diag=str(bamor))

    cache = build_default_cache()
    feasible = cache.profiles("freqduet_cpu_ablation_c17_32")
    boundary = cache.get("freqduet_cpu_ablation_c17_32", 8)
    check("module56 service cache exposes feasible profiles and capacity boundary",
          [record.profile for record in feasible] == [1, 2, 4]
          and boundary is not None
          and boundary.capacity_boundary
          and math.isclose(cache.get("freqduet_cpu_ablation_c17_32", 4).aggregate_rate, 0.878373516, rel_tol=1e-9),
          diag=str([record.snapshot() for record in cache.profiles("freqduet_cpu_ablation_c17_32", include_boundaries=True)]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_cpu_c17_32",),
    )
    check("production load certificate can certify the measured FreqDuet sub-bucket alone",
          report["mapped_counts"] == {"freqduet_cpu_ablation_c17_32": 1}
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module57_freqduet_runner_exact_config_is_strictly_measured(check, sch):
    row = {
        "id": "freq-runner-c9",
        "project": "freqduet",
        "signature": "freqduet/auto-adopted/p2000161",
        "description": "FreqDuet direct runner exact config",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 10,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "/usr/bin/python3 runner_v3.py --config "
            "configs_freqduet/F_allfreq_alllayers_hiro.yaml "
            "--episodes 20 --seed 123 --no-resume"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet runner_v3 c9_16 exact config maps after module57 strict measurement",
          cls["workload_key"] == "freqduet_runner_v3_allfreq_alllayers_c9_16"
          and cls["mapping_mode"] == "strict_measured",
          diag=str(cls))

    other_config = dict(row)
    other_config["id"] = "freq-runner-other-config"
    other_config["cmd"] = other_config["cmd"].replace(
        "F_allfreq_alllayers_hiro.yaml",
        "F_freqduet_timetable_hiro.yaml",
    )
    other_cls = classify_record(other_config, include_representative=False)
    check("module57 classifier does not generalize runner_v3 to other configs",
          other_cls["workload_key"] is None and other_cls["reason"] == "unmapped_cpu",
          diag=str(other_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_runner_v3_allfreq_alllayers_c9_16")
    check("module57 service cache exposes the full measured runner_v3 c9_16 curve",
          [record.profile for record in profiles] == [1, 2, 4, 8]
          and math.isclose(
              cache.get("freqduet_runner_v3_allfreq_alllayers_c9_16", 8).aggregate_rate,
              0.3627567171,
              rel_tol=1e-9,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_runner_v3_allfreq_alllayers_c9_16",),
    )
    check("production load certificate can certify the measured runner_v3 exact-config slice alone",
          report["mapped_counts"] == {"freqduet_runner_v3_allfreq_alllayers_c9_16": 1}
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module58_freqduet_c9_ablation_uses_parsed_production_units(check, sch):
    row = {
        "id": "freq-c9",
        "project": "FreqDuet",
        "signature": "FreqDuet/noharm_screen_20260601/node001_s0",
        "description": "FreqDuet c9 ablation shard",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 13,
        "ram_mb": 32768,
        "cmd": (
            "python scripts/run_freqduet_ablation.py "
            "--configs F_freqduet_terminal_main_hiro,F_freqduet_gen_highnoise_main_hiro,"
            "F_freqduet_gen_odshift_main_hiro,F_freqduet_gen_rushshift_main_hiro "
            "--seeds 7,11,17,23,31,37,42,43,53,61,71,83,97,109,123,127,149,456,789,2026 "
            "--episodes 100 --workers 13 --worker-threads 1 --job-start 0 --job-end 13"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c9_16 ablation shard maps after module58 strict measurement",
          cls["workload_key"] == "freqduet_cpu_ablation_c9_16"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 1300.0),
          diag=str(cls))

    unknown_units = dict(row)
    unknown_units["id"] = "freq-c9-unknown-units"
    unknown_units["cmd"] = (
        "python scripts/run_freqduet_ablation.py --configs $CONFIGS --seeds $SEEDS "
        "--episodes 100 --workers 13 --worker-threads 1"
    )
    unknown_cls = classify_record(unknown_units, include_representative=False)
    check("module58 classifier keeps unknown-size ablation commands unmeasured",
          unknown_cls["workload_key"] is None and unknown_cls["reason"] == "unmapped_cpu",
          diag=str(unknown_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_cpu_ablation_c9_16")
    check("module58 service cache exposes the measured c9_16 ablation curve",
          [record.profile for record in profiles] == [1, 2, 4, 8]
          and math.isclose(
              cache.get("freqduet_cpu_ablation_c9_16", 8).aggregate_rate,
              1.240651142,
              rel_tol=1e-9,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_cpu_ablation_c9_16",),
    )
    check("production load certificate sums parsed c9_16 ablation units",
          report["mapped_counts"] == {"freqduet_cpu_ablation_c9_16": 1}
          and math.isclose(report["mapped_units"]["freqduet_cpu_ablation_c9_16"], 1300.0)
          and math.isclose(report["lambda"]["freqduet_cpu_ablation_c9_16"], 1300.0 / 86400.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module60_freqduet_c3_ablation_completed_history_profile1(check, sch):
    row = {
        "id": "freq-c3",
        "project": "freqduet",
        "signature": "freqduet/c3_8_ablation_shard",
        "description": "FreqDuet c3_8 ablation shard",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 5,
        "ram_mb": 8192,
        "cmd": (
            "python -u scripts/run_freqduet_ablation.py "
            "--configs F_freqduet_gen_highnoise_main_hiro,F_freqduet_gen_odshift_main_hiro "
            "--seeds 42,123,456,789,2026 "
            "--episodes 40 --last-k 20 --workers 5 --worker-threads 1 "
            "--job-start 5 --job-end 10 --clean"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c3_8 ablation shard maps after module60 completed-history certificate",
          cls["workload_key"] == "freqduet_cpu_ablation_c3_8_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 200.0),
          diag=str(cls))

    direct_runner = dict(row)
    direct_runner["id"] = "freq-c3-runner"
    direct_runner["cmd"] = (
        "python3 runner_v3.py --config configs_freqduet/F_freqduet_haar_hiro.yaml "
        "--episodes 20 --seed 123"
    )
    direct_cls = classify_record(direct_runner, include_representative=False)
    check("module60 classifier does not map direct runner_v3 records",
          direct_cls["workload_key"] is None and direct_cls["reason"] == "unmapped_cpu",
          diag=str(direct_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_cpu_ablation_c3_8_completed_history")
    check("module60 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("freqduet_cpu_ablation_c3_8_completed_history", 1).aggregate_rate,
              0.011071600978054631,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_cpu_ablation_c3_8_completed_history",),
    )
    check("production load certificate sums parsed c3_8 ablation units",
          report["mapped_counts"] == {"freqduet_cpu_ablation_c3_8_completed_history": 1}
          and math.isclose(report["mapped_units"]["freqduet_cpu_ablation_c3_8_completed_history"], 200.0)
          and math.isclose(
              report["lambda"]["freqduet_cpu_ablation_c3_8_completed_history"],
              200.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module61_freqduet_c33_ablation_completed_history_profile1(check, sch):
    row = {
        "id": "freq-c33",
        "project": "FreqDuet",
        "signature": "FreqDuet/c33_64_ablation_shard",
        "description": "FreqDuet c33_64 ablation shard",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 48,
        "ram_mb": 98304,
        "cmd": (
            "python scripts/run_freqduet_ablation.py "
            "--configs F_freqduet_terminal_main_hiro,F_freqduet_gen_highnoise_main_hiro "
            "--seeds 42,123,456,789,2026 "
            "--episodes 40 --last-k 20 --workers 48 --worker-threads 1 "
            "--job-start 0 --job-end 38 --clean"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c33_64 ablation shard maps after module61 completed-history certificate",
          cls["workload_key"] == "freqduet_cpu_ablation_c33_64_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 1520.0),
          diag=str(cls))

    direct_runner = dict(row)
    direct_runner["id"] = "freq-c33-runner"
    direct_runner["cmd"] = (
        "python runner_v3.py --config configs_freqduet/F_freqduet_gen_rushshift_main_hiro.yaml "
        "--episodes 40 --seed 2026"
    )
    direct_cls = classify_record(direct_runner, include_representative=False)
    check("module61 classifier does not map c33_64 direct runner_v3 records",
          direct_cls["workload_key"] is None and direct_cls["reason"] == "unmapped_cpu",
          diag=str(direct_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_cpu_ablation_c33_64_completed_history")
    check("module61 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("freqduet_cpu_ablation_c33_64_completed_history", 1).aggregate_rate,
              0.3915115908840592,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_cpu_ablation_c33_64_completed_history",),
    )
    check("production load certificate sums parsed c33_64 ablation units",
          report["mapped_counts"] == {"freqduet_cpu_ablation_c33_64_completed_history": 1}
          and math.isclose(report["mapped_units"]["freqduet_cpu_ablation_c33_64_completed_history"], 1520.0)
          and math.isclose(
              report["lambda"]["freqduet_cpu_ablation_c33_64_completed_history"],
              1520.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module59_simple_sac_sumo_eval_completed_history_profile1(check, sch):
    row = {
        "id": "simple-sumo",
        "project": "SimpleSAC",
        "signature": "H2Oplus/r3_eval_nosnap_wsrl_nosnap_s42_s1001_od0.6",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 2,
        "ram_mb": 581,
        "cmd": (
            "bash /home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC/"
            "run_multiseed_eval.sh wsrl_nosnap_s42 1001 0.6"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("SimpleSAC clean SUMO eval command maps after module59 completed-history certificate",
          cls["workload_key"] == "sumo_eval_simple_sac_c_le2"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 1.0),
          diag=str(cls))

    complex_batch = dict(row)
    complex_batch["id"] = "simple-sumo-complex"
    complex_batch["cmd"] = (
        "bash -lc 'for method in wsrl_nosnap_s42 rlpd_nosnap_s42; do "
        "bash run_multiseed_eval.sh \"$method\" 1001 0.6; done'"
    )
    complex_cls = classify_record(complex_batch, include_representative=False)
    check("module59 classifier does not map complex SimpleSAC bash batches",
          complex_cls["workload_key"] is None and complex_cls["reason"] == "unmapped_cpu",
          diag=str(complex_cls))

    cache = build_default_cache()
    profiles = cache.profiles("sumo_eval_simple_sac_c_le2")
    check("module59 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("sumo_eval_simple_sac_c_le2", 1).aggregate_rate,
              0.001525163882271866,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_sumo_eval_simple_sac_c_le2",),
    )
    check("production load certificate can certify the SimpleSAC eval slice with profile1",
          report["mapped_counts"] == {"sumo_eval_simple_sac_c_le2": 1}
          and math.isclose(report["mapped_units"]["sumo_eval_simple_sac_c_le2"], 1.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_production_coverage_drilldown_separates_population_and_obligations(check, sch):
    rows = [
        {
            "id": "bench",
            "project": "ScheduleurmBench",
            "signature": "ScheduleurmBench/q01_gpu_heavy_jax_matmul/profile_1",
            "submitted_at": 900.0,
            "status": "done",
            "est_vram_mb": 1000,
        },
        {
            "id": "cancel",
            "project": "tmp",
            "signature": "TEST/cpu-training-justification",
            "submitted_at": 901.0,
            "status": "cancelled",
            "est_vram_mb": 0,
            "cmd": "python train.py --device cpu",
        },
        {
            "id": "rl",
            "project": "RE-SAC",
            "signature": "RE-SAC/review/sac_Ant-v2_1",
            "submitted_at": 902.0,
            "status": "done",
            "est_vram_mb": 1000,
            "cmd": "python -m jax_experiments.train --algo sac --env Ant-v2",
        },
        {
            "id": "sumo",
            "project": "TransitDuet",
            "signature": "TransitDuet/eval/main",
            "submitted_at": 903.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 2,
            "cmd": "python eval_operational.py --sumo",
        },
    ]
    report = build_coverage_drilldown(records=rows, window_days=1.0, now_ts=1000.0)
    completed = report["views"]["completed_active_production"]["representative"]
    obligations = report["completed_active_production_obligations"]
    check("production coverage drilldown excludes benchmark/test from production view",
          completed["record_count_window"] == 2
          and completed["mapped_task_count"] == 1
          and completed["unmapped_task_count"] == 1,
          diag=str(report))
    check("production coverage drilldown emits SUMO/transit measurement obligation",
          any(row["bucket"] == "cpu_sumo_transit_eval_or_control"
              and row["status"] == "measurement_required"
              for row in obligations["top_buckets"]),
          diag=str(obligations))
    check("population label excludes benchmark and cancellation",
          population_label(rows[0])["label"] == "excluded_benchmark"
          and population_label(rows[1])["label"] == "excluded_cancelled",
          diag=str([population_label(row) for row in rows]))
    check("bucket obligation maps representative RL but not unmeasured transit CPU",
          bucket_obligation(rows[2])["status"] == "mapped"
          and bucket_obligation(rows[3])["bucket"] == "cpu_sumo_transit_eval_or_control",
          diag=str([bucket_obligation(row) for row in rows]))


def test_production_bucket_probe_manifest_splits_cpu_sumo_sub_buckets(check, sch):
    rows = [
        {
            "id": "bamor",
            "project": "BAMOR",
            "signature": "BAMOR/diagnostic/uniform/steps100000",
            "description": "BAMOR diagnostic uniform seed 0",
            "submitted_at": 900.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 8,
            "ram_mb": 8192,
            "cmd": "python train_compare_baselines.py --method uniform --device cpu --total_steps 100000",
        },
        {
            "id": "freq",
            "project": "FreqHRLNative",
            "signature": "FreqHRLNative/native-promotion-persistent-stress",
            "description": "Freq-HRL native Transit validation",
            "submitted_at": 901.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 72,
            "ram_mb": 120000,
            "cmd": "python -m freq_hrl.experiments.transit.native_promotion_replan_validation --workers 72",
        },
    ]
    report = build_probe_manifest(records=rows, window_days=1.0, now_ts=1000.0)
    sub = {row["sub_bucket"]: row for row in report["sub_buckets"]}
    check("production bucket probe manifest keeps distinct CPU/SUMO sub-buckets",
          "bamor_cpu_training|c_3_8" in sub
          and "transit_freqhrl_cpu_validation|c_65p" in sub,
          diag=str(report))
    check("production bucket probe manifest proposes concurrency profiles",
          report["probe_grid"]["task_concurrency_profiles"] == [1, 2, 4, 8]
          and report["theorem_status"] == "measurement_required",
          diag=str(report))


def test_production_cpu_workload_curve_plan_renders_cpu_only_wrapped_tasks(check, sch):
    rendered = render_cpu_workload_command(
        "python run_eval.py --seed {seed} --episodes {total_units} --out {output_root}/{run_name}",
        run_id="r1",
        sub_bucket="freqduet_cpu_ablation|c_17_32",
        phase="profile_2_per_resource",
        profile=2,
        index=1,
        seed_base=10,
        total_units=50,
        output_root="out",
        node="local",
        cpu_cores=16,
    )
    check("production CPU workload template renders deterministic identity",
          rendered["seed"] == 11
          and rendered["values"]["seed_csv"] == "11"
          and rendered["values"]["progress_total_units"] == 50
          and "freqduet_cpu_ablation_c_17_32" in rendered["run_name"]
          and "--episodes 50" in rendered["cmd"],
          diag=str(rendered))
    rendered_many = render_cpu_workload_command(
        "python run_eval.py --seeds {seed_csv} --episodes {total_units}",
        run_id="r1",
        sub_bucket="freqduet_cpu_ablation|c_17_32",
        phase="profile_2_per_resource",
        profile=2,
        index=1,
        seed_base=100,
        total_units=5,
        output_root="out",
        node="local",
        cpu_cores=16,
        work_items_per_task=4,
    )
    check("production CPU workload template expands seed CSV for worker-heavy tasks",
          rendered_many["seed"] == 104
          and rendered_many["values"]["seed_csv"] == "104,105,106,107"
          and rendered_many["values"]["progress_total_units"] == 20,
          diag=str(rendered_many))
    plan = build_submission_plan(
        run_id="r1",
        sub_bucket="freqduet_cpu_ablation|c_17_32",
        profiles=[1, 2],
        cmd_template="python run_eval.py --seed {seed} --episodes {total_units}",
        node="local",
        cwd="/tmp/scheduleurm-prod-cpu",
        output_root="out",
        seed_base=20,
        total_units=40,
        work_items_per_task=1,
        cpu_cores=16,
        ram_mb=32000,
        project="ScheduleurmBench",
        signature_prefix="ScheduleurmBench/prod_cpu",
        progress_unit="episode",
        progress_wrapper="/tmp/wrapper.py",
    )
    tasks = [task for row in plan["profiles"] for task in row["tasks"]]
    check("production CPU workload plan creates CPU-only profile tasks",
          plan["task_count"] == 3
          and all(task["vram_mb"] == 0 and task["require_node"] == "local" for task in tasks)
          and "progress_wrapper.py" not in tasks[0]["cmd"]
          and "/tmp/wrapper.py" in tasks[0]["cmd"],
          diag=str(plan))

    win_plan = build_submission_plan(
        run_id="r2",
        sub_bucket="freqduet_cpu_ablation|c_17_32",
        profiles=[1],
        cmd_template="python scripts/run_freqduet_ablation.py --seeds {seed_csv} --episodes {total_units} --logs-dir {output_root}/{run_name}",
        node="jtl110cpu2",
        cwd="/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        output_root="out",
        seed_base=30,
        total_units=5,
        work_items_per_task=24,
        cpu_cores=24,
        ram_mb=65536,
        project="ScheduleurmBench",
        signature_prefix="ScheduleurmBench/prod_cpu",
        progress_unit="episode",
        progress_wrapper=r"F:\pkg\algorithm\experiments\csv_progress_wrapper.py",
        progress_wrapper_kind="csv",
        progress_csv_glob="{output_root}/{run_name}/*/diagnostics.csv",
        progress_poll_s=2.0,
        child_shell="argv",
    )
    win_cmd = win_plan["profiles"][0]["tasks"][0]["cmd"]
    check("production CPU workload plan supports Windows CSV progress wrapper",
          "csv_progress_wrapper.py" in win_cmd
          and "--total 120" in win_cmd
          and "--csv-glob" in win_cmd
          and "diagnostics.csv" in win_cmd
          and "--seeds 30,31,32" in win_cmd
          and "bash -lc" not in win_cmd,
          diag=win_cmd)


def test_live_validation_compares_replay_and_observed_jct(check, sch):
    replay = {
        "results": [
            {
                "policy": "calibrated_guarded_knee",
                "profiles": {"w": 2},
                "makespan_s": 100.0,
                "mean_flow_s": 80.0,
                "p90_flow_s": 102.0,
                "completed_jobs": 2,
            }
        ]
    }
    live = {
        "run_id": "unit_live",
        "jobs": [
            {"job_id": "j0", "arrival_s": 0.0, "completion_s": 58.0},
            {"job_id": "j1", "arrival_s": 0.0, "completion_s": 102.0},
        ],
    }
    report = compare_replay_to_live(replay, live, max_relative_error=0.10)
    check("live validation accepts small replay-to-live JCT error",
          report["usable_for_live_sanity"]
          and report["completed_jobs_match"]
          and report["observed"]["completed_jobs"] == 2,
          diag=str(report))

    censored = compare_replay_to_live(
        replay,
        {
            "run_id": "unit_censored",
            "jobs": [
                {"job_id": "j0", "flow_s": 60.0},
                {"job_id": "j1", "flow_s": 100.0, "censored": True},
            ],
        },
    )
    check("live validation refuses censored completion rows for theorem-facing sanity",
          censored["observed"]["censored_jobs"] == 1
          and not censored["usable_for_live_sanity"],
          diag=str(censored))


def test_composed_live_proxy_reuses_trace_replay_semantics(check, sch):
    source = sch.tmp / "profile_1_per_resource_summary.json"
    source.write_text(
        """{
  "phase": "profile_1_per_resource",
  "aggregate_active_rate_unit_s": 10.0,
  "placement_valid": true,
  "rate_units": ["unit"],
  "rates_unit_s": [10.0],
  "running_count": 1,
  "status_counts": {"running": 1}
}
""",
        encoding="utf-8",
    )
    replay = {
        "results": [
            {
                "policy": "calibrated_guarded_knee",
                "profiles": {"cpu_heavy_local_bench": 1},
                "makespan_s": 25600.0,
                "mean_flow_s": 12850.0,
                "p90_flow_s": 23000.0,
                "completed_jobs": 256,
            }
        ]
    }
    live = build_composed_live_report(
        replay,
        taskset_name="q10_cpu_host_bound",
        source_paths={"cpu_heavy_local_bench": [source]},
        policy="calibrated_guarded_knee",
        run_id="unit_composed_live",
    )
    comparison = compare_replay_to_live(
        replay,
        live,
        policy="calibrated_guarded_knee",
        max_relative_error=0.01,
    )
    check("composed live proxy emits uncensored job completions on the replay trace",
          comparison["usable_for_live_sanity"]
          and live["profiles"] == {"cpu_heavy_local_bench": 1}
          and len(live["jobs"]) == 256,
          diag=str(comparison))


def test_capacity_lp_and_drift_margin(check, sch):
    cap = solve_capacity_slack([
        {"action_id": "serve_i", "service_vector": {"i": 1.0, "j": 0.0}},
        {"action_id": "serve_j", "service_vector": {"i": 0.0, "j": 1.0}},
    ], {"i": 0.3, "j": 0.2})
    check("capacity LP recovers two-action toy slack",
          cap["status"] == "optimal" and math.isclose(cap["delta"], 0.25, abs_tol=1e-8),
          diag=str(cap))

    zero = solve_capacity_slack([
        {"action_id": "serve_i", "service_vector": {"i": 1.0, "j": 0.0}},
        {"action_id": "serve_j", "service_vector": {"i": 0.0, "j": 1.0}},
    ], {"i": 0.5, "j": 0.5})
    check("capacity LP refuses zero-slack theorem certificate",
          zero["status"] == "optimal" and math.isclose(zero["delta"], 0.0, abs_tol=1e-8)
          and not zero["usable_for_theorem"],
          diag=str(zero))

    drift = drift_margin_certificate(
        delta=0.5,
        L=0.1,
        rho=0.5,
        epsilon_est=0.1,
        beta=0.05,
        alpha1=0.1,
        B=1.0,
        P0=2.0,
        alpha0=1.0,
        alpha=1.0,
    )
    check("drift margin reports eta and finite-set threshold",
          math.isclose(drift["eta"], 0.2)
          and drift["finite_set_threshold_N"] == 25
          and drift["usable_for_theorem"],
          diag=str(drift))

    failed = drift_margin_certificate(
        delta=0.1,
        L=1.0,
        rho=0.2,
        epsilon_est=0.1,
        beta=0.0,
        alpha1=0.0,
        B=1.0,
        P0=0.0,
        alpha0=0.0,
    )
    summary = summarize_certificate(failed)
    check("report marks nonpositive eta as FAIL",
          summary["status"] == "FAIL" and not summary["usable_for_theorem"],
          diag=str(summary))

    open_summary = summarize_certificate({
        "usable_for_theorem": True,
        "eta": 0.1,
        "delta": 0.2,
        "L": 0.0,
        "rho": 0.0,
        "epsilon_est": 0.0,
        "P0": 0.0,
        "beta": 0.0,
        "alpha0": 0.0,
        "alpha1": 0.0,
    })
    check("report keeps missing constants as empirical-open",
          open_summary["status"] == "EMPIRICAL-OPEN"
          and "NA" in open_summary["calibration_table_md"]
          and not open_summary["usable_for_theorem"],
          diag=str(open_summary))

    complete_summary = summarize_certificate({
        "usable_for_theorem": True,
        "Amax_i": {"i": 2.0},
        "Smax_i": {"i": 3.0},
        "B": 10.0,
        "eta": 0.1,
        "delta": 0.2,
        "L": 0.0,
        "rho": 0.0,
        "epsilon_est": 0.0,
        "P0": 0.0,
        "beta": 0.0,
        "alpha0": 0.0,
        "alpha1": 0.0,
    })
    check("report accepts complete per-class theorem constants",
          complete_summary["status"] == "PASS"
          and complete_summary["usable_for_theorem"]
          and "EMPIRICAL-OPEN" not in complete_summary["calibration_table_md"],
          diag=str(complete_summary))
