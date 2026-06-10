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


def test_module71_c9_native_and_runner_residual_completed_history_profile1(check, sch):
    native = {
        "id": "module71-native-c9",
        "project": "TransitDuet",
        "signature": "TransitDuet/native/c9",
        "description": "Transit native promotion c9_16",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 16,
        "ram_mb": 8192,
        "cwd": "/home/erzhu419/mine_code/TransitDuet",
        "cmd": (
            "PYTHONPATH=transit_hrl python3 -m "
            "freq_hrl.experiments.transit.native_promotion_replan_validation "
            "--preset persistent_stress --stress-profile final_delta_floor_reward_wait_v31 "
            "--seed-index-start 0 --seed-index-end 8 --episodes 1"
        ),
    }
    native_cls = classify_record(native, include_representative=False)
    check("Module71 maps residual c9_16 native-promotion records with parsed seed-episode units",
          native_cls["workload_key"] == "transit_native_promotion_c9_16_residual_completed_history"
          and native_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(native_cls["units"]), 8.0),
          diag=str(native_cls))

    bounded = dict(native)
    bounded["id"] = "module71-native-bounded-c9"
    bounded["cmd"] = bounded["cmd"].replace(
        "final_delta_floor_reward_wait_v31",
        "bounded_wait_nofinal_v14",
    )
    bounded_cls = classify_record(bounded, include_representative=False)
    check("Module71 isolates bounded_wait_nofinal_v14 c9_16 native records",
          bounded_cls["workload_key"] == "transit_native_promotion_c9_16_bounded_wait_completed_history"
          and math.isclose(float(bounded_cls["units"]), 8.0),
          diag=str(bounded_cls))

    native_pyc = {
        **native,
        "id": "module71-native-python-c",
        "project": "FreqHRL",
        "cmd": (
            "python3 -c \"from freq_hrl.experiments.transit import "
            "native_promotion_replan_validation as v; "
            "seeds=list(range(4)); v.run_validation(seeds=seeds, episodes=2)\""
        ),
    }
    native_pyc_cls = classify_record(native_pyc, include_representative=False)
    check("Module71 accepts python -c native run_validation seed lists by AST",
          native_pyc_cls["workload_key"] == "transit_native_promotion_c9_16_residual_completed_history"
          and math.isclose(float(native_pyc_cls["units"]), 8.0),
          diag=str(native_pyc_cls))

    runner = {
        "id": "module71-runner-c9",
        "project": "freqduet",
        "signature": "freqduet/auto-adopted/p432269",
        "description": "residual c9_16 runner_v3 config",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 12,
        "ram_mb": 8192,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "python runner_v3.py --config "
            "configs_freqduet/F_freqduet_gen_odshift_allfreq_hiro.yaml "
            "--episodes 40 --seed 2026 --no-resume"
        ),
    }
    runner_cls = classify_record(runner, include_representative=False)
    check("Module71 maps residual c9_16 runner_v3 records with episode units",
          runner_cls["workload_key"] == "freqduet_runner_v3_c9_16_residual_completed_history"
          and runner_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(runner_cls["units"]), 40.0),
          diag=str(runner_cls))

    exact = dict(runner)
    exact["id"] = "module71-exact-module57"
    exact["cmd"] = exact["cmd"].replace(
        "F_freqduet_gen_odshift_allfreq_hiro.yaml",
        "F_allfreq_alllayers_hiro.yaml",
    )
    exact_cls = classify_record(exact, include_representative=False)
    check("Module71 runner residual classifier does not steal Module57 exact config",
          exact_cls["workload_key"] == "freqduet_runner_v3_allfreq_alllayers_c9_16",
          diag=str(exact_cls))

    cache = build_default_cache()
    expected_rates = {
        "transit_native_promotion_c9_16_bounded_wait_completed_history": 0.0037922214849335067,
        "transit_native_promotion_c9_16_residual_completed_history": 0.03508487702580752,
        "freqduet_runner_v3_c9_16_residual_completed_history": 0.03045966967156635,
    }
    check("Module71 service cache exposes separated c9_16 native and runner lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({key: [record.snapshot() for record in cache.profiles(key)]
                    for key in expected_rates}))

    report = build_production_load_certificate(
        records=[native, runner],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_transit_native_promotion_c9_16_residual_completed_history",
            "production_freqduet_runner_v3_c9_16_residual_completed_history",
        ),
    )
    check("Production load certificate sums Module71 c9_16 parsed units",
          report["mapped_counts"] == {
              "transit_native_promotion_c9_16_residual_completed_history": 1,
              "freqduet_runner_v3_c9_16_residual_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["transit_native_promotion_c9_16_residual_completed_history"], 8.0)
          and math.isclose(report["mapped_units"]["freqduet_runner_v3_c9_16_residual_completed_history"], 40.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module72_cfcmt_sumo_cle2_completed_history_profile1(check, sch):
    base = {
        "project": "CFCMT",
        "signature": "CFCMT/auto-adopted/p72",
        "description": "CFCMT c_le2 production row",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1024,
        "cwd": "/home/erzhu419/mine_code/CFCMT",
    }
    rows = [
        {
            **base,
            "id": "module72-feed",
            "cmd": (
                "python3 H2Oplus/bus_h2o/data/gtfs_to_h2o_xlsx.py "
                "--city MBTA --gtfs-dir gtfs --all-routes"
            ),
            "expected_key": "cfcmt_feed_conversion_c_le2_completed_history",
            "expected_units": 1.0,
        },
        {
            **base,
            "id": "module72-env",
            "cmd": (
                "python3 H2Oplus/bus_h2o/data/validate_h2o_city_env.py "
                "env_dir --max-steps 20000 --out validation_smoke.json"
            ),
            "expected_key": "cfcmt_env_validation_c_le2_completed_history",
            "expected_units": 20000.0,
        },
        {
            **base,
            "id": "module72-generation",
            "cmd": (
                "python3 -m cf_h2o.eval.sumo_apc_avl_sumo_generation "
                "--duration-sec 3600 --run-sumo --out generation.json"
            ),
            "expected_key": "cfcmt_sumo_generation_c_le2_completed_history",
            "expected_units": 3600.0,
        },
        {
            **base,
            "id": "module72-snapshot",
            "cmd": (
                "python3 -m cf_h2o.eval.sumo_apc_avl_snapshot_generation "
                "--stage2-report cf_h2o/results/sumo_apc_avl_sumo_generation_control_full_4h.json "
                "--snapshot-period 60 --out snapshot.json"
            ),
            "expected_key": "cfcmt_snapshot_generation_c_le2_completed_history",
            "expected_units": 240.0,
        },
        {
            **base,
            "id": "module72-rollout",
            "cmd": (
                "python3 cf_h2o/eval/sumo_policy_rollout_validation.py "
                "--stage2-report cf_h2o/results/sumo_apc_avl_sumo_generation_control_full_4h.json "
                "--max-events-per-city-policy 40 --policies no_hold,daganzo_policy "
                "--out rollout.json"
            ),
            "expected_key": "cfcmt_policy_rollout_c_le2_completed_history",
            "expected_units": 80.0,
        },
        {
            **base,
            "id": "module72-phase2",
            "cmd": (
                "python3 cf_h2o/eval/traffic_signal_sumo_phase2.py "
                "--out traffic_signal_sumo_phase2.json"
            ),
            "expected_key": "cfcmt_traffic_signal_phase2_c_le2_completed_history",
            "expected_units": 1.0,
        },
    ]
    for row in rows:
        expected_key = row.pop("expected_key")
        expected_units = row.pop("expected_units")
        cls = classify_record(row, include_representative=False)
        check(f"Module72 maps CFCMT c_le2 row {row['id']} to its script-level certificate",
              cls["workload_key"] == expected_key
              and cls["mapping_mode"] == "strict_measured"
              and math.isclose(float(cls["units"]), expected_units),
              diag=str(cls))
        row["expected_key"] = expected_key
        row["expected_units"] = expected_units

    cache = build_default_cache()
    expected_rates = {
        "cfcmt_feed_conversion_c_le2_completed_history": 0.0006767211093289617,
        "cfcmt_env_validation_c_le2_completed_history": 11.792971438978132,
        "cfcmt_sumo_generation_c_le2_completed_history": 18.52709054097842,
        "cfcmt_snapshot_generation_c_le2_completed_history": 0.028279002327485144,
        "cfcmt_policy_rollout_c_le2_completed_history": 10.848108919227917,
        "cfcmt_traffic_signal_phase2_c_le2_completed_history": 0.0012016200150934825,
    }
    check("Module72 service cache exposes six separated CFCMT c_le2 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({key: [record.snapshot() for record in cache.profiles(key)]
                    for key in expected_rates}))

    report_rows = [
        {key: value for key, value in row.items() if not key.startswith("expected_")}
        for row in rows
    ]
    report = build_production_load_certificate(
        records=report_rows,
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_cfcmt_feed_conversion_c_le2_completed_history",
            "production_cfcmt_env_validation_c_le2_completed_history",
            "production_cfcmt_sumo_generation_c_le2_completed_history",
            "production_cfcmt_snapshot_generation_c_le2_completed_history",
            "production_cfcmt_policy_rollout_c_le2_completed_history",
            "production_cfcmt_traffic_signal_phase2_c_le2_completed_history",
        ),
    )
    check("Production load certificate sums Module72 CFCMT parsed units",
          report["mapped_counts"] == {key: 1 for key in expected_rates}
          and math.isclose(report["mapped_units"]["cfcmt_feed_conversion_c_le2_completed_history"], 1.0)
          and math.isclose(report["mapped_units"]["cfcmt_env_validation_c_le2_completed_history"], 20000.0)
          and math.isclose(report["mapped_units"]["cfcmt_sumo_generation_c_le2_completed_history"], 3600.0)
          and math.isclose(report["mapped_units"]["cfcmt_snapshot_generation_c_le2_completed_history"], 240.0)
          and math.isclose(report["mapped_units"]["cfcmt_policy_rollout_c_le2_completed_history"], 80.0)
          and math.isclose(report["mapped_units"]["cfcmt_traffic_signal_phase2_c_le2_completed_history"], 1.0)
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


def test_module66_freqduet_runner_c3_completed_history_profile1(check, sch):
    row = {
        "id": "freq-runner-c3",
        "project": "freqduet",
        "signature": "freqduet/auto-adopted/p1975560",
        "description": "FreqDuet c3_8 direct runner task",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 5,
        "ram_mb": 4096,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "/usr/bin/python3 runner_v3.py --config "
            "configs_freqduet/F_freqduet_haar_hiro.yaml "
            "--episodes 20 --seed 123 --no-resume --upper-warmup-eps 10"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c3_8 direct runner_v3 maps after module66 completed-history certificate",
          cls["workload_key"] == "freqduet_runner_v3_c3_8_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 20.0),
          diag=str(cls))

    ablation = dict(row)
    ablation["id"] = "freq-c3-ablation-still-module60"
    ablation["cmd"] = (
        "python -u scripts/run_freqduet_ablation.py "
        "--configs F_freqduet_gen_highnoise_main_hiro,F_freqduet_gen_odshift_main_hiro "
        "--seeds 42,123 --episodes 40 --last-k 20 --workers 5 --worker-threads 1 "
        "--job-start 0 --job-end 2 --clean"
    )
    ablation_cls = classify_record(ablation, include_representative=False)
    check("module66 classifier does not steal c3_8 ablation records from module60",
          ablation_cls["workload_key"] == "freqduet_cpu_ablation_c3_8_completed_history"
          and math.isclose(float(ablation_cls["units"]), 80.0),
          diag=str(ablation_cls))

    non_freqduet = dict(row)
    non_freqduet["id"] = "not-freqduet-runner"
    non_freqduet["project"] = "other"
    non_freqduet["signature"] = "other/runner"
    non_freqduet["cwd"] = "/tmp/other"
    non_freqduet_cls = classify_record(non_freqduet, include_representative=False)
    check("module66 classifier requires FreqDuet project/path/signature evidence",
          non_freqduet_cls["workload_key"] is None
          and non_freqduet_cls["reason"] == "unmapped_cpu",
          diag=str(non_freqduet_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_runner_v3_c3_8_completed_history")
    check("module66 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("freqduet_runner_v3_c3_8_completed_history", 1).aggregate_rate,
              0.006679260715538253,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_runner_v3_c3_8_completed_history",),
    )
    check("production load certificate sums c3_8 runner_v3 episode units",
          report["mapped_counts"] == {"freqduet_runner_v3_c3_8_completed_history": 1}
          and math.isclose(report["mapped_units"]["freqduet_runner_v3_c3_8_completed_history"], 20.0)
          and math.isclose(
              report["lambda"]["freqduet_runner_v3_c3_8_completed_history"],
              20.0 / 86400.0,
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


def test_module62_freqduet_runner_cle2_completed_history_profile1(check, sch):
    row = {
        "id": "freq-runner-cle2",
        "project": "freqduet",
        "signature": "freqduet/c_le2_runner_v3",
        "description": "FreqDuet c_le2 runner_v3 task",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1024,
        "cmd": (
            "python3 runner_v3.py --config configs_freqduet/F_freqduet_terminal_hiro.yaml "
            "--episodes 40 --seed 123 --no-resume"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("FreqDuet c_le2 runner_v3 maps after module62 completed-history certificate",
          cls["workload_key"] == "freqduet_runner_v3_c_le2_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 40.0),
          diag=str(cls))

    ablation = dict(row)
    ablation["id"] = "freq-cle2-ablation"
    ablation["cmd"] = (
        "python scripts/run_freqduet_ablation.py --configs F_freqduet_terminal_main_hiro "
        "--seeds 42 --episodes 20 --workers 1 --worker-threads 1"
    )
    ablation_cls = classify_record(ablation, include_representative=False)
    check("module76 classifier maps c_le2 ablation records after completed-history certificate",
          ablation_cls["workload_key"] == "freqduet_cpu_ablation_c_le2_completed_history"
          and math.isclose(float(ablation_cls["units"]), 20.0),
          diag=str(ablation_cls))

    cache = build_default_cache()
    profiles = cache.profiles("freqduet_runner_v3_c_le2_completed_history")
    check("module62 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("freqduet_runner_v3_c_le2_completed_history", 1).aggregate_rate,
              0.010924044632484135,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_freqduet_runner_v3_c_le2_completed_history",),
    )
    check("production load certificate sums parsed c_le2 runner_v3 episode units",
          report["mapped_counts"] == {"freqduet_runner_v3_c_le2_completed_history": 1}
          and math.isclose(report["mapped_units"]["freqduet_runner_v3_c_le2_completed_history"], 40.0)
          and math.isclose(
              report["lambda"]["freqduet_runner_v3_c_le2_completed_history"],
              40.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module76_cle2_residual_completed_history_splits(check, sch):
    ablation = {
        "id": "module76-ablation-cle2",
        "project": "freqduet",
        "signature": "FreqDuet/promotion_cooldown_base_20260531/jtl110cpu2",
        "description": "FreqDuet c_le2 run_freqduet_ablation task",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 4096,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "python -u scripts/run_freqduet_ablation.py "
            "--configs F_freqduet_terminal_final_nopromotion_hiro,F_freqduet_terminal_final_promotion_hiro "
            "--seeds 42,123,456,789,2026 --episodes 40 --last-k 20 "
            "--worker-threads 1 --workers 8 --job-start 5 --job-end 10 --clean"
        ),
    }
    ablation_cls = classify_record(ablation, include_representative=False)
    check("Module76 maps c_le2 run_freqduet_ablation.py with parsed episode units",
          ablation_cls["workload_key"] == "freqduet_cpu_ablation_c_le2_completed_history"
          and ablation_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(ablation_cls["units"]), 200.0),
          diag=str(ablation_cls))

    baseline = dict(ablation)
    baseline["id"] = "module76-baseline-rule"
    baseline["signature"] = "freqduet/auto-adopted/p2712945"
    baseline["cmd"] = (
        "python3 run_baseline_rule.py --upper rule_fixed --episodes 20 "
        "--seed 789 --demand_noise 0.15 --fleet_mode elastic --fleet_min 8 --fleet_max 16"
    )
    baseline_cls = classify_record(baseline, include_representative=False)
    check("Module76 maps c_le2 run_baseline_rule.py with parsed episodes",
          baseline_cls["workload_key"] == "freqduet_baseline_rule_c_le2_completed_history"
          and math.isclose(float(baseline_cls["units"]), 20.0),
          diag=str(baseline_cls))

    preflight = dict(ablation)
    preflight["id"] = "module76-preflight"
    preflight["signature"] = "FreqDuet/valueguard-preflight-remote-v2"
    preflight["cmd"] = (
        "set -euo pipefail\n"
        "PY=/env/bin/python\n"
        "$PY -m py_compile runner_v3.py scripts/run_freqduet_ablation.py\n"
        "echo DONE"
    )
    preflight_cls = classify_record(preflight, include_representative=False)
    check("Module76 maps c_le2 preflight separately from ablation despite py_compile text",
          preflight_cls["workload_key"] == "freqduet_preflight_c_le2_completed_history"
          and math.isclose(float(preflight_cls["units"]), 1.0),
          diag=str(preflight_cls))

    analysis = {
        "id": "module76-analysis",
        "project": "TransitDuet",
        "signature": "TransitDuet/leakage-no-tradeoff-matrix-v27-v5",
        "description": "Transit/FreqHRL analysis matrix c_le2 job",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 4096,
        "cwd": "/home/zhengliang01/scheduleurm_work/TransitDuet",
        "cmd": (
            "PYTHONPATH=transit_hrl /env/bin/python -m "
            "freq_hrl.experiments.leakage_no_tradeoff_matrix "
            "--results-root transit_hrl/results --min-pairs 5 --output-dir out"
        ),
    }
    analysis_cls = classify_record(analysis, include_representative=False)
    check("Module76 maps Transit/FreqHRL analysis matrix jobs as completed analysis units",
          analysis_cls["workload_key"] == "transit_freqhrl_analysis_matrix_c_le2_completed_history"
          and math.isclose(float(analysis_cls["units"]), 1.0),
          diag=str(analysis_cls))

    merge = dict(analysis)
    merge["id"] = "module76-merge"
    merge["signature"] = "TransitDuet/native-real-demand-waitaware-v2-merge"
    merge["cmd"] = (
        "PYTHONPATH=transit_hrl /env/bin/python -m "
        "freq_hrl.experiments.transit.merge_native_real_demand_shards "
        "--input-dirs a b c --min-pairs 24 --output-dir out"
    )
    merge_cls = classify_record(merge, include_representative=False)
    check("Module76 maps Transit/FreqHRL merge jobs as completed merge units",
          merge_cls["workload_key"] == "transit_freqhrl_merge_c_le2_completed_history"
          and math.isclose(float(merge_cls["units"]), 1.0),
          diag=str(merge_cls))

    spin = dict(ablation)
    spin["id"] = "module76-spin"
    spin["project"] = "python"
    spin["signature"] = "python/auto-adopted/p423895"
    spin["cwd"] = "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python"
    spin["cmd"] = "/env/bin/python /home/zhengliang01/scheduleurm_work/tmp/freqduet_autoadopt_spin.py"
    spin_cls = classify_record(spin, include_representative=False)
    check("Module76 leaves auto-adopt spin helpers unmeasured",
          spin_cls["workload_key"] is None and spin_cls["reason"] == "unmapped_cpu",
          diag=str(spin_cls))

    cache = build_default_cache()
    expected_rates = {
        "freqduet_cpu_ablation_c_le2_completed_history": 0.013767135197622387,
        "freqduet_baseline_rule_c_le2_completed_history": 0.2799606783210553,
        "freqduet_preflight_c_le2_completed_history": 0.005813123703207098,
        "transit_freqhrl_analysis_matrix_c_le2_completed_history": 0.005627377518782017,
        "transit_freqhrl_merge_c_le2_completed_history": 0.006077690095218778,
    }
    check("Module76 service cache exposes five c_le2 residual profile1 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({
              key: [record.snapshot() for record in cache.profiles(key)]
              for key in expected_rates
          }))

    report = build_production_load_certificate(
        records=[ablation, baseline, preflight, analysis, merge],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_freqduet_cpu_ablation_c_le2_completed_history",
            "production_freqduet_baseline_rule_c_le2_completed_history",
            "production_freqduet_preflight_c_le2_completed_history",
            "production_transit_freqhrl_analysis_matrix_c_le2_completed_history",
            "production_transit_freqhrl_merge_c_le2_completed_history",
        ),
    )
    check("production load certificate sums Module76 c_le2 residual units",
          report["mapped_counts"] == {
              "freqduet_cpu_ablation_c_le2_completed_history": 1,
              "freqduet_baseline_rule_c_le2_completed_history": 1,
              "freqduet_preflight_c_le2_completed_history": 1,
              "transit_freqhrl_analysis_matrix_c_le2_completed_history": 1,
              "transit_freqhrl_merge_c_le2_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["freqduet_cpu_ablation_c_le2_completed_history"], 200.0)
          and math.isclose(report["mapped_units"]["freqduet_baseline_rule_c_le2_completed_history"], 20.0)
          and math.isclose(report["mapped_units"]["freqduet_preflight_c_le2_completed_history"], 1.0)
          and math.isclose(report["mapped_units"]["transit_freqhrl_analysis_matrix_c_le2_completed_history"], 1.0)
          and math.isclose(report["mapped_units"]["transit_freqhrl_merge_c_le2_completed_history"], 1.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module77_bamor_cle2_training_completed_history_profile1(check, sch):
    compare = {
        "id": "bamor-cle2-compare",
        "project": "BAMOR",
        "signature": "BAMOR/auto-adopted/p720106",
        "description": "BAMOR c_le2 train compare production record",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1412,
        "cwd": "/home/erzhu419/mine_code/BAMOR",
        "cmd": (
            "python3 train_compare_baselines.py --method uniform --total_steps 5000 "
            "--switch_interval 300 --seed 0 --seeds 3 --save_dir out "
            "--device cuda --eval_freq 0"
        ),
    }
    compare_cls = classify_record(compare, include_representative=False)
    check("Module77 maps BAMOR c_le2 train_compare_baselines despite cuda device flag",
          compare_cls["workload_key"] == "bamor_train_compare_c_le2_completed_history"
          and compare_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(compare_cls["units"]), 15000.0),
          diag=str(compare_cls))

    mujoco = dict(compare)
    mujoco["id"] = "bamor-cle2-mujoco"
    mujoco["signature"] = "BAMOR/auto-adopted/p4054856"
    mujoco["cmd"] = (
        "python3 train_bamor_mujoco.py --env mo-mountaincarcontinuous-v0 "
        "--method bamor --total_steps 20000 --switch_interval 10000 "
        "--seed 0 --num_seeds 3 --save_dir out --eval_freq 20000 "
        "--device cuda --hidden 64"
    )
    mujoco_cls = classify_record(mujoco, include_representative=False)
    check("Module77 maps BAMOR c_le2 train_bamor_mujoco with parsed num_seeds times total_steps",
          mujoco_cls["workload_key"] == "bamor_mujoco_c_le2_completed_history"
          and math.isclose(float(mujoco_cls["units"]), 60000.0),
          diag=str(mujoco_cls))

    shard = dict(compare)
    shard["id"] = "bamor-cle2-shard"
    shard["signature"] = "BAMOR/diagnostic-shard/diagnostic_v2_generic_hpc_20260608_113450/offset0/node001"
    shard["ram_mb"] = 65536
    shard["cmd"] = (
        "/env/bin/python run_bamor_diagnostic_shard.py --start 0 --end 2 "
        "--item-offset 0 --methods 'bamor_v2_generic bamor_v2_contrast_generic' "
        "--seed-start 0 --seeds 20 --total-steps 100000 --switch-interval 300 "
        "--save-dir out --device cpu --eval-freq 0 --workers 2 --max-workers 12 "
        "--threads-per-run 2"
    )
    shard_cls = classify_record(shard, include_representative=False)
    check("Module77 maps BAMOR c_le2 diagnostic shard with parsed shard training steps",
          shard_cls["workload_key"] == "bamor_diagnostic_shard_c_le2_completed_history"
          and math.isclose(float(shard_cls["units"]), 200000.0),
          diag=str(shard_cls))

    cache = build_default_cache()
    expected_rates = {
        "bamor_train_compare_c_le2_completed_history": 20.30524282845025,
        "bamor_mujoco_c_le2_completed_history": 23.68602883211671,
        "bamor_diagnostic_shard_c_le2_completed_history": 101.33950173403711,
    }
    check("Module77 service cache exposes three BAMOR c_le2 profile1 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({
              key: [record.snapshot() for record in cache.profiles(key)]
              for key in expected_rates
          }))

    report = build_production_load_certificate(
        records=[compare, mujoco, shard],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_bamor_train_compare_c_le2_completed_history",
            "production_bamor_mujoco_c_le2_completed_history",
            "production_bamor_diagnostic_shard_c_le2_completed_history",
        ),
    )
    check("production load certificate sums BAMOR c_le2 training-step units",
          report["mapped_counts"] == {
              "bamor_train_compare_c_le2_completed_history": 1,
              "bamor_mujoco_c_le2_completed_history": 1,
              "bamor_diagnostic_shard_c_le2_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["bamor_train_compare_c_le2_completed_history"], 15000.0)
          and math.isclose(report["mapped_units"]["bamor_mujoco_c_le2_completed_history"], 60000.0)
          and math.isclose(report["mapped_units"]["bamor_diagnostic_shard_c_le2_completed_history"], 200000.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module78_sumo_cle2_residual_completed_history_profile1(check, sch):
    base = {
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 1024,
    }
    offline = {
        **base,
        "id": "module78-offline",
        "project": "offline-sumo",
        "signature": "offline-sumo/eval-v2/operational",
        "cwd": "/home/erzhu419/mine_code/offline-sumo",
        "cmd": (
            "/env/bin/python -u eval_operational.py --n_workers 12 "
            "--out_csv experiment_output/eval_operational_v3.csv"
        ),
    }
    h2o = {
        **base,
        "id": "module78-h2o-shell",
        "project": "SimpleSAC",
        "signature": "H2Oplus/p0-p1_default_eval_s42",
        "cwd": "/home/erzhu419/mine_code/sumo-rl/H2Oplus/SimpleSAC",
        "cmd": (
            "bash -lc 'for method in p4_nojtt_ep20_s42 p4_nojtt_ep40_s42; do "
            "bash run_multiseed_eval.sh \"$method\" 1001 1.0; done'"
        ),
    }
    zsw = {
        **base,
        "id": "module78-zsw-metrics",
        "project": "ZSW_platform",
        "signature": "ZSW_platform/auto-adopted/p21690",
        "cwd": "/home/erzhu419/zsw_tsp_m0_gpu1/ZSW_platform",
        "cmd": (
            "/env/bin/python TSP_only/src/oracle/m1_metrics_parser.py "
            "--tripinfo out.tripinfo.xml --personinfo out.personinfo.xml --stopinfo out.stopinfo.xml"
        ),
    }
    resco = {
        **base,
        "id": "module78-resco",
        "project": "config",
        "signature": "config/auto-adopted/p48578",
        "cwd": "/home/erzhu419/mine_code/CFCMT/H2Oplus/downloads/traffic_signal_resco/repo/resco_benchmark/config",
        "cmd": (
            "/usr/bin/python3 main.py @arterial4x4 @MPLight episodes:100 testing:5 "
            "save_console_log:False libsumo:True gui:False save_model:False seed:131"
        ),
    }
    nature_extract = {
        **base,
        "id": "module78-nature-extract",
        "project": "Nature_Emissions_gpu1_balanced_p100_20260605_153448",
        "signature": "Nature_Emissions_gpu1_balanced_p100_20260605_153448/auto-adopted/p1803",
        "cwd": "/home/erzhu419/Nature_Emissions_gpu1_balanced_p100_20260605_153448",
        "cmd": (
            "/env/bin/python scripts/real_road_mapping/extract_open_berlin_corridor_actual_demand.py "
            "--output-root runs --output-suffix routeguard050_v1"
        ),
    }
    nature_sumo = {
        **base,
        "id": "module78-nature-sumo",
        "project": "Nature_Emissions_gpu1_balanced_p100_20260605_153448",
        "signature": "Nature_Emissions_gpu1_balanced_p100_20260605_153448/auto-adopted/p4444",
        "cwd": "/home/erzhu419/Nature_Emissions_gpu1_balanced_p100_20260605_153448",
        "cmd": "/env/bin/sumo -c runs/baseline_human/configs/human_baseline.sumocfg --fcd-output out.xml",
    }

    expected = {
        "module78-offline": "offline_sumo_eval_c_le2_completed_history",
        "module78-h2o-shell": "h2oplus_shell_eval_c_le2_completed_history",
        "module78-zsw-metrics": "zsw_metrics_parser_c_le2_completed_history",
        "module78-resco": "resco_config_eval_c_le2_completed_history",
        "module78-nature-extract": "nature_emissions_extract_c_le2_completed_history",
        "module78-nature-sumo": "nature_emissions_sumo_c_le2_completed_history",
    }
    rows = [offline, h2o, zsw, resco, nature_extract, nature_sumo]
    mapped = {row["id"]: classify_record(row, include_representative=False) for row in rows}
    check("Module78 maps each sumo_eval_cpu c_le2 residual command shape separately",
          all(
              mapped[row_id]["workload_key"] == key
              and mapped[row_id]["mapping_mode"] == "strict_measured"
              and math.isclose(float(mapped[row_id]["units"]), 1.0)
              for row_id, key in expected.items()
          ),
          diag=str(mapped))

    clean_simple = {
        **base,
        "id": "module78-clean-simple",
        "project": "SimpleSAC",
        "signature": "H2Oplus/r3_eval_nosnap_wsrl_nosnap_s42_s1001_od0.6",
        "cmd": "bash /repo/SimpleSAC/run_multiseed_eval.sh wsrl_nosnap_s42 1001 0.6",
    }
    clean_cls = classify_record(clean_simple, include_representative=False)
    check("Module78 does not steal clean Module59 SimpleSAC eval records",
          clean_cls["workload_key"] == "sumo_eval_simple_sac_c_le2",
          diag=str(clean_cls))

    cache = build_default_cache()
    expected_rates = {
        "offline_sumo_eval_c_le2_completed_history": 1.2678824791597223e-05,
        "h2oplus_shell_eval_c_le2_completed_history": 7.962432212763291e-05,
        "zsw_metrics_parser_c_le2_completed_history": 0.058559362044645305,
        "resco_config_eval_c_le2_completed_history": 7.378534661398382e-05,
        "nature_emissions_extract_c_le2_completed_history": 0.022218150425893826,
        "nature_emissions_sumo_c_le2_completed_history": 0.003339202281791623,
    }
    check("Module78 service cache exposes six profile1 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({
              key: [record.snapshot() for record in cache.profiles(key)]
              for key in expected_rates
          }))

    report = build_production_load_certificate(
        records=rows,
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_offline_sumo_eval_c_le2_completed_history",
            "production_h2oplus_shell_eval_c_le2_completed_history",
            "production_zsw_metrics_parser_c_le2_completed_history",
            "production_resco_config_eval_c_le2_completed_history",
            "production_nature_emissions_extract_c_le2_completed_history",
            "production_nature_emissions_sumo_c_le2_completed_history",
        ),
    )
    check("production load certificate sums Module78 production-job units",
          report["mapped_counts"] == {key: 1 for key in expected_rates}
          and all(math.isclose(report["mapped_units"][key], 1.0) for key in expected_rates)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module79_transit_c3_8_native_completed_history_profile1(check, sch):
    base = {
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 4,
        "ram_mb": 30000,
        "cwd": "/home/zhengliang01/scheduleurm_work/TransitDuet/transit_hrl",
    }
    promotion = {
        **base,
        "id": "module79-promotion",
        "project": "FreqHRLNative",
        "signature": "FreqHRLNative/native-promotion-persistent-stress-candidate-targetguard347-v8-fixed-interval-cli",
        "cmd": (
            "PY=/env/bin/python; SEEDS=$($PY -c \"seeds=[3551,3581,3681,3961,4071,4101]; "
            "print(' '.join(map(str, seeds[2:6])))\"); "
            "$PY -m freq_hrl.experiments.transit.native_promotion_replan_validation "
            "--preset persistent_stress --variants interval_only --seeds $SEEDS "
            "--episodes 1 --min-pairs 1 --workers 4 --output-dir out"
        ),
    }
    batch = {
        **base,
        "id": "module79-batch",
        "project": "TransitDuet",
        "signature": "TransitDuet/freqhrl-native-real-demand-v8-24pair-node003",
        "cmd": (
            "PYTHONPATH=transit_hrl /env/bin/python -m "
            "freq_hrl.experiments.transit.native_real_demand_control_validation "
            "--sources afc apc --seeds 31 41 51 61 71 81 91 101 111 121 131 141 "
            "--episodes 1 --device cpu --max-series 8 --min-bins 20 --limit 8 "
            "--min-pairs 10 --output-dir out"
        ),
    }
    alighting = {
        **base,
        "id": "module79-alighting",
        "project": "TransitDuet",
        "signature": "TransitDuet/freqhrl-native-real-demand-alighting-safe-v2-24pair-node001-31_41",
        "cmd": (
            "PYTHONPATH=transit_hrl /env/bin/python -m "
            "freq_hrl.experiments.transit.native_real_demand_control_validation "
            "--control-profile alighting_safe_v2 --sources afc apc --seeds 31 41 "
            "--episodes 1 --device cpu --min-pairs 2 --output-dir out"
        ),
    }
    expected = {
        "module79-promotion": (
            "transit_native_promotion_c3_8_persistent_stress_completed_history",
            4.0,
        ),
        "module79-batch": (
            "transit_native_real_demand_batch_c3_8_completed_history",
            48.0,
        ),
        "module79-alighting": (
            "transit_native_real_demand_alighting_c3_8_completed_history",
            8.0,
        ),
    }
    rows = [promotion, batch, alighting]
    mapped = {row["id"]: classify_record(row, include_representative=False) for row in rows}
    check("Module79 maps c3_8 Transit/FreqHRL native command shapes separately",
          all(
              mapped[row_id]["workload_key"] == key
              and mapped[row_id]["mapping_mode"] == "strict_measured"
              and math.isclose(float(mapped[row_id]["units"]), units)
              for row_id, (key, units) in expected.items()
          ),
          diag=str(mapped))

    cache = build_default_cache()
    expected_rates = {
        "transit_native_promotion_c3_8_persistent_stress_completed_history": 0.008254465907208857,
        "transit_native_real_demand_batch_c3_8_completed_history": 0.01341091825756977,
        "transit_native_real_demand_alighting_c3_8_completed_history": 0.01943298920817775,
    }
    check("Module79 service cache exposes three profile1 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({
              key: [record.snapshot() for record in cache.profiles(key)]
              for key in expected_rates
          }))

    report = build_production_load_certificate(
        records=rows,
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_transit_native_promotion_c3_8_persistent_stress_completed_history",
            "production_transit_native_real_demand_batch_c3_8_completed_history",
            "production_transit_native_real_demand_alighting_c3_8_completed_history",
        ),
    )
    check("production load certificate sums Module79 parsed native units",
          report["mapped_counts"] == {key: 1 for key in expected_rates}
          and math.isclose(
              report["mapped_units"]["transit_native_promotion_c3_8_persistent_stress_completed_history"],
              4.0,
          )
          and math.isclose(
              report["mapped_units"]["transit_native_real_demand_batch_c3_8_completed_history"],
              48.0,
          )
          and math.isclose(
              report["mapped_units"]["transit_native_real_demand_alighting_c3_8_completed_history"],
              8.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module63_native_promotion_c17_seedrange_completed_history_profile1(check, sch):
    row = {
        "id": "native-c17",
        "project": "TransitDuet",
        "signature": "TransitDuet/native-promotion-v25/node001-0-64",
        "description": "Transit native promotion c17_32 seed range",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 32,
        "ram_mb": 8192,
        "cmd": (
            "PYTHONPATH=transit_hrl python3 -m "
            "freq_hrl.experiments.transit.native_promotion_replan_validation "
            "--preset persistent_stress --stress-profile reward_floor_throughput_v25 "
            "--seed-index-start 0 --seed-index-end 64 --seed-base 31 "
            "--seed-step 10 --episodes 2 --workers 32 --output-dir out"
        ),
    }
    cls = classify_record(row, include_representative=False)
    check("Transit native promotion c17_32 seed range maps after module63 certificate",
          cls["workload_key"] == "transit_native_promotion_c17_32_seedrange_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 128.0),
          diag=str(cls))

    no_range = dict(row)
    no_range["id"] = "native-c17-no-range"
    no_range["cmd"] = (
        "PYTHONPATH=transit_hrl python3 -m "
        "freq_hrl.experiments.transit.native_promotion_replan_validation "
        "--preset persistent_stress --episodes 1 --workers 32 --output-dir out"
    )
    no_range_cls = classify_record(no_range, include_representative=False)
    check("module63 classifier leaves native commands without seed-index range unmeasured",
          no_range_cls["workload_key"] is None and no_range_cls["reason"] == "unmapped_cpu",
          diag=str(no_range_cls))

    cache = build_default_cache()
    profiles = cache.profiles("transit_native_promotion_c17_32_seedrange_completed_history")
    check("module63 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("transit_native_promotion_c17_32_seedrange_completed_history", 1).aggregate_rate,
              0.07545606639567005,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[row],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_transit_native_promotion_c17_32_seedrange_completed_history",),
    )
    check("production load certificate sums native seed-range episode units",
          report["mapped_counts"] == {"transit_native_promotion_c17_32_seedrange_completed_history": 1}
          and math.isclose(
              report["mapped_units"]["transit_native_promotion_c17_32_seedrange_completed_history"],
              128.0,
          )
          and math.isclose(
              report["lambda"]["transit_native_promotion_c17_32_seedrange_completed_history"],
              128.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module74_c17_residual_completed_history_splits_runner_longtrain_native(check, sch):
    runner = {
        "id": "runner-c17",
        "project": "freqduet",
        "signature": "freqduet/c17-runner",
        "description": "FreqDuet c17 direct runner",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 24,
        "ram_mb": 8192,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "python runner_v3.py --config configs_freqduet/F_freqduet_terminal_hiro.yaml "
            "--episodes 40 --seed 123 --no-resume"
        ),
    }
    runner_cls = classify_record(runner, include_representative=False)
    check("Module74 maps c17_32 runner_v3 records with episode units",
          runner_cls["workload_key"] == "freqduet_runner_v3_c17_32_completed_history"
          and runner_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(runner_cls["units"]), 40.0),
          diag=str(runner_cls))

    longtrain = {
        "id": "longtrain-c17",
        "project": "FreqDuet",
        "signature": "FreqDuet/paper-longtrain/c17",
        "description": "FreqDuet paper longtrain shard",
        "submitted_at": 901.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 30,
        "ram_mb": 32768,
        "cwd": "/home/erzhu419/mine_code/TransitDuet/FreqDuet/freqduet",
        "cmd": (
            "PYTHON=/env/bin/python EPISODES=200 WORKERS=30 "
            "bash scripts/run_freqduet_paper_longtrain_matrix.sh "
            "--job-start 150 --job-end 180 --skip-existing --no-aggregate"
        ),
    }
    longtrain_cls = classify_record(longtrain, include_representative=False)
    check("Module74 maps paper longtrain as one conservative shard unit",
          longtrain_cls["workload_key"] == "freqduet_paper_longtrain_c17_32_completed_history"
          and math.isclose(float(longtrain_cls["units"]), 1.0),
          diag=str(longtrain_cls))

    native = {
        "id": "native-c17-residual",
        "project": "TransitDuet",
        "signature": "TransitDuet/native-c17-residual",
        "description": "Transit native promotion c17 residual",
        "submitted_at": 902.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 32,
        "ram_mb": 32768,
        "cmd": (
            "python3 -c \"from pathlib import Path; "
            "from freq_hrl.experiments.transit import native_promotion_replan_validation as v; "
            "v.run_validation(output_dir=Path('out'), "
            "config_path=v.TRANSIT_DUET_ROOT / 'configs_freqduet' / 'T.yaml', "
            "seeds=[201 + 10*i for i in range(0, 32)], episodes=2, "
            "device='cpu', min_pairs=32, workers=16)\""
        ),
    }
    native_cls = classify_record(native, include_representative=False)
    check("Module74 maps c17_32 native residual seed comprehensions",
          native_cls["workload_key"] == "transit_native_promotion_c17_32_residual_completed_history"
          and math.isclose(float(native_cls["units"]), 64.0),
          diag=str(native_cls))

    seedrange = dict(native)
    seedrange["id"] = "native-c17-still-module63"
    seedrange["cmd"] = (
        "python3 -m freq_hrl.experiments.transit.native_promotion_replan_validation "
        "--seed-index-start 0 --seed-index-end 32 --episodes 1 --workers 32"
    )
    seedrange_cls = classify_record(seedrange, include_representative=False)
    check("Module74 residual classifier does not steal Module63 seed-index records",
          seedrange_cls["workload_key"] == "transit_native_promotion_c17_32_seedrange_completed_history",
          diag=str(seedrange_cls))

    cache = build_default_cache()
    checks = {
        "freqduet_runner_v3_c17_32_completed_history": 0.0176335126709709,
        "freqduet_paper_longtrain_c17_32_completed_history": 0.00006430664806364469,
        "transit_native_promotion_c17_32_residual_completed_history": 0.05909746405816389,
    }
    check("Module74 service cache exposes three conservative profile1 c17 residual classes",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in checks.items()
          ),
          diag=str({key: [record.snapshot() for record in cache.profiles(key)] for key in checks}))

    report = build_production_load_certificate(
        records=[runner, longtrain, native],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_freqduet_runner_v3_c17_32_completed_history",
            "production_freqduet_paper_longtrain_c17_32_completed_history",
            "production_transit_native_promotion_c17_32_residual_completed_history",
        ),
    )
    check("production load certificate certifies Module74 c17 residual split",
          report["mapped_counts"] == {
              "freqduet_paper_longtrain_c17_32_completed_history": 1,
              "freqduet_runner_v3_c17_32_completed_history": 1,
              "transit_native_promotion_c17_32_residual_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["freqduet_runner_v3_c17_32_completed_history"], 40.0)
          and math.isclose(report["mapped_units"]["freqduet_paper_longtrain_c17_32_completed_history"], 1.0)
          and math.isclose(report["mapped_units"]["transit_native_promotion_c17_32_residual_completed_history"], 64.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module68_native_promotion_c33_seed_units_completed_history_profile1(check, sch):
    batch = {
        "id": "native-c33-batch",
        "project": "TransitDuet",
        "signature": "TransitDuet/freq-hrl-native-wait-aware-guarded",
        "description": "Transit native promotion c33_64 guarded batch",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 60,
        "ram_mb": 131072,
        "cmd": (
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
            "freq_hrl.experiments.transit.native_promotion_replan_validation "
            "--config transit_hrl/freq_transitduet/configs_freqduet/T_freqhrl_native_full.yaml "
            "--seeds 201 211 221 231 --episodes 2 --min-pairs 4 "
            "--offpolicy-replay-updates 3 --workers 60 --output-dir out"
        ),
    }
    batch_cls = classify_record(batch, include_representative=False)
    check("Transit native promotion c33_64 CLI seed-list maps to module68 batch class",
          batch_cls["workload_key"] == "transit_native_promotion_c33_64_batch_completed_history"
          and batch_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(batch_cls["units"]), 8.0),
          diag=str(batch_cls))

    pyc = dict(batch)
    pyc["id"] = "native-c33-pyc"
    pyc["project"] = "FreqHRL"
    pyc["signature"] = "freq_hrl_native_learned_wait_same070_cap2_512seed_w16"
    pyc["cmd"] = (
        "bash -lc \"set -e; python3 -c \\\"from pathlib import Path; "
        "from freq_hrl.experiments.transit import native_promotion_replan_validation as v; "
        "seeds=[201 + 10*i for i in range(0,86)]; "
        "v.run_validation(output_dir=Path('out'), config_path=v.PERSISTENT_STRESS_CONFIG, "
        "seeds=seeds, episodes=1, device='cpu', min_pairs=86, workers=16)\\\"\""
    )
    pyc_cls = classify_record(pyc, include_representative=False)
    check("module68 parser proves bash-wrapped python-c seed comprehension by AST",
          pyc_cls["workload_key"] == "transit_native_promotion_c33_64_batch_completed_history"
          and math.isclose(float(pyc_cls["units"]), 86.0),
          diag=str(pyc_cls))

    single = dict(batch)
    single["id"] = "native-c33-single"
    single["project"] = "FreqHRLNative"
    single["signature"] = "FreqHRLNative/native-promotion-persistent-stress-single"
    single["cpu_cores"] = 38
    single["cmd"] = (
        "python3 -m freq_hrl.experiments.transit.native_promotion_replan_validation "
        "--seeds 31 --episodes 1 --workers 38 --output-dir out"
    )
    single_cls = classify_record(single, include_representative=False)
    check("module68 isolates c33_64 single-seed smoke records from batch class",
          single_cls["workload_key"] == "transit_native_promotion_c33_64_single_seed_completed_history"
          and math.isclose(float(single_cls["units"]), 1.0),
          diag=str(single_cls))

    unparseable = dict(batch)
    unparseable["id"] = "native-c33-unparseable"
    unparseable["cmd"] = (
        "python3 -m freq_hrl.experiments.transit.native_promotion_replan_validation "
        "--seeds $SEEDS --episodes 1 --workers 60 --output-dir out"
    )
    unparseable_cls = classify_record(unparseable, include_representative=False)
    check("module68 leaves unparseable shell-expanded seed lists unmeasured",
          unparseable_cls["workload_key"] is None and unparseable_cls["reason"] == "unmapped_cpu",
          diag=str(unparseable_cls))

    cache = build_default_cache()
    batch_profiles = cache.profiles("transit_native_promotion_c33_64_batch_completed_history")
    single_profiles = cache.profiles("transit_native_promotion_c33_64_single_seed_completed_history")
    check("module68 service cache exposes separated c33_64 native lower services",
          [record.profile for record in batch_profiles] == [1]
          and [record.profile for record in single_profiles] == [1]
          and math.isclose(
              cache.get("transit_native_promotion_c33_64_batch_completed_history", 1).aggregate_rate,
              0.06020432113562225,
              rel_tol=1e-12,
          )
          and math.isclose(
              cache.get("transit_native_promotion_c33_64_single_seed_completed_history", 1).aggregate_rate,
              0.003367969313750444,
              rel_tol=1e-12,
          ),
          diag=str({
              "batch": [record.snapshot() for record in batch_profiles],
              "single": [record.snapshot() for record in single_profiles],
          }))

    report = build_production_load_certificate(
        records=[pyc],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_transit_native_promotion_c33_64_batch_completed_history",),
    )
    check("production load certificate sums module68 parsed c33_64 seed-episode units",
          report["mapped_counts"] == {"transit_native_promotion_c33_64_batch_completed_history": 1}
          and math.isclose(
              report["mapped_units"]["transit_native_promotion_c33_64_batch_completed_history"],
              86.0,
          )
          and math.isclose(
              report["lambda"]["transit_native_promotion_c33_64_batch_completed_history"],
              86.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module69_c65p_completed_history_profile1(check, sch):
    ablation = {
        "id": "freq-c65p-ablation",
        "project": "FreqDuet",
        "signature": "FreqDuet/final-driftfb/node003/s2",
        "description": "FreqDuet c65p ablation shard",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 80,
        "ram_mb": 98304,
        "cmd": (
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "python scripts/run_freqduet_ablation.py "
            "--configs F_freqduet_terminal_main_hiro,F_freqduet_gen_highnoise_main_hiro "
            "--seeds 42,123,456,789 "
            "--episodes 40 --last-k 20 --workers 80 --worker-threads 1 "
            "--job-start 0 --job-end 16 --skip-existing"
        ),
    }
    ablation_cls = classify_record(ablation, include_representative=False)
    check("Module69 maps c65p run_freqduet_ablation.py with parsed episode units",
          ablation_cls["workload_key"] == "freqduet_cpu_ablation_c65p_completed_history"
          and ablation_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(ablation_cls["units"]), 640.0),
          diag=str(ablation_cls))

    non_strict_threads = dict(ablation)
    non_strict_threads["id"] = "freq-c65p-ablation-threads2"
    non_strict_threads["cmd"] = non_strict_threads["cmd"].replace("--worker-threads 1", "--worker-threads 2")
    non_strict_cls = classify_record(non_strict_threads, include_representative=False)
    check("Module69 rejects c65p ablation commands with multi-thread workers",
          non_strict_cls["workload_key"] is None and non_strict_cls["reason"] == "unmapped_cpu",
          diag=str(non_strict_cls))

    promoted = dict(ablation)
    promoted["id"] = "freq-c65p-promoted"
    promoted["signature"] = "FreqDuet/promoted-ep100-genbase-r1/node001_s0"
    promoted["description"] = "FreqDuet promoted ep100 gen+external one-batch"
    promoted["cpu_cores"] = 84
    promoted["cmd"] = (
        "PYTHON_BIN=/env/bin/python THREADS=1 bash "
        "scripts/run_freqduet_promoted_ep100_hpc_batch.sh "
        "--job-start 0 --job-end 84 --workers 84 --worker-threads 1 "
        "--shard-label r1_node001_s0"
    )
    promoted_cls = classify_record(promoted, include_representative=False)
    check("Module69 maps promoted ep100 c65p shell batches with job-count units",
          promoted_cls["workload_key"] == "freqduet_promoted_ep100_c65p_completed_history"
          and math.isclose(float(promoted_cls["units"]), 8400.0),
          diag=str(promoted_cls))

    promoted_bad = dict(promoted)
    promoted_bad["id"] = "freq-c65p-promoted-ep50"
    promoted_bad["cmd"] = "EPISODES=50 " + promoted_bad["cmd"]
    promoted_bad_cls = classify_record(promoted_bad, include_representative=False)
    check("Module69 rejects promoted shell batches with non-ep100 override",
          promoted_bad_cls["workload_key"] is None and promoted_bad_cls["reason"] == "unmapped_cpu",
          diag=str(promoted_bad_cls))

    native = dict(ablation)
    native["id"] = "native-c65p-shell-pyc"
    native["project"] = "FreqHRLNative"
    native["signature"] = "FreqHRLNative/native-promotion-persistent-stress-riskcap-512seed-py310-v5"
    native["description"] = "Freq-HRL native c65p command-substitution seed generator"
    native["cpu_cores"] = 71
    native["cmd"] = (
        "export PYTHONDONTWRITEBYTECODE=1; PY=/env/bin/python3.10; "
        "SEEDS=$($PY -c \"print(' '.join(str(201 + 10*i) for i in range(0, 71)))\"); "
        "$PY -m freq_hrl.experiments.transit.native_promotion_replan_validation "
        "--preset persistent_stress --seeds $SEEDS --episodes 2 "
        "--min-pairs 64 --workers 71 --output-dir out"
    )
    native_cls = classify_record(native, include_representative=False)
    check("Module69 proves c65p shell-generated native seed lists by AST",
          native_cls["workload_key"] == "transit_native_promotion_c65p_completed_history"
          and math.isclose(float(native_cls["units"]), 142.0),
          diag=str(native_cls))

    cache = build_default_cache()
    ablation_profiles = cache.profiles("freqduet_cpu_ablation_c65p_completed_history")
    promoted_profiles = cache.profiles("freqduet_promoted_ep100_c65p_completed_history")
    native_profiles = cache.profiles("transit_native_promotion_c65p_completed_history")
    check("Module69 service cache exposes three separated c65p lower services",
          [record.profile for record in ablation_profiles] == [1]
          and [record.profile for record in promoted_profiles] == [1]
          and [record.profile for record in native_profiles] == [1]
          and math.isclose(
              cache.get("freqduet_cpu_ablation_c65p_completed_history", 1).aggregate_rate,
              1.0997359841532857,
              rel_tol=1e-12,
          )
          and math.isclose(
              cache.get("freqduet_promoted_ep100_c65p_completed_history", 1).aggregate_rate,
              1.7274785454630845,
              rel_tol=1e-12,
          )
          and math.isclose(
              cache.get("transit_native_promotion_c65p_completed_history", 1).aggregate_rate,
              0.18213138004622983,
              rel_tol=1e-12,
          ),
          diag=str({
              "ablation": [record.snapshot() for record in ablation_profiles],
              "promoted": [record.snapshot() for record in promoted_profiles],
              "native": [record.snapshot() for record in native_profiles],
          }))

    report = build_production_load_certificate(
        records=[ablation, promoted, native],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_freqduet_cpu_ablation_c65p_completed_history",
            "production_freqduet_promoted_ep100_c65p_completed_history",
            "production_transit_native_promotion_c65p_completed_history",
        ),
    )
    check("Production load certificate sums Module69 c65p parsed units",
          report["mapped_counts"] == {
              "freqduet_cpu_ablation_c65p_completed_history": 1,
              "freqduet_promoted_ep100_c65p_completed_history": 1,
              "transit_native_promotion_c65p_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["freqduet_cpu_ablation_c65p_completed_history"], 640.0)
          and math.isclose(report["mapped_units"]["freqduet_promoted_ep100_c65p_completed_history"], 8400.0)
          and math.isclose(report["mapped_units"]["transit_native_promotion_c65p_completed_history"], 142.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module70_transit_freqhrl_c_le2_completed_history_profile1(check, sch):
    base = {
        "submitted_at": 900.0,
        "status": "done",
        "project": "TransitDuet",
        "signature": "TransitDuet/auto-adopted/p",
        "description": "auto-adopted TransitDuet c_le2",
        "cwd": "/home/erzhu419/mine_code/TransitDuet",
        "est_vram_mb": 0,
        "cpu_cores": 2,
        "ram_mb": 1024,
    }
    sweep = {
        **base,
        "id": "module70-sweep",
        "cmd": (
            "python3 -m freq_hrl.experiments.trading.promotion_sweep "
            "--seeds 42 123 456 789 2026 --steps 720 --assets 3 "
            "--thresholds 0.00035 0.00050 0.00065 0.00080 "
            "--ratios 0.20 0.30 0.40 "
            "--mid-gains 0.0 0.5 1.0 --output-dir out"
        ),
    }
    sweep_cls = classify_record(sweep, include_representative=False)
    check("Module70 maps TransitDuet trading sweep c_le2 with script-default grid units",
          sweep_cls["workload_key"] == "transit_trading_sweep_c_le2_completed_history"
          and math.isclose(float(sweep_cls["units"]), 15552000.0),
          diag=str(sweep_cls))

    policy = {
        **base,
        "id": "module70-policy",
        "cmd": (
            "python3 -m freq_hrl.experiments.trading.policy_entry "
            "--mode train --policy pg_linear "
            "--train-seeds 42 123 456 789 2026 "
            "--eval-seeds 31415 27182 16180 11235 4242 "
            "--steps 720 --assets 3 --pg-iterations 32 --output-dir out"
        ),
    }
    policy_cls = classify_record(policy, include_representative=False)
    check("Module70 maps TransitDuet trading policy c_le2 with train plus eval units",
          policy_cls["workload_key"] == "transit_trading_policy_c_le2_completed_history"
          and math.isclose(float(policy_cls["units"]), 356400.0),
          diag=str(policy_cls))

    surrogate = {
        **base,
        "id": "module70-surrogate",
        "cmd": (
            "python3 -m freq_hrl.experiments.transit.gap_closure_validation "
            "--train-seeds 11 23 --eval-seeds 101 131 "
            "--steps 96 --iterations 3 --output-dir out"
        ),
    }
    surrogate_cls = classify_record(surrogate, include_representative=False)
    check("Module70 maps Transit surrogate c_le2 with four-variant corridor-step units",
          surrogate_cls["workload_key"] == "transit_surrogate_validation_c_le2_completed_history"
          and math.isclose(float(surrogate_cls["units"]), 6144.0),
          diag=str(surrogate_cls))

    native = {
        **base,
        "id": "module70-native",
        "cmd": (
            "python3 -m freq_hrl.experiments.transit.native_promotion_replan_validation "
            "--seeds 31 41 51 61 71 81 91 101 --episodes 1 "
            "--min-pairs 64 --output-dir out"
        ),
    }
    native_cls = classify_record(native, include_representative=False)
    check("Module70 native promotion c_le2 does not multiply work units by min-pairs",
          native_cls["workload_key"] == "transit_native_promotion_c_le2_completed_history"
          and math.isclose(float(native_cls["units"]), 32.0),
          diag=str(native_cls))

    native_control = {
        **base,
        "id": "module70-native-control",
        "cmd": (
            "python3 -m freq_hrl.experiments.transit.native_real_demand_control_validation "
            "--sources afc apc --seeds 31 41 51 --episodes 1 "
            "--max-series 2 --min-bins 4 --limit 1000 --min-pairs 3 --output-dir out"
        ),
    }
    native_control_cls = classify_record(native_control, include_representative=False)
    check("Module70 maps native real-demand control c_le2 with source variant episode units",
          native_control_cls["workload_key"] == "transit_native_control_c_le2_completed_history"
          and math.isclose(float(native_control_cls["units"]), 12.0),
          diag=str(native_control_cls))

    import_smoke = {
        **base,
        "id": "module70-import-smoke",
        "cpu_cores": 1,
        "ram_mb": 4096,
        "cmd": (
            "F:\\v\\Scripts\\python.exe -c \"import sys; sys.path.insert(0, 'transit_hrl'); "
            "from freq_hrl.experiments.transit import native_promotion_replan_validation as v; "
            "print('IMPORT_OK')\" & exit /b %ERRORLEVEL%"
        ),
    }
    import_cls = classify_record(import_smoke, include_representative=False)
    check("Module70 keeps Windows import smoke as its own singleton class",
          import_cls["workload_key"] == "transit_freqhrl_import_smoke_c_le2_completed_history"
          and math.isclose(float(import_cls["units"]), 1.0),
          diag=str(import_cls))

    cache = build_default_cache()
    expected_rates = {
        "transit_trading_sweep_c_le2_completed_history": 899.8910320624362,
        "transit_trading_policy_c_le2_completed_history": 650.1093258685299,
        "transit_surrogate_validation_c_le2_completed_history": 84.11024942110117,
        "transit_native_promotion_c_le2_completed_history": 0.00540690633698809,
        "transit_native_control_c_le2_completed_history": 0.02835729525976647,
        "transit_freqhrl_import_smoke_c_le2_completed_history": 0.001756136750968169,
    }
    check("Module70 service cache exposes six separated c_le2 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({key: [record.snapshot() for record in cache.profiles(key)]
                    for key in expected_rates}))

    report = build_production_load_certificate(
        records=[sweep, policy, surrogate, native, native_control, import_smoke],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_transit_trading_sweep_c_le2_completed_history",
            "production_transit_trading_policy_c_le2_completed_history",
            "production_transit_surrogate_validation_c_le2_completed_history",
            "production_transit_native_promotion_c_le2_completed_history",
            "production_transit_native_control_c_le2_completed_history",
            "production_transit_freqhrl_import_smoke_c_le2_completed_history",
        ),
    )
    check("Production load certificate sums Module70 c_le2 parsed units",
          report["mapped_counts"] == {
              "transit_trading_sweep_c_le2_completed_history": 1,
              "transit_trading_policy_c_le2_completed_history": 1,
              "transit_surrogate_validation_c_le2_completed_history": 1,
              "transit_native_promotion_c_le2_completed_history": 1,
              "transit_native_control_c_le2_completed_history": 1,
              "transit_freqhrl_import_smoke_c_le2_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["transit_trading_sweep_c_le2_completed_history"], 15552000.0)
          and math.isclose(report["mapped_units"]["transit_trading_policy_c_le2_completed_history"], 356400.0)
          and math.isclose(report["mapped_units"]["transit_surrogate_validation_c_le2_completed_history"], 6144.0)
          and math.isclose(report["mapped_units"]["transit_native_promotion_c_le2_completed_history"], 32.0)
          and math.isclose(report["mapped_units"]["transit_native_control_c_le2_completed_history"], 12.0)
          and math.isclose(report["mapped_units"]["transit_freqhrl_import_smoke_c_le2_completed_history"], 1.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module64_bamor_c3_training_completed_history_profile1(check, sch):
    compare = {
        "id": "bamor-compare",
        "project": "BAMOR",
        "signature": "BAMOR/diagnostic/uniform/steps100000/switch300/eval0",
        "description": "BAMOR diagnostic uniform seed 0",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 8,
        "ram_mb": 8192,
        "cwd": "/home/erzhu419/mine_code/BAMOR",
        "cmd": (
            "python train_compare_baselines.py --method uniform --total_steps 100000 "
            "--switch_interval 300 --seed 0 --seeds 1 --save_dir out "
            "--device cpu --eval_freq 0"
        ),
    }
    cls = classify_record(compare, include_representative=False)
    check("BAMOR c3_8 train_compare_baselines maps to module67 script-level certificate",
          cls["workload_key"] == "bamor_train_compare_c3_8_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 100000.0),
          diag=str(cls))

    mujoco = dict(compare)
    mujoco["id"] = "bamor-mujoco"
    mujoco["signature"] = "BAMOR/mujoco/mo-halfcheetah-v5/bamor/steps50000"
    mujoco["cmd"] = (
        "python -u train_bamor_mujoco.py --env mo-halfcheetah-v5 --method bamor "
        "--total_steps 50000 --switch_interval 25000 --seed 0 --num_seeds 2 "
        "--save_dir out --eval_freq 50000 --device cpu --hidden 64"
    )
    mujoco_cls = classify_record(mujoco, include_representative=False)
    check("BAMOR c3_8 train_bamor_mujoco maps parsed num_seeds times total_steps",
          mujoco_cls["workload_key"] == "bamor_mujoco_c3_8_completed_history"
          and math.isclose(float(mujoco_cls["units"]), 100000.0),
          diag=str(mujoco_cls))

    shard = dict(compare)
    shard["id"] = "bamor-shard"
    shard["signature"] = "BAMOR/diagnostic-shard/offset56/node004"
    shard["cmd"] = (
        "python run_bamor_diagnostic_shard.py --start 58 --end 66 --item-offset 56 "
        "--methods 'uniform bamor_oracle_cone bamor_oracle bamor_fixed_cone bamor "
        "bamor_ns bamor_ns_eps bamor_nc fixed' --seed-start 0 --seeds 20 "
        "--total-steps 100000 --switch-interval 300 --save-dir out --device cpu "
        "--eval-freq 0 --workers 8 --max-workers 12 --threads-per-run 2"
    )
    shard_cls = classify_record(shard, include_representative=False)
    check("BAMOR c3_8 diagnostic shard maps parsed shard items times total_steps",
          shard_cls["workload_key"] == "bamor_diagnostic_shard_c3_8_completed_history"
          and math.isclose(float(shard_cls["units"]), 800000.0),
          diag=str(shard_cls))

    all_method = dict(compare)
    all_method["id"] = "bamor-all-method"
    all_method["cmd"] = (
        "python train_compare_baselines.py --method all --total_steps 100000 "
        "--seed 0 --seeds 1 --device cpu"
    )
    all_cls = classify_record(all_method, include_representative=False)
    check("module64 classifier leaves method-all BAMOR commands unmeasured",
          all_cls["workload_key"] is None and all_cls["reason"] == "unmapped_cpu",
          diag=str(all_cls))

    cache = build_default_cache()
    compare_profiles = cache.profiles("bamor_train_compare_c3_8_completed_history")
    mujoco_profiles = cache.profiles("bamor_mujoco_c3_8_completed_history")
    shard_profiles = cache.profiles("bamor_diagnostic_shard_c3_8_completed_history")
    check("module67 service cache exposes script-level BAMOR profile1 lower services",
          [record.profile for record in compare_profiles] == [1]
          and [record.profile for record in mujoco_profiles] == [1]
          and [record.profile for record in shard_profiles] == [1]
          and math.isclose(
              cache.get("bamor_train_compare_c3_8_completed_history", 1).aggregate_rate,
              10.055276060265134,
              rel_tol=1e-12,
          )
          and math.isclose(
              cache.get("bamor_mujoco_c3_8_completed_history", 1).aggregate_rate,
              37.51019245700603,
              rel_tol=1e-12,
          )
          and math.isclose(
              cache.get("bamor_diagnostic_shard_c3_8_completed_history", 1).aggregate_rate,
              412.6128023797939,
              rel_tol=1e-12,
          ),
          diag=str({
              "compare": [record.snapshot() for record in compare_profiles],
              "mujoco": [record.snapshot() for record in mujoco_profiles],
              "shard": [record.snapshot() for record in shard_profiles],
          }))

    report = build_production_load_certificate(
        records=[shard],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_bamor_diagnostic_shard_c3_8_completed_history",),
    )
    check("production load certificate sums BAMOR c3_8 training-step units",
          report["mapped_counts"] == {"bamor_diagnostic_shard_c3_8_completed_history": 1}
          and math.isclose(report["mapped_units"]["bamor_diagnostic_shard_c3_8_completed_history"], 800000.0)
          and math.isclose(
              report["lambda"]["bamor_diagnostic_shard_c3_8_completed_history"],
              800000.0 / 86400.0,
          )
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module75_bamor_c9_16_training_completed_history_profile1(check, sch):
    compare = {
        "id": "bamor-c9-compare",
        "project": "BAMOR",
        "signature": "BAMOR/diagnostic/bamor_v2_seq/steps300",
        "description": "BAMOR c9 train compare baseline",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 10,
        "ram_mb": 16384,
        "cwd": "/home/erzhu419/mine_code/BAMOR",
        "cmd": (
            "python train_compare_baselines.py --method bamor_v2_seq "
            "--total_steps 300 --seeds 1 --save_dir out --eval_freq 0 "
            "--device cpu"
        ),
    }
    compare_cls = classify_record(compare, include_representative=False)
    check("BAMOR c9_16 train_compare_baselines maps to module75 script-level certificate",
          compare_cls["workload_key"] == "bamor_train_compare_c9_16_completed_history"
          and compare_cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(compare_cls["units"]), 300.0),
          diag=str(compare_cls))

    mujoco = dict(compare)
    mujoco["id"] = "bamor-c9-mujoco"
    mujoco["signature"] = "BAMOR/mujoco/mo-mountaincarcontinuous-v0/uniform/steps6000"
    mujoco["cpu_cores"] = 9
    mujoco["cmd"] = (
        "python3 train_bamor_mujoco.py --env mo-mountaincarcontinuous-v0 "
        "--method uniform --total_steps 6000 --num_seeds 1 --save_dir out "
        "--device cpu --eval_freq 100000 --hidden 64"
    )
    mujoco_cls = classify_record(mujoco, include_representative=False)
    check("BAMOR c9_16 train_bamor_mujoco maps parsed num_seeds times total_steps",
          mujoco_cls["workload_key"] == "bamor_mujoco_c9_16_completed_history"
          and math.isclose(float(mujoco_cls["units"]), 6000.0),
          diag=str(mujoco_cls))

    shard = dict(compare)
    shard["id"] = "bamor-c9-shard"
    shard["signature"] = "BAMOR/diagnostic-shard/offset0/node-c9"
    shard["cpu_cores"] = 11
    shard["cmd"] = (
        "/env/bin/python run_bamor_diagnostic_shard.py --start 8 --end 19 "
        "--item-offset 0 --methods 'uniform_ctx_oracle bamor_ctx_oracle "
        "bamor_ctx_oracle_cone' --seed-start 0 --seeds 20 "
        "--total-steps 100000 --switch-interval 300 --save-dir out "
        "--device cpu --eval-freq 0 --workers 11 --max-workers 10 "
        "--threads-per-run 2"
    )
    shard_cls = classify_record(shard, include_representative=False)
    check("BAMOR c9_16 diagnostic shard maps parsed shard items times total_steps",
          shard_cls["workload_key"] == "bamor_diagnostic_shard_c9_16_completed_history"
          and math.isclose(float(shard_cls["units"]), 1100000.0),
          diag=str(shard_cls))

    c3_boundary = dict(compare)
    c3_boundary["id"] = "bamor-c3-boundary"
    c3_boundary["cpu_cores"] = 8
    c3_boundary_cls = classify_record(c3_boundary, include_representative=False)
    check("module75 c9_16 parser does not steal the c3_8 upper boundary",
          c3_boundary_cls["workload_key"] == "bamor_train_compare_c3_8_completed_history",
          diag=str(c3_boundary_cls))

    cache = build_default_cache()
    expected_rates = {
        "bamor_train_compare_c9_16_completed_history": 1.50794692809882,
        "bamor_mujoco_c9_16_completed_history": 37.69062579924275,
        "bamor_diagnostic_shard_c9_16_completed_history": 1051.9833126514195,
    }
    check("module75 service cache exposes script-level BAMOR c9_16 profile1 lower services",
          all(
              [record.profile for record in cache.profiles(key)] == [1]
              and math.isclose(cache.get(key, 1).aggregate_rate, rate, rel_tol=1e-12)
              for key, rate in expected_rates.items()
          ),
          diag=str({
              key: [record.snapshot() for record in cache.profiles(key)]
              for key in expected_rates
          }))

    report = build_production_load_certificate(
        records=[compare, mujoco, shard],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=(
            "production_bamor_train_compare_c9_16_completed_history",
            "production_bamor_mujoco_c9_16_completed_history",
            "production_bamor_diagnostic_shard_c9_16_completed_history",
        ),
    )
    check("production load certificate sums BAMOR c9_16 training-step units",
          report["mapped_counts"] == {
              "bamor_train_compare_c9_16_completed_history": 1,
              "bamor_mujoco_c9_16_completed_history": 1,
              "bamor_diagnostic_shard_c9_16_completed_history": 1,
          }
          and math.isclose(report["mapped_units"]["bamor_train_compare_c9_16_completed_history"], 300.0)
          and math.isclose(report["mapped_units"]["bamor_mujoco_c9_16_completed_history"], 6000.0)
          and math.isclose(report["mapped_units"]["bamor_diagnostic_shard_c9_16_completed_history"], 1100000.0)
          and report["global_coverage_usable_for_theorem"]
          and report["mapped_capacity_usable_for_theorem"],
          diag=str(report))


def test_module65_zsw_tsp_sumo_cle2_completed_history_profile1(check, sch):
    baseline = {
        "id": "zsw-baseline",
        "project": "ZSW_platform",
        "signature": "ZSW_platform/auto-adopted/p24710",
        "description": "ZSW TSP baseline SUMO runner",
        "submitted_at": 900.0,
        "status": "done",
        "est_vram_mb": 0,
        "cpu_cores": 1,
        "ram_mb": 512,
        "cwd": "/home/erzhu419/zsw_tsp_m0_gpu1/ZSW_platform",
        "cmd": (
            "python TSP_only/src/oracle/baseline_runner.py --config cfg.sumocfg "
            "--duration 18000 --backend libsumo --seed 1 --tls-program 120s "
            "--output out.json"
        ),
    }
    cls = classify_record(baseline, include_representative=False)
    check("ZSW c_le2 baseline_runner maps after module65 certificate",
          cls["workload_key"] == "zsw_tsp_sumo_eval_c_le2_completed_history"
          and cls["mapping_mode"] == "strict_measured"
          and math.isclose(float(cls["units"]), 18000.0),
          diag=str(cls))

    m21 = dict(baseline)
    m21["id"] = "zsw-m21"
    m21["project"] = "zsw_tsp_m0_gpu1"
    m21["signature"] = "zsw_tsp_m0_gpu1/auto-adopted/p11407"
    m21["cmd"] = (
        "python /home/erzhu419/zsw_tsp_m0_gpu1/ZSW_platform/TSP_only/src/oracle/"
        "m21_cycle_conserving_tsp_runner.py --duration 18000 --seed 1 "
        "--backend libsumo --tls-program 120s --output out.json"
    )
    m21_cls = classify_record(m21, include_representative=False)
    check("ZSW c_le2 m21 runner maps parsed duration units",
          m21_cls["workload_key"] == "zsw_tsp_sumo_eval_c_le2_completed_history"
          and math.isclose(float(m21_cls["units"]), 18000.0),
          diag=str(m21_cls))

    missing_duration = dict(baseline)
    missing_duration["id"] = "zsw-no-duration"
    missing_duration["cmd"] = "python TSP_only/src/oracle/baseline_runner.py --output out.json"
    missing_cls = classify_record(missing_duration, include_representative=False)
    check("module65 classifier leaves ZSW runner without duration unmeasured",
          missing_cls["workload_key"] is None and missing_cls["reason"] == "unmapped_cpu",
          diag=str(missing_cls))

    non_zsw = dict(baseline)
    non_zsw["id"] = "not-zsw"
    non_zsw["project"] = "other"
    non_zsw["signature"] = "other/runner"
    non_zsw["cwd"] = "/tmp/other"
    non_zsw_cls = classify_record(non_zsw, include_representative=False)
    check("module65 classifier requires ZSW project/path/signature evidence",
          non_zsw_cls["workload_key"] is None and non_zsw_cls["reason"] == "unmapped_cpu",
          diag=str(non_zsw_cls))

    cache = build_default_cache()
    profiles = cache.profiles("zsw_tsp_sumo_eval_c_le2_completed_history")
    check("module65 service cache exposes only profile1 completed-history lower service",
          [record.profile for record in profiles] == [1]
          and math.isclose(
              cache.get("zsw_tsp_sumo_eval_c_le2_completed_history", 1).aggregate_rate,
              5.896139229545579,
              rel_tol=1e-12,
          ),
          diag=str([record.snapshot() for record in profiles]))

    report = build_production_load_certificate(
        records=[m21],
        window_days=1.0,
        now_ts=1000.0,
        taskset_names=("production_zsw_tsp_sumo_eval_c_le2_completed_history",),
    )
    check("production load certificate sums ZSW c_le2 simulated-second units",
          report["mapped_counts"] == {"zsw_tsp_sumo_eval_c_le2_completed_history": 1}
          and math.isclose(report["mapped_units"]["zsw_tsp_sumo_eval_c_le2_completed_history"], 18000.0)
          and math.isclose(
              report["lambda"]["zsw_tsp_sumo_eval_c_le2_completed_history"],
              18000.0 / 86400.0,
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
        {
            "id": "external-stdin",
            "project": "TransitDuet",
            "signature": "TransitDuet/auto-adopted/p123",
            "description": "auto-adopted: TransitDuet on local:CPU-only (1 procs)",
            "submitted_at": 904.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 1,
            "cmd": "python3 -",
            "origin": "external",
            "auto_adopted": True,
            "scheduler_id": None,
            "log_path": None,
        },
        {
            "id": "external-waiter",
            "project": "TransitDuet",
            "signature": "TransitDuet/auto-adopted/p456",
            "description": "auto-adopted: TransitDuet on local:CPU-only (1 procs)",
            "submitted_at": 905.0,
            "status": "done",
            "est_vram_mb": 0,
            "cpu_cores": 1,
            "cmd": "python3 /home/erzhu419/mine_code/scheduleurm/skill/scheduler.py wait-for --task-id t1",
            "origin": "external",
            "auto_adopted": True,
            "scheduler_id": None,
            "log_path": None,
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
    check("population label excludes unobservable external adopted control processes",
          population_label(rows[4])["label"] == "excluded_external_auto_adopted_unobservable"
          and population_label(rows[5])["label"] == "excluded_external_auto_adopted_unobservable"
          and not population_label(rows[4])["include_completed_active"]
          and not population_label(rows[5])["include_completed_active"],
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
