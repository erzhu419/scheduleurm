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
from algorithm.experiments.oracle_audit import audit_slots
from algorithm.experiments.penalty_fit import fit_penalty_envelope
from algorithm.experiments.report import summarize_certificate
from algorithm.experiments.service_model import (
    calibrate_service_lower_bounds,
    estimate_epsilon_est,
)
from algorithm.experiments.slot_builder import progress_service_units, queue_step


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
