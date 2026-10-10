"""Readout tests use real FIFO replay, then exercise pairing and causal gates."""
from copy import deepcopy

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_timing import THREAD_ENV
from algorithm.experiments.integer_capacity_decision_delay_replay import simulate_delayed_trajectory
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments import small_dimension_exact_native_decision_delay_summary as summary


def _model(node="tiny"):
    return FrameModel(node, "native_delay_summary", 8, ("a", "b", "c"), (1.0, 1.0, 1.0),
                      (FrameOption("a", (1, 0, 0), 2.0, ((2.0,), (), ())),))


def _runtime():
    return {"cpu_affinity": [0], "gc_enabled": True,
            "thread_environment": dict.fromkeys(THREAD_ENV, "1"),
            "threadpool_info": [{"user_api": "blas", "num_threads": 1}]}


def _trajectory(model, protocol, tape, label, delays=(0.0, 0.0)):
    times = iter(value for delay in delays for value in (0, int(delay * 1e9)))
    result, _ = simulate_delayed_trajectory(
        model, tape, protocol["arrival_horizon"], protocol["observation_end"],
        protocol["reference_budget"] if label == "full_reference" else protocol["lp_node_budget"],
        select_canonical_count_space_bnb_frame, policy=label, clock=lambda: next(times))
    calls = result["decision_count"]
    result.update(selector_metrics={
        "calls": calls, "failures": 0, "canonical_exact_count": calls,
        "score_mismatch_count": 0, "canonical_action_mismatch_count": 0,
        "reference_incomplete_count": 0,
        "whole_work": {"search_work": {"lp_calls": calls}, "small_dimension_lp": {"exact_calls": calls}},
    }, decision_diagnostics=[{"errors": []} for _ in result["frames"]],
        reference_audit={"incomplete_count": 0})
    return result


def _case(nodes=("tiny",), loads=(.8,), seeds=(41,)):
    models = tuple(_model(node) for node in nodes)
    protocol = {
        "schema_version": 1, "experiment": "small_dimension_exact_native_decision_delay_development_v1",
        "registered_at_utc": "2026-09-29T00:00:00+00:00",
        "models": [model.snapshot() for model in models], "lanes": [8], "loads": list(loads),
        "arrival_seeds": list(seeds), "labels": list(summary.LABELS), "zero_labels": list(summary.ZERO_LABELS),
        "repeats": 6, "arrival_horizon": 10.0, "observation_end": 12.0,
        "cpu": 0, "reference_budget": 4096, "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "oracle_state_budget": 5_000_000,
        "float_relative_tolerance": 1e-9, "minimum_pooled_flow_improvement": .01,
        "tapes": [], "timing_orders": {}, "execution_order": [[model.node, 8] for model in models],
        "prior_decisions": {"prior": "UTILITY_HOLD"}, "limitations": "Development fixture.",
    }
    rows = []
    for model in models:
        row = {"node": model.node, "lanes": 8, "runtime_start": _runtime(), "runtime_end": _runtime(), "runs": []}
        for load in loads:
            for seed in seeds:
                tape_id = f"{model.node}:{load}:{seed}"
                tape = [(0.0, 0, 0), (.1, 0, 1), (.2, 0, 2)]
                protocol["tapes"].append({"tape_id": tape_id, "node": model.node, "lanes": 8,
                                          "load": load, "seed": seed, "arrivals": tape})
                orders = [list(summary.LABELS), list(reversed(summary.LABELS))] * 3
                protocol["timing_orders"][tape_id] = orders
                row["runs"].append({
                    "tape_id": tape_id, "load": load, "seed": seed,
                    "zero": {label: _trajectory(model, protocol, tape, label) for label in summary.ZERO_LABELS},
                    "measured": [{"repeat": repeat, "order": order,
                                  "policies": {label: _trajectory(model, protocol, tape, label) for label in order}}
                                 for repeat, order in enumerate(orders)],
                    "reference_audit": {"calls": 3, "oracle_calls": 0, "failures": 0,
                                        "incomplete_count": 0, "work": {"search_work": {"lp_calls": 3}, "oracle": {"states": 0}}},
                })
        rows.append(row)
    return models, protocol, rows


def _replace_measured(case, label, delays, repeats=None, run_index=0):
    models, protocol, rows = case
    run = rows[0]["runs"][run_index]
    tape = next(tape["arrivals"] for tape in protocol["tapes"] if tape["tape_id"] == run["tape_id"])
    for repeat in run["measured"]:
        if repeats is None or repeat["repeat"] in repeats:
            repeat["policies"][label] = _trajectory(models[0], protocol, tape, label, delays)


def _refresh(result, protocol):
    result["snapshots"] = {key: trajectory_snapshot(result["jobs"], protocol[key], 3)
                           for key in ("arrival_horizon", "observation_end")}


def test_registered_24_tapes_keep_360_outputs_and_16_seed_level_groups():
    result = summary.summarize(*_case(nodes=("linux", "node007", "gpu", "gpu2"), loads=(.8, .95), seeds=(41, 42, 43)))
    assert result["tape_count"] == 24
    assert result["trajectory_count"] == 360
    assert len(result["groups"]) == 16
    assert all(group["seed_count"] == 3 and group["repeats_per_seed"] == 6 for group in result["groups"])
    assert result["integration_gate_passed"] and result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    assert result["decision"].endswith("PRESERVED")
    assert result["prior_hold_retained"]


def test_policy_work_and_offline_references_are_separate_real_call_ledgers():
    result = summary.summarize(*_case())
    assert result["totals"]["zero"]["full_reference"]["calls"] == 2
    assert result["totals"]["zero"]["candidate"]["calls"] == 2
    assert result["totals"]["measured"]["candidate"]["calls"] == 12
    assert result["totals"]["measured"]["candidate"]["whole_work"]["search_work"]["lp_calls"] == 12
    assert result["reference_audit_totals"]["calls"] == 3
    assert result["reference_audit_totals"]["work"]["search_work"]["lp_calls"] == 3


def test_common_intersection_includes_every_repeat_and_counts_repeats_within_seed():
    case = _case(seeds=(41, 42))
    _replace_measured(case, "candidate", (0.0, 15.0), repeats=(5,))
    result = summary.summarize(*case)
    pair = result["paired_runs"][0]
    for snapshot in pair["snapshots"].values():
        assert snapshot["common_cohort_count"] == 1
        assert snapshot["common_job_ids"] == [[0, 0]]
        assert snapshot["measured"]["candidate"]["unfinished_jobs"] == pytest.approx(2 / 6)
        assert snapshot["zero"]["full_reference"]["common_mean_flow_s"] == 2.0
    assert all(group["seed_count"] == 2 and group["repeats_per_seed"] == 6 for group in result["groups"])
    assert all(group["measured"]["candidate"]["unfinished_jobs"] == pytest.approx(2 / 6) for group in result["groups"])
    assert result["integration_gate_passed"]
    assert not result["utility_gate_passed"] and not result["flow_gain_established"]


def test_a_cancelled_final_decision_is_paid_once_and_remains_audited():
    case = _case()
    _replace_measured(case, "candidate", (0.0, 15.0), repeats=(0,))
    result = summary.summarize(*case)
    costs = result["totals"]["measured"]["candidate"]
    assert costs["calls"] == costs["canonical_exact_count"] == 12
    assert costs["executed_frame_count"] == 11
    assert costs["cancelled_decision_count"] == 1
    assert costs["delay_s"] == 15.0
    assert result["integration_gate_passed"]


def test_faster_completed_jobs_cannot_hide_unfinished_or_restricted_flow_regression():
    case = _case()
    _replace_measured(case, "direct", (.5, .5))
    _replace_measured(case, "candidate", (0.0, 15.0))
    result = summary.summarize(*case)
    assert result["pooled_flow_improvement"] > .01
    assert result["integration_gate_passed"]
    assert not result["utility_gate_passed"]
    assert not result["flow_gain_established"]
    for group in result["groups"]:
        checks = group["comparison"]["checks"]
        assert checks["common_mean_flow_s"] and checks["common_p90_flow_s"]
        assert not checks["unfinished_jobs"] and not checks["restricted_mean_flow_s"]


def test_all_utility_groups_and_one_percent_gain_are_both_needed():
    case = _case()
    _replace_measured(case, "direct", (.2, .2))
    _replace_measured(case, "candidate", (.1, .1))
    result = summary.summarize(*case)
    assert result["integration_gate_passed"] and result["utility_gate_passed"] and result["flow_gain_established"]
    assert result["decision"].endswith("FLOW_GAIN")
    assert result["pooled_flow_improvement"] > .01
    assert result["totals"]["measured"]["candidate"]["delay_s"] == pytest.approx(1.2)
    case = _case()
    _replace_measured(case, "direct", (.001, .001))
    result = summary.summarize(*case)
    assert result["utility_gate_passed"]
    assert 0 < result["pooled_flow_improvement"] < .01
    assert not result["flow_gain_established"]


def test_empty_cohort_cannot_pass_utility_even_if_counts_are_equal():
    case = _case()
    _replace_measured(case, "candidate", (15.0,), repeats=(0,))
    _replace_measured(case, "direct", (15.0,), repeats=(0,))
    result = summary.summarize(*case)
    assert result["integration_gate_passed"]
    assert all(not group["common_cohorts_nonempty"] for group in result["groups"])
    assert not result["utility_gate_passed"] and not result["flow_gain_established"]


@pytest.mark.parametrize("label", summary.LABELS)
def test_both_zero_controllers_must_match_full_reference_trajectory(label):
    models, protocol, rows = _case()
    zero = rows[0]["runs"][0]["zero"][label]
    zero["jobs"][0]["completion"] += .1
    _refresh(zero, protocol)
    result = summary.summarize(models, protocol, rows)
    assert result["execution_checks_passed"]
    assert not result["zero_trajectory_identity_passed"]
    assert not result["integration_gate_passed"]


@pytest.mark.parametrize("failure", ("uncertified", "score", "action", "reference", "diagnostic", "budget", "job_id", "snapshot"))
def test_invalid_canonical_reference_or_execution_evidence_cannot_pass(failure):
    models, protocol, rows = _case()
    candidate = rows[0]["runs"][0]["measured"][0]["policies"]["candidate"]
    if failure == "uncertified":
        candidate["selector_metrics"]["canonical_exact_count"] -= 1
    elif failure in ("score", "action"):
        key = "score_mismatch_count" if failure == "score" else "canonical_action_mismatch_count"
        candidate["selector_metrics"][key] += 1
    elif failure == "reference":
        candidate["selector_metrics"]["reference_incomplete_count"] += 1
    elif failure == "diagnostic":
        candidate["decision_diagnostics"][0]["errors"].append("LP budget exceeded")
    elif failure == "budget":
        candidate["budget"] += 1
    elif failure == "job_id":
        candidate["jobs"][1]["index"] = 0
    else:
        candidate["snapshots"]["observation_end"]["unfinished_jobs"] += 1
    result = summary.summarize(models, protocol, rows)
    assert not result["integration_gate_passed"] and not result["flow_gain_established"]


@pytest.mark.parametrize("failure", ("double_total", "double_dispatch", "double_next_start", "cancelled_service", "zero_delay"))
def test_single_charge_clock_and_cancelled_dispatch_are_required(failure):
    case = _case()
    _replace_measured(case, "candidate", (.1, 15.0), repeats=(0,))
    models, protocol, rows = case
    candidate = rows[0]["runs"][0]["measured"][0]["policies"]["candidate"]
    if failure == "double_total":
        candidate["total_decision_delay_s"] *= 2
    elif failure == "double_dispatch":
        candidate["frames"][0]["dispatch_time"] += .1
    elif failure == "double_next_start":
        candidate["frames"][1]["start"] += .1
    elif failure == "cancelled_service":
        candidate["frames"][-1]["executed_service"] = candidate["frames"][-1]["service"]
    else:
        zero = rows[0]["runs"][0]["zero"]["candidate"]
        zero["frames"][0]["decision_delay_s"] = .1
        zero["total_decision_delay_s"] = .1
    result = summary.summarize(models, protocol, rows)
    assert not result["execution_checks_passed"] and not result["integration_gate_passed"]


@pytest.mark.parametrize("failure", ("cpu", "gc", "blas_env", "blas_pool"))
def test_registered_cpu_gc_and_actual_blas_threads_are_required(failure):
    models, protocol, rows = _case()
    runtime = rows[0]["runtime_end"]
    if failure == "cpu":
        runtime["cpu_affinity"] = [1]
    elif failure == "gc":
        runtime["gc_enabled"] = False
    elif failure == "blas_env":
        runtime["thread_environment"]["OPENBLAS_NUM_THREADS"] = "2"
    else:
        runtime["threadpool_info"][0]["num_threads"] = 2
    result = summary.summarize(models, protocol, rows)
    assert not result["runtime_checks_passed"] and not result["integration_gate_passed"]


@pytest.mark.parametrize("failure", ("cell", "run", "repeat", "duplicate_repeat", "order", "policy", "zero_policy", "unused_tape"))
def test_missing_or_changed_registered_coverage_is_rejected(failure):
    models, protocol, rows = _case()
    run = rows[0]["runs"][0]
    if failure == "cell":
        rows.append(deepcopy(rows[0]))
    elif failure == "run":
        rows[0]["runs"] = []
    elif failure == "repeat":
        run["measured"].pop()
    elif failure == "duplicate_repeat":
        run["measured"][5]["repeat"] = 0
    elif failure == "order":
        run["measured"][0]["order"] = list(reversed(run["measured"][0]["order"]))
    elif failure == "policy":
        del run["measured"][0]["policies"]["direct"]
    elif failure == "zero_policy":
        del run["zero"]["direct"]
    else:
        protocol["tapes"].append({**protocol["tapes"][0], "tape_id": "unused"})
    with pytest.raises(ValueError):
        summary.summarize(models, protocol, rows)
