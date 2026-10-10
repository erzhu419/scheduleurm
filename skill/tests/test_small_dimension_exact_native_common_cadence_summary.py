"""Common-clock readout on a real tiny FIFO replay with injected call clocks."""
from copy import deepcopy

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_timing import THREAD_ENV
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments.small_dimension_exact_native_common_cadence_replay import (
    simulate_common_cadence_trajectory, simulate_fixed_reference_trajectory,
)
from algorithm.experiments import small_dimension_exact_native_common_cadence_summary as summary


def _model(node="tiny"):
    return FrameModel(node, "common_cadence_summary", 8, ("a", "b", "c"), (1.0, 1.0, 1.0),
                      (FrameOption("a", (1, 0, 0), 2.0, ((2.0,), (), ())),))


def _runtime():
    return {"cpu_affinity": [0], "gc_enabled": True,
            "thread_environment": dict.fromkeys(THREAD_ENV, "1"),
            "threadpool_info": [{"user_api": "blas", "num_threads": 1}]}


def _trajectory(model, protocol, tape, label, wall=None, cpu=None):
    wall = (1.2, 1.2) if label == "full_reference" and wall is None else wall
    wall = ((.8, .8) if label == "candidate" else (.9, .9)) if wall is None else wall
    cpu = tuple(value / 2 for value in wall) if cpu is None else cpu
    window = protocol["computation_window_s"]
    wall_ticks = iter(value for seconds in wall for value in (0, int(seconds * 1e9)))
    cpu_ticks = iter(value for seconds in cpu for value in (0, int(seconds * 1e9)))
    simulate = simulate_fixed_reference_trajectory if label == "full_reference" else simulate_common_cadence_trajectory
    result, _ = simulate(
        model, tape, protocol["arrival_horizon"], protocol["observation_end"],
        protocol["reference_budget"] if label == "full_reference" else protocol["lp_node_budget"],
        select_canonical_count_space_bnb_frame, window,
        wall_clock=lambda: next(wall_ticks), cpu_clock=lambda: next(cpu_ticks),
        **({} if label == "full_reference" else {"policy": label}))
    calls = result["decision_count"]
    result.update(
        selector_metrics={"calls": calls, "failures": 0, "canonical_exact_count": calls,
                          "score_mismatch_count": 0, "canonical_action_mismatch_count": 0,
                          "reference_incomplete_count": 0,
                          "operational_elapsed_s": sum(frame["selector_wall_s"] for frame in result["frames"]),
                          "operational_cpu_s": sum(frame["selector_cpu_s"] for frame in result["frames"]),
                          "whole_work": {"search_work": {"lp_calls": calls}, "cold_constructions": calls}},
        decision_diagnostics=[{"errors": []} for _ in result["frames"]],
        reference_audit={"incomplete_count": 0})
    return result


def _case(nodes=("tiny",), loads=(.8,), seeds=(41,)):
    models = tuple(_model(node) for node in nodes)
    protocol = {
        "schema_version": 1, "experiment": "small_dimension_exact_native_common_cadence_development_v1",
        "registered_at_utc": "2026-09-29T00:00:00+00:00", "lanes": [8], "loads": list(loads),
        "arrival_seeds": list(seeds), "labels": list(summary.LABELS), "repeats": 6,
        "arrival_horizon": 10.0, "observation_end": 12.0, "computation_window_s": 1.0,
        "cpu": 0, "reference_budget": 4096, "initial_lp_budget": 144, "lp_node_budget": 256,
        "integer_state_budget": 8192, "oracle_state_budget": 5_000_000,
        "float_relative_tolerance": 1e-9, "maximum_group_wall_ratio": 1.05, "maximum_pooled_wall_ratio": .95,
        "tapes": [], "timing_orders": {}, "execution_order": [[model.node, 8] for model in models],
        "prior_decisions": {"free_clock": "UTILITY_HOLD"}, "limitations": "Development fixture.",
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
                    "reference": _trajectory(model, protocol, tape, "full_reference"),
                    "measured": [{"repeat": repeat, "order": order,
                                  "policies": {label: _trajectory(model, protocol, tape, label) for label in order}}
                                 for repeat, order in enumerate(orders)],
                    "reference_audit": {"calls": 3, "oracle_calls": 0, "failures": 0, "incomplete_count": 0,
                                        "work": {"search_work": {"lp_calls": 3}, "oracle": {"states": 0}}},
                })
        rows.append(row)
    return models, protocol, rows


def _replace(case, label, wall, cpu=None, repeats=None, run_index=0):
    models, protocol, rows = case
    run = rows[0]["runs"][run_index]
    tape = next(tape["arrivals"] for tape in protocol["tapes"] if tape["tape_id"] == run["tape_id"])
    if label == "full_reference":
        run["reference"] = _trajectory(models[0], protocol, tape, label, wall, cpu)
    else:
        for repeat in run["measured"]:
            if repeats is None or repeat["repeat"] in repeats:
                repeat["policies"][label] = _trajectory(models[0], protocol, tape, label, wall, cpu)


def _refresh(result, protocol):
    result["snapshots"] = {key: trajectory_snapshot(result["jobs"], protocol[key], 3)
                           for key in ("arrival_horizon", "observation_end")}


def test_all_24_tapes_keep_312_outputs_eight_groups_nine_speed_gates():
    result = summary.summarize(*_case(nodes=("linux", "node007", "gpu", "gpu2"), loads=(.8, .95), seeds=(41, 42, 43)))
    assert result["tape_count"] == 24 and result["trajectory_count"] == 312
    assert len(result["groups"]) == 8 and result["gate_count"] == 9 and result["failed_gate_count"] == 0
    assert all(group["seed_count"] == 3 and group["repeats_per_seed"] == 6 for group in result["groups"])
    assert result["common_cadence_gate_passed"] and result["decision"].endswith("TIMING_ADVANCE")
    assert result["prior_hold_retained"] and result["prior_decisions"]["free_clock"] == "UTILITY_HOLD"
    assert not result["flow_gain_established"]


def test_actual_call_costs_exclude_padding_and_offline_reference_work():
    result = summary.summarize(*_case())
    measured = result["totals"]["measured"]
    assert measured["candidate"]["calls"] == measured["direct"]["calls"] == 12
    assert measured["candidate"]["whole_work"]["cold_constructions"] == 12
    assert measured["candidate"]["delay_s"] == measured["direct"]["delay_s"] == 12.0
    assert measured["candidate"]["selector_wall_s"] == pytest.approx(9.6)
    assert measured["candidate"]["window_padding_s"] == pytest.approx(2.4)
    assert result["totals"]["reference"]["calls"] == 2
    assert result["totals"]["reference"]["selector_wall_s"] == pytest.approx(2.4)
    assert result["totals"]["reference"]["window_overrun_count"] == 2
    assert result["reference_audit_totals"]["calls"] == 3
    assert result["pooled_mean_actual_costs_sum"]["candidate"]["wall_s"] == pytest.approx(1.6)
    assert result["pooled_timing_comparison"]["wall_s_ratio"] == pytest.approx(8 / 9)
    assert result["actual_call_wall_p99_ratio"] == pytest.approx(8 / 9)


def test_full_reference_is_a_normative_clock_not_a_deployable_deadline_arm():
    case = _case()
    _replace(case, "full_reference", (100.0, 100.0), (90.0, 90.0))
    result = summary.summarize(*case)
    assert result["clock_accounting_passed"] and result["no_overrun_gate_passed"]
    assert result["trajectory_identity_passed"] and result["common_cadence_gate_passed"]
    assert result["totals"]["reference"]["selector_wall_s"] == 200.0
    assert result["totals"]["reference"]["delay_s"] == 2.0


def test_cpu_and_p99_are_descriptive_and_do_not_add_unregistered_speed_gates():
    case = _case()
    _replace(case, "candidate", (.6, 1.0), (1.5, 1.5))
    _replace(case, "direct", (.9, .9), (.1, .1))
    result = summary.summarize(*case)
    assert result["pooled_timing_comparison"]["cpu_s_ratio"] == 15.0
    assert result["actual_call_wall_p99_ratio"] > 1.05
    assert result["timing_gate_passed"] and result["common_cadence_gate_passed"]


def test_overrun_is_paid_recorded_and_holds_after_all_repeats_are_kept():
    case = _case()
    _replace(case, "candidate", (.8, 2.0), repeats=(5,))
    result = summary.summarize(*case)
    assert result["execution_checks_passed"] and result["integration_gate_passed"]
    assert not result["no_overrun_gate_passed"] and not result["trajectory_identity_passed"]
    assert result["decision"].endswith("CLOCK_HOLD")
    assert result["totals"]["measured"]["candidate"]["calls"] == 12
    assert result["totals"]["measured"]["candidate"]["delay_s"] == 13.0
    assert result["totals"]["measured"]["candidate"]["window_overrun_count"] == 1


def test_small_overrun_holds_even_when_every_registered_speed_gate_passes():
    case = _case()
    _replace(case, "candidate", (.01, 1.001), repeats=(5,))
    result = summary.summarize(*case)
    assert result["timing_gate_passed"]
    assert not result["no_overrun_gate_passed"] and not result["common_cadence_gate_passed"]


def test_final_cancelled_solve_stays_paid_and_audited_without_dispatch():
    case = _case()
    _replace(case, "candidate", (.8, 15.0), repeats=(0,))
    result = summary.summarize(*case)
    costs = result["totals"]["measured"]["candidate"]
    assert result["execution_checks_passed"] and result["integration_gate_passed"]
    assert costs["calls"] == costs["canonical_exact_count"] == 12
    assert costs["cancelled_decision_count"] == 1 and costs["executed_frame_count"] == 11
    assert costs["selector_wall_s"] == pytest.approx(23.8)
    assert costs["window_overrun_s"] == 14.0
    assert not result["common_cadence_gate_passed"]
    for cohort in result["paired_runs"][0]["common_completed_cohorts"].values():
        assert cohort["count"] == 1 and cohort["job_ids"] == [[0, 0]]
        assert len(cohort["policies"]) == 13
        assert cohort["policies"]["reference"]["mean_flow_s"] == 3.0


def test_six_repeats_average_within_tape_before_seed_costs_are_summed():
    case = _case(seeds=(41, 42))
    _replace(case, "candidate", (.2, .2), repeats=(0,))
    _replace(case, "candidate", (.5, .5), run_index=1)
    result = summary.summarize(*case)
    expected = (.4 + 5 * 1.6) / 6 + 1.0
    assert result["pooled_mean_actual_costs_sum"]["candidate"]["wall_s"] == pytest.approx(expected)
    assert result["groups"][0]["seed_count"] == 2
    assert result["groups"][0]["mean_actual_costs_sum"]["candidate"]["wall_s"] == pytest.approx(expected)


def test_pooled_five_percent_saving_is_required_after_identity_and_group_gates():
    case = _case()
    _replace(case, "candidate", (.9, .9))
    result = summary.summarize(*case)
    assert result["common_clock_gate_passed"] and result["groups"][0]["wall_gate_passed"]
    assert not result["timing_gate_passed"] and result["decision"].endswith("TIMING_HOLD")


def test_one_bad_group_cannot_hide_behind_a_faster_pooled_ratio():
    case = _case(loads=(.8, .95))
    _replace(case, "candidate", (.1, .1))
    _replace(case, "candidate", (1.0, 1.0), run_index=1)
    result = summary.summarize(*case)
    assert result["pooled_timing_comparison"]["wall_s_ratio"] < .95
    assert result["common_clock_gate_passed"] and not result["timing_gate_passed"]
    assert result["failed_gate_count"] == 1


@pytest.mark.parametrize("field", ("start", "dispatch_time", "queue", "option_indices", "service", "duration", "executed_service", "dispatched"))
def test_identity_covers_every_registered_execution_field(field):
    _, _, rows = _case()
    run = rows[0]["runs"][0]
    candidate = deepcopy(run["reference"])
    value = candidate["frames"][0][field]
    candidate["frames"][0][field] = ([99] if isinstance(value, (list, tuple)) else
                                      not value if isinstance(value, bool) else value + .1)
    match = summary._trajectory_identity(candidate, run["reference"])
    assert not match["passed"] and match["first_mismatch"]["field"] == field


@pytest.mark.parametrize("field", ("jobs", "snapshots"))
def test_identity_covers_all_jobs_and_both_cutoff_snapshots(field):
    _, _, rows = _case()
    reference = rows[0]["runs"][0]["reference"]
    candidate = deepcopy(reference)
    if field == "jobs":
        candidate["jobs"][0]["completion"] += .1
    else:
        candidate["snapshots"]["arrival_horizon"]["unfinished_jobs"] += 1
    match = summary._trajectory_identity(candidate, reference)
    assert not match["passed"] and match["first_mismatch"]["field"] == field


def test_future_job_during_window_cannot_be_substituted_into_the_old_action():
    case = _case()
    models, protocol, rows = case
    candidate = rows[0]["runs"][0]["measured"][0]["policies"]["candidate"]
    candidate["jobs"][0]["start"], candidate["jobs"][1]["start"] = candidate["jobs"][1]["start"], candidate["jobs"][0]["start"]
    candidate["jobs"][0]["completion"], candidate["jobs"][1]["completion"] = candidate["jobs"][1]["completion"], candidate["jobs"][0]["completion"]
    _refresh(candidate, protocol)
    result = summary.summarize(models, protocol, rows)
    assert not result["causal_fifo_checks_passed"] and not result["integration_gate_passed"]


@pytest.mark.parametrize("failure", ("runtime_as_padding", "cpu_total", "wall_total", "padding", "overrun", "overrun_count", "applied_delay", "window"))
def test_clock_and_real_runtime_ledgers_are_checked_separately(failure):
    models, protocol, rows = _case()
    candidate = rows[0]["runs"][0]["measured"][0]["policies"]["candidate"]
    if failure == "runtime_as_padding":
        candidate["selector_metrics"]["operational_elapsed_s"] = candidate["total_decision_delay_s"]
    elif failure == "cpu_total":
        candidate["total_selector_cpu_s"] += .1
    elif failure == "wall_total":
        candidate["total_selector_wall_s"] += .1
    elif failure in ("padding", "overrun"):
        candidate["frames"][0]["window_" + failure + "_s"] += .1
    elif failure == "overrun_count":
        candidate["window_overrun_count"] = 1
    elif failure == "applied_delay":
        candidate["frames"][0]["decision_delay_s"] = .8
    else:
        candidate["computation_window_s"] = 2.0
    result = summary.summarize(models, protocol, rows)
    assert not result["clock_accounting_passed"] and not result["common_cadence_gate_passed"]


@pytest.mark.parametrize("failure", ("uncertified", "score", "action", "reference", "diagnostic", "budget", "snapshot"))
def test_canonical_reference_certificate_and_execution_failures_hold(failure):
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
        candidate["decision_diagnostics"][0]["errors"].append("LP fee mismatch")
    elif failure == "budget":
        candidate["budget"] += 1
    else:
        candidate["snapshots"]["observation_end"]["unfinished_jobs"] += 1
    result = summary.summarize(models, protocol, rows)
    assert not result["integration_gate_passed"] and not result["common_cadence_gate_passed"]


@pytest.mark.parametrize("failure", ("cpu", "gc", "blas_env", "blas_pool"))
def test_cpu_gc_and_thread_provenance_remain_required(failure):
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


@pytest.mark.parametrize("failure", ("cell", "run", "repeat", "duplicate_repeat", "order", "policy", "unbalanced", "unused_tape"))
def test_missing_or_changed_registered_pairing_is_rejected(failure):
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
    elif failure == "unbalanced":
        for repeat in run["measured"]:
            repeat["order"] = list(summary.LABELS)
        protocol["timing_orders"][run["tape_id"]] = [list(summary.LABELS)] * 6
    else:
        protocol["tapes"].append({**protocol["tapes"][0], "tape_id": "unused"})
    with pytest.raises(ValueError):
        summary.summarize(models, protocol, rows)
