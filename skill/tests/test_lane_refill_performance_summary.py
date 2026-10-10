"""Performance gates preserve tails, literal lane commitments and whole-call fees."""
import pytest

from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.integer_capacity_trajectory_replay import trajectory_snapshot
from algorithm.experiments import lane_refill_performance_summary as summary


def _trajectory(model, tape, protocol, label, plan):
    jobs = [{"arrival": float(time), "coordinate": coordinate, "index": index,
             "start": None, "completion": None} for time, coordinate, index in sorted(tape)]
    frames, events, available = [], [], [0.] * model.lanes
    end = protocol["observation_end"]
    exposure = lambda start, finish: max(0., min(finish, end) - start)
    occupied = barrier = idle = 0.
    for start, indices in plan:
        free = tuple(lane for lane, finish in enumerate(available) if finish == start)
        lanes = free[:1] if label == "refill_single" else free
        assert len(indices) == len(lanes)
        waiting = [[job for job in jobs if job["coordinate"] == coordinate
                    and job["arrival"] <= start and job["start"] is None]
                   for coordinate in range(len(model.workloads))]
        queue = tuple(map(len, waiting))
        duration = max((1., *(model.options[index].duration for index in indices if index is not None)))
        offsets = [[] for _ in model.workloads]
        for lane, index in zip(lanes, indices, strict=True):
            option = None if index is None else model.options[index]
            local = 1. if option is None else max(1., option.duration)
            local_end = start + local
            reserved_end = start + duration if label == "sync_joint" else local_end
            events.append({"lane": lane, "start": start, "option_index": index,
                           "local_end": local_end, "reserved_end": reserved_end})
            available[lane] = reserved_end
            barrier += exposure(local_end, reserved_end)
            if option is None:
                idle += exposure(start, local_end)
            else:
                occupied += exposure(start, local_end)
                for coordinate, values in enumerate(option.completion_offsets):
                    offsets[coordinate].extend(values)
        service = tuple(map(len, offsets))
        for coordinate, values in enumerate(offsets):
            for job, offset in zip(waiting[coordinate], sorted(values)):
                job["start"], job["completion"] = start, start + offset
        frames.append({"start": start, "dispatch_time": start, "decision_delay_s": 0.,
                       "queue": queue, "lane_ids": lanes, "lanes_called": len(lanes),
                       "option_indices": indices, "service": service, "executed_service": service,
                       "duration": duration, "canonical_exact": True, "dispatched": True,
                       "selector_wall_s": .01})
    calls = len(frames)
    work = {"calls": calls, "search_work": {"lp_calls": 3 * calls, "lp_matrix_coefficients": 7 * calls},
            "shared_integer_states": 11 * calls, "max_shared_integer_states": 11,
            "small_dimension_lp_work": {"rational_conversions": 13 * calls}}
    return {"policy": label, "dispatch_rule": summary.RULES[label], "budget": protocol["lp_node_budget"],
            "arrival_horizon": protocol["arrival_horizon"], "observation_end": end,
            "frame_count": calls, "decision_count": calls, "executed_frame_count": calls,
            "total_decision_delay_s": 0., "frames": frames, "jobs": jobs, "lane_events": events,
            "last_frame_end": max(available), "last_decision_end": frames[-1]["start"],
            "invariants": {"resource_feasible": True, "lane_nonoverlap": True},
            "snapshots": {key: trajectory_snapshot(jobs, protocol[key], len(model.workloads))
                          for key in summary.CUTOFFS},
            "occupied_lane_time_s": occupied, "barrier_idle_lane_time_s": barrier,
            "scheduled_idle_lane_time_s": idle,
            "waiting_area_s": sum(min(end, job["start"] if job["start"] is not None else end)
                                  - job["arrival"] for job in jobs),
            "total_selector_wall_s": .01 * calls,
            "selector_metrics": {**dict.fromkeys(summary.SCALAR_COSTS, 0),
                                 "calls": calls, "canonical_exact_count": calls, "whole_work": work},
            "decision_diagnostics": [{"frame": index, "lanes_called": frame["lanes_called"],
                                      "actual_lp_budget": 144, "lp_calls": 3,
                                      "shared_integer_states": 11, "canonical_exact": True,
                                      "reference_complete": True, "errors": []}
                                     for index, frame in enumerate(frames)],
            "reference_audit": {"calls": calls, "oracle_calls": 0, "failures": 0,
                                "incomplete_count": 0, "work": {"search_work": {"lp_calls": calls}}}}


def _case(*, seeds=(41,), equal=False, tail=None):
    if tail is None:
        slow = 2. if equal else 10.
        model = FrameModel("toy", "toy", 2, ("a", "b"), (1., 1.), (
            FrameOption("fast", (1, 0), 2., ((2.,), ())),
            FrameOption("slow", (0, 1), slow, ((), (slow,))),
            FrameOption("faster", (1, 0), 1., ((1.,), ())),))
        tape = ((0., 0, 0), (0., 0, 1), (0., 0, 2), (0., 1, 0))
        plans = {"sync_joint": [(0., (0, 1)), (slow, (0, 0))],
                 "refill_joint": ([(0., (0, 1)), (2., (0, 0))] if equal else
                                  [(0., (0, 1)), (2., (0,)), (4., (0,))]),
                 "refill_single": ([(0., (0,)), (0., (1,)), (2., (0,)), (2., (0,))] if equal else
                                   [(0., (0,)), (0., (1,)), (2., (0,)), (4., (0,))])}
        end = 20.
    else:
        model = FrameModel("toy", "toy", 2, ("a",), (1.,), (
            FrameOption("fast", (1,), 2., ((2.,),)),
            FrameOption("slow", (1,), 20., ((20.,),)),
            FrameOption("long", (1,), 30., ((30.,),)),))
        tape = tuple((0., 0, index) for index in range(4))
        plans = {"sync_joint": [(0., (0, 1)), (20., (0, 0))],
                 "refill_joint": [(0., (0, 0)), (2., (0, 2))],
                 "refill_single": [(0., (0,)), (0., (0,)), (2., (0,)), (2., (2,))]}
        end = tail
    protocol = {"schema_version": 1, "experiment": "toy", "registered_at_utc": "frozen",
                "labels": summary.LABELS, "lanes": [2], "loads": [.8], "arrival_seeds": list(seeds),
                "execution_order": [["toy", 2]], "arrival_horizon": 5., "observation_end": end,
                "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
                "float_relative_tolerance": 1e-9, "minimum_pooled_flow_improvement": .01,
                "expected_tape_count": len(seeds), "expected_trajectory_count": 3 * len(seeds),
                "expected_group_count": 2, "prior_decisions": {"free_clock": "UTILITY_HOLD"},
                "limitations": "toy development", "tapes": []}
    row = {"node": "toy", "lanes": 2, "runs": []}
    for seed in seeds:
        tape_id = f"toy:{seed}"
        protocol["tapes"].append({"node": "toy", "lanes": 2, "load": .8, "seed": seed,
                                  "tape_id": tape_id, "arrivals": tape})
        policies = {label: _trajectory(model, tape, protocol, label, plan) for label, plan in plans.items()}
        calls = sum(result["selector_metrics"]["calls"] for result in policies.values())
        row["runs"].append({"load": .8, "seed": seed, "tape_id": tape_id, "arrival_count": len(tape),
                            "policies": policies,
                            "reference_audit": {"calls": calls, "oracle_calls": 0, "failures": 0,
                                                "incomplete_count": 0,
                                                "work": {"search_work": {"lp_calls": calls}}}})
    return [model], protocol, [row]


def test_refill_advances_on_flow_and_sums_actual_work_without_summing_a_maximum():
    result = summary.summarize(*_case(seeds=(41, 42)))
    assert result["decision"] == "LANE_REFILL_PERFORMANCE_DEVELOPMENT_ADVANCE"
    assert result["trajectory_count"] == 6 and result["utility_group_count"] == 2
    assert result["pooled_flow_at_T"]["sync_joint"]["restricted_mean_flow_s"] == 9.
    assert result["pooled_flow_at_T"]["refill_joint"]["restricted_mean_flow_s"] == 5.5
    assert result["totals"]["refill_joint"]["calls"] == 6
    assert result["totals"]["refill_single"]["calls"] == 8
    assert result["totals"]["refill_joint"]["whole_work"]["search_work"]["lp_calls"] == 18
    assert result["totals"]["refill_joint"]["whole_work"]["max_shared_integer_states"] == 11
    assert result["totals"]["sync_joint"]["barrier_idle_lane_time_s"] == 16.
    assert result["prior_hold_retained"]


def test_mean_gain_cannot_hide_shared_cohort_p90_regression():
    result = summary.summarize(*_case(tail=40.))
    group = next(group for group in result["groups"] if group["cutoff"] == "observation_end")
    assert result["integration_gate_passed"] and result["flow_gain_threshold_passed"]
    assert group["comparisons"]["sync_joint"]["checks"]["common_mean_flow_s"]
    assert not group["comparisons"]["sync_joint"]["checks"]["common_p90_flow_s"]
    assert result["decision"].endswith("UTILITY_HOLD")


def test_better_restricted_mean_and_completed_cohort_cannot_hide_censored_tail():
    result = summary.summarize(*_case(tail=25.))
    group = next(group for group in result["groups"] if group["cutoff"] == "observation_end")
    assert result["integration_gate_passed"] and result["flow_gain_threshold_passed"]
    assert group["policies"]["refill_joint"]["unfinished_jobs"] == 1
    assert group["policies"]["sync_joint"]["unfinished_jobs"] == 0
    assert group["comparisons"]["sync_joint"]["checks"]["common_p90_flow_s"]
    assert not group["comparisons"]["sync_joint"]["checks"]["unfinished_jobs"]
    cohort = result["paired_runs"][0]["snapshots"]["observation_end"]["common_cohort"]
    assert cohort["count"] == 3
    assert result["decision"].endswith("UTILITY_HOLD")


def test_wall_time_does_not_advance_clock_or_determine_performance_gate():
    case = _case()
    before = summary.summarize(*case)
    current = case[2][0]["runs"][0]["policies"]["refill_joint"]
    for frame in current["frames"]:
        frame["selector_wall_s"] = 500.
    current["total_selector_wall_s"] = 500. * len(current["frames"])
    after = summary.summarize(*case)
    assert after["decision"] == before["decision"]
    assert after["pooled_flow_at_T"] == before["pooled_flow_at_T"]
    assert after["totals"]["refill_joint"]["total_selector_wall_s"] == 1500.


def test_single_lane_control_is_reported_but_does_not_change_registered_promotion_gate():
    models, protocol, rows = _case()
    policies = rows[0]["runs"][0]["policies"]
    tape = protocol["tapes"][0]["arrivals"]
    policies["refill_single"] = _trajectory(models[0], tape, protocol, "refill_single",
                                           [(0., (2,)), (0., (1,)), (1., (2,)), (2., (2,))])
    result = summary.summarize(models, protocol, rows)
    assert result["performance_gate_passed"]
    assert not result["refill_single_utility_descriptive_passed"]
    assert result["pooled_flow_improvement"]["refill_single"]["restricted_mean_flow_s"] < 0


def test_identical_flow_preserves_utility_but_holds_the_one_percent_gain():
    result = summary.summarize(*_case(equal=True))
    assert result["integration_gate_passed"] and result["utility_gate_passed"]
    assert result["pooled_flow_improvement"]["sync_joint"]["restricted_mean_flow_s"] == 0
    assert result["decision"].endswith("GAIN_HOLD")


def test_a_lane_cannot_be_freed_before_the_complete_local_option_ends():
    case = _case()
    current = case[2][0]["runs"][0]["policies"]["refill_joint"]
    current["lane_events"][1]["local_end"] = 1.
    current["lane_events"][1]["reserved_end"] = 1.
    result = summary.summarize(*case)
    assert not result["lane_accounting_passed"]
    assert result["decision"].endswith("EXECUTION_HOLD")


def test_nonzero_compute_delay_and_per_call_cap_violation_are_execution_failures():
    case = _case()
    current = case[2][0]["runs"][0]["policies"]["refill_joint"]
    current["frames"][0]["decision_delay_s"] = 1.
    current["decision_diagnostics"][0]["shared_integer_states"] = 8193
    result = summary.summarize(*case)
    assert not result["clock_accounting_passed"] and not result["execution_checks_passed"]
    assert result["decision"].endswith("EXECUTION_HOLD")


def test_missing_cell_or_tape_coverage_is_rejected():
    models, protocol, rows = _case(seeds=(41, 42))
    rows[0]["runs"].pop()
    with pytest.raises(ValueError, match="load/seed"):
        summary.summarize(models, protocol, rows)
    with pytest.raises(ValueError, match="node/R"):
        summary.summarize(models, protocol, [])


def test_incomplete_reference_retains_all_trajectories_and_holds():
    case = _case()
    current = case[2][0]["runs"][0]["policies"]["refill_joint"]
    current["selector_metrics"]["reference_incomplete_count"] = 1
    current["reference_audit"]["incomplete_count"] = 1
    current["decision_diagnostics"][0]["reference_complete"] = False
    result = summary.summarize(*case)
    assert result["trajectory_count"] == 3 and result["totals"]["refill_joint"]["calls"] == 3
    assert not result["references_complete"]
    assert result["decision"].endswith("REFERENCE_HOLD")


def test_missing_canonical_proof_holds_even_when_the_flow_gain_is_large():
    case = _case()
    current = case[2][0]["runs"][0]["policies"]["refill_joint"]
    current["frames"][0]["canonical_exact"] = False
    current["decision_diagnostics"][0]["canonical_exact"] = False
    current["selector_metrics"]["canonical_exact_count"] -= 1
    result = summary.summarize(*case)
    assert result["flow_gain_threshold_passed"] and not result["canonical_action_gate_passed"]
    assert result["decision"].endswith("INTEGRATION_HOLD")


def test_snapshot_edits_and_fifo_slot_swaps_are_not_accepted_as_performance():
    models, protocol, rows = _case()
    current = rows[0]["runs"][0]["policies"]["refill_joint"]
    first, second = current["jobs"][0], current["jobs"][1]
    first["start"], second["start"] = second["start"], first["start"]
    first["completion"], second["completion"] = second["completion"], first["completion"]
    for cutoff in summary.CUTOFFS:
        current["snapshots"][cutoff] = trajectory_snapshot(current["jobs"], protocol[cutoff], 2)
    current["snapshots"]["observation_end"]["restricted_mean_flow_s"] -= 1.
    result = summary.summarize(models, protocol, rows)
    assert not result["causal_fifo_checks_passed"] and not result["execution_checks_passed"]
    assert result["decision"].endswith("EXECUTION_HOLD")
