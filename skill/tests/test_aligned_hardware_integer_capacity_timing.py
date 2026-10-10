"""Check timing boundaries and decisions without running the hardware benchmark."""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import permutations
import json
from types import SimpleNamespace

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_timing as timing
from algorithm.experiments import aligned_hardware_integer_capacity_timing_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_development import _metrics


LABELS = ("old_shared", "fractional_shared", "integer_capacity")


@pytest.fixture
def model():
    return FrameModel(
        node="timing_test", scenario_id="small_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(FrameOption("a", (1, 0, 0), 1.0, ((1.0,), (), ())),),
    )


def _metric(calls):
    result = _metrics()
    result.update(calls=calls, score_exact_count=calls, canonical_exact_count=calls,
                  reference_comparable_count=calls, score_reference_comparable_count=calls,
                  deterministic_repeat_mismatch_count=0)
    return result


def _synthetic_case(model, queues=((2, 0, 0),)):
    protocol = {
        "schema_version": 1, "experiment": "synthetic_timing", "registered_at_utc": "test",
        "models": [model.snapshot()], "lanes": [4], "labels": list(LABELS),
        "repeats": 6, "cpu": 0, "budgets": {"4": 144}, "reference_budget": 4096,
        "pooled_ratio_limit": 0.95, "cell_ratio_limit": 1.0,
        "queue_panels": {"4": [{"panel": "fresh_test", "seed": 123, "queues": list(map(list, queues))}]},
        "limitations": "Synthetic test data.",
    }
    states = [{
        "panel": "fresh_test", "seed": 123, "queues": list(queue),
        "wall_ns": {label: [80 if label == "integer_capacity" else 100] * 6 for label in LABELS},
        "cpu_ns": {label: [70 if label == "integer_capacity" else 90] * 6 for label in LABELS},
    } for queue in queues]
    row = {
        "node": model.node, "lanes": 4, "queue_count": len(states), "states": states,
        "uncertified_reference_count": 0,
        "correctness": {"reference": _metric(len(states)),
                        **{label: _metric(6 * len(states)) for label in LABELS}},
        **{key: {"cpu_affinity": [0]} for key in
           ("runtime_before_warmup", "runtime_after_warmup", "runtime_end")},
    }
    return protocol, row


def test_summary_accepts_complete_timings_with_both_controls(model):
    protocol, row = _synthetic_case(model)
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"].endswith("ADVANCE")
    assert result["queue_count"] == 1
    assert result["timed_selector_calls"] == 18
    assert result["untimed_reference_calls"] == 1


@pytest.mark.parametrize("failure", (
    "reference", "incorrect_score", "incorrect_action", "nonexact", "invalid_certificate",
    "nondeterministic", "missing_reference_call", "missing_timed_call", "cpu_affinity",
))
def test_failed_correctness_or_execution_never_advances(model, failure):
    protocol, row = _synthetic_case(model)
    if failure == "reference":
        row["uncertified_reference_count"] = 1
        row["correctness"]["reference"]["canonical_exact_count"] = 0
    elif failure == "incorrect_score":
        row["correctness"]["integer_capacity"]["score_mismatch_count"] = 1
    elif failure == "incorrect_action":
        row["correctness"]["integer_capacity"]["canonical_action_mismatch_count"] = 1
    elif failure == "nonexact":
        row["correctness"]["integer_capacity"]["canonical_exact_count"] -= 1
    elif failure == "invalid_certificate":
        row["correctness"]["fractional_shared"]["failures"] = 1
    elif failure == "nondeterministic":
        row["correctness"]["old_shared"]["deterministic_repeat_mismatch_count"] = 1
    elif failure == "missing_reference_call":
        row["correctness"]["reference"]["calls"] = 0
    elif failure == "missing_timed_call":
        row["correctness"]["integer_capacity"]["calls"] -= 1
    else:
        row["runtime_end"]["cpu_affinity"] = [0, 1]
    assert not summary.summarize((model,), protocol, [row])["decision"].endswith("ADVANCE")


def test_original_controls_may_remain_truthfully_inexact(model):
    protocol, row = _synthetic_case(model)
    for label in LABELS[:2]:
        row["correctness"][label].update(score_exact_count=0, canonical_exact_count=0,
                          score_mismatch_count=6, canonical_action_mismatch_count=6,
                          cap_reached_count=6, uncertified_cap_reached_count=6)
    assert summary.summarize((model,), protocol, [row])["decision"].endswith("ADVANCE")


@pytest.mark.parametrize("control", LABELS[:2])
def test_each_control_cell_gate_blocks_despite_pooled_benefit(model, control):
    protocol, first = _synthetic_case(model)
    other = replace(model, node="other_hardware")
    second = deepcopy(first)
    second["node"] = other.node
    protocol["models"].append(other.snapshot())
    first["states"][0]["wall_ns"][control] = [70] * 6  # New method takes 80.
    second["states"][0]["wall_ns"][control] = [1000] * 6
    result = summary.summarize((model, other), protocol, [first, second])
    assert not result["decision"].endswith("ADVANCE")


def test_primary_cost_is_sum_of_state_medians_not_minimum_or_pooled_repeats(model):
    protocol, row = _synthetic_case(model, queues=((1, 0, 0), (2, 0, 0), (3, 0, 0)))
    # Per-state medians: controls 100 + 100 + 1000 = 1200; new 90 + 90 + 850 = 1030.
    # Minimums favor controls (3 vs 15); repeat-pooled medians favor controls (100 vs 175).
    for state, control, candidate in zip(
        row["states"],
        ([1, 100, 100, 100, 100, 100], [1, 100, 100, 100, 100, 100], [1, 1000, 1000, 1000, 1000, 1000]),
        ([5, 5, 5, 175, 900, 900], [5, 5, 5, 175, 900, 900], [5, 5, 850, 850, 900, 900]),
    ):
        state["wall_ns"] = {LABELS[0]: control, LABELS[1]: control, LABELS[2]: candidate}
    result = summary.summarize((model,), protocol, [row])
    assert result["decision"].endswith("ADVANCE")
    assert result["queue_count"] == 3
    assert result["timed_selector_calls"] == 54
    assert result["untimed_reference_calls"] == 3
    assert result["latency"]["integer_capacity"]["sum_state_median_wall_ns"] == 1030
    assert result["latency"]["old_shared"]["sum_state_median_wall_ns"] == 1200


@pytest.mark.parametrize("failure", ("missing_repeat", "missing_state", "duplicate_state", "wrong_queue", "missing_cpu"))
def test_missing_or_wrong_measurement_coverage_is_rejected(model, failure):
    protocol, row = _synthetic_case(model)
    if failure == "missing_repeat":
        row["states"][0]["wall_ns"]["integer_capacity"].pop()
    elif failure == "missing_state":
        row["states"] = []
    elif failure == "duplicate_state":
        row["states"].append(deepcopy(row["states"][0]))
    elif failure == "wrong_queue":
        row["states"][0]["queues"] = [3, 0, 0]
    else:
        del row["states"][0]["cpu_ns"]["integer_capacity"]
    with pytest.raises((ValueError, KeyError)):
        summary.summarize((model,), protocol, [row])


def test_registered_order_balances_every_state_across_all_six_permutations():
    orders = timing.timing_orders("test_node", 8, 64)
    assert orders == timing.timing_orders("test_node", 8, 64)
    assert len(orders["state_orders"]) == 6
    assert all(sorted(order) == list(range(64)) for order in orders["state_orders"])
    assert len(orders["permutation_indices"]) == 64
    choices = list(permutations(LABELS))
    for order in orders["permutation_indices"]:
        assert sorted(order) == list(range(6))
        for label in LABELS:
            assert [choices[index].index(label) for index in order].count(0) == 2
            assert [choices[index].index(label) for index in order].count(1) == 2
            assert [choices[index].index(label) for index in order].count(2) == 2


def test_protocol_reuses_complete_successful_roster_and_budgets():
    from algorithm.experiments.aligned_hardware_integer_capacity_development import load_catalogues
    models = load_catalogues()
    previous = json.loads(timing.CONFIRMATION_PROTOCOL.read_text())
    protocol = timing.build_protocol(models)
    assert protocol == timing.build_protocol(models)
    for key in ("models", "queue_panels", "panel_seeds", "lanes", "budgets", "reference_budget"):
        assert protocol[key] == previous[key]
    assert protocol["budgets"] == {"4": 144, "8": 288, "16": 576}
    assert protocol["reference_budget"] == 4096
    assert protocol["repeats"] == 6
    assert protocol["pooled_ratio_limit"] == 0.95
    assert protocol["cell_ratio_limit"] == 1.0
    assert len(protocol["execution_order"]) == 12
    assert sum(len(timing.state_roster(protocol, lanes)) for lanes in protocol["lanes"]) == 768


@dataclass(frozen=True)
class _FakeWork:
    operations: int = 1


@dataclass(frozen=True)
class _FakeSelection:
    label: str
    lp_calls: int = 1
    changed_path: int = 0


@dataclass(frozen=True)
class _FakeResult:
    selection: _FakeSelection
    preprocessing: _FakeWork = _FakeWork()
    capacity_propagation: _FakeWork = _FakeWork()


@pytest.mark.parametrize("nondeterministic", (False, True))
def test_whole_selector_is_timed_with_references_and_checks_outside(model, monkeypatch, nondeterministic):
    protocol, _ = _synthetic_case(model, ((1, 0, 0), (2, 0, 0)))
    protocol.update(
        selector_permutations=list(permutations(LABELS)), search_counters=["lp_calls"],
        timing_orders={f"{model.node}:4": timing.timing_orders(model.node, 4, 2)},
    )
    clock = {"now": 0, "active": False}
    trace, calls = [], []

    def wall_clock():
        clock["active"] = not clock["active"]
        trace.append("start" if clock["active"] else "end")
        return clock["now"]

    def reference(options, queue, lanes, budget):
        assert not clock["active"]
        calls.append(("reference", queue, lanes, budget, False))
        clock["now"] += 500
        return _FakeSelection("reference")

    def selector(label):
        def invoke(options, queue, lanes, budget):
            calls.append((label, queue, lanes, budget, clock["active"]))
            for phase, duration in (("preprocessing", 3), ("search", 5), ("result", 2)):
                if clock["active"]:
                    trace.append(phase)
                clock["now"] += duration
            changed = int(nondeterministic and label == "old_shared" and
                          sum(call[0] == label and call[1] == queue and call[-1] for call in calls) > 1)
            return _FakeResult(_FakeSelection(label, changed_path=changed))
        return invoke

    def record(metrics, selected, *args):
        assert not clock["active"]
        trace.append("reference_check" if selected.label == "reference" else "check")
        clock["now"] += 1000
        metrics["calls"] += 1
        metrics["score_exact_count"] += 1
        metrics["canonical_exact_count"] += 1

    def oracle(*args):
        assert not clock["active"]
        clock["now"] += 100
        return "expected"

    runtime = SimpleNamespace(
        _metrics=_metrics, _record=record, enumerate_frames=oracle, oracle=oracle,
        reference_key=lambda *args: "expected", select_canonical_count_space_bnb_frame=reference,
        select_shared_threshold_canonical_requirement_dominance_frame=selector("old_shared"),
        select_fractional_shared_threshold_canonical_requirement_dominance_frame=selector("fractional_shared"),
        select_integer_capacity_shared_threshold_frame=selector("integer_capacity"),
    )
    real_asdict = timing.asdict

    def untimed_asdict(value):
        assert not clock["active"]
        clock["now"] += 1000
        return real_asdict(value)

    def cpu_clock():
        clock["now"] += 7  # A CPU clock read must not enter the primary wall interval.
        return clock["now"]

    monkeypatch.setattr(timing, "asdict", untimed_asdict)
    monkeypatch.setattr(timing, "perf_counter_ns", wall_clock)
    monkeypatch.setattr(timing, "process_time_ns", cpu_clock)
    monkeypatch.setattr(timing, "runtime_provenance", lambda: {"cpu_affinity": [0]})
    row = timing.timing_cell(model, 4, protocol, runtime, progress=False)
    assert not clock["active"]
    assert row["correctness"]["reference"]["calls"] == 2
    assert len([call for call in calls if call[0] == "reference"]) == 2
    for label in LABELS:
        assert row["correctness"][label]["calls"] == 12
        assert row["warmup_calls_by_label"][label] == 2
        assert len([call for call in calls if call[0] == label and call[-1]]) == 12
        assert all(state["wall_ns"][label] == [10] * 6 for state in row["states"])
        assert all(state["cpu_ns"][label] == [17] * 6 for state in row["states"])
    assert all(call[3] == (4096 if call[0] == "reference" else 144) for call in calls)
    block = ["start", "preprocessing", "search", "result", "end"] * 3 + ["check"] * 3
    assert trace == ["reference_check"] * 2 + block * 12
    assert row["correctness"]["old_shared"]["deterministic_repeat_mismatch_count"] == (10 if nondeterministic else 0)
