"""Synthetic runner tests for borrowed references and three-arm measurement."""
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
from itertools import permutations
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as confirmation
from algorithm.experiments import aligned_hardware_resumable_probe_development as previous_experiment
from algorithm.experiments import aligned_hardware_certificate_routed_development as experiment
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


@pytest.fixture
def case(monkeypatch):
    provenance = lambda: {"cpu_affinity": [0]}
    monkeypatch.setattr(confirmation, "runtime_provenance", provenance)
    monkeypatch.setattr(experiment, "runtime_provenance", provenance)
    options = tuple(FrameOption(str(i), requirements, duration, ((), ()))
                    for i, (requirements, duration) in enumerate((
                        ((1, 0), 5.), ((0, 1), 2.), ((2, 1), 5.),
                        ((1, 2), 3.), ((3, 0), 2.), ((0, 3), 3.))))
    model = FrameModel(node="synthetic", scenario_id="routed", lanes=4,
                       workloads=("a", "b"), scales=(1., 1.), options=options)
    protocol = {
        "registered_at_utc": "strict-prior", "budget": 3, "integer_state_budget": 100000,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "queue_panels": {"4": [{"panel": "development", "cohort": "old_regression", "seed": None,
             "states": [{"state_id": str(i), "queue": queue}
                        for i, queue in enumerate(([1, 6], [0, 0], [1, 1]))]}]},
    }
    prior = confirmation.screen_cell(model, 4, protocol, runtime, progress=False)
    for state in prior["states"]:
        state["expected_initial_lp"] = previous_experiment.previous_initial(state)
        state["expected_control_result"] = previous_experiment.confirmed_result(state)
        state["previous_integer_trigger"] = bool(state["completed"]["score_proof"]["attempted"]
                                                 or state["completed"]["completion"]["attempted"])
        state["screen_result"] = asdict(select_resumable_probe_frame(options, state["queue"], 4, 3, 64, 0))
    protocol.update(
        registered_at_utc="routed-test", prior_registered_at_utc="probe-prior", experiment="synthetic-routed",
        initial_lp_budget=3, lp_node_budget=64, integer_state_budget=0,
        control_integer_state_budget=100000, long_lp_budget=4096, models=[model.snapshot()], lanes=[4],
        execution_order=[[model.node, 4]], labels=list(experiment.LABELS), repeats=6,
        warmups_per_label=2, ordinary_per_cell=1, cpu=0,
        score_gain_limit=.95, state_ratio_limit=1.05, nonregression_limit=1.05, trigger_long_ratio_limit=1.,
        limitations="Tiny synthetic model only.",
    )
    indices = experiment.timing_roster(prior, 1)
    protocol["timing_indices"] = {"synthetic:4": indices}
    protocol["timing_orders"] = {"synthetic:4": experiment.timing_orders("synthetic", 4, len(indices))}
    return model, protocol, prior


def test_screen_and_timing_borrow_references_and_use_registered_budgets(case, monkeypatch):
    model, protocol, prior = case
    calls = []

    def wrapping(label, selector):
        def call(*args):
            calls.append((label, args[1], args[3:]))
            return selector(*args)
        return call

    def forbidden(*args, **kwargs):
        raise AssertionError("development must reuse existing references")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", forbidden)
    from algorithm.experiments import integer_capacity_canonical_oracle
    monkeypatch.setattr(integer_capacity_canonical_oracle, "canonical_at_certified_score", forbidden)
    old = deepcopy(prior)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior,
        wrapping("candidate", select_certificate_routed_frame),
        wrapping("probe", select_resumable_probe_frame),
        wrapping("long", select_strict_score_proof_canonical_completion_frame), False)
    indices = protocol["timing_indices"]["synthetic:4"]
    assert len(calls) == 3 + len(indices) * 18 + 6
    assert calls[:3] == [("candidate", tuple(s["queue"]), (3, 64, 0)) for s in prior["states"]]
    assert [c[0] for c in calls[3:9]] == list(experiment.LABELS) * 2
    assert all(args == ((4096, 100000) if label == "long" else (3, 64, 0)) for label, _, args in calls)
    expected_order = []
    orders = protocol["timing_orders"]["synthetic:4"]
    for repeat, state_order in enumerate(orders["state_orders"]):
        for position in state_order:
            queue = tuple(prior["states"][indices[position]]["queue"])
            expected_order.extend((label, queue) for label in orders["label_orders"][position][repeat])
    assert [(label, queue) for label, queue, _ in calls[9:]] == expected_order
    assert prior == old
    assert row["screen_quality_passed"]
    for state, source in zip(row["states"], prior["states"]):
        assert state["expected_probe_result"] == source["screen_result"]
        assert state["score_unproved"] == (not source["expected_initial_lp"]["selection"]["score_exact"])
        assert state["reference"] == source["reference"]
        assert state["oracle"] == source["oracle"]


def test_screen_nonclosure_preserves_all_timing_and_warmups(case):
    model, protocol, prior = case
    protocol["lp_node_budget"] = protocol["initial_lp_budget"]
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior, progress=False)
    assert not row["screen_quality_passed"]
    assert row["screen_failures"]
    for index in protocol["timing_indices"]["synthetic:4"]:
        state = row["states"][index]
        assert set(state["first_results"]) == set(experiment.LABELS)
        assert all(len(state["wall_ns"][label]) == 6 for label in experiment.LABELS)
    assert len(row["warmups"]) == 6


def test_three_arm_blocks_exclude_audits_and_serialization(case, monkeypatch):
    model, protocol, prior = case
    events, clocks = [], {"wall": 0, "cpu": 0}

    def clock(label):
        events.append(label)
        clocks[label] += 100
        return clocks[label]

    monkeypatch.setattr(experiment, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(experiment, "process_time_ns", lambda: clock("cpu"))
    original_record, original_asdict = runtime._record, experiment.asdict

    def record(*args, **kwargs):
        events.append("audit")
        return original_record(*args, **kwargs)

    def serialize(value):
        events.append("serialize")
        return original_asdict(value)

    def wrapping(selector):
        def call(*args):
            events.append("selector")
            return selector(*args)
        return call

    monkeypatch.setattr(runtime, "_record", record)
    monkeypatch.setattr(experiment, "asdict", serialize)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior,
        wrapping(select_certificate_routed_frame), wrapping(select_resumable_probe_frame),
        wrapping(select_strict_score_proof_canonical_completion_frame), False)
    measured = events[events.index("cpu"):]
    seen = 0
    for i, event in enumerate(measured):
        if event == "selector":
            assert measured[i - 2:i + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
            seen += 1
        elif event in ("audit", "serialize"):
            assert seen % 3 == 0
    assert seen == 18 * len(protocol["timing_indices"]["synthetic:4"])
    for index in protocol["timing_indices"]["synthetic:4"]:
        for label in experiment.LABELS:
            assert row["states"][index]["wall_ns"][label] == [100] * 6


def test_repeated_path_changes_are_saved_without_discarding_calls(case):
    model, protocol, prior = case
    counter = 0

    def changing(*args):
        nonlocal counter
        counter += 1
        result = select_certificate_routed_frame(*args)
        return replace(result, status=f"observed-{counter}")

    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior,
                                         candidate_selector=changing, progress=False)
    for index in protocol["timing_indices"]["synthetic:4"]:
        mismatches = row["states"][index]["repeat_mismatches"]["candidate"]
        assert [change["repeat_index"] for change in mismatches] == list(range(1, 6))
    assert counter == 3 + 2 + 6 * len(protocol["timing_indices"]["synthetic:4"])


def test_timing_orders_balance_all_six_permutations():
    orders = experiment.timing_orders("synthetic", 4, 17)
    assert orders == experiment.timing_orders("synthetic", 4, 17)
    for per_state in orders["label_orders"]:
        assert set(per_state) == set(permutations(experiment.LABELS))
        assert len(per_state) == 6
    assert len(orders["state_orders"]) == 6
    assert all(sorted(order) == list(range(17)) for order in orders["state_orders"])


@pytest.mark.parametrize("changed", ("source", "queue", "budget", "timing_order", "timing_subset"))
def test_registration_rejects_changed_sources_and_design(case, monkeypatch, tmp_path, changed):
    model, prior_protocol, prior = case
    prior_protocol = deepcopy(prior_protocol)
    prior_protocol.update(registered_at_utc="probe-prior", execution_sources={"frozen.py": "original"})
    (tmp_path / "frozen.py").write_text("original")
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({"decision": "RESUMABLE_PROBE_DEVELOPMENT_ADVANCE",
        **{key: True for key in ("certificate_checks_passed", "effective_references_complete", "quality_gate_passed",
                                 "execution_checks_passed", "development_timing_gate_passed")}}))
    prior_path = tmp_path / "prior_protocol.json"
    prior_path.write_text(json.dumps(prior_protocol))
    with gzip.open(_cell_path(prior_output, model.node, 4), "wt") as target:
        json.dump({"registered_at_utc": "probe-prior", "cell": prior}, target)
    for name, value in (("ROOT", tmp_path), ("PRIOR_OUTPUT", prior_output), ("PRIOR_PROTOCOL", prior_path),
                        ("EXTRA_SOURCES", ()), ("INITIAL_LP_BUDGET", 3), ("LP_NODE_BUDGET", 64),
                        ("INTEGER_STATE_BUDGET", 0), ("ORDINARY_PER_CELL", 1)):
        monkeypatch.setattr(experiment, name, value)
    protocol = experiment.build_protocol((model,))
    protocol["registered_at_utc"] = "frozen"
    if changed == "source":
        (tmp_path / "frozen.py").write_text("modified")
    elif changed == "queue":
        protocol["queue_panels"]["4"][0]["states"][0]["queue"][0] += 1
    elif changed == "budget":
        protocol["integer_state_budget"] += 1
    elif changed == "timing_order":
        protocol["timing_orders"]["synthetic:4"]["label_orders"][0][0].reverse()
    else:
        protocol["timing_indices"]["synthetic:4"].pop()
    with pytest.raises(ValueError):
        experiment.verify_protocol(protocol, (model,))
