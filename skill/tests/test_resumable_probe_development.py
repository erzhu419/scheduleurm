"""Synthetic runner integration and real call/measurement accounting."""
from copy import deepcopy
import gzip
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as confirmation
from algorithm.experiments import aligned_hardware_resumable_probe_development as experiment
from algorithm.experiments import resumable_probe_development_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
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
    model = FrameModel(node="synthetic", scenario_id="probe", lanes=4,
                       workloads=("a", "b"), scales=(1., 1.), options=options)
    protocol = {
        "registered_at_utc": "prior", "budget": 3, "integer_state_budget": 100000,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "queue_panels": {"4": [{"panel": "development", "cohort": "old_regression", "seed": None,
             "states": [{"state_id": str(i), "queue": queue}
                        for i, queue in enumerate(([1, 6], [0, 0], [1, 1]))]}]},
    }
    prior = confirmation.screen_cell(model, 4, protocol, runtime, progress=False)
    protocol.update(
        registered_at_utc="probe-test", prior_registered_at_utc="prior", experiment="synthetic-probe",
        initial_lp_budget=3, lp_node_budget=64, integer_state_budget=0,
        control_integer_state_budget=100000, models=[model.snapshot()], lanes=[4],
        execution_order=[[model.node, 4]], labels=["candidate", "control"], repeats=6,
        warmups_per_label=2, ordinary_per_cell=1, cpu=0,
        trigger_ratio_limit=.90, trigger_state_ratio_limit=1.05, ordinary_ratio_limit=1.05,
        limitations="Tiny synthetic model only.",
    )
    indices = experiment.timing_roster(prior, 1)
    protocol["timing_indices"] = {"synthetic:4": indices}
    protocol["timing_orders"] = {"synthetic:4": experiment.timing_orders("synthetic", 4, len(indices), 6)}
    assert prior["states"][0]["completed"]["completion"]["attempted"]
    return model, protocol, prior


def test_real_screen_timing_summary_reuses_references_and_counts_calls(case, monkeypatch):
    model, protocol, prior = case
    calls = []

    def candidate(*args):
        calls.append(("candidate", args[1], args[3:6]))
        return select_resumable_probe_frame(*args)

    def control(*args):
        calls.append(("control", args[1], args[3:5]))
        return select_strict_score_proof_canonical_completion_frame(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("timing must reuse existing references")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", forbidden)
    from algorithm.experiments import integer_capacity_canonical_oracle
    monkeypatch.setattr(integer_capacity_canonical_oracle, "canonical_at_certified_score", forbidden)
    old = deepcopy(prior)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior, candidate, control, False)
    result = summary.summarize(protocol, [row])
    indices = protocol["timing_indices"]["synthetic:4"]
    count = len(indices)
    assert len(calls) == 3 + count * 12 + 4
    assert calls[:3] == [("candidate", tuple(s["queue"]), (3, 64, 0)) for s in prior["states"]]
    assert [c[0] for c in calls[3:7]] == ["candidate", "control"] * 2
    expected_order = []
    order = protocol["timing_orders"]["synthetic:4"]
    for repeat, state_order in enumerate(order["state_orders"]):
        for position in state_order:
            queue = tuple(prior["states"][indices[position]]["queue"])
            expected_order.extend((label, queue) for label in order["label_orders"][position][repeat])
    assert [(label, queue) for label, queue, _ in calls[7:]] == expected_order
    assert result["queue_count"] == 3
    assert result["executed_selector_calls"] == len(calls)
    assert result["quality_gate_passed"]
    assert not result["accounting_failures"]
    assert prior == old
    assert row["states"][0]["screen_result"]["resumed"]


def test_honest_screen_nonclosure_retains_entire_registered_timing(case):
    model, protocol, prior = case
    protocol["lp_node_budget"] = protocol["initial_lp_budget"]
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior, progress=False)
    result = summary.summarize(protocol, [row])
    assert not row["screen_quality_passed"]
    assert not result["quality_gate_passed"]
    assert result["executed_selector_calls"] == 3 + 12 * len(protocol["timing_indices"]["synthetic:4"]) + 4
    for index in protocol["timing_indices"]["synthetic:4"]:
        state = row["states"][index]
        assert all(len(state["wall_ns"][label]) == 6 for label in protocol["labels"])
    assert len(row["warmups"]) == 4


def test_whole_selector_clocks_exclude_audits_and_pairs_stay_adjacent(case, monkeypatch):
    model, protocol, prior = case
    events, clocks = [], {"wall": 0, "cpu": 0}

    def clock(label):
        events.append(label)
        clocks[label] += 100
        return clocks[label]

    monkeypatch.setattr(experiment, "perf_counter_ns", lambda: clock("wall"))
    monkeypatch.setattr(experiment, "process_time_ns", lambda: clock("cpu"))
    original_record = runtime._record

    def record(*args, **kwargs):
        events.append("audit")
        return original_record(*args, **kwargs)

    def candidate(*args):
        events.append("selector")
        return select_resumable_probe_frame(*args)

    def control(*args):
        events.append("selector")
        return select_strict_score_proof_canonical_completion_frame(*args)

    monkeypatch.setattr(runtime, "_record", record)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, prior, candidate, control, False)
    measured = events[events.index("cpu"):]
    seen = 0
    for i, event in enumerate(measured):
        if event == "selector":
            assert measured[i - 2:i + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
            seen += 1
        elif event == "audit":
            assert seen % 2 == 0
    assert seen == 12 * len(protocol["timing_indices"]["synthetic:4"])
    for index in protocol["timing_indices"]["synthetic:4"]:
        for label in protocol["labels"]:
            assert row["states"][index]["wall_ns"][label] == [100] * 6


@pytest.mark.parametrize("changed", ("source", "queue", "budget", "timing_order"))
def test_registration_rejects_changes_before_execution(case, monkeypatch, tmp_path, changed):
    model, prior_protocol, prior = case
    prior_protocol = deepcopy(prior_protocol)
    prior_protocol.update(registered_at_utc="prior", integer_state_budget=100000,
                          execution_sources={"frozen.py": "original"})
    (tmp_path / "frozen.py").write_text("original")
    prior_output = tmp_path / "prior.json"
    prior_output.write_text(json.dumps({"decision": "STRICT_SCORE_PROOF_DEVELOPMENT_ADVANCE",
                                       "certificate_checks_passed": True, "effective_references_complete": True}))
    prior_path = tmp_path / "prior_protocol.json"
    prior_path.write_text(json.dumps(prior_protocol))
    with gzip.open(_cell_path(prior_output, model.node, 4), "wt") as target:
        json.dump({"registered_at_utc": "prior", "cell": prior}, target)
    for name, value in (("ROOT", tmp_path), ("PRIOR_OUTPUT", prior_output), ("PRIOR_PROTOCOL", prior_path),
                        ("EXTRA_SOURCES", ()), ("INITIAL_LP_BUDGET", 3), ("ORDINARY_PER_CELL", 1)):
        monkeypatch.setattr(experiment, name, value)
    protocol = experiment.build_protocol((model,))
    protocol["registered_at_utc"] = "frozen"
    if changed == "source":
        (tmp_path / "frozen.py").write_text("modified")
    elif changed == "queue":
        protocol["queue_panels"]["4"][0]["states"][0]["queue"][0] += 1
    elif changed == "budget":
        protocol["integer_state_budget"] += 1
    else:
        protocol["timing_orders"]["synthetic:4"]["label_orders"][0][0].reverse()
    with pytest.raises(ValueError):
        experiment.verify_protocol(protocol, (model,))
