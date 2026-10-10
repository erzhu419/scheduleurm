"""Fresh-reference order, frozen three-arm timing, and synthetic integration."""
from copy import deepcopy
from dataclasses import replace
import gzip
from itertools import permutations
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_resumable_probe_confirmation as experiment
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as strict_confirmation
from algorithm.experiments import resumable_probe_confirmation_summary as summary
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame


@pytest.fixture
def case(monkeypatch):
    provenance = lambda: {"cpu_affinity": [0]}
    monkeypatch.setattr(experiment, "runtime_provenance", provenance)
    monkeypatch.setattr(strict_confirmation, "runtime_provenance", provenance)
    pairs = (((1, 0), 5.), ((0, 1), 2.), ((2, 1), 5.), ((1, 2), 3.), ((3, 0), 2.), ((0, 3), 3.))
    options = tuple(FrameOption(str(i), req, duration, ((), ())) for i, (req, duration) in enumerate(pairs))
    model = FrameModel(node="synthetic", scenario_id="probe", lanes=4, workloads=("a", "b"), scales=(1., 1.), options=options)
    protocol = {
        "experiment": "synthetic_confirmation", "registered_at_utc": "test", "models": [model.snapshot()], "lanes": [4],
        "initial_lp_budget": 3, "lp_node_budget": 64, "integer_state_budget": 0,
        "control_integer_state_budget": 100000, "long_lp_budget": 64,
        "reference_budget": 4096, "oracle_state_budget": 5000000,
        "labels": ["candidate", "legacy", "long"], "repeats": 6, "warmups_per_label": 2,
        "timing_fresh_per_panel": 1, "cpu": 0, "panel_seeds": [123], "queues_per_panel": 1,
        "historical_trigger_keys": [{"node": "synthetic", "lanes": 4, "queue": [1, 6]}],
        "execution_order": [["synthetic", 4]], "fresh_nonregression_limit": 1.05,
        "trigger_legacy_ratio_limit": .9, "trigger_legacy_state_limit": 1.05,
        "trigger_long_ratio_limit": 1., "trigger_long_p99_limit": 1.05,
        "tail_state_limit": 1.05, "fresh_speedup_limit": .95, "limitations": "Synthetic only.",
        "queue_panels": {"4": [
            {"panel": "fresh", "cohort": "fresh_confirmation", "seed": 123,
             "states": [{"state_id": "fresh:123:0", "queue": [0, 0]}]},
            {"panel": "hard", "cohort": "old_regression", "seed": None,
             "states": [{"state_id": "hard:0", "queue": [1, 6]}]},
            {"panel": "tail", "cohort": "ordinary_tail_regression", "seed": None,
             "states": [{"state_id": "tail:0", "queue": [1, 1]}]},
        ]},
    }
    indices = experiment.timing_indices(protocol, model.node, 4)
    protocol["timing_indices"] = {"synthetic:4": indices}
    protocol["timing_orders"] = {"synthetic:4": experiment.timing_orders("synthetic", 4, len(indices))}
    return model, protocol


def test_three_arm_orders_are_complete_balanced_permutations(case):
    _, protocol = case
    order = protocol["timing_orders"]["synthetic:4"]
    assert order == experiment.timing_orders("synthetic", 4, 3)
    for labels in order["label_orders"]:
        assert sorted(map(tuple, labels)) == sorted(permutations(experiment.LABELS))
    assert all(sorted(indices) == [0, 1, 2] for indices in order["state_orders"])


def test_timing_subgroups_use_frozen_exact_node_queue_keys(case):
    _, protocol = case
    assert experiment.timing_indices(protocol, "synthetic", 4) == [0, 1, 2]
    assert experiment.timing_indices(protocol, "other_catalogue", 4) == [0, 2]
    fresh = protocol["queue_panels"]["4"][0]["states"]
    fresh.append({"state_id": "fresh:123:1", "queue": [0, 1]})
    assert experiment.timing_indices(protocol, "synthetic", 4) == [0, 2, 3]


def test_new_reference_precedes_independent_legacy_and_candidate_with_full_timing(case, monkeypatch):
    model, protocol = case
    events = []
    original_reference = runtime.select_canonical_count_space_bnb_frame

    def reference(*args):
        events.append(("reference", args[1]))
        return original_reference(*args)

    def candidate(*args):
        events.append(("candidate", args[1]))
        return select_resumable_probe_frame(*args)

    def legacy(*args):
        events.append(("legacy", args[1]))
        return select_strict_score_proof_canonical_completion_frame(*args)

    def longer(*args):
        events.append(("long", args[1]))
        return select_strict_score_proof_canonical_completion_frame(*args)

    def no_oracle(*args, **kwargs):
        raise AssertionError("complete full references must not invoke conditional oracles")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, legacy, longer, no_oracle, False)
    result = summary.summarize(protocol, [row])
    queues = [tuple(state["queue"]) for state in experiment.state_roster(protocol, 4)]
    assert events[:9] == [(label, queue) for queue in queues for label in ("reference", "legacy", "candidate")]
    assert [label for label, _ in events[9:15]] == list(experiment.LABELS) * 2
    expected_order = []
    order = protocol["timing_orders"]["synthetic:4"]
    for repeat, positions in enumerate(order["state_orders"]):
        for position in positions:
            expected_order.extend((label, queues[position]) for label in order["label_orders"][position][repeat])
    assert events[15:] == expected_order
    assert len(events) == 3 * 3 + 3 * 18 + 6
    assert result["candidate_quality_gate_passed"] and result["comparison_quality_gate_passed"] and result["certificate_checks_passed"]
    assert result["executed_reference_calls"] == 3 and result["executed_oracle_calls"] == 0
    assert result["total_executed_lp_selector_calls"] == len(events)
    assert row["states"][1]["screen_results"]["candidate"]["resumed"]
    for state in row["states"]:
        assert state["expected_initial_lp"]["selection"] == state["screen_results"]["legacy"]["baseline_selection"]


def test_measurement_block_has_three_selectors_before_any_audit_or_serialization(case, monkeypatch):
    model, protocol = case
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

    def serialize(*args, **kwargs):
        events.append("serialize")
        return original_asdict(*args, **kwargs)

    def candidate(*args):
        events.append("selector")
        return select_resumable_probe_frame(*args)

    def control(*args):
        events.append("selector")
        return select_strict_score_proof_canonical_completion_frame(*args)

    monkeypatch.setattr(runtime, "_record", record)
    monkeypatch.setattr(experiment, "asdict", serialize)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, control, control, progress=False)
    measured, count = events[events.index("cpu"):], 0
    for i, event in enumerate(measured):
        if event == "selector":
            assert measured[i - 2:i + 3] == ["cpu", "wall", "selector", "wall", "cpu"]
            count += 1
        elif event in ("audit", "serialize"):
            assert count % 3 == 0
    assert count == 54
    assert all(state["wall_ns"][label] == [100] * 6 for state in row["states"] for label in experiment.LABELS)


def test_screen_nonclosure_keeps_all_three_arm_timing_and_warmups(case):
    model, protocol = case
    protocol["lp_node_budget"] = protocol["initial_lp_budget"]
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, progress=False)
    result = summary.summarize(protocol, [row])
    assert not row["states"][1]["screen_results"]["candidate"]["selection"]["canonical_exact"]
    assert not result["candidate_quality_gate_passed"]
    assert result["total_executed_lp_selector_calls"] == 69
    assert len(row["warmups"]) == 6
    assert all(len(state["wall_ns"][label]) == 6 for state in row["states"] for label in experiment.LABELS)


def test_conditional_reference_oracle_runs_before_both_arms(case, monkeypatch):
    model, protocol = case
    events = []
    original_reference = runtime.select_canonical_count_space_bnb_frame
    from algorithm.experiments.integer_capacity_canonical_oracle import canonical_at_certified_score

    def reference(*args):
        events.append("reference")
        result = original_reference(*args)
        return replace(result, exact=False, canonical_exact=False, canonical_tie_regions_remaining=1,
                       frontier_regions_remaining=1)

    def oracle(*args, **kwargs):
        events.append("oracle")
        assert kwargs["state_visit_budget"] == protocol["oracle_state_budget"]
        return canonical_at_certified_score(*args, **kwargs)

    def candidate(*args):
        events.append("candidate")
        return select_resumable_probe_frame(*args)

    def legacy(*args):
        events.append("legacy")
        return select_strict_score_proof_canonical_completion_frame(*args)

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, legacy, legacy, oracle, False)
    assert events[:12] == ["reference", "oracle", "legacy", "candidate"] * 3
    assert all(not state["reference_complete"] and state["effective_reference_complete"] for state in row["states"])
    assert all(state["effective_reference_source"] == "independent_score_face_oracle" for state in row["states"])


def test_queue_design_excludes_history_and_pairs_regressions(monkeypatch):
    monkeypatch.setattr(experiment, "LANES", (4,))
    monkeypatch.setattr(experiment, "PANEL_SEEDS", (2026092791, 2026092792))
    monkeypatch.setattr(experiment, "QUEUES_PER_PANEL", 4)
    monkeypatch.setattr(experiment, "PRIOR_FRESH_TRIGGERS", {4: ((1, 2, 3),)})
    monkeypatch.setattr(experiment, "ORDINARY_TAIL_QUEUES", {4: ((4, 5, 6),)})
    previous = {"4": [
        {"panel": "oldfresh", "cohort": "fresh_confirmation", "seed": 5, "states": [{"state_id": "oldfresh:0", "queue": [7, 8, 9]}]},
        {"panel": "reg", "cohort": "old_regression", "seed": None, "states": [{"state_id": "old:0", "queue": [1, 1, 1]}]},
    ]}
    exclusions = {"4": {(7, 8, 9), (1, 1, 1), (1, 2, 3), (4, 5, 6)}}
    original = deepcopy(previous)
    panels = experiment.queue_design(previous, exclusions)["4"]
    fresh = [tuple(state["queue"]) for panel in panels if panel["cohort"] == "fresh_confirmation" for state in panel["states"]]
    assert len(fresh) == len(set(fresh)) == 8
    assert not set(fresh) & exclusions["4"]
    assert previous == original
    assert [panel["cohort"] for panel in panels[2:]] == ["old_regression", "prior_fresh_trigger_regression", "ordinary_tail_regression"]


@pytest.mark.parametrize("change", ("source", "queue", "budget", "timing_order"))
def test_registration_rejects_execution_changes(case, monkeypatch, tmp_path, change):
    model, template = case
    strict = {**template, "budget": 3, "integer_state_budget": 100000, "execution_sources": {"frozen.py": "original"}}
    strict["registered_at_utc"] = "strict"
    strict_row = strict_confirmation.screen_cell(model, 4, strict, runtime, progress=False)
    prior = {**template, "registered_at_utc": "development", "execution_sources": {"frozen.py": "original"}}
    (tmp_path / "frozen.py").write_text("original")
    prior_output = tmp_path / "dev.json"
    prior_output.write_text(json.dumps({"decision": "RESUMABLE_PROBE_DEVELOPMENT_ADVANCE", **{key: True for key in (
        "certificate_checks_passed", "effective_references_complete", "quality_gate_passed", "execution_checks_passed", "development_timing_gate_passed")}}))
    prior_path, strict_path, strict_output = (tmp_path / name for name in ("dev_protocol.json", "strict_protocol.json", "strict.json"))
    prior_path.write_text(json.dumps(prior))
    strict_path.write_text(json.dumps(strict))
    with gzip.open(_cell_path(strict_output, "synthetic", 4), "wt") as target:
        json.dump({"registered_at_utc": "strict", "cell": strict_row}, target)
    changes = {"ROOT": tmp_path, "PRIOR_OUTPUT": prior_output, "PRIOR_PROTOCOL": prior_path,
        "STRICT_PROTOCOL": strict_path, "STRICT_OUTPUT": strict_output, "EXTRA_SOURCES": (),
        "STATIC_PROTOCOLS": (), "FRONTIER_PROTOCOL": tmp_path / "frontier.json", "LANES": (4,),
        "PANEL_SEEDS": (2026092791,), "QUEUES_PER_PANEL": 2, "TIMING_FRESH_PER_PANEL": 1,
        "INITIAL_LP_BUDGET": 3, "LP_NODE_BUDGET": 64, "INTEGER_STATE_BUDGET": 0,
        "PRIOR_FRESH_TRIGGERS": {}, "ORDINARY_TAIL_QUEUES": {}, "EXPECTED_TRIGGER_COUNT": 1}
    for name, value in changes.items():
        monkeypatch.setattr(experiment, name, value)
    monkeypatch.setattr(experiment, "static_exclusions", lambda previous: {"4": set()})
    protocol = experiment.build_protocol((model,))
    protocol["registered_at_utc"] = "frozen"
    if change == "source":
        (tmp_path / "frozen.py").write_text("changed")
    elif change == "queue":
        protocol["queue_panels"]["4"][0]["states"][0]["queue"][0] += 1
    elif change == "budget":
        protocol["integer_state_budget"] += 1
    else:
        protocol["timing_orders"]["synthetic:4"]["label_orders"][0][0].reverse()
    with pytest.raises(ValueError):
        experiment.verify_protocol(protocol, (model,))
