"""Fresh-reference order, frozen three-arm timing, and synthetic integration."""
from copy import deepcopy
from dataclasses import replace
import gzip
from itertools import permutations
import json

import pytest

from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
from algorithm.experiments import aligned_hardware_certificate_routed_confirmation as experiment
from algorithm.experiments import aligned_hardware_strict_score_proof_confirmation as strict_confirmation
from algorithm.experiments.certificate_routed_development_summary import audit_result
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.resumable_probe_frame_candidates import select_resumable_probe_frame
from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
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
        "labels": ["candidate", "probe", "long"], "repeats": 6, "warmups_per_label": 2,
        "timing_fresh_per_panel": 1, "cpu": 0, "panel_seeds": [123], "queues_per_panel": 1,
        "historical_trigger_keys": [{"node": "synthetic", "lanes": 4, "queue": [1, 6]}],
        "score_unproved_keys": [], "previous_untimed_probe_keys": [],
        "execution_order": [["synthetic", 4]], "fresh_nonregression_limit": 1.05,
        "score_gain_limit": .95, "state_ratio_limit": 1.05, "nonregression_limit": 1.05,
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


def test_new_reference_precedes_independent_probe_and_candidate_with_full_timing(case, monkeypatch):
    model, protocol = case
    events = []
    original_reference = runtime.select_canonical_count_space_bnb_frame

    def reference(*args):
        events.append(("reference", args[1]))
        return original_reference(*args)

    def candidate(*args):
        events.append(("candidate", args[1]))
        return select_certificate_routed_frame(*args)

    def probe(*args):
        events.append(("probe", args[1]))
        return select_resumable_probe_frame(*args)

    def longer(*args):
        events.append(("long", args[1]))
        return select_strict_score_proof_canonical_completion_frame(*args)

    def no_oracle(*args, **kwargs):
        raise AssertionError("complete full references must not invoke conditional oracles")

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, probe, longer, no_oracle, False)
    queues = [tuple(state["queue"]) for state in experiment.state_roster(protocol, 4)]
    assert events[:9] == [(label, queue) for queue in queues for label in ("reference", "probe", "candidate")]
    assert [label for label, _ in events[9:15]] == list(experiment.LABELS) * 2
    expected_order = []
    order = protocol["timing_orders"]["synthetic:4"]
    for repeat, positions in enumerate(order["state_orders"]):
        for position in positions:
            expected_order.extend((label, queues[position]) for label in order["label_orders"][position][repeat])
    assert events[15:] == expected_order
    assert len(events) == 3 * 3 + 3 * 18 + 6
    for state in row["states"]:
        expected = runtime.reference_key(experiment.selection_from_record(state["reference"]), tuple(state["queue"]), len(model.options))
        for label in experiment.LABELS:
            _, errors, quality = audit_result(state["first_results"][label], state, protocol, model.snapshot(), 4, label, expected)
            assert not errors and quality
    assert row["states"][1]["screen_results"]["candidate"]["resumed"]
    for state in row["states"]:
        assert state["expected_initial_lp"]["selection"] == state["screen_results"]["probe"]["initial_lp"]["selection"]


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
        return select_certificate_routed_frame(*args)

    def probe(*args):
        events.append("selector")
        return select_resumable_probe_frame(*args)

    def control(*args):
        events.append("selector")
        return select_strict_score_proof_canonical_completion_frame(*args)

    monkeypatch.setattr(runtime, "_record", record)
    monkeypatch.setattr(experiment, "asdict", serialize)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, probe, control, progress=False)
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
    assert not row["states"][1]["screen_results"]["candidate"]["selection"]["canonical_exact"]
    assert all(not state["screen_results"]["candidate"]["selection"]["canonical_exact"] for state in row["states"] if state["queue"] == [1, 6])
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
        return select_certificate_routed_frame(*args)

    def probe(*args):
        events.append("probe")
        return select_resumable_probe_frame(*args)

    monkeypatch.setattr(runtime, "select_canonical_count_space_bnb_frame", reference)
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, candidate, probe, select_strict_score_proof_canonical_completion_frame, oracle, False)
    assert events[:12] == ["reference", "oracle", "probe", "candidate"] * 3
    assert all(not state["reference_complete"] and state["effective_reference_complete"] for state in row["states"])
    assert all(state["effective_reference_source"] == "independent_score_face_oracle" for state in row["states"])


def test_queue_design_excludes_history_and_pairs_regressions(monkeypatch):
    monkeypatch.setattr(experiment, "LANES", (4,))
    monkeypatch.setattr(experiment, "PANEL_SEEDS", (2026092791, 2026092792))
    monkeypatch.setattr(experiment, "QUEUES_PER_PANEL", 4)
    monkeypatch.setattr(experiment, "PREVIOUS_UNTIMED_PROBE_QUEUES", {4: ((1, 2, 3),)})
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
    assert [panel["cohort"] for panel in panels[2:]] == ["old_regression", "previously_untimed_probe_regression", "ordinary_tail_regression"]


@pytest.fixture
def registration_case(case, monkeypatch, tmp_path):
    model, template = case
    previous = {**deepcopy(template), "registered_at_utc": "previous", "execution_sources": {"frozen.py": "original"}}
    row = experiment.screen_and_time_cell(model, 4, previous, runtime, progress=False)
    prior = {**deepcopy(previous), "registered_at_utc": "development"}
    strict = {**deepcopy(previous), "registered_at_utc": "strict"}
    paths = {name: tmp_path / name for name in ("prior.json", "previous.json", "prior_protocol.json", "previous_protocol.json", "strict_protocol.json")}
    paths["prior.json"].write_text(json.dumps({"decision": "CERTIFICATE_ROUTED_DEVELOPMENT_ADVANCE", **{key: True for key in (
        "certificate_checks_passed", "effective_references_complete", "quality_gate_passed", "execution_checks_passed", "development_timing_gate_passed")}}))
    paths["previous.json"].write_text(json.dumps({"decision": "RESUMABLE_PROBE_CONFIRMATION_TIMING_HOLD", **{key: True for key in (
        "certificate_checks_passed", "effective_references_complete", "candidate_quality_gate_passed", "comparison_quality_gate_passed", "execution_checks_passed")}}))
    for name, protocol in (("prior_protocol.json", prior), ("previous_protocol.json", previous), ("strict_protocol.json", strict)):
        paths[name].write_text(json.dumps(protocol))
    (tmp_path / "frozen.py").write_text("original")
    with gzip.open(_cell_path(paths["previous.json"], "synthetic", 4), "wt") as target:
        json.dump({"registered_at_utc": "previous", "cell": row}, target)
    score_count = int(not row["states"][1]["screen_results"]["candidate"]["initial_lp"]["selection"]["score_exact"])
    changes = {"ROOT": tmp_path, "PRIOR_OUTPUT": paths["prior.json"], "PRIOR_PROTOCOL": paths["prior_protocol.json"],
        "PREVIOUS_OUTPUT": paths["previous.json"], "PREVIOUS_PROTOCOL": paths["previous_protocol.json"],
        "STRICT_PROTOCOL": paths["strict_protocol.json"], "STRICT_OUTPUT": tmp_path / "strict.json", "EXTRA_SOURCES": (),
        "STATIC_PROTOCOLS": (), "FRONTIER_PROTOCOL": tmp_path / "frontier.json", "LANES": (4,),
        "PANEL_SEEDS": (2026092801,), "QUEUES_PER_PANEL": 2, "TIMING_FRESH_PER_PANEL": 1,
        "INITIAL_LP_BUDGET": 3, "LP_NODE_BUDGET": 64, "INTEGER_STATE_BUDGET": 0,
        "PREVIOUS_UNTIMED_PROBE_QUEUES": {}, "ORDINARY_TAIL_QUEUES": {}, "EXPECTED_TRIGGER_COUNT": 1,
        "EXPECTED_SCORE_UNPROVED_COUNT": score_count}
    for name, value in changes.items():
        monkeypatch.setattr(experiment, name, value)
    monkeypatch.setattr(experiment, "static_exclusions", lambda old: {"4": set()})
    return model, paths


def test_previous_timing_hold_does_not_block_new_quality_valid_registration(registration_case):
    model, _ = registration_case
    protocol = experiment.build_protocol((model,))
    assert protocol["previous_decision"] == "RESUMABLE_PROBE_CONFIRMATION_TIMING_HOLD"
    assert protocol["historical_trigger_keys"] == [{"node": "synthetic", "lanes": 4, "queue": [1, 6]}]
    assert (0, 0) in map(tuple, protocol["excluded_static_queues"]["4"])
    assert (1, 6) in map(tuple, protocol["excluded_static_queues"]["4"])


@pytest.mark.parametrize("change", ("source", "queue", "budget", "timing_order"))
def test_registration_rejects_execution_changes(registration_case, change):
    model, paths = registration_case
    protocol = experiment.build_protocol((model,))
    protocol["registered_at_utc"] = "frozen"
    if change == "source":
        (paths["prior.json"].parent / "frozen.py").write_text("changed")
    elif change == "queue":
        protocol["queue_panels"]["4"][0]["states"][0]["queue"][0] += 1
    elif change == "budget":
        protocol["integer_state_budget"] += 1
    else:
        protocol["timing_orders"]["synthetic:4"]["label_orders"][0][0].reverse()
    with pytest.raises(ValueError):
        experiment.verify_protocol(protocol, (model,))


def test_failed_prior_quality_rejects_registration(registration_case):
    model, paths = registration_case
    result = json.loads(paths["previous.json"].read_text())
    result["candidate_quality_gate_passed"] = False
    paths["previous.json"].write_text(json.dumps(result))
    with pytest.raises(ValueError, match="quality-valid"):
        experiment.build_protocol((model,))


def test_same_tail_panel_is_extended_without_duplicate_queues(monkeypatch):
    monkeypatch.setattr(experiment, "LANES", (4,))
    monkeypatch.setattr(experiment, "PANEL_SEEDS", ())
    monkeypatch.setattr(experiment, "PREVIOUS_UNTIMED_PROBE_QUEUES", {})
    monkeypatch.setattr(experiment, "ORDINARY_TAIL_QUEUES", {4: ((1, 2, 3), (2, 3, 4))})
    previous = {"4": [{"panel": "ordinary_tail_regression", "cohort": "ordinary_tail_regression", "seed": None,
        "states": [{"state_id": "old:tail", "queue": [1, 2, 3]}]}]}
    panels = experiment.queue_design(previous, {"4": set()})["4"]
    assert len(panels) == 1
    assert [tuple(state["queue"]) for state in panels[0]["states"]] == [(1, 2, 3), (2, 3, 4)]
    assert panels[0]["states"][0]["state_id"] == "old:tail"


def test_regression_flags_are_exact_node_keys_not_live_certificates(case):
    model, protocol = case
    protocol["score_unproved_keys"] = [{"node": "other", "lanes": 4, "queue": [1, 6]}]
    protocol["previous_untimed_probe_keys"] = [{"node": "synthetic", "lanes": 4, "queue": [1, 6]}]
    row = experiment.screen_and_time_cell(model, 4, protocol, runtime, progress=False)
    assert not any(state["score_unproved_regression"] for state in row["states"])
    assert [state["previous_untimed_probe_trigger"] for state in row["states"]] == [False, True, False]
