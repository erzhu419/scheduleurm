from dataclasses import replace
from fractions import Fraction
from itertools import product
import json

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments import aligned_hardware_shared_threshold_scaling_screen as screen
from algorithm.experiments.aligned_hardware_frame_model import FrameModel, FrameOption


@pytest.fixture
def model():
    return FrameModel(
        node="scaling_test", scenario_id="small_real_count_space", lanes=2,
        workloads=("a", "b", "c"), scales=(1.0, 1.0, 1.0),
        options=(
            FrameOption("a_early", (1, 0, 0), 3.0, ((3.0,), (), ())),
            FrameOption("a_late", (1, 0, 0), 1.0, ((1.0,), (), ())),
            FrameOption("b", (0, 1, 0), 2.0, ((), (2.0,), ())),
        ),
    )


@pytest.fixture
def reference(model):
    queue = (3, 3, 0)
    selected = select_canonical_count_space_bnb_frame(model.options, queue, 2, 4096)
    expected = screen.oracle(screen.enumerate_frames(model.options, 2), queue)
    assert selected.canonical_exact
    return selected, expected


def _inexact_alternative(reference):
    return replace(
        reference, option_counts=(0, 0, 1), option_indices=(2, None),
        service=(0, 1, 0), duration=2.0, score=1.5, upper_bound=6.75,
        gap=5.25, epsilon=5.25 / 6, exact=False, score_exact=False,
        canonical_exact=False, lp_node_budget=3, lp_calls=3,
    )


@pytest.mark.parametrize("lanes", (4, 8, 16))
def test_queue_panel_scales_every_coordinate_without_changing_membership(lanes):
    expected = tuple(tuple(value * (lanes // 2) for value in queue)
                     for queue in product((0, 1, 3, 4, 10), repeat=3))
    assert screen.scaled_queues(lanes) == expected
    assert len(expected) == 125


def test_checkpoint_accepts_feasible_inexact_choice_with_covering_bound(model, reference):
    selected, expected = reference
    candidate = _inexact_alternative(selected)
    checks = screen.check_selection(candidate, model.options, (3, 3, 0), 2, 3, expected)
    assert candidate.option_indices != selected.option_indices
    assert not any(checks.values()), checks


@pytest.mark.parametrize("change", (
    {"score_exact": True},
    {"score_exact": True, "canonical_exact": True, "exact": True},
    {"upper_bound": 5.0, "gap": 3.5, "epsilon": 3.5 / 6},
))
def test_checkpoint_rejects_false_certificate_or_uncovered_optimum(model, reference, change):
    selected, expected = reference
    candidate = replace(_inexact_alternative(selected), **change)
    assert any(screen.check_selection(candidate, model.options, (3, 3, 0), 2, 3, expected).values())


def test_equal_score_wrong_action_is_allowed_until_canonical_exact_is_claimed(model):
    # Equal-duration aliases have the same service score, but only index zero
    # is the full catalogue's canonical choice.
    options = (replace(model.options[1], action_id="first"),
               replace(model.options[1], action_id="second"))
    queue = (3, 0, 0)
    selected = select_canonical_count_space_bnb_frame(options, queue, 2, 4096)
    expected = screen.oracle(screen.enumerate_frames(options, 2), queue)
    alternate = replace(selected, option_counts=(0, 2), option_indices=(1, 1),
                        canonical_exact=False, exact=False)
    assert not any(screen.check_selection(alternate, options, queue, 2, 4096, expected).values())
    falsely_certified = replace(alternate, canonical_exact=True, exact=True)
    assert any(screen.check_selection(falsely_certified, options, queue, 2, 4096, expected).values())


@pytest.mark.parametrize("flag", ("exact", "score_exact", "canonical_exact"))
def test_large_budget_alone_does_not_turn_reference_into_oracle(reference, flag):
    selected, expected = reference
    assert screen.reference_key(selected, (3, 3, 0), len(selected.option_counts)) == expected
    incomplete = replace(selected, **{flag: False})
    assert incomplete.lp_node_budget == 4096
    assert screen.reference_key(incomplete, (3, 3, 0), len(selected.option_counts)) is None


@pytest.mark.parametrize("queue", ((0, 0, 0), (2, 6, 0), (6, 6, 0), (20, 2, 0)))
def test_r4_reference_matches_independent_multiset_oracle(model, queue):
    selected = select_canonical_count_space_bnb_frame(model.options, queue, 4, 4096)
    expected = screen.oracle(screen.enumerate_frames(model.options, 4), queue)
    assert screen.reference_key(selected, queue, len(model.options)) == expected
    assert screen.selection_score(selected, queue) == -expected[0]
    assert not any(screen.check_selection(selected, model.options, queue, 4, 4096, expected).values())


def test_r4_oracle_catches_false_large_budget_reference(model):
    queue = (3, 3, 0)
    selected = select_canonical_count_space_bnb_frame(model.options, queue, 4, 4096)
    expected = screen.oracle(screen.enumerate_frames(model.options, 4), queue)
    fabricated = replace(selected, option_counts=(0, 0, 1), option_indices=(2, None, None, None),
                         service=(0, 1, 0), duration=2.0, score=1.5, upper_bound=1.5,
                         gap=0.0, epsilon=0.0)
    assert screen.reference_key(fabricated, queue, len(model.options)) != expected
    assert any(screen.check_selection(fabricated, model.options, queue, 4, 4096, expected).values())


def test_protocol_is_json_roundtrip_stable_and_records_scaled_panel(model):
    protocol = screen.build_protocol((model,))
    assert json.loads(json.dumps(protocol)) == protocol
    assert protocol["models"] == [json.loads(json.dumps(model.snapshot()))]


def test_invalid_checkpoint_action_is_rejected_without_an_exact_reference(model, reference):
    selected, _ = reference
    overdrawn = replace(selected, service=(4, 0, 0), option_counts=(0, 4, 0),
                        option_indices=(1, 1), duration=1.0, score=12.0,
                        upper_bound=12.0, gap=0.0, epsilon=0.0)
    assert any(screen.check_selection(overdrawn, model.options, (3, 3, 0), 2, 4096).values())


def test_exact_fraction_score_is_reconstructed_from_action_service(reference):
    selected, _ = reference
    assert screen.selection_score(selected, (3, 3, 0)) == Fraction(6)


def test_uncertified_upper_bound_must_not_round_below_exact_optimum():
    options = (FrameOption("one_third", (1,), 3.0, ((3.0,),)),)
    queue = (1,)
    selected = select_canonical_count_space_bnb_frame(options, queue, 1, 4096)
    expected = screen.oracle(screen.enumerate_frames(options, 1), queue)
    assert -expected[0] == Fraction(1, 3)
    assert Fraction.from_float(selected.upper_bound) < Fraction(1, 3)
    # Exact certificates publish the nearest float but preserve the exact
    # score through the selected integer service and binary-float duration.
    assert not any(screen.check_selection(selected, options, queue, 1, 4096, expected).values())
    uncertified = replace(selected, exact=False, score_exact=False, canonical_exact=False)
    assert any(screen.check_selection(uncertified, options, queue, 1, 4096, expected).values())


def test_uncertified_reference_is_retained_without_false_checkpoint_mismatches(model, monkeypatch):
    original = screen.select_canonical_count_space_bnb_frame

    def incomplete_reference(options, queue, lanes, budget):
        result = original(options, queue, lanes, budget)
        if budget != 4096:
            return result
        return replace(result, option_counts=(0, 0, 4),
                       option_indices=(2, 2, 2, 2, None, None, None, None),
                       service=(0, 4, 0), duration=2.0, score=8.0,
                       upper_bound=16.0, gap=8.0, epsilon=8.0 / 12,
                       exact=False, score_exact=False, canonical_exact=False)

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", incomplete_reference)
    protocol = screen.build_protocol((model,))
    protocol.update(lanes=[8], base_queue_levels=[1], registered_at_utc="test")
    row = screen.screen_cell(model, 8, protocol, progress=False)
    assert row["uncertified_reference_count"] == 1
    assert row["reference"]["failures"] == 0
    for budget in row["budgets"].values():
        for label in ("full", "shared"):
            metrics = budget[label]
            assert metrics["failures"] == 0
            assert metrics["reference_comparable_count"] == 0
            assert metrics["score_mismatch_count"] == 0
            assert metrics["canonical_action_mismatch_count"] == 0
    summary = screen.summarize((model,), protocol, [row])
    assert summary["decision"] == "INCOMPLETE_REFERENCE"
    assert not summary["references_complete"]
    assert not summary["deployment_action_gate_passed"]


def test_protocol_drift_stops_before_any_selector_call(model, tmp_path, monkeypatch):
    output = tmp_path / "scaling.json"
    monkeypatch.setattr(screen, "load_catalogues", lambda: (model,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    protocol_path = tmp_path / "scaling_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["models"][0]["options"][0]["duration"] += 1
    protocol_path.write_text(json.dumps(protocol))

    def unexpected_search(*args, **kwargs):
        raise AssertionError("drifted protocol reached a selector")

    monkeypatch.setattr(screen, "select_canonical_count_space_bnb_frame", unexpected_search)
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    with pytest.raises(ValueError, match="registered models or protocol changed"):
        screen.main()
    assert not output.exists()


def test_cli_accepts_external_output_directory(model, tmp_path, monkeypatch):
    output = tmp_path / "scaling.json"
    monkeypatch.setattr(screen, "load_catalogues", lambda: (model,))
    monkeypatch.setattr(screen, "LANES", (4,))
    monkeypatch.setattr(screen, "QUEUE_LEVELS", (0,))
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output), "--register"])
    screen.main()
    monkeypatch.setattr("sys.argv", ["screen", "--output", str(output)])
    screen.main()
    result = json.loads(output.read_text())
    assert result["queue_count"] == 1
    assert result["selector_calls"] == 5
    assert result["oracle_queue_count"] == 1
    assert result["certificate_checks_passed"]
    assert result["references_complete"]
    assert result["protocol_path"] == str(tmp_path / "scaling_protocol.json")
