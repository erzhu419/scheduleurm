"""Mock both search arms and retain their ordinary route and shared probe budget."""
from dataclasses import asdict, dataclass, replace

import pytest

import algorithm.late_score_proof_certificate_routed_frame_candidates as routed
import algorithm.resumable_probe_frame_candidates as probe_module
from algorithm.score_face_canonical_completion_frame_candidates import ScoreFaceCompletionWork
from algorithm.strict_score_proof_canonical_completion_frame_candidates import StrictScoreProofWork


@dataclass(frozen=True)
class Selection:
    lp_node_budget: int
    lp_calls: int
    score_exact: bool
    canonical_exact: bool
    score: float
    upper_bound: float
    gap: float
    epsilon: float


@dataclass(frozen=True)
class Stage:
    selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]
    root_pruning: tuple[str, ...]
    dual_reuse: tuple[str, ...]


@dataclass(frozen=True)
class Probe:
    selection: Selection
    baseline_selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]


@pytest.fixture(params=(
    ("select_branch_reserved_late_score_proof_frame", "BranchReservedIntegerCapacitySearch"),
    ("select_dual_reuse_late_score_proof_frame", "DualReuseIntegerCapacitySearch"),
))
def arm(request):
    return request.param


def stage(cap, *, score_exact=False, canonical_exact=False):
    score_exact = score_exact or canonical_exact
    return Stage(Selection(cap, cap - 2, score_exact, canonical_exact, 1.0,
                           1.0 if score_exact else 2.0,
                           0.0 if score_exact else 1.0,
                           0.0 if score_exact else 1.0),
                 ("catalogue_once",), ("cumulative_capacity",),
                 ("reserved_roots",), ("cumulative_dual_work",))


def install_search(monkeypatch, arm, initial, final=None):
    constructions, caps = [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            assert cap in (144, 256) and len(caps) <= 2
            assert final is not None or cap == 144
            return initial if cap == 144 else final

    monkeypatch.setattr(routed, arm[1], Search)
    return constructions, caps


def install_probe(monkeypatch, *, score_exact=True, canonical_exact=False):
    calls = []

    def probe(options, queue, lanes, source, budget, idle_duration):
        calls.append((options, queue, lanes, source, budget, idle_duration))
        return Probe(replace(source.selection, score_exact=score_exact,
                             canonical_exact=canonical_exact), source.selection,
                     source.preprocessing, source.capacity_propagation)

    monkeypatch.setattr(routed, "_probe", probe)
    return calls


def select(arm, budgets=(144, 256, 8192)):
    return getattr(routed, arm[0])(("catalogue_option",), (11, 3), 8, *budgets)


def test_initial_canonical_snapshot_skips_every_probe_and_resume(monkeypatch, arm):
    initial = stage(144, canonical_exact=True)
    constructions, caps = install_search(monkeypatch, arm, initial)
    probes = install_probe(monkeypatch)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144] and not probes
    assert result.initial_lp is result.final_lp is initial
    assert result.selection is initial.selection and not result.resumed
    assert result.status == "LP_INITIAL_COMPLETE" and result.probe_stage == "none"


def test_initial_certified_probe_can_finish_on_the_original_route(monkeypatch, arm):
    initial = stage(144, score_exact=True)
    constructions, caps = install_search(monkeypatch, arm, initial)
    probes = install_probe(monkeypatch, canonical_exact=True)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144] and len(probes) == 1
    assert probes[0][3:] == (initial, 8192, 1.0)
    assert result.status == "INTEGER_PROBE_COMPLETE" and result.probe_stage == "initial"
    assert result.final_lp is initial and not result.resumed


def test_exhausted_initial_probe_is_never_repeated_after_lp_continuation(monkeypatch, arm):
    initial, final = stage(144, score_exact=True), stage(256, score_exact=True)
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    probes = install_probe(monkeypatch)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256] and len(probes) == 1
    assert probes[0][3] is initial
    assert result.selection is final.selection and result.final_lp is final
    assert result.status == "LP_RESUME_INCOMPLETE" and result.probe_stage == "initial"


def test_unproved_initial_score_still_goes_directly_to_lp256_and_can_close(monkeypatch, arm):
    initial, final = stage(144), stage(256, canonical_exact=True)
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    probes = install_probe(monkeypatch)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256] and not probes
    assert result.selection is final.selection and result.final_lp is final
    assert result.status == "LP_RESUME_COMPLETE" and result.probe_stage == "none"


def test_certified_final_score_keeps_the_original_resumed_probe_route(monkeypatch, arm):
    initial, final = stage(144), stage(256, score_exact=True)
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    probes = install_probe(monkeypatch, canonical_exact=True)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256] and len(probes) == 1
    assert probes[0][3:] == (final, 8192, 1.0)
    assert result.status == "INTEGER_PROBE_COMPLETE" and result.probe_stage == "resumed"
    assert result.selection is result.probe.selection


@pytest.mark.parametrize("probe_score_exact,probe_canonical_exact", (
    (True, True), (True, False), (False, False),
))
def test_open_final_score_gets_one_late_probe_without_changing_raw_snapshots(
        monkeypatch, arm, probe_score_exact, probe_canonical_exact):
    initial, final = stage(144), stage(256)
    initial_before, final_before = asdict(initial), asdict(final)
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    probes = install_probe(monkeypatch, score_exact=probe_score_exact,
                           canonical_exact=probe_canonical_exact)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256] and len(probes) == 1
    assert constructions[0] == ((("catalogue_option",), (11, 3), 8), {"idle_duration": 1.0})
    assert probes[0][3:] == (final, 8192, 1.0)
    assert result.initial_lp is initial and result.final_lp is final and result.resumed
    assert result.selection is result.probe.selection
    assert result.probe.baseline_selection is final.selection
    assert result.probe.preprocessing is final.preprocessing
    assert result.probe.capacity_propagation is final.capacity_propagation
    assert result.probe_stage == "late"
    assert result.status == ("LATE_PROBE_COMPLETE" if probe_canonical_exact else "LATE_PROBE_INCOMPLETE")
    assert asdict(initial) == initial_before and asdict(final) == final_before
    assert (result.initial_lp_budget, result.lp_node_budget, result.integer_state_budget) == (144, 256, 8192)


@pytest.mark.parametrize("proof_complete", (False, True))
def test_existing_probe_shares_one_budget_between_strict_proof_and_completion(
        monkeypatch, arm, proof_complete):
    initial, final = stage(144), stage(256)
    constructions, caps = install_search(monkeypatch, arm, initial, final)
    proof_calls, completion_calls = [], []
    proof_states = 700 if proof_complete else 8192

    def prove(options, queue, lanes, source, budget, idle_duration):
        proof_calls.append((source, budget))
        return StrictScoreProofWork("COMPLETE" if proof_complete else "BUDGET_EXHAUSTED",
                                    True, proof_complete, budget, states_visited=proof_states)

    def complete(options, queue, lanes, source, budget, idle_duration):
        assert proof_complete and source.score_exact and budget == 8192 - proof_states
        completion_calls.append((source, budget))
        return replace(source, canonical_exact=True), ScoreFaceCompletionWork(
            "COMPLETE", True, True, budget, states_visited=3)

    monkeypatch.setattr(probe_module, "_prove_no_strict_improvement", prove)
    monkeypatch.setattr(probe_module, "_complete_canonical", complete)
    monkeypatch.setattr(routed, "_probe", probe_module._probe)
    result = select(arm)
    assert len(constructions) == 1 and caps == [144, 256]
    assert proof_calls == [(final.selection, 8192)]
    assert len(completion_calls) == int(proof_complete)
    assert result.probe.score_proof.states_visited == proof_states
    assert result.probe.completion.state_budget == 8192 - proof_states
    assert result.probe.score_proof.states_visited + result.probe.completion.states_visited <= 8192
    assert result.probe.baseline_selection is final.selection
    assert not final.selection.score_exact
    assert result.probe_stage == "late"
    assert result.status == ("LATE_PROBE_COMPLETE" if proof_complete else "LATE_PROBE_INCOMPLETE")


@pytest.mark.parametrize("budgets", ((-1, 256, 8192), (257, 256, 8192), (144, 256, -1)))
def test_invalid_budgets_fail_before_search_construction(monkeypatch, arm, budgets):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid budgets must fail before search construction")

    monkeypatch.setattr(routed, arm[1], unexpected)
    with pytest.raises(ValueError):
        select(arm, budgets)
