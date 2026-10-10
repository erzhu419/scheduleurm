"""Exercise short-prefix routing with fake snapshots and no LP or integer work."""
from dataclasses import asdict, dataclass, replace

import pytest

import algorithm.short_lookahead_certificate_routed_frame_candidates as routed
from algorithm.root_pruned_integer_capacity_frame_candidates import RootPruningWork


@dataclass(frozen=True)
class Selection:
    lp_node_budget: int
    lp_calls: int
    score_exact: bool
    canonical_exact: bool
    frontier_regions_remaining: int
    inherited_unsolved_regions: int


@dataclass(frozen=True)
class Stage:
    selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]
    root_pruning: RootPruningWork


@dataclass(frozen=True)
class Probe:
    selection: Selection
    baseline_selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]


def stage(cap, *, complete=False, score_exact=True, frontier=5, unsolved=None):
    if complete:
        frontier, unsolved = 0, 0
    if unsolved is None:
        unsolved = frontier
    return Stage(Selection(cap, cap - 4, score_exact, complete, frontier, unsolved),
                 ("preprocessed_once",), ("cumulative_capacity_work",),
                 RootPruningWork(8, 4, 2, 2, 1))


def install(monkeypatch, snapshots, *, probe_completes=()):
    constructions, caps, probes = [], [], []
    outcomes = iter(probe_completes)

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            assert cap in snapshots, f"unexpected LP cap {cap}"
            return snapshots[cap]

    def probe(options, queue, lanes, source, budget, idle_duration):
        assert source.selection.score_exact, "strict score proof is never permitted"
        probes.append((options, queue, lanes, source, budget, idle_duration))
        complete = next(outcomes)
        return Probe(replace(source.selection, canonical_exact=complete), source.selection,
                     source.preprocessing, source.capacity_propagation)

    monkeypatch.setattr(routed, "BranchReservedIntegerCapacitySearch", Search)
    monkeypatch.setattr(routed, "_probe", probe)
    return constructions, caps, probes


def select(*, initial=144, total=256, integer=8192):
    return routed.select_short_lookahead_certificate_routed_frame(
        ("catalogue_option",), (11, 3), 8, initial, total, integer)


def test_five_unsolved_regions_close_in_five_steps_without_a_probe(monkeypatch):
    snapshots = {144: stage(144)}
    snapshots.update({cap: stage(cap, frontier=149 - cap) for cap in range(145, 149)})
    snapshots[149] = stage(149, complete=True)
    constructions, caps, probes = install(monkeypatch, snapshots)
    result = select()
    assert len(constructions) == 1
    assert constructions[0] == ((("catalogue_option",), (11, 3), 8), {"idle_duration": 1.0})
    assert caps == [144, 145, 146, 147, 148, 149] and not probes
    assert result.status == "LP_LOOKAHEAD_COMPLETE" and result.resumed
    assert result.initial_lp is snapshots[144] and result.final_lp is snapshots[149]
    assert result.selection is snapshots[149].selection
    assert result.lookahead == tuple(snapshots[cap] for cap in range(145, 150))
    assert result.final_lp.selection.lp_node_budget == 149
    assert result.lp_node_budget == 256 and result.probe_stage == "none"
    assert result.final_lp.selection.lp_calls - result.initial_lp.selection.lp_calls == 5
    assert (result.final_lp.selection.lp_calls
            + result.final_lp.root_pruning.roots_seen - result.final_lp.root_pruning.roots_solved) == 149


def test_mixed_frontier_keeps_the_original_initial_probe_route(monkeypatch):
    initial = stage(144, frontier=3, unsolved=2)
    constructions, caps, probes = install(monkeypatch, {144: initial}, probe_completes=(True,))
    result = select()
    assert len(constructions) == 1 and caps == [144]
    assert result.lookahead == () and not result.resumed
    assert result.status == "INTEGER_PROBE_COMPLETE" and result.probe_stage == "initial"
    assert probes[0][3:] == (initial, 8192, 1.0)
    assert result.initial_lp is result.final_lp is initial


def test_retained_solved_region_stops_the_prefix_and_probes_that_snapshot(monkeypatch):
    initial = stage(144, frontier=5)
    prefix = stage(145, frontier=5, unsolved=4)
    constructions, caps, probes = install(monkeypatch, {144: initial, 145: prefix},
                                          probe_completes=(True,))
    result = select()
    assert len(constructions) == 1 and caps == [144, 145]
    assert result.lookahead == (prefix,) and result.final_lp is prefix
    assert result.probe_stage == "lookahead" and result.resumed
    assert result.status == "INTEGER_PROBE_COMPLETE"
    assert probes[0][3:] == (prefix, 8192, 1.0)
    assert result.probe.baseline_selection is prefix.selection
    assert result.probe.preprocessing is prefix.preprocessing
    assert result.probe.capacity_propagation is prefix.capacity_propagation


def test_unproved_initial_score_goes_directly_to_full_lp_cap(monkeypatch):
    initial = stage(144, score_exact=False)
    final = stage(256, complete=True)
    constructions, caps, probes = install(monkeypatch, {144: initial, 256: final})
    result = select()
    assert len(constructions) == 1 and caps == [144, 256] and not probes
    assert result.lookahead == () and result.final_lp is final
    assert result.status == "LP_RESUME_COMPLETE" and result.resumed
    assert result.probe_stage == "none"


def test_unproved_initial_score_probes_only_the_certified_full_snapshot(monkeypatch):
    initial = stage(144, score_exact=False)
    final = stage(256, frontier=3, unsolved=2)
    constructions, caps, probes = install(monkeypatch, {144: initial, 256: final},
                                          probe_completes=(True,))
    result = select()
    assert len(constructions) == 1 and caps == [144, 256]
    assert result.lookahead == () and result.final_lp is final
    assert result.probe_stage == "resumed" and result.status == "INTEGER_PROBE_COMPLETE"
    assert probes[0][3:] == (final, 8192, 1.0)


def test_exhausted_lookahead_probe_is_not_repeated_after_full_continuation(monkeypatch):
    initial = stage(144, frontier=2)
    prefix = stage(145, frontier=2, unsolved=1)
    final = stage(256, frontier=2, unsolved=1)
    constructions, caps, probes = install(monkeypatch,
        {144: initial, 145: prefix, 256: final}, probe_completes=(False,))
    result = select()
    assert len(constructions) == 1 and caps == [144, 145, 256]
    assert len(probes) == 1 and probes[0][3:] == (prefix, 8192, 1.0)
    assert result.lookahead == (prefix,) and result.final_lp is final
    assert result.selection is final.selection
    assert result.status == "LP_RESUME_INCOMPLETE" and result.probe_stage == "lookahead"
    assert result.resumed and result.integer_state_budget == 8192
    # Prefix snapshots preserve their original cumulative work after resume.
    assert tuple(s.selection.lp_calls for s in result.lookahead) == (141,)
    assert asdict(result)["final_lp"]["selection"]["lp_calls"] == 252


def test_prefix_stays_within_total_cap_without_repeating_that_cap(monkeypatch):
    snapshots = {144: stage(144), 145: stage(145, frontier=4), 146: stage(146, frontier=3)}
    constructions, caps, probes = install(monkeypatch, snapshots, probe_completes=(False,))
    result = select(total=146)
    assert len(constructions) == 1 and caps == [144, 145, 146]
    assert len(probes) == 1 and probes[0][3] is snapshots[146]
    assert result.lookahead == (snapshots[145], snapshots[146])
    assert result.final_lp is snapshots[146] and result.lp_node_budget == 146
    assert result.status == "LP_RESUME_INCOMPLETE" and result.resumed


def test_canonical_initial_snapshot_skips_every_later_stage(monkeypatch):
    initial = stage(144, complete=True)
    constructions, caps, probes = install(monkeypatch, {144: initial})
    result = select()
    assert len(constructions) == 1 and caps == [144] and not probes
    assert result.lookahead == () and result.probe is None
    assert result.initial_lp is result.final_lp is initial
    assert result.status == "LP_INITIAL_COMPLETE" and not result.resumed
    assert result.probe_stage == "none"


def test_equal_caps_do_not_claim_a_resume_or_repeat_the_initial_probe(monkeypatch):
    initial = stage(144, frontier=5)
    constructions, caps, probes = install(monkeypatch, {144: initial}, probe_completes=(False,))
    result = select(total=144)
    assert len(constructions) == 1 and caps == [144] and len(probes) == 1
    assert result.lookahead == () and result.final_lp is initial
    assert not result.resumed and result.probe_stage == "initial"
    assert result.status == "LP_RESUME_INCOMPLETE"


@pytest.mark.parametrize("budgets", ((-1, 256, 8192), (145, 144, 8192),
                                     (144, 256, -1), (144.5, 256, 8192)))
def test_invalid_budgets_fail_before_search_construction(monkeypatch, budgets):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid budgets must not initialize a search")

    monkeypatch.setattr(routed, "BranchReservedIntegerCapacitySearch", unexpected)
    with pytest.raises(ValueError):
        routed.select_short_lookahead_certificate_routed_frame(
            ("catalogue_option",), (11, 3), 8, *budgets)
