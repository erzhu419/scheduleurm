"""Keep the original certificate routing with one fake dual-reuse frontier."""
from dataclasses import dataclass, replace

import pytest

import algorithm.dual_reuse_certificate_routed_frame_candidates as routed
from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseWork
from algorithm.root_pruned_integer_capacity_frame_candidates import RootPruningWork


@dataclass(frozen=True)
class Selection:
    lp_node_budget: int
    lp_calls: int
    score_exact: bool
    canonical_exact: bool
    frontier_regions_remaining: int = 5
    inherited_unsolved_regions: int = 5


@dataclass(frozen=True)
class Stage:
    selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]
    root_pruning: RootPruningWork
    dual_reuse: DualReuseWork


@dataclass(frozen=True)
class Probe:
    selection: Selection
    baseline_selection: Selection
    preprocessing: tuple[str, ...]
    capacity_propagation: tuple[str, ...]


def stage(cap, *, score_exact=True, complete=False):
    return Stage(Selection(cap, cap - 2, score_exact, complete), ("one_catalogue",),
                 ("cumulative_capacity",), RootPruningWork(4, 2, 1, 1, 0),
                 DualReuseWork(cap - 2, cap, cap // 2, cap // 4, 5 * cap, 4 * cap, 2 * cap))


def install(monkeypatch, initial, final=None, *, probe_complete=False):
    constructions, caps, probes = [], [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            assert cap in (144, 256), "no short-lookahead routing is used"
            assert len(caps) <= 2
            return initial if cap == 144 else final

    def probe(options, queue, lanes, source, budget, idle_duration):
        assert source.selection.score_exact, "integer strict score proof is not permitted"
        probes.append((source, budget, idle_duration))
        return Probe(replace(source.selection, canonical_exact=probe_complete), source.selection,
                     source.preprocessing, source.capacity_propagation)

    monkeypatch.setattr(routed, "DualReuseIntegerCapacitySearch", Search)
    monkeypatch.setattr(routed, "_probe", probe)
    return constructions, caps, probes


def select():
    return routed.select_dual_reuse_certificate_routed_frame(
        ("catalogue_option",), (11, 3), 8, 144, 256, 8192)


@pytest.mark.parametrize("initial_score_exact", (False, True))
def test_exhausted_probe_runs_once_and_charges_only_the_cumulative_final_frontier(
        monkeypatch, initial_score_exact):
    initial, final = stage(144, score_exact=initial_score_exact), stage(256)
    constructions, caps, probes = install(monkeypatch, initial, final)
    result = select()
    assert len(constructions) == 1 and caps == [144, 256]
    assert constructions[0] == ((("catalogue_option",), (11, 3), 8), {"idle_duration": 1.0})
    assert probes == [(initial if initial_score_exact else final, 8192, 1.0)]
    assert result.initial_lp is initial and result.final_lp is final and result.resumed
    assert result.selection is (final.selection if initial_score_exact else result.probe.selection)
    assert result.probe_stage == ("initial" if initial_score_exact else "resumed")
    assert result.status == "LP_RESUME_INCOMPLETE"
    assert result.final_lp.dual_reuse is final.dual_reuse
    assert result.final_lp.selection.lp_calls == 254


def test_initial_canonical_certificate_skips_resume_and_integer_probe(monkeypatch):
    initial = stage(144, complete=True)
    constructions, caps, probes = install(monkeypatch, initial)
    result = select()
    assert len(constructions) == 1 and caps == [144] and not probes
    assert result.initial_lp is result.final_lp is initial
    assert result.selection is initial.selection and not result.resumed
    assert result.status == "LP_INITIAL_COMPLETE" and result.probe_stage == "none"


def test_initial_certified_probe_reuses_metadata_and_can_end_the_search(monkeypatch):
    initial = stage(144)
    constructions, caps, probes = install(monkeypatch, initial, probe_complete=True)
    result = select()
    assert len(constructions) == 1 and caps == [144]
    assert probes == [(initial, 8192, 1.0)]
    assert result.initial_lp is result.final_lp is initial and not result.resumed
    assert result.probe.baseline_selection is initial.selection
    assert result.probe.preprocessing is initial.preprocessing
    assert result.probe.capacity_propagation is initial.capacity_propagation
    assert result.status == "INTEGER_PROBE_COMPLETE" and result.probe_stage == "initial"


def test_unproved_initial_score_goes_directly_to_full_lp_and_can_close(monkeypatch):
    initial, final = stage(144, score_exact=False), stage(256, complete=True)
    constructions, caps, probes = install(monkeypatch, initial, final)
    result = select()
    assert len(constructions) == 1 and caps == [144, 256] and not probes
    assert result.final_lp is final and result.selection is final.selection
    assert result.status == "LP_RESUME_COMPLETE" and result.probe_stage == "none"
    assert result.resumed


@pytest.mark.parametrize("budgets", ((-1, 256, 8192), (257, 256, 8192), (144, 256, -1)))
def test_invalid_budgets_do_not_construct_a_frontier(monkeypatch, budgets):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid budgets must fail before constructing the search")

    monkeypatch.setattr(routed, "DualReuseIntegerCapacitySearch", unexpected)
    with pytest.raises(ValueError):
        routed.select_dual_reuse_certificate_routed_frame(
            ("catalogue_option",), (11, 3), 8, *budgets)
