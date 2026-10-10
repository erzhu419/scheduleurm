"""The reserved kernel retains one frontier and one integer-probe budget."""
from dataclasses import dataclass, replace

import pytest

import algorithm.branch_reserved_integer_capacity_frame_candidates as engine
import algorithm.branch_reserved_certificate_routed_frame_candidates as routed


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@dataclass(frozen=True)
class Probe:
    selection: object


@pytest.mark.parametrize("initial_score_exact", (False, True))
def test_registered_budgets_reuse_one_frontier_and_do_not_retry_a_probe(monkeypatch, initial_score_exact):
    options, queue, lanes = (Option((1,), 1.0),), (1,), 1
    raw = engine.BranchReservedIntegerCapacitySearch(options, queue, lanes).advance_to(64)

    def stage(cap, calls, score_exact):
        selected = replace(raw.selection, lp_node_budget=cap, lp_calls=calls,
            exact=False, canonical_exact=False, score_exact=score_exact,
            upper_bound=1.0 if score_exact else 2.0, gap=0.0 if score_exact else 1.0,
            epsilon=0.0 if score_exact else 1.0, frontier_regions_remaining=1,
            canonical_tie_regions_remaining=int(score_exact))
        return replace(raw, selection=selected)

    initial, final = stage(144, 1, initial_score_exact), stage(256, 2, True)
    constructions, caps, probes = [], [], []

    class Search:
        def __init__(self, *args, **kwargs):
            constructions.append((args, kwargs))

        def advance_to(self, cap):
            caps.append(cap)
            assert len(caps) <= 2
            return initial if len(caps) == 1 else final

    def probe(options, queue, lanes, source, budget, idle_duration):
        probes.append((source, budget, idle_duration))
        return Probe(source.selection)

    monkeypatch.setattr(routed, "BranchReservedIntegerCapacitySearch", Search)
    monkeypatch.setattr(routed, "_probe", probe)
    result = routed.select_branch_reserved_certificate_routed_frame(options, queue, lanes, 144, 256, 8192)
    assert len(constructions) == 1 and caps == [144, 256]
    assert constructions[0] == ((options, queue, lanes), {"idle_duration": 1.0})
    assert probes == [(initial if initial_score_exact else final, 8192, 1.0)]
    assert result.probe_stage == ("initial" if initial_score_exact else "resumed")
    assert result.initial_lp is initial and result.final_lp is final and result.resumed
    assert result.selection is final.selection
    assert result.selection.lp_calls == result.final_lp.selection.lp_calls == 2
    assert result.final_lp.root_pruning is final.root_pruning
    assert result.status == "LP_RESUME_INCOMPLETE"
    assert (result.initial_lp_budget, result.lp_node_budget, result.integer_state_budget) == (144, 256, 8192)
