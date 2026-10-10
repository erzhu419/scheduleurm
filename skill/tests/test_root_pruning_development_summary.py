"""Gates retain low-budget losses and exclude zero-only work savings."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json

import pytest

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.resumable_integer_capacity_frame_candidates import IntegerCapacitySearch
from algorithm.root_pruned_integer_capacity_frame_candidates import RootPrunedIntegerCapacitySearch
from algorithm.certificate_routed_frame_candidates import select_certificate_routed_frame
from algorithm.root_pruned_certificate_routed_frame_candidates import select_root_pruned_certificate_routed_frame
from algorithm.experiments.root_pruning_development_summary import audit_root_stage, summarize


@dataclass(frozen=True)
class Option:
    requirements: tuple[int, ...]
    duration: float


@pytest.fixture
def records():
    options, caps = (Option((1,), 1.0),), (1, 2, 144, 256)
    states = []
    for queue, source in (((4,), "development"), ((0,), "new_zero")):
        traces = {}
        for label, cls in (("old", IntegerCapacitySearch), ("new", RootPrunedIntegerCapacitySearch)):
            search = cls(options, queue, 2)
            traces[label] = {str(cap): asdict(search.advance_to(cap)) for cap in caps}
        states.append({"state_id": source, "queue": queue, "source": source, "caps": caps,
            "reference": asdict(select_canonical_count_space_bnb_frame(options, queue, 2, 4096)),
            "oracle": None, "effective_reference_complete": True, "traces": traces,
            "old_routed": asdict(select_certificate_routed_frame(options, queue, 2, 144, 256, 8192)),
            "routed": asdict(select_root_pruned_certificate_routed_frame(options, queue, 2, 144, 256, 8192))})
    protocol = {"models": [{"node": "toy", "options": [asdict(option) for option in options]}],
        "cells": [{"node": "toy", "lanes": 2, "states": states}], "registered_at_utc": "fixed",
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "prior_decisions": {"confirmation": "CERTIFICATE_ROUTED_CONFIRMATION_TIMING_HOLD"},
        "limitations": "Development only."}
    rows = [{"node": "toy", "lanes": 2, "registered_at_utc": "fixed", "states": states,
             "new_reference_calls": 1, "new_old_routed_calls": 1}]
    return json.loads(json.dumps(protocol)), json.loads(json.dumps(rows))


def test_zero_queue_savings_cannot_advance_regular_work_gate(records):
    summary = summarize(*records)
    assert summary["execution_checks_passed"] and summary["routed_quality_gate_passed"]
    assert summary["finite_budget_gate_passed"] and not summary["work_gate_passed"]
    assert summary["decision"] == "ROOT_PRUNING_DEVELOPMENT_HOLD"
    assert summary["regular_state_count"] == 1 and summary["state_count"] == 2
    all_work = summary["all_executed_trace_and_routed_work"]
    assert all_work["routed_new"]["search_work"]["lp_calls"] < all_work["routed_old_borrowed_or_zero_control"]["search_work"]["lp_calls"]
    assert summary["nonzero_routed_lp_ratio"] == 1.0
    assert summary["executed_interface_calls"] == 19
    assert summary["native_search_constructions"] == 7
    assert summary["new_reference_calls"] == summary["new_old_routed_control_calls"] == 1
    assert summary["prior_hold_retained"]


def test_valid_low_budget_score_loss_is_preserved_and_blocks_finite_gate(records):
    protocol, rows = records
    selected = rows[0]["states"][0]["traces"]["new"]["1"]["selection"]
    selected.update(option_counts=[0], option_indices=[None, None], service=[0], score=0.,
        exact=False, score_exact=False, canonical_exact=False, upper_bound=8., gap=8., epsilon=2.,
        frontier_regions_remaining=1)
    summary = summarize(protocol, rows)
    assert summary["execution_checks_passed"] and summary["routed_quality_gate_passed"]
    assert not summary["finite_budget_gate_passed"]
    assert len(summary["finite_budget_score_regressions"]) == 1
    loss = summary["finite_budget_score_regressions"][0]
    assert loss["lp_budget"] == 1 and loss["old_score"] == "8" and loss["new_score"] == "0"


def test_bound_below_independent_reference_is_invalid(records):
    protocol, rows = records
    rows[0]["states"][0]["traces"]["new"]["1"]["selection"].update(
        option_counts=[0], option_indices=[None, None], service=[0], score=0.,
        exact=False, score_exact=False, canonical_exact=False, upper_bound=7., gap=7., epsilon=1.75,
        frontier_regions_remaining=1)
    summary = summarize(protocol, rows)
    assert summary["decision"] == "INVALID_EXECUTION"
    assert any("bound_below_optimum" in message for failure in summary["failures"] for message in failure["errors"])


def test_root_partition_and_tie_subset_have_distinct_meanings(records):
    stage = deepcopy(records[1][0]["states"][0]["traces"]["new"]["1"])
    assert audit_root_stage(stage) == []
    stage["root_pruning"]["roots_skipped_canonical_tie"] = 1
    assert "canonical-tie skips must be a subset of bound skips" in audit_root_stage(stage)
    stage["root_pruning"].update(roots_skipped_canonical_tie=0, roots_skipped_bound=1)
    assert "root counters do not partition all visited strata" in audit_root_stage(stage)
