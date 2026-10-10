"""Post-outcome audit repair on three saved real records, without new searches."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from algorithm.experiments import settled_root_dual_face_checked_development_summary as checked
from algorithm.experiments import settled_root_dual_face_development_summary as frozen
from algorithm.experiments.score_face_completion_confirmation_summary import _reference


@pytest.fixture(scope="module")
def saved_records():
    base = Path(__file__).resolve().parents[2] / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928"
    protocol = json.loads(Path(str(base) + "_protocol.json").read_text())
    targets = {8: {(30, 4, 12)}, 64: {(212, 13, 75), (293, 28, 101)}}
    records = {}
    for lanes, queues in targets.items():
        with gzip.open(str(base) + f"_cell_gpu_a_r{lanes}.json.gz", "rt") as stream:
            row = json.load(stream)["cell"]
        for state in row["states"]:
            if tuple(state["queue"]) in queues:
                records[lanes, tuple(state["queue"])] = state
    model = next(model for model in protocol["models"] if model["node"] == "gpu_a")
    return protocol, model, records


@pytest.mark.parametrize("lanes,queue", ((8, (30, 4, 12)), (64, (212, 13, 75)), (64, (293, 28, 101))))
def test_saved_real_lazy_prunes_are_not_charged_twice_and_fee_mutations_are_rejected(saved_records, lanes, queue):
    protocol, model, records = saved_records
    state = records[lanes, queue]
    result, phase = state["routed"], state["routed"]["root_settle_work"]
    expected, reference_errors = _reference(state, protocol, len(model["options"]))
    assert not reference_errors
    _, old_errors, old_quality = frozen._audit_settled_root_result(result, state, protocol, model, lanes, "candidate", expected)
    assert old_quality and old_errors == ["settlement head, free pop/split/singleton and stopping diagnostics differ from native heap progress"]
    _, errors, quality = checked._audit_settled_root_result(result, state, protocol, model, lanes, "candidate", expected)
    assert not errors and quality
    before, after = result["root_lp"]["selection"], result["settled_lp"]["selection"]
    assert before["bound_pruned_regions"] == 7 and after["bound_pruned_regions"] == 9
    assert phase["frontier_regions_before"] - before["frontier_regions_remaining"] == 3
    assert phase["frontier_regions_after"] - after["frontier_regions_remaining"] == 0
    assert phase["pruned_regions"] == 3
    cumulative_delta = (after["bound_pruned_regions"] - phase["frontier_regions_after"] + after["frontier_regions_remaining"]
        - before["bound_pruned_regions"] + phase["frontier_regions_before"] - before["frontier_regions_remaining"])
    assert cumulative_delta == 5
    work = checked._work([result], "candidate")
    assert work["search_work"]["lp_calls"] == after["lp_calls"] == before["lp_calls"]
    assert work["root_settle_work"]["snapshot_heap_entries_read"] == 0
    assert work["root_settle_diagnostics"]["pruned_regions"] == 3
    mutated = deepcopy(result)
    mutated["root_settle_work"]["snapshot_heap_entries_read"] += 2
    _, errors, _ = checked._audit_settled_root_result(mutated, state, protocol, model, lanes, "candidate", expected)
    assert errors
