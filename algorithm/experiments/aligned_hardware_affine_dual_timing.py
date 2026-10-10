"""Repeat the complete frozen three-arm timing with exact affine dual updates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
from statistics import median

from algorithm.experiments import aligned_hardware_residual_capacity_timing as previous
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_timing_20260928.json"
DEVELOPMENT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_affine_dual_development_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_prune_first_dual_timing_20260928.json"
SOURCES = ("algorithm/experiments/aligned_hardware_affine_dual_timing.py",
           "algorithm/experiments/affine_dual_timing_summary.py")
LABELS, REPEATS = previous.LABELS, previous.REPEATS
WARMUPS_PER_LABEL, ORDER_SEED = previous.WARMUPS_PER_LABEL, previous.ORDER_SEED
timing_orders = previous.timing_orders
runtime_setup = previous.runtime_setup


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    old = json.loads(_protocol_path(PREVIOUS_OUTPUT).read_text())
    development = json.loads(_protocol_path(DEVELOPMENT_OUTPUT).read_text())
    old_result = json.loads(PREVIOUS_OUTPUT.read_text())
    dev_result = json.loads(DEVELOPMENT_OUTPUT.read_text())
    if (old_result["decision"] != "PRUNE_FIRST_DUAL_LOCAL_TIMING_HOLD" or not all(
            old_result[key] for key in ("execution_checks_passed", "references_complete",
                                        "quality_gate_passed", "repeat_paths_stable",
                                        "frozen_residual_results_reproduced"))):
        raise ValueError("retain the complete prior prune-first timing HOLD")
    if (dev_result["decision"] != "AFFINE_DUAL_DEVELOPMENT_ADVANCE" or not all(
            dev_result[key] for key in ("development_gate_passed", "behavior_preserved",
                "quality_gate_passed", "work_gate_passed", "references_complete",
                "execution_checks_passed"))):
        raise ValueError("complete affine dual development is required before timing")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    sources = {}
    for source in (old, development):
        if source["models"] != snapshots:
            raise ValueError("cached catalogue models differ from the frozen development")
        if tuple(source[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
                "reference_budget", "oracle_state_budget")) != (144, 256, 8192, 4096, 5_000_000):
            raise ValueError("retain the frozen selector and reference budgets")
        for path, literal in source["execution_sources"].items():
            if (ROOT / path).read_text() != literal:
                raise ValueError(f"frozen source changed: {path}")
            sources[path] = literal
    if tuple(old[key] for key in ("labels", "repeats", "warmups_per_label", "order_seed",
            "long_lp_budget", "control_integer_state_budget", "nonregression_limit",
            "trigger_speedup_limit", "expected_state_count", "expected_timed_calls",
            "expected_warmup_calls", "expected_selector_calls")) != (
                list(LABELS), 6, 2, ORDER_SEED, 4096, 100_000, 1.05, .95, 754, 13572, 96, 13668):
        raise ValueError("retain the complete timing design, gates and call accounting")
    counts = {group: sum(state["timing_group"] == group for cell in old["cells"] for state in cell["states"])
              for group in ("ordinary_development", "historical_regression", "zero_queue",
                            "trigger_control", "new_trigger")}
    if len(old["cells"]) != 16 or counts != dict(ordinary_development=688,
            historical_regression=34, zero_queue=16, trigger_control=8, new_trigger=8):
        raise ValueError("retain the complete 754-state timing roster")
    if [(cell["node"], cell["lanes"]) for cell in development["cells"]] != [
            (cell["node"], cell["lanes"]) for cell in old["cells"]]:
        raise ValueError("development must preserve every original timing cell in order")
    protocol = deepcopy(old)
    for cell, dev_cell in zip(protocol["cells"], development["cells"], strict=True):
        if len(cell["states"]) != len(dev_cell["states"]):
            raise ValueError("development must preserve the complete 754-state timing roster")
        for index, (state, original) in enumerate(zip(cell["states"], dev_cell["states"], strict=True)):
            extra = {key: original[key] for key in ("affine_source_index", "affine_source_state_id")}
            if {key: value for key, value in original.items() if key not in extra} != state:
                raise ValueError("development state metadata differs from the original timing roster")
            state.update(extra, timing_source="affine_development", timing_source_index=index,
                         timing_source_state_id=original["state_id"])
    failure_keys = development["previous_timing_failure_keys"]
    focus_keys = development["affine_focus_keys"]
    registered = {(cell["node"], cell["lanes"], tuple(state["queue"]))
                  for cell in old["cells"] for state in cell["states"]}
    key_set = lambda items: {(item["node"], item["lanes"], tuple(item["queue"])) for item in items}
    if (len(failure_keys) != 4 or len(key_set(failure_keys)) != 4
            or len(focus_keys) != 6 or len(key_set(focus_keys)) != 6
            or not key_set(failure_keys) <= key_set(focus_keys) <= registered):
        raise ValueError("retain all four original and both new timing failures in the complete roster")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol.pop("registered_at_utc", None)
    protocol.update(experiment="affine_dual_equal_quality_timing_v1",
        cpu=min(os.sched_getaffinity(0)), execution_sources=sources,
        previous_timing_failure_keys=deepcopy(failure_keys),
        affine_focus_keys=deepcopy(focus_keys),
        reference_sources={"affine_development": {
            "path": str(DEVELOPMENT_OUTPUT.relative_to(ROOT)),
            "registered_at_utc": development["registered_at_utc"]}},
        prior_decisions={**old["prior_decisions"],
            "prune_first_dual_timing": old_result["decision"],
            "affine_dual_development": dev_result["decision"]},
        comparison="Candidate uses exact affine updates of retained parent dual objectives after the same cheap pruning. Branch remains branch-reserved residual-capacity routing, both144/256/shared8192. Long remains the strict-score-proof selector with LP4096/shared100000. All arms must finish score/canonical proofs and match independent borrowed references. All three clocks are newly measured here; prior HOLD clocks are not controls.",
        roster="The exact prior754 states, groups, controls and six-permutation orders are retained. References and expected complete outputs come from affine development. No new sampling, screening, references or oracle calls.",
        accounting="13572 timed and96 warmup calls, separately reported. Whole selector clocks include affine preparation and updates, native pruning, frontier export, proof DP, retained-vector copies, canonical prefixes/DFS and queries. Real dual RHS/read costs and affine preparation/coefficient/delta costs are reported separately, never replaced by hypothetical full-RHS equivalents. Borrowed references, native snapshots and axis outputs are not new execution. Finish all states after honest nonclosure.",
        limitations="Local descriptive timing reuses the same754 development and conditional-confirmation states, four cached physical catalogues and virtual R8/16/32/64. Paired repeats are dependent. This does not establish independent ordinary-seed speed, physical scaling or flow-time benefit. Five-percent gates remain operational rather than significance tests. Earlier timing HOLD results are retained and the paper remains untouched.")
    return json.loads(json.dumps(protocol))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered sources, roster, budgets or timing design changed")


def _selectors():
    from algorithm.affine_dual_residual_capacity_frame_candidates import select_affine_dual_residual_capacity_frontier_frame
    from algorithm.residual_capacity_frontier_frame_candidates import select_branch_reserved_residual_capacity_frontier_frame
    from algorithm.strict_score_proof_canonical_completion_frame_candidates import select_strict_score_proof_canonical_completion_frame
    return {"candidate": select_affine_dual_residual_capacity_frontier_frame,
            "branch": select_branch_reserved_residual_capacity_frontier_frame,
            "long": select_strict_score_proof_canonical_completion_frame}


def run_cell(model, cell, protocol, runtime, selectors=None, progress=True):
    return previous.run_cell(model, cell, protocol, runtime,
        selectors=_selectors() if selectors is None else selectors, progress=progress)


def summarize(protocol, rows):
    from algorithm.experiments.affine_dual_timing_summary import summarize as summarize_affine
    result = summarize_affine(protocol, rows)
    failed = {(item["node"], item["lanes"], tuple(item["queue"]))
              for item in protocol["previous_timing_failure_keys"]}
    result["previous_timing_failure_states"] = [{
        "node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"], "queue": state["queue"],
        "candidate_median_wall_ms": median(state["wall_ns"]["candidate"]) / 1e6,
        "comparisons": {label: {"wall_ratio": median(state["wall_ns"]["candidate"]) /
                                   median(state["wall_ns"][label])} for label in LABELS[1:]}}
        for row in rows for state in row["states"]
        if (row["node"], row["lanes"], tuple(state["queue"])) in failed]
    focus = {(item["node"], item["lanes"], tuple(item["queue"]))
             for item in protocol["affine_focus_keys"]}
    result["affine_focus_states"] = [{
        "node": row["node"], "lanes": row["lanes"], "state_id": state["state_id"],
        "queue": state["queue"], "timing_group": state["timing_group"],
        "candidate_median_wall_ms": median(state["wall_ns"]["candidate"]) / 1e6,
        "comparisons": {label: {"wall_ratio": median(state["wall_ns"]["candidate"]) /
                                   median(state["wall_ns"][label])} for label in LABELS[1:]}}
        for row in rows for state in row["states"]
        if (row["node"], row["lanes"], tuple(state["queue"])) in focus]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = _protocol_path(args.output)
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "states": protocol["expected_state_count"],
                          "calls": protocol["expected_selector_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if args.output.exists() or any(_cell_path(args.output, cell["node"], cell["lanes"]).exists()
                                  for cell in protocol["cells"]):
        raise FileExistsError("affine dual timing outputs already exist")
    rows = []
    selectors = _selectors()
    for cell in protocol["cells"]:
        model = next(model for model in models if model.node == cell["node"])
        row = run_cell(model, cell, protocol, runtime, selectors=selectors)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, stream,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    result = summarize(protocol, rows)
    result["protocol_path"] = str(protocol_path)
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
