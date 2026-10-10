"""Characterize fixed-selector budget frontiers on reused queues and old hard cases."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from itertools import product
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_integer_capacity_large_r import (
    screen_cell, _cell_path, runtime_setup,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_budget_frontier_development_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_large_r_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
LANES = (8, 16, 32, 64)
LABELS = ("old_shared", "fractional_shared", "integer_capacity")
BUDGET_GRID = (9, 18, 36, 72, 144, 288, 576)
REFERENCE_BUDGET = 4096
RETAINED_WITNESSES = (
    {"name": "old_r8_6_6_37", "node": "gpu_a", "lanes": 8, "queue": (6, 6, 37),
     "origin": "integer_capacity_development", "prior_integer_lp_calls": 90},
    {"name": "old_r16_35_32_77", "node": "gpu_b", "lanes": 16, "queue": (35, 32, 77),
     "origin": "integer_capacity_development", "prior_integer_lp_calls": 97},
    {"name": "trajectory_r8_9_12_39", "node": "gpu_a", "lanes": 8, "queue": (9, 12, 39),
     "origin": "trajectory_control_action_mismatch", "prior_integer_lp_calls": 30},
)
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_integer_capacity_budget_frontier.py",
    "algorithm/experiments/integer_capacity_budget_frontier_summary.py",
)


def queue_design(previous_panels):
    """Retain prior states, then add deduplicated radius-one hard-case neighborhoods."""
    panels = {str(lanes): deepcopy(previous_panels.get(str(lanes), [])) for lanes in LANES}
    seen = {lanes: {tuple(state["queue"]) for panel in panels[str(lanes)]
                    for state in panel["states"]} for lanes in LANES}
    for witness in RETAINED_WITNESSES:
        lanes = witness["lanes"]
        states = []
        for offsets in product((-1, 0, 1), repeat=3):
            queue = tuple(q + offset for q, offset in zip(witness["queue"], offsets, strict=True))
            if queue in seen[lanes]:
                continue
            states.append({"state_id": f"hard:{witness['name']}:{','.join(map(str, offsets))}",
                           "queue": queue, "anchor": witness["name"], "offsets": offsets})
            seen[lanes].add(queue)
        panels[str(lanes)].append({"panel": f"hard_{witness['name']}", "states": states})
    return panels


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    required = ("certificate_checks_passed", "references_complete",
                "scaled_action_gate_passed", "scaled_cost_gate_passed")
    if not all(prior[key] for key in required):
        raise ValueError("large-R development must pass before characterizing its lower-budget frontier")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen execution source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_budget_frontier_development_v1",
        "models": previous["models"], "lanes": LANES, "labels": LABELS,
        "queue_panels": queue_design(previous["queue_panels"]),
        "retained_witnesses": RETAINED_WITNESSES,
        "budget_grid": BUDGET_GRID,
        "unique_budgets": {str(lanes): BUDGET_GRID for lanes in LANES},
        "reference_budget": REFERENCE_BUDGET,
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_result": {key: prior[key] for key in ("decision", *required)},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "prior_protocol_path": str(PRIOR_PROTOCOL.relative_to(ROOT)),
        "sampler": "Reuse all 76 registered large-R queues at each of R16/R32/R64. Add Cartesian radius-one neighborhoods (27 queues each) of three retained hard cases at their original R; deduplicate against earlier queues within R. Evaluate the identical roster on all four fixed catalogues; keep old panels and each neighborhood separate. No new random draws or independent confirmation.",
        "calls": "One cold full K4096 reference per node/R/Q, then three independent cold selectors at every cap in the registered grid. No candidate caching, warm starts, escalation or reference rescue. All cells serial on one pinned CPU using the frozen runtime setup.",
        "accounting": "Count every actual reference and selector call; reference cost is separate. LP, matrix, preprocessing and integer-propagation work retained at each cap. Single-call elapsed time is diagnostic only.",
        "reference_policy": "Only a complete full-reference canonical certificate makes an action comparable. Self-certification never rescues an incomplete reference. Observed canonical equality requires score plus duration and canonical option indices; score equality and internal canonical closure remain separate.",
        "boundary_definition": "For each selector and condition, the smallest TESTED cap for which all registered states pass at that cap AND every tested higher cap. Report separately observed canonical agreement and certified canonical closure. Null when no suffix passes or execution/reference is incomplete. This is a sampled grid boundary, not the minimum LP budget or a universal bound.",
        "gap_definition": "Per-state certificate gap divided by the reported upper bound when positive, otherwise zero. Report zero-upper/zero-score counts and zero-gap canonical nonclosure separately; zero score gap does not establish canonical optimality.",
        "main_gate": "Characterization requires complete coverage, valid calls, complete references, and integer-capacity certified canonical reference agreement at the largest cap. Low-cap nonclosure and regressions are retained as results. No strict low-cap cost-advancement gate.",
        "search_counters": previous["search_counters"],
        "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Targeted computational development on reused queues and hard-case neighborhoods; not independent validation. Four fixed cached physical catalogues with virtual identical lanes; no additional physical scaling or measured latency evidence. Prior trajectory and decision-delay utility HOLD results remain unchanged; this screen establishes no flow-time benefit.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered models, sources, queues, caps or frontier protocol changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, models)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.run:
        for node, lanes in protocol["execution_order"]:
            if _cell_path(args.output, node, lanes).exists():
                raise FileExistsError(_cell_path(args.output, node, lanes))
    rows = []
    for node, lanes in protocol["execution_order"]:
        path = _cell_path(args.output, node, lanes)
        if args.merge_cells:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                saved = json.load(source)
            if saved["registered_at_utc"] != protocol["registered_at_utc"]:
                raise ValueError("cell registration differs from protocol")
            row = saved["cell"]
        else:
            model = next(model for model in models if model.node == node)
            row = screen_cell(model, lanes, protocol, runtime)
            with gzip.open(path, "xt", encoding="utf-8") as target:
                json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row},
                          target, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.integer_capacity_budget_frontier_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
