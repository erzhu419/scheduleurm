"""Confirm the frozen K144 selector on new queues, with separate hard-case regressions."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.aligned_hardware_integer_capacity_large_r import (
    screen_cell, _cell_path, runtime_setup,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_k144_confirmation_20260927.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_budget_frontier_development_20260927.json"
PRIOR_PROTOCOL = PRIOR_OUTPUT.with_name(PRIOR_OUTPUT.stem + "_protocol.json")
STATIC_PROTOCOLS = (
    ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_development_20260927_protocol.json",
    ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927_protocol.json",
)
LANES = (8, 16, 32, 64)
LABELS = ("old_shared", "fractional_shared", "integer_capacity")
BUDGET = 144
REFERENCE_BUDGET = 4096
PANEL_SEEDS = (2026092751, 2026092752, 2026092753, 2026092754)
QUEUES_PER_PANEL = 64
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_integer_capacity_k144_confirmation.py",
    "algorithm/experiments/integer_capacity_k144_confirmation_summary.py",
)


def static_exclusions(previous):
    excluded = {str(lanes): {tuple(state["queue"])
                            for panel in previous["queue_panels"][str(lanes)]
                            for state in panel["states"]} for lanes in LANES}
    for path in STATIC_PROTOCOLS:
        old = json.loads(path.read_text(encoding="utf-8"))
        for lanes in LANES:
            excluded[str(lanes)].update(tuple(queue)
                                       for panel in old["queue_panels"].get(str(lanes), [])
                                       for queue in panel["queues"])
    return excluded


def queue_design(previous_panels, excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    panels = {}
    for lanes in LANES:
        seen = set(excluded[str(lanes)])
        panels[str(lanes)] = []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({
                "panel": f"fresh_{seed}", "seed": seed, "cohort": "fresh",
                "states": [{"state_id": f"fresh:{seed}:{index}", "queue": queue, "cohort": "fresh"}
                           for index, queue in enumerate(queues)],
            })
            seen.update(queues)
        for old in previous_panels[str(lanes)]:
            if old["panel"].startswith("hard_"):
                panel = deepcopy(old)
                panel["cohort"] = "regression"
                for state in panel["states"]:
                    state["cohort"] = "regression"
                panels[str(lanes)].append(panel)
    return panels


def build_protocol(models):
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    required = ("certificate_checks_passed", "references_complete", "high_cap_action_gate_passed")
    chosen = next(curve for curve in prior["curves"] if curve["budget"] == BUDGET)
    if (not all(prior[key] for key in required)
            or chosen["label_rates"]["integer_capacity"]["certified_canonical_action_equal_count"] != prior["queue_count"]):
        raise ValueError("K144 development must close every reference-matching canonical certificate")
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen execution source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    excluded = static_exclusions(previous)
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_k144_fresh_confirmation_v1",
        "models": previous["models"], "lanes": LANES, "labels": LABELS,
        "budget_grid": [BUDGET], "unique_budgets": {str(lanes): [BUDGET] for lanes in LANES},
        "reference_budget": REFERENCE_BUDGET,
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": queue_design(previous["queue_panels"], excluded),
        "retained_witnesses": previous["retained_witnesses"],
        "excluded_static_queues": {lanes: sorted(queues) for lanes, queues in excluded.items()},
        "excluded_protocol_paths": [str(path.relative_to(ROOT)) for path in (*STATIC_PROTOCOLS, PRIOR_PROTOCOL)],
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_result": {key: prior[key] for key in ("decision", *required)},
        "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "sampler": "For each R use random.Random(panel_seed+1000*R), each coordinate randint(0,5R). Reject the three listed static protocol rosters, native/scaled grids and earlier accepted draws at that R. Four panels of64 distinct queues per R, shared across four nodes; no cross-R coupling. This is conditional uniform sampling without replacement, not a claim of no overlap with all historical trajectory states.",
        "regression_roster": "All three hard-case radius-one neighborhood panels from the frozen budget-frontier protocol, at their original R and copied across nodes. Re-execute atK144; report separately from fresh queues and exclude from fresh cost comparisons.",
        "calls": "One cold full K4096 reference and three independent cold K144 selectors per state, serial on one pinned CPU. No candidate cache, warm starts, escalation or reference rescue. Complete all frozen panels even on honest nonclosure or mismatch; preserve a HOLD result without replacement seeds or retuning.",
        "accounting": "References charged separately; LP, matrix, preprocessing and integer-propagation counters retained. Single-call elapsed time is diagnostic only.",
        "main_gate": "All calls valid and references complete. Fresh confirmation requires integer-capacity canonical certification and exact reference score/duration/canonical-index agreement on every fresh state. Hard regressions require the same on their separate roster; overall confirmation requires both. Control nonclosure is reported and does not invalidate a correct candidate.",
        "secondary_cost_gate": "Fresh-only integer-capacity LP and matrix totals no greater than BOTH controls in every node/R cell and strictly smaller pooled against BOTH. Per-state regressions retained. Cost is secondary and cannot rescue a failed correctness confirmation or negate a successful fixed-cap correctness confirmation.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "Only queue panels are new under a frozen sampling law; fixed cached catalogues and virtual identical lanes. Queues are paired across nodes and are not independent node-state observations. Single-cap confirmation does not identify a minimum or universal budget. Prior trajectory/decision-delay utility HOLD results remain unchanged; no new physical scaling, measured latency or flow-time benefit claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered models, sources, queues, cap or confirmation protocol changed")


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
    from algorithm.experiments.integer_capacity_k144_confirmation_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
