"""Fixed-selector development screen at virtual R16/R32/R64 and two caps."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_large_r_development_20260927.json"
PRIOR_PROTOCOL = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_timing_20260927_protocol.json"
PRIOR_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_decision_delay_development_20260927.json"
LANES = (16, 32, 64)
LABELS = ("old_shared", "fractional_shared", "integer_capacity")
PANEL_SEEDS = (2026092721, 2026092722, 2026092723, 2026092724)
QUEUES_PER_PANEL = 16
REFERENCE_BUDGET = 4096
EXTRA_SOURCES = (
    "algorithm/experiments/aligned_hardware_integer_capacity_large_r.py",
    "algorithm/experiments/integer_capacity_large_r_summary.py",
)


def boundary_queues(lanes):
    return ((0, 0, 0), (1, 1, 1),
            (lanes - 1, lanes, lanes + 1), (lanes, lanes + 1, lanes - 1),
            (lanes + 1, lanes - 1, lanes),
            (3 * lanes - 1, lanes, 1), (1, 3 * lanes - 1, lanes), (lanes, 1, 3 * lanes - 1),
            (3 * lanes + 1, lanes, 1), (1, 3 * lanes + 1, lanes), (lanes, 1, 3 * lanes + 1),
            (5 * lanes, 5 * lanes, 5 * lanes))


def queue_design():
    """Couple queue density across scales without forcing even counts."""
    panels, seen = {}, {}
    for lanes in LANES:
        boundary = boundary_queues(lanes)
        seen[lanes] = set(boundary)
        panels[str(lanes)] = [{"panel": "boundary", "seed": None, "states": [
            {"state_id": f"edge:{index}", "queue": queue} for index, queue in enumerate(boundary)]}]
    families = []
    for seed in PANEL_SEEDS:
        rng = random.Random(seed)
        selected = {lanes: [] for lanes in LANES}
        while len(selected[16]) < QUEUES_PER_PANEL:
            draws = [(rng.randrange(80), rng.getrandbits(1), rng.getrandbits(1)) for _ in range(3)]
            queues = {16: tuple(b for b, _, _ in draws),
                      32: tuple(2 * b + u for b, u, _ in draws),
                      64: tuple(4 * b + 2 * u + v for b, u, v in draws)}
            if any(queues[lanes] in seen[lanes] for lanes in LANES):
                continue
            state_id = f"random:{seed}:{len(selected[16])}"
            families.append({"state_id": state_id, "seed": seed, "coordinate_draws": draws})
            for lanes in LANES:
                selected[lanes].append({"state_id": state_id, "queue": queues[lanes]})
                seen[lanes].add(queues[lanes])
        for lanes in LANES:
            panels[str(lanes)].append({"panel": f"random_{seed}", "seed": seed, "states": selected[lanes]})
    return panels, families


def require_prior_integration():
    previous = json.loads(PRIOR_OUTPUT.read_text(encoding="utf-8"))
    if not previous["integration_gate_passed"]:
        raise ValueError("decision-delay integration must pass before larger-R development")
    return {key: previous[key] for key in ("decision", "integration_gate_passed", "utility_gate_passed", "flow_gain_established")}


def build_protocol(models):
    prior = require_prior_integration()
    previous = json.loads(PRIOR_PROTOCOL.read_text(encoding="utf-8"))
    for path, frozen in previous["execution_sources"].items():
        if (ROOT / path).read_text(encoding="utf-8") != frozen:
            raise ValueError(f"frozen selector source changed: {path}")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("frozen catalogue models changed")
    panels, families = queue_design()
    regimes = {"fixed_576": {str(lanes): 576 for lanes in LANES},
               "scaled_36r": {str(lanes): 36 * lanes for lanes in LANES}}
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "integer_capacity_large_r_development_v1",
        "models": previous["models"], "lanes": LANES, "labels": LABELS,
        "panel_seeds": PANEL_SEEDS, "queues_per_panel": QUEUES_PER_PANEL,
        "queue_panels": panels, "queue_families": families,
        "budget_regimes": regimes,
        "unique_budgets": {str(lanes): sorted({caps[str(lanes)] for caps in regimes.values()}) for lanes in LANES},
        "reference_budget": REFERENCE_BUDGET,
        "execution_order": [[model.node, lanes] for model in models for lanes in LANES],
        "prior_result": prior, "prior_result_path": str(PRIOR_OUTPUT.relative_to(ROOT)),
        "sampler": "Per coordinate draw independent b uniform0..79 and fair bits u,v from random.Random(panel_seed). Q16=b,Q32=2b+u,Q64=4b+2u+v; the unconditioned marginal is uniform0..5R-1. Reject the entire family if any scale duplicates its boundary or earlier queue. Four panels of16 families shared across scales and nodes. Twelve boundary states including5R endpoint are registered and reported separately.",
        "calls": "One cold full K4096 reference per node/R/Q, then three independent cold selectors at every distinct cap, fixed576 and36R. AtR16 the equal caps are one actual call per selector and two reporting views. AtR32/64 both caps are executed independently. No candidate caching, warm starts, escalation, or reference rescue.",
        "accounting": "LP, matrix, candidate and preprocessing work per actual call; integer propagation separate. Full-reference work excluded from method comparisons and not paid again for the second cap. Single-call elapsed time is diagnostic only; run serial on one pinned CPU, not a repeated paired timing claim.",
        "reference_policy": "Original full catalogue canonical selector at K4096. Any incomplete reference remains incomplete even when a candidate self-certifies. Internal canonical certificate rate, reference-comparable rate and action agreement remain separate; no exhaustive large-R oracle is claimed.",
        "main_gate": "Complete references and valid certificates for every actual call; at36R integer canonical exact and score/action matching every reference. Integer LP and matrix totals no greater than BOTH controls in each node/R cell, and both strictly smaller pooled. Boundary and random panel results retained separately.",
        "fixed_cap_role": "K576 is a diagnostic budget-scaling control. Honest nonclosure under this cap does not invalidate a successful36R gate; invalid certificates under either cap do. Its exactness, gaps, disagreement, reference coverage and cost remain reported.",
        "search_counters": previous["search_counters"], "capacity_counters": previous["capacity_counters"],
        "execution_sources": {path: (ROOT / path).read_text(encoding="utf-8")
                              for path in (*previous["execution_sources"], *EXTRA_SOURCES)},
        "limitations": "New computational development screen, not independent confirmation. Four fixed cached point catalogues, virtual identical-lane R16/R32/R64; options and physical measurements do not grow. Paired queue families and catalogue states are not independent observations. No physical scaling, latency, trajectory or flow-time benefit claim; prior utility HOLD results remain unchanged.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered models, sources, queues, caps or scaling protocol changed")


def screen_cell(model, lanes, protocol, runtime=None, progress=True):
    if runtime is None:
        from algorithm.experiments import aligned_hardware_integer_capacity_development as runtime
    start = perf_counter()
    selectors = dict(zip(LABELS, (
        runtime.select_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_fractional_shared_threshold_canonical_requirement_dominance_frame,
        runtime.select_integer_capacity_shared_threshold_frame), strict=True))
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "option_count": len(model.options), "states": [], "runtime_start": runtime_provenance()}
    for panel in protocol["queue_panels"][str(lanes)]:
        for state in panel["states"]:
            queue = tuple(state["queue"])
            started = perf_counter()
            reference = runtime.select_canonical_count_space_bnb_frame(
                model.options, queue, lanes, protocol["reference_budget"])
            elapsed = perf_counter() - started
            expected = runtime.reference_key(reference, queue, len(model.options))
            reference_metrics = runtime._metrics()
            runtime._record(reference_metrics, reference, model, queue, lanes,
                            protocol["reference_budget"], expected, elapsed_s=elapsed)
            saved = {**state, "panel": panel["panel"], "reference": asdict(reference),
                     "reference_elapsed_s": elapsed, "reference_complete": expected is not None,
                     "reference_metrics": reference_metrics, "budgets": {}}
            for budget in protocol["unique_budgets"][str(lanes)]:
                results = {}
                for label, selector in selectors.items():
                    started = perf_counter()
                    result = selector(model.options, queue, lanes, budget)
                    elapsed = perf_counter() - started
                    metrics = runtime._metrics()
                    runtime._record(metrics, result.selection, model, queue, lanes, budget,
                                    expected, asdict(result.preprocessing), reference, elapsed)
                    results[label] = {"selection": asdict(result.selection), "metrics": metrics,
                                      "capacity_propagation": asdict(result.capacity_propagation)
                                      if hasattr(result, "capacity_propagation") else {}}
                saved["budgets"][str(budget)] = results
            row["states"].append(saved)
        if progress:
            print(f"{model.node} R{lanes}: {panel['panel']} {len(panel['states'])} states complete", flush=True)
    row["runtime_end"] = runtime_provenance()
    row["operational_elapsed_s"] = perf_counter() - start
    return row


def _cell_path(output, node, lanes):
    return output.with_name(f"{output.stem}_cell_{node}_r{lanes}.json.gz")


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
    from algorithm.experiments.integer_capacity_large_r_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "executed_selector_calls")}))


if __name__ == "__main__":
    main()
