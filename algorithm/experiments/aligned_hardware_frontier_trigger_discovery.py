"""Discover fresh unproved LP frontiers before validating every triggered queue."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_setup, runtime_provenance
from algorithm.experiments.aligned_hardware_integer_capacity_large_r import _cell_path
from algorithm.experiments.aligned_hardware_dual_reuse_development import _caps
from algorithm.experiments.aligned_hardware_frontier_score_proof_confirmation import run_cell

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_trigger_discovery_20260928.json"
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_frontier_score_proof_confirmation_20260928.json"
LANES = 64
PANEL_SEEDS = (2026092851, 2026092852, 2026092853, 2026092854)
QUEUES_PER_PANEL = 1024
BLOCK_SIZE = 64
MAX_ROUNDS = 16
MIN_TRIGGER_QUEUES = 2
MIN_TRIGGER_SEEDS = 2
SOURCES = (
    "algorithm/experiments/aligned_hardware_frontier_trigger_discovery.py",
    "algorithm/experiments/frontier_trigger_discovery_summary.py",
)


def fresh_design(excluded):
    from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues
    seen, panels = set(excluded), []
    for seed in PANEL_SEEDS:
        queues = sample_queues(LANES, seed, QUEUES_PER_PANEL, excluded=seen)
        panels.append({"seed": seed, "panel": f"trigger_{seed}", "states": [
            {"state_id": f"trigger:{seed}:{index}", "queue": queue, "seed": seed,
             "index": index, "panel": f"trigger_{seed}", "cohort": "conditional_trigger_validation"}
            for index, queue in enumerate(queues)]})
        seen.update(queues)
    return panels


def build_protocol(models):
    previous = json.loads(PREVIOUS_OUTPUT.read_text())
    previous_path = PREVIOUS_OUTPUT.with_name(PREVIOUS_OUTPUT.stem + "_protocol.json")
    old = json.loads(previous_path.read_text())
    if previous["decision"] != "FRONTIER_SCORE_PROOF_FRESH_CONFIRMATION_ADVANCE" or not all(previous[key] for key in (
            "execution_checks_passed", "effective_references_complete", "confirmation_gate_passed")):
        raise ValueError("the completed independent frontier confirmation advance is required")
    snapshots = json.loads(json.dumps([model.snapshot() for model in models]))
    if snapshots != old["models"] or tuple(old[key] for key in (
            "initial_lp_budget", "lp_node_budget", "integer_state_budget")) != (144, 256, 8192):
        raise ValueError("cached catalogues or the frozen routing budgets changed")
    sources = dict(old["execution_sources"])
    repair = previous["summary_repair"]
    sources[repair["file"]] = repair["source"]
    for path, literal in sources.items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen source changed: {path}")
    sources.update({path: (ROOT / path).read_text() for path in SOURCES})
    excluded = set(map(tuple, old["excluded_static_queues"][str(LANES)]))
    for cell in old["cells"]:
        if cell["lanes"] == LANES:
            excluded.update(tuple(state["queue"]) for state in cell["states"])
    panels = fresh_design(excluded)
    decisions = {**previous["prior_decisions"], "frontier_score_proof_confirmation": previous["decision"]}
    limits = ("New conditional frontier cases under the original R64 queue law and four cached catalogues. "
              "Screen-selected queues do not establish a population trigger rate, wall-time, flow-time or physical-scaling gain. "
              "All prior HOLDs and the preceding 4150-state ADVANCE remain unchanged; the paper is not edited.")
    template = {"schema_version": 1, "experiment": "frontier_trigger_conditional_validation_v1",
        "models": snapshots, "lanes": [LANES], "cpu": min(os.sched_getaffinity(0)),
        **{key: old[key] for key in ("initial_lp_budget", "lp_node_budget", "integer_state_budget",
            "reference_budget", "oracle_state_budget", "reference_source", "rule")},
        "historical_regression_keys": [], "failure_regression_keys": [],
        "prior_decisions": decisions, "execution_sources": sources, "limitations": limits,
        "reference_policy": "After the trigger roster is locked, obtain each state's independent fullK4096 reference before either method. Only that reference's certified score can request its independent5M canonical oracle. All incomplete references and method failures remain recorded.",
        "accounting": "For every triggered queue retain all four catalogues, execute two independent native ladders M,M+1,144,256 and two cold144/256/8192 routers. Charge cumulative native256 and routed final_lp once, including every spent proof/canonical state, dual arithmetic and frontier export. All references and discovery calls are separate.",
        "gate": "Complete frozen conditional roster, valid certificates and paid-budget identities, independent effective references complete, and both cold final actions canonical exact and reference-matching. Candidate native256 must reproduce its screen256 snapshot. Comparative costs and finite-budget score ordering remain diagnostic; no strict gain is required."}
    return json.loads(json.dumps({"schema_version": 1, "experiment": "frontier_trigger_discovery_v1",
        "models": snapshots, "lanes": LANES, "cpu": template["cpu"], "panel_seeds": PANEL_SEEDS,
        "queues_per_panel": QUEUES_PER_PANEL, "block_size": BLOCK_SIZE, "max_rounds": MAX_ROUNDS,
        "min_trigger_queues": MIN_TRIGGER_QUEUES, "min_trigger_seeds": MIN_TRIGGER_SEEDS,
        "queue_panels": panels, "excluded_static_queues": {str(LANES): sorted(excluded)},
        "initial_lp_budget": 144, "lp_node_budget": 256, "integer_state_budget": 8192,
        "expected_max_screen_calls": len(models) * len(PANEL_SEEDS) * QUEUES_PER_PANEL,
        "validation_template": template, "execution_sources": sources, "prior_decisions": decisions,
        "prior_protocol_path": str(previous_path.relative_to(ROOT)),
        "prior_registered_at_utc": old["registered_at_utc"],
        "prior_summary_repair": {key: repair[key] for key in ("file", "reason", "source_diff")},
        "sampler": "Four new seeds2026092851-54, unchanged Random(seed+1000R)/randint(0,5R) law atR64. Exclude all stored preceding static exclusions and all precedingR64 states; reject earlier accepted queues across seeds. Freeze1024 distinct queues per seed before screening.",
        "discovery_rule": "Each round executes64 queued draws per seed, paired across all four catalogues, in seed/queue/model order. Each screen is one cold candidate DualReuseIntegerCapacitySearch.advance_to(256). A queue triggers iff any candidate native endpoint has score_exact=False. No reference, oracle, integer probe or old-method result enters discovery. Finish the entire round; stop at the first completed round with at least2 distinct trigger queues from at least2 seeds, otherwise finish all16 rounds. Retain all trigger queues through the stop, including additional triggers in the stopping round.",
        "validation_rule": "Write an immutable conditional validation roster before any reference or cold router. Every triggered queue retains all four catalogues, even catalogues that did not trigger. Run all states without deleting failures, replacement, changed budgets or extending the discovery limit.",
        "gate": "Screen source/certificate/accounting and first-complete-round stopping identities hold. At least2 new trigger queues from at least2 seeds, followed by complete independent references and both final canonical certificates on every paired state. Too few triggers is TRIGGER_INSUFFICIENT, not method success. Costs and all failed/exhausted attempts stay visible; gains are not a gate.",
        "limitations": limits}))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("frozen discovery sources, queues, catalogue or stopping rule changed")


def run_discovery(models, protocol, search_factory=None, progress=True, on_round=None):
    if search_factory is None:
        from algorithm.dual_reuse_integer_capacity_frame_candidates import DualReuseIntegerCapacitySearch
        search_factory = DualReuseIntegerCapacitySearch
    discovery = {"registered_at_utc": protocol["registered_at_utc"],
        "runtime_start": runtime_provenance(), "rounds": [], "trigger_states": [],
        "stop_reason": "screen_limit_reached"}
    hits = {}
    for round_index in range(protocol["max_rounds"]):
        record = {"round_index": round_index, "states": []}
        start = round_index * protocol["block_size"]
        end = start + protocol["block_size"]
        for panel in protocol["queue_panels"]:
            for registered in panel["states"][start:end]:
                queue = tuple(registered["queue"])
                for model in models:
                    stage = asdict(search_factory(model.options, queue, protocol["lanes"]).advance_to(protocol["lp_node_budget"]))
                    record["states"].append({**deepcopy(registered), "node": model.node,
                        "lanes": protocol["lanes"], "round_index": round_index, "stage": stage})
                    if not stage["selection"]["score_exact"]:
                        hit = hits.setdefault(registered["state_id"], {**deepcopy(registered), "trigger_nodes": []})
                        hit["trigger_nodes"].append(model.node)
        discovery["rounds"].append(record)
        if on_round is not None:
            on_round(record)
        seeds = {state["seed"] for state in hits.values()}
        if progress:
            print(f"screen round {round_index + 1}/{protocol['max_rounds']}: {len(record['states'])} calls, "
                  f"{len(hits)} trigger queues from {len(seeds)} seeds", flush=True)
        if len(hits) >= protocol["min_trigger_queues"] and len(seeds) >= protocol["min_trigger_seeds"]:
            discovery["stop_reason"] = "two_seed_trigger_threshold"
            break
    discovery["trigger_states"] = list(hits.values())
    discovery["runtime_end"] = runtime_provenance()
    return json.loads(json.dumps(discovery))


def validation_protocol(protocol, discovery, models):
    validation = deepcopy(protocol["validation_template"])
    validation.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        discovery_registered_at_utc=protocol["registered_at_utc"],
        trigger_states=deepcopy(discovery["trigger_states"]), screen_stop_reason=discovery["stop_reason"])
    validation["cells"] = [{"node": model.node, "lanes": protocol["lanes"], "states": [
        {**deepcopy(state), "source": "fresh_confirmation", "historical_regression": False, "caps": _caps(model)}
        for state in discovery["trigger_states"]]} for model in models]
    count = len(models) * len(discovery["trigger_states"])
    validation.update(expected_state_count=count, expected_fresh_state_count=count, expected_failure_regression_count=0,
        expected_selector_calls=count * 2, expected_native_advance_calls=count * 8,
        expected_interface_calls=count * 10, expected_native_constructions=count * 4, expected_new_reference_calls=count)
    return json.loads(json.dumps(validation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    runtime = runtime_setup()
    models = runtime.load_catalogues()
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        protocol = build_protocol(models)
        protocol["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        runtime._write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "max_screen_calls": protocol["expected_max_screen_calls"]}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    if list(args.output.parent.glob(args.output.stem + "*.json*")) != [protocol_path]:
        raise FileExistsError("discovery outputs already exist")
    round_paths = []

    def save_round(record):
        path = args.output.with_name(args.output.stem + f"_round_{record['round_index'] + 1:02d}.json.gz")
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "round": record}, stream, separators=(",", ":"), allow_nan=False)
        round_paths.append(str(path))

    discovery = run_discovery(models, protocol, on_round=save_round)
    discovery_path = args.output.with_name(args.output.stem + "_discovery.json")
    runtime._write_new(discovery_path, {**{key: value for key, value in discovery.items() if key != "rounds"}, "round_paths": round_paths})
    validation = validation_protocol(protocol, discovery, models)
    validation_path = args.output.with_name(args.output.stem + "_validation_protocol.json")
    runtime._write_new(validation_path, validation)
    print(f"locked {len(discovery['trigger_states'])} trigger queues; beginning independent references and paired validation", flush=True)
    rows = []
    for model, cell in zip(models, validation["cells"]):
        row = run_cell(model, cell, validation, runtime)
        with gzip.open(_cell_path(args.output, model.node, cell["lanes"]), "xt", encoding="utf-8") as stream:
            json.dump({"registered_at_utc": validation["registered_at_utc"], "cell": row}, stream, separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.frontier_trigger_discovery_summary import summarize
    summary = summarize(protocol, discovery, validation, rows)
    summary.update(protocol_path=str(protocol_path), discovery_path=str(discovery_path), validation_protocol_path=str(validation_path))
    runtime._write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("decision", "screened_queue_count", "trigger_queue_count", "validation_state_count")}))


if __name__ == "__main__":
    main()
