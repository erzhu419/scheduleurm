"""Fresh-queue confirmation, conditional on the frozen development gate passing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from algorithm.experiments import aligned_hardware_integer_capacity_development as development
from algorithm.experiments.aligned_hardware_shared_threshold_queue_confirmation import sample_queues


ROOT = development.ROOT
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_integer_capacity_queue_confirmation_20260927.json"
DEVELOPMENT_OUTPUT = development.DEFAULT_OUTPUT
DEVELOPMENT_PROTOCOL = DEVELOPMENT_OUTPUT.with_name(DEVELOPMENT_OUTPUT.stem + "_protocol.json")
PANEL_SEEDS = (2026092705, 2026092706, 2026092707, 2026092708)
QUEUES_PER_PANEL = 64
EXECUTION_SOURCES = (*development.EXECUTION_SOURCES,
    "algorithm/experiments/aligned_hardware_integer_capacity_queue_confirmation.py")
screen_cell = development.screen_cell


def require_development_advance():
    if not DEVELOPMENT_OUTPUT.exists():
        raise ValueError("development result is incomplete; confirmation cannot be registered")
    result = json.loads(DEVELOPMENT_OUTPUT.read_text(encoding="utf-8"))
    flags = ("certificate_checks_passed", "references_complete", "development_action_gate_passed", "development_search_cost_gate_passed")
    if result["decision"] != "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE" or not all(result[key] for key in flags):
        raise ValueError("development action and cost gates must pass before confirmation registration")


def build_protocol(models):
    previous = json.loads(DEVELOPMENT_PROTOCOL.read_text(encoding="utf-8"))
    panels = {}
    for lanes in previous["lanes"]:
        seen = {tuple(queue) for panel in previous["queue_panels"][str(lanes)] for queue in panel["queues"]}
        panels[str(lanes)] = []
        for seed in PANEL_SEEDS:
            queues = sample_queues(lanes, seed, QUEUES_PER_PANEL, excluded=seen)
            panels[str(lanes)].append({"panel": f"fresh_{seed}", "seed": seed, "queues": queues})
            seen.update(queues)
    protocol = {key: value for key, value in previous.items() if key != "registered_at_utc"}
    protocol.update(
        experiment="integer_capacity_fresh_queue_confirmation_v1",
        models=[model.snapshot() for model in models],
        panel_seeds=PANEL_SEEDS, queues_per_panel=QUEUES_PER_PANEL, queue_panels=panels,
        sampler="Python random.Random(panel_seed + 1000*R); each coordinate randint(0,5*R); reject all frozen development queues, native/scaled grids, and earlier draws across fresh panels within R; conditional uniform without replacement",
        queue_source=str(DEVELOPMENT_PROTOCOL.relative_to(ROOT)),
        prior_development=str(DEVELOPMENT_OUTPUT.relative_to(ROOT)), retained_witnesses=[],
        execution_sources={path: (ROOT / path).read_text(encoding="utf-8") for path in EXECUTION_SOURCES},
        limitations="Only queues are fresh under a fixed sampling law; cached point models and algorithms remain fixed. Virtual identical-lane scaling; queues paired across nodes, so node-state comparisons are not independent observations. Propagation work is separate. No new physical measurement, trajectory, flow-time, or total wall-time claim.",
    )
    return json.loads(json.dumps(protocol))


def summarize(models, protocol, rows):
    result = development.summarize(models, protocol, rows)
    for kind in ("action", "search_cost"):
        result[f"confirmation_{kind}_gate_passed"] = result.pop(f"development_{kind}_gate_passed")
    result["decision"] = {
        "INTEGER_CAPACITY_DEVELOPMENT_ADVANCE": "INTEGER_CAPACITY_FRESH_QUEUE_CONFIRMATION_ADVANCE",
        "DEVELOPMENT_ACTION_REVIEW_REQUIRED": "CONFIRMATION_ACTION_REVIEW_REQUIRED",
    }.get(result["decision"], result["decision"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--cell", help="registered NODE:R cell")
    mode.add_argument("--merge-cells", action="store_true")
    args = parser.parse_args()
    if args.register:
        require_development_advance()
    models = development.load_catalogues()
    expected = build_protocol(models)
    protocol_path = args.output.with_name(args.output.stem + "_protocol.json")
    if args.register:
        expected["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        development._write_new(protocol_path, expected)
        print(protocol_path)
        return
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered models, execution source, or protocol changed; register a new output")
    if args.cell:
        node, lanes_text = args.cell.rsplit(":", 1)
        lanes = int(lanes_text)
        model = next(model for model in models if model.node == node)
        if lanes not in protocol["lanes"]:
            raise ValueError("unregistered lane count")
        output = development._cell_path(args.output, node, lanes)
        if output.exists():
            raise FileExistsError(output)
        development._write_new(output, {"protocol": protocol, "cell": screen_cell(model, lanes, protocol)})
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for model in models:
        for lanes in protocol["lanes"]:
            if args.merge_cells:
                saved = json.loads(development._cell_path(args.output, model.node, lanes).read_text(encoding="utf-8"))
                if saved["protocol"] != protocol:
                    raise ValueError("cell protocol differs from registered protocol")
                rows.append(saved["cell"])
            else:
                rows.append(screen_cell(model, lanes, protocol))
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    development._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "queue_count", "comparison_search_work", "capacity_propagation")}))


if __name__ == "__main__":
    main()
