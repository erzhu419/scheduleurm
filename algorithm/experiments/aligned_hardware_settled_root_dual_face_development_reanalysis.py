"""Correct snapshot prune accounting on the saved run, without rerunning it."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928.json"
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_settled_root_dual_face_development_20260928_checked.json"
SOURCES = (
    "algorithm/experiments/settled_root_dual_face_checked_development_summary.py",
    "algorithm/experiments/aligned_hardware_settled_root_dual_face_development_reanalysis.py",
    "algorithm/experiments/aligned_hardware_settled_root_dual_face_checked_timing.py",
    "algorithm/experiments/settled_root_dual_face_checked_timing_summary.py",
)
DIAGNOSTIC = "settlement head, free pop/split/singleton and stopping diagnostics differ from native heap progress"


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol():
    original = json.loads(_protocol_path(ORIGINAL_OUTPUT).read_text())
    result = json.loads(ORIGINAL_OUTPUT.read_text())
    if (result["decision"] != "INVALID_EXECUTION" or result["state_count"] != 2802
            or result["executed_interface_calls"] != 2802 or not result["references_complete"]
            or not result["source_native_behavior_preserved"] or result["nonclosures"]
            or len(result["failures"]) != 3
            or any(failure["errors"] != [DIAGNOSTIC] for failure in result["failures"])):
        raise ValueError("reuse the complete saved run with only the three diagnosed snapshot-prune audit failures")
    for path, literal in original["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"original frozen source changed: {path}")
    protocol = deepcopy(original)
    protocol["execution_sources"].update({path: (ROOT / path).read_text() for path in SOURCES})
    protocol["summary_reanalysis"] = {
        "data_path": str(ORIGINAL_OUTPUT.relative_to(ROOT)),
        "registered_at_utc": original["registered_at_utc"],
        "original_decision": result["decision"],
        "actual_selector_calls": 0,
        "actual_native_advance_calls": 0,
        "rule": "Post-outcome audit repair only. Native snapshot bound_pruned_regions includes the currently prunable heap entries. Remove raw-minus-retained counts from both snapshots before comparing the cumulative native prune increment with settlement popped prunes. Reuse all2802 original records and every original reference, budget and gate. Change no selector, action, fee, state or timing order. Preserve the original INVALID_EXECUTION result and all old files.",
    }
    return protocol


def verify_protocol(protocol):
    if {name: value for name, value in protocol.items()
            if name != "summary_reanalysis_registered_at_utc"} != build_protocol():
        raise ValueError("registered audit correction, source records or frozen algorithm changed")


def reanalyze(protocol, loader=None, summarizer=None):
    if loader is None:
        from algorithm.experiments.aligned_hardware_root_pruning_development import load_cell
        loader = load_cell
    if summarizer is None:
        from algorithm.experiments.settled_root_dual_face_checked_development_summary import summarize
        summarizer = summarize
    source = protocol["summary_reanalysis"]
    rows = [loader(ROOT / source["data_path"], cell["node"], cell["lanes"], source["registered_at_utc"])
            for cell in protocol["cells"]]
    result = summarizer(protocol, rows)
    result.update(summary_reanalysis=deepcopy(source),
        summary_reanalysis_registered_at_utc=protocol["summary_reanalysis_registered_at_utc"],
        actual_selector_calls_during_reanalysis=0, actual_native_advance_calls_during_reanalysis=0,
        inherited_selector_calls=result["executed_interface_calls"])
    return result


def _write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    protocol_path = _protocol_path(args.output)
    if args.register:
        protocol = build_protocol()
        protocol["summary_reanalysis_registered_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_new(protocol_path, protocol)
        print(json.dumps({"protocol": str(protocol_path), "inherited_states": 2802, "new_selector_calls": 0}))
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol)
    if args.output.exists():
        raise FileExistsError("checked development summary already exists")
    result = reanalyze(protocol)
    result["protocol_path"] = str(protocol_path)
    _write_new(args.output, result)
    print(json.dumps({name: result[name] for name in ("decision", "state_count", "actual_selector_calls_during_reanalysis")}))


if __name__ == "__main__":
    main()
