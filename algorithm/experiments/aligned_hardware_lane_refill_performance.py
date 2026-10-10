"""Performance-first development: synchronous barriers versus local lane renewals."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_integer_capacity_timing import runtime_provenance, runtime_setup


ROOT = Path(__file__).resolve().parents[2]
PREVIOUS_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_small_dimension_exact_native_decision_delay_development_20260929.json"
DEFAULT_OUTPUT = ROOT / "md/experiment_artifacts/aligned_hardware_lane_refill_performance_development_20260929.json"
LABELS = ("sync_joint", "refill_joint", "refill_single")
DISPATCH_RULES = dict(zip(LABELS, ("synchronous", "refill", "single"), strict=True))
SOURCES = (
    "algorithm/experiments/aligned_hardware_lane_refill_performance.py",
    "algorithm/experiments/lane_refill_performance_replay.py",
    "algorithm/experiments/lane_refill_performance_summary.py",
)


def _protocol_path(output):
    return output.with_name(output.stem + "_protocol.json")


def build_protocol(models):
    previous = json.loads(_protocol_path(PREVIOUS_OUTPUT).read_text())
    result = json.loads(PREVIOUS_OUTPUT.read_text())
    if not all(result[key] for key in ("execution_checks_passed", "certificate_checks_passed",
            "references_complete", "canonical_action_gate_passed", "integration_gate_passed")):
        raise ValueError("complete preceding native replay correctness is required")
    if json.loads(json.dumps([model.snapshot() for model in models])) != previous["models"]:
        raise ValueError("retain the original complete-job service catalogues")
    for path, literal in previous["execution_sources"].items():
        if (ROOT / path).read_text() != literal:
            raise ValueError(f"frozen native source changed: {path}")
    frozen = {key: previous[key] for key in ("models", "lanes", "loads", "arrival_seeds",
        "arrival_horizon", "observation_end", "initial_lp_budget", "lp_node_budget",
        "integer_state_budget", "reference_budget", "oracle_state_budget", "budgets",
        "capacities", "tapes", "execution_order", "float_relative_tolerance", "cpu")}
    if (len(models), tuple(frozen["lanes"]), tuple(frozen["loads"]), tuple(frozen["arrival_seeds"]),
            len(frozen["tapes"])) != (4, (8,), (.80, .95), (41, 42, 43), 24):
        raise ValueError("retain the original24 development tapes")
    return json.loads(json.dumps({
        "schema_version": 1, "experiment": "aligned_complete_job_lane_refill_performance_development_v1",
        **frozen, "labels": LABELS, "dispatch_rules": DISPATCH_RULES,
        "idle_duration": 1.0, "minimum_pooled_flow_improvement": .01,
        "expected_tape_count": 24, "expected_trajectory_count": 72, "expected_group_count": 16,
        "previous_protocol_path": str(_protocol_path(PREVIOUS_OUTPUT).relative_to(ROOT)),
        "previous_result_path": str(PREVIOUS_OUTPUT.relative_to(ROOT)),
        "prior_decisions": {**result["prior_decisions"], "small_dimension_exact_native_decision_delay": result["decision"]},
        "execution_sources": {**previous["execution_sources"],
                              **{path: (ROOT / path).read_text() for path in SOURCES}},
        "design": "All24 original development arrival tapes, four unchanged independent-lane complete-job catalogues, virtualR8, loads0.80/0.95, seeds41/42/43,H600/T1200. Three actual native-controller trajectories per tape,72 total: synchronous joint selection; joint refill of all currently free lanes; serial one-lane refill control. Every call keeps144 initial/256 totalLP/8192 integer-state caps. All original local options and job completion offsets are retained. No new arrival seeds, method/parameter sweep, policy selection cache or candidate rescue.",
        "clock": "Service time only, without measured computation delay or a1s computation window. At a lane-availability event release only already arrived FIFO jobs. Synchronous joint dispatch reserves all lanes until the selected global duration; refill dispatch releases each occupied lane after max(1,its complete local option.duration). Idle lanes renew after1s. Serial control invokes the same native selector withR1 for the first free lane and repeats at the same time while more lanes remain free. Arrival events themselves never trigger a dispatch, so original1s idle polling is held fixed. Stop starting decisions atT; retain scheduled completions pastT.",
        "lane_semantics": "Local option.duration includes every target/resident complete-job phase, including resident drain and finalization. An earlier target completion offset never frees a loaded lane. Busy lanes retain commitments and their admitted jobs do not re-enter waiting Q. Oldest admitted jobs receive earliest completion slots within each cold selection. Per-lane service independence is the cached model assumption; no unmeasured cross-lane co-location is introduced.",
        "audit": "Only after all three genuine policy trajectories on a tape, audit every returned native prefix and full work ledger against K4096 reference on that actual(free-lane count,Q). Maintain a separate reference cache for each free-lane count; cache only offline reference results, never native policy results. Only reference's own score-exact/canonical-open result may invoke the unchanged5M score-face oracle. Actual returned stage caps and whole144/256/8192 fees are separate; all native/rational/root/probe/evidence work and every cold call are paid.",
        "primary": "Algorithm scheduling performance, not solver speed. AtH/T report every released-job restricted mean flow, waiting area, unfinished/queued/inflight counts, and one identical completed-ID intersection across all three policies for mean/p90. Primary contrast is refill_joint versus sync_joint (global-barrier ablation); refill_single is the ordinary serial per-lane MaxWeight control, whose relative outcomes are descriptive and distinguish joint choice from simple refill.",
        "utility_gate": "Refill_joint must not increase restricted mean, shared-cohort mean/p90 or unfinished counts versus sync_joint in any of16node/load/H,T groups, averaging the three paired seeds first. Relative float tolerance1e-9 only; exact integer counts and no empty common cohorts. Keep every regression. Promotion additionally requires at least1% lower pooled per-tape ALL-JOB restricted mean atT. Wall/CPU/LP/IP savings do not determine promotion.",
        "accounting": "One fresh serial CPU-pinned process, BLAS1 and GC enabled. Measure whole-selector wall only as an auxiliary diagnostic, never model time or a promotion gate. Retain total/per-released-job cold calls, logicalLP, integer visits and all whole-controller fee families. Equal per-call budgets do not imply equal cumulative search work; async may make more calls. Clip busy/barrier-idle/scheduled-idle lane intervals atT and retain waiting area for all released jobs.",
        "limitations": "Development mechanism screen on reused seeds and deterministic cached independent lanes, not independent confirmation or physical cluster performance. Catalogue/load pairs share seeds. Asynchronous lane dispatch is a new aligned execution rule; the old synchronous-frame drift proof is not automatically an asynchronous stability proof. Original service-capacity arrival rates are fixed; no new lower-confidence guarantee. Retain all historical performance/timing HOLDs. No wall-speed, broad superiority or production deployment claim.",
    }))


def verify_protocol(protocol, models):
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != build_protocol(models):
        raise ValueError("registered lane dispatch, sources, budgets, tapes or performance criteria changed")


def audit_lane_calls(model, trajectory, originals, references, protocol, auditor=None, reference_factory=None):
    """Audit real free-lane states using the existing complete native auditor."""
    from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import audit_trajectory, ReplayReferenceAudit
    from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import SCALAR_COSTS, _add_work

    auditor = audit_trajectory if auditor is None else auditor
    reference_factory = ReplayReferenceAudit if reference_factory is None else reference_factory
    if len(originals) != len(trajectory["frames"]):
        raise ValueError("every lane decision needs its original native result")
    metrics = {**dict.fromkeys(SCALAR_COSTS, 0), "whole_work": {}, "operational_elapsed_s": 0.0}
    diagnostics, audit = [], {"calls": 0, "oracle_calls": 0, "failures": 0, "incomplete_count": 0, "work": {}}
    for lanes in sorted({frame["lanes_called"] for frame in trajectory["frames"]}):
        if lanes not in references:
            references[lanes] = reference_factory(replace(model, lanes=lanes), protocol)
        indices = [number for number, frame in enumerate(trajectory["frames"]) if frame["lanes_called"] == lanes]
        # The old auditor's elapsed field describes its timed call, not dispatch.
        frames = [{**trajectory["frames"][number], "decision_delay_s": trajectory["frames"][number]["selector_wall_s"]}
                  for number in indices]
        shadow = {"policy": "candidate", "frames": frames}
        auditor(replace(model, lanes=lanes), shadow, [originals[number] for number in indices], references[lanes], protocol)
        for key in SCALAR_COSTS:
            metrics[key] += shadow["selector_metrics"][key]
        metrics["operational_elapsed_s"] += shadow["selector_metrics"]["operational_elapsed_s"]
        work = shadow["selector_metrics"]["whole_work"]
        previous_max = metrics["whole_work"].get("max_shared_integer_states", 0)
        _add_work(metrics["whole_work"], work)
        if "max_shared_integer_states" in work:
            metrics["whole_work"]["max_shared_integer_states"] = max(previous_max, work["max_shared_integer_states"])
        diagnostics.extend({**entry, "frame": indices[entry["frame"]], "lanes_called": lanes}
                           for entry in shadow["decision_diagnostics"])
        for key in audit:
            if key == "work":
                _add_work(audit[key], shadow["reference_audit"][key])
            else:
                audit[key] += shadow["reference_audit"][key]
    trajectory.update(selector_metrics=metrics, decision_diagnostics=sorted(diagnostics, key=lambda entry: entry["frame"]),
                      reference_audit=audit)


def run_cell(model, lanes, protocol, dependencies=None, progress=True):
    if dependencies is None:
        from algorithm.experiments.lane_refill_performance_replay import simulate_lane_refill_trajectory
        from algorithm.experiments.small_dimension_exact_native_decision_delay_audit import selectors
        dependencies = {"simulate": simulate_lane_refill_trajectory, "selectors": selectors,
                        "audit": audit_lane_calls, "provenance": runtime_provenance}
    virtual_model, started = replace(model, lanes=lanes), perf_counter()
    row = {"node": model.node, "lanes": lanes, "physical_lanes": model.lanes,
           "runtime_start": dependencies["provenance"](), "runs": []}
    selected = dependencies["selectors"](protocol)["candidate"]
    for tape in protocol["tapes"]:
        if (tape["node"], tape["lanes"]) != (model.node, lanes):
            continue
        run = {key: tape[key] for key in ("tape_id", "load", "seed")}
        run["arrival_count"] = len(tape["arrivals"])
        run["policies"], pending, references = {}, [], {}
        for label in protocol["labels"]:
            trajectory, originals = dependencies["simulate"](virtual_model, tape["arrivals"],
                protocol["arrival_horizon"], protocol["observation_end"], protocol["lp_node_budget"], selected,
                policy=label, dispatch_rule=protocol["dispatch_rules"][label])
            run["policies"][label] = trajectory
            pending.append((trajectory, originals))
        for trajectory, originals in pending:
            dependencies["audit"](virtual_model, trajectory, originals, references, protocol)
        from algorithm.experiments.small_dimension_exact_native_decision_delay_summary import _add_work
        run["reference_audit"] = {"calls": 0, "oracle_calls": 0, "failures": 0, "incomplete_count": 0, "work": {}}
        for reference in references.values():
            summary = reference.summary()
            for key in run["reference_audit"]:
                if key == "work":
                    _add_work(run["reference_audit"][key], summary[key])
                else:
                    run["reference_audit"][key] += summary[key]
        row["runs"].append(run)
        if progress:
            print(f"{model.node} R{lanes}: load={run['load']} seed={run['seed']} sync/refill/single performance trajectories audited", flush=True)
    row["runtime_end"] = dependencies["provenance"]()
    row["operational_elapsed_s"] = perf_counter() - started
    return row


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
        print(protocol_path, flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol, models)
    paths = [args.output.with_name(f"{args.output.stem}_cell_{node}_r{lanes}.json.gz")
             for node, lanes in protocol["execution_order"]]
    if any(path.exists() for path in (args.output, *paths)):
        raise FileExistsError("registered performance outputs already exist")
    rows = []
    for (node, lanes), path in zip(protocol["execution_order"], paths, strict=True):
        row = run_cell(next(model for model in models if model.node == node), lanes, protocol)
        with gzip.open(path, "xt") as target:
            json.dump({"registered_at_utc": protocol["registered_at_utc"], "cell": row}, target,
                      separators=(",", ":"), allow_nan=False)
        rows.append(row)
    from algorithm.experiments.lane_refill_performance_summary import summarize
    result = summarize(models, protocol, rows)
    result["protocol_path"] = str(protocol_path.resolve())
    runtime._write_new(args.output, result)
    print(json.dumps({key: result[key] for key in ("decision", "integration_gate_passed", "utility_gate_passed", "flow_gain_established")}), flush=True)


if __name__ == "__main__":
    main()
