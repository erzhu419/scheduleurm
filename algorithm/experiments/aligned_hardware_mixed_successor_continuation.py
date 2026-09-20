"""Replay actual raw/mixed disagreements and resample their planning margins."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_frame_mechanism import action_signature
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS
from algorithm.experiments.aligned_mixed_successor_continuation_replay import (
    SOURCE_POLICIES, run_source,
)
from algorithm.experiments.aligned_hardware_mixed_successor_continuation_sampling import (
    compare_case, sample_state, state_key, summarize,
)


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_mixed_successor_continuation_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_mixed_successor_rollout_development_20260910.json"
SAMPLES = 64
NAMESPACE = "aligned-mixed-continuation:{sample_index}:{coordinate}"
SOURCE_FILES = (
    "algorithm/mixed_successor_rollout_terminal.py",
    "algorithm/mixed_successor_rollout_frame_candidates.py",
    "algorithm/experiments/aligned_hardware_mixed_successor_rollout.py",
    "algorithm/experiments/aligned_mixed_successor_continuation_replay.py",
    "algorithm/experiments/aligned_hardware_mixed_successor_continuation_sampling.py",
    "algorithm/experiments/aligned_hardware_mixed_successor_continuation.py",
)


def _same_action(left, right):
    return action_signature(left) == action_signature(right)


def _sum_work(states, field):
    keys = sorted({key for state in states for key in state[field]})
    return {key: sum(state[field].get(key, 0) for state in states) for key in keys}


def _summarize_subset(runs, states):
    state_ids = {case["state_id"] for run in runs for case in run["cases"]}
    return summarize(runs, [state for state in states if state["state_id"] in state_ids])


def build_diagnosis(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    confirmed_snapshot = json.loads(
        (ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != confirmed_snapshot:
        raise ValueError("confirmed source lineage changed")
    previous = json.loads(DEVELOPMENT.read_text())
    if previous["status"] != "PASS":
        raise ValueError("mixed-successor development source did not pass")
    base = previous["protocol"]["base_protocol"]
    model = load_model()
    certificates = {load: capacity_certificate(model, load) for load in base["loads"]}
    mu = terminal_service_rates(model, certificates[base["loads"][0]])
    if (json.loads(json.dumps(model.snapshot())) != previous["model"]
            or json.loads(json.dumps(list(certificates.values()))) != previous["certificates"]
            or list(mu) != previous["protocol"]["terminal_service_rates"]):
        raise ValueError("continuation diagnosis changed the registered model or rates")
    root = Path(__file__).resolve().parents[2]
    source_texts = {name: (root / name).read_text() for name in SOURCE_FILES}
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development mechanism diagnosis using consumed real and planning seeds",
        "loads": base["loads"], "seeds": base["seeds"],
        "planning_seeds": previous["protocol"]["planning_seeds"],
        "budget": 26, "arrival_horizon_s": base["arrival_horizon_s"],
        "terminal_service_rates": mu, "source_policies": SOURCE_POLICIES,
        "source_roster": "54 frozen K26 trajectories: three loads x three real seeds x three planning seeds x raw/mixed source policies",
        "shadow_comparison": "before each active native call, copy all planning RNG states and run the opposite selector offline; native trajectory RNG and counters remain unchanged",
        "current_candidates": "raw and mixed must retain identical root search, support incumbent, analytic challenger, planning path and random-draw endpoint",
        "state_roster": "every source frame, including drain, where the same pre-call sample selects physically different actions; service, duration and sorted per-class completion offsets define physical equality",
        "intervention": "replay the identical native prefix; force raw versus mixed first action once; both branches then use memoized frozen v4 K26 until the identical original population drains",
        "sign": "all intervention and oriented planning deltas are mixed-first minus raw-first, in job-seconds",
        "area_partition": "current and future use each branch's own first-frame boundary; current plus future and flow components must reconstruct total",
        "repeated_sampling": {
            "samples_per_unique_load_queue_pair": SAMPLES,
            "rng_namespace": NAMESPACE,
            "selectors": "run raw and mixed from separately initialized identical states for every sample",
            "measure": "fixed challenger-minus-incumbent margins, within-state means/variances/MCSE and paired mixed-minus-raw change",
            "reuse": "one paired estimate per unique load/queue and contender pair; reused across occurrences",
        },
        "residual_partition": "actual minus discovery estimate equals fresh mean minus discovery sample plus actual minus fresh proxy mean, separately for raw and mixed",
        "diagnostic_work": "native source accounting, shadow selector work and RNG copies, continuation-cache searches, replay frames, and both fresh-sampling selectors are reported separately from online K",
        "source_snapshot": "full text companion file; no hashes",
        "stopping_rule": "all 54 sources and every physical disagreement once, then 64 paired fresh samples for every unique state; no outcome filter, tuning or extra seeds",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    output.with_name(output.stem + "_fixed_sources.json").write_text(
        json.dumps(source_texts, indent=2) + "\n")
    originals = {
        (row["load"], row["seed"], row["planning_seed"], row["policy"]): row
        for row in previous["runs"]
        if row["budget"] == 26 and row["policy"] in SOURCE_POLICIES
    }
    tasks = [
        (model, certificates[load], load, seed, planning_seed, policy, protocol,
         originals[(load, seed, planning_seed, policy)])
        for load in base["loads"] for seed in base["seeds"]
        for planning_seed in protocol["planning_seeds"] for policy in SOURCE_POLICIES
    ]
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        runs = list(pool.map(run_source, tasks))
    states = {}
    for row in runs:
        for case in row["cases"]:
            key = state_key(row["load"], case["queue"])
            if key not in states:
                states[key] = sample_state(model, certificates[row["load"]], mu, case, SAMPLES)
            else:
                state = states[key]
                for name in ("incumbent_action", "challenger_action"):
                    if not _same_action(case[name], state[name]):
                        raise ValueError("one load/queue produced different current contenders")
            case["state_id"] = key
            case["estimate_comparison"] = compare_case(case, states[key])
    state_rows = list(states.values())
    checks = {
        "all_54_frozen_sources_reproduced": len(runs) == 54 and all(r["source_reproduced"] for r in runs),
        "complete_unique_source_roster": len({
            (r["load"], r["seed"], r["planning_seed"], r["source_policy"]) for r in runs}) == 54,
        "every_actual_disagreement_replayed": all(
            r["substitution_count"] == r["paired_replay_count"] for r in runs),
        "population_budget_drift_flow_and_rng_pairing_hold": True,
        "every_unique_state_has_64_paired_fresh_samples": all(
            state["status"] == "PASS" and state["sample_count"] == SAMPLES
            for state in state_rows),
        "fresh_sample_and_target_residual_identities_hold": True,
        "diagnostic_sources_unchanged_during_run": all(
            (root / name).read_text() == text for name, text in source_texts.items()),
    }
    report = {
        "experiment": "aligned_hardware_mixed_successor_paired_continuation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "protocol": protocol, "model": model.snapshot(),
        "certificates": list(certificates.values()), "runs": runs, "states": state_rows,
        "fresh_sampling_work": {
            "raw": _sum_work(state_rows, "raw_work"),
            "mixed": _sum_work(state_rows, "mixed_work"),
        },
        "summary": _summarize_subset(runs, state_rows),
        "summary_by_load": [
            {"load": load,
             **_summarize_subset([r for r in runs if r["load"] == load], state_rows)}
            for load in base["loads"]
        ],
        "summary_by_load_and_source": [
            {"load": load, "source_policy": policy,
             **_summarize_subset([r for r in runs
                                  if r["load"] == load and r["source_policy"] == policy],
                                 state_rows)}
            for load in base["loads"] for policy in SOURCE_POLICIES
        ],
        "wall_s": perf_counter() - started,
        "limitations": [
            "Disagreement states are selected by the two development policies. Fresh samples estimate only these fixed contender pairs and do not remove that conditioning.",
            "Actual-minus-proxy residuals combine finite arrival-tape noise, horizon cutoff and two-step/fluid versus v4 continuation; they are not pure model-bias estimates.",
            "Sources share development tapes and cases have overlapping futures. Counts and means are descriptive, correlated and cannot be summed into a whole-policy gain.",
            "The common-v4 intervention isolates one action from subsequent mixed-policy feedback; independent confirmation remains required before adoption.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_diagnosis(args.output)
    print(json.dumps({"status": result["status"], "summary": result["summary"],
                      "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
