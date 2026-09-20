"""Paired single-action continuation diagnosis on every development substitution."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import accumulate
import json
from pathlib import Path
from statistics import mean, median

from algorithm.experiments.aligned_lookahead_continuation import (
    DecisionPairs, PrefixReplay, expected_parts, realized_current_area,
)
from algorithm.experiments.aligned_frame_flow_decomposition import decompose_flow
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals, simulate_frames
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description, action_signature
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS, COMPONENTS


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_lookahead_continuation_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_lookahead_development_20260910.json"
AREA_TOLERANCE = 1e-6


class Discovery:
    def __init__(self, pairs):
        self.pairs = pairs
        self.now = 0.0
        self.prefix, self.frames, self.switches = [], [], []

    def __call__(self, options, queues, lanes, budget):
        support, lookahead = self.pairs.get(queues)
        descriptions = [action_description(self.pairs.model, a) for a in (support, lookahead)]
        if action_signature(descriptions[0]) != action_signature(descriptions[1]):
            self.switches.append(len(self.frames))
        self.prefix.append((tuple(queues), lookahead))
        self.frames.append({"start_s": self.now, "duration_s": lookahead.duration,
                            "queue_before": tuple(queues), "departures": lookahead.service,
                            "option_indices": lookahead.option_indices})
        self.now += lookahead.duration
        return lookahead


def action_audit(model, queues, action, start, horizon, rates, service_rates, times, sums, class_times):
    predicted = expected_parts(model, queues, action, rates, service_rates)
    tau = action.duration
    exposure = min(tau, max(0.0, horizon - start))
    remaining = tuple(q - d for q, d in zip(queues, action.service, strict=True))
    base_area = predicted["current_area"] - sum(rates) * tau * tau / 2
    truncated_current = base_area + sum(rates) * (tau * exposure - exposure * exposure / 2)
    truncated_terminal = sum(((q + rate * exposure) ** 2 + rate * exposure) / (2 * mu)
                             for q, rate, mu in zip(remaining, rates, service_rates, strict=True))
    next_queue = tuple(q + bisect_right(t, start + tau) - bisect_right(t, start)
                       for q, t in zip(remaining, class_times, strict=True))
    return {
        "predicted_current_area": predicted["current_area"], "predicted_terminal_area": predicted["terminal_area"],
        "predicted_total_area": predicted["total"],
        "truncated_expected_current_area": truncated_current,
        "truncated_expected_terminal_area": truncated_terminal,
        "realized_current_area": realized_current_area(model, queues, action, start, times, sums),
        "realized_next_queue": next_queue,
        "realized_terminal_value": sum(q * q / (2 * mu) for q, mu in zip(next_queue, service_rates, strict=True)),
    }


def paired_effect(before, after, audits, population):
    """All deltas are lookahead-first minus v4-first, in job-seconds."""
    delta = {key: audits[1][key] - audits[0][key] for key in audits[0] if key != "realized_next_queue"}
    total = population * (after["metrics"]["mean_flow_s"] - before["metrics"]["mean_flow_s"])
    future = total - delta["realized_current_area"]
    cutoff = (delta["truncated_expected_current_area"] + delta["truncated_expected_terminal_area"]
              - delta["predicted_total_area"])
    arrival_current = delta["realized_current_area"] - delta["truncated_expected_current_area"]
    arrival_terminal = delta["realized_terminal_value"] - delta["truncated_expected_terminal_area"]
    continuation = future - delta["realized_terminal_value"]
    components = {key: population * (after["decomposition"][key] - before["decomposition"][key]) for key in COMPONENTS}
    return {
        "predicted_total_delta_job_s": delta["predicted_total_area"],
        "realized_total_delta_job_s": total, "mean_flow_delta_s": total / population,
        "realized_current_delta_job_s": delta["realized_current_area"], "realized_future_delta_job_s": future,
        "cutoff_correction_delta_job_s": cutoff,
        "current_arrival_residual_delta_job_s": arrival_current,
        "next_queue_residual_delta_job_s": arrival_terminal,
        "continuation_residual_delta_job_s": continuation,
        "prediction_residual_delta_job_s": total - delta["predicted_total_area"],
        "residual_identity_error_job_s": cutoff + arrival_current + arrival_terminal + continuation - (total - delta["predicted_total_area"]),
        "flow_component_deltas_job_s": components,
        "flow_component_identity_error_job_s": sum(components.values()) - total,
        "outcome": "improved" if total < -AREA_TOLERANCE else "worsened" if total > AREA_TOLERANCE else "unchanged",
    }


def _checked_run(model, tape, horizon, certificate, recorder):
    metrics = simulate_frames(model, tape, horizon, certificate, 26, selector=recorder)
    parts = decompose_flow(model, tape, recorder.frames, horizon)
    if (metrics["completed_jobs"] != len(tape) or metrics["maximum_candidate_attempts_per_frame"] > 26
            or metrics["maximum_queue_recursion_violation"] != 0
            or metrics["maximum_expected_drift_bound_violation"] > 1e-8
            or abs(parts["mean_flow_s"] - metrics["mean_flow_s"]) > 1e-8):
        raise ValueError("continuation replay failed population, budget, drift or flow reconstruction")
    return {"metrics": metrics, "decomposition": parts}


def _run_seed(task):
    model, certificate, load, seed, protocol, original = task
    horizon = protocol["arrival_horizon_s"]
    rates, mu = certificate["arrival_rates"], protocol["terminal_service_rates"]
    tape = poisson_arrivals(rates, horizon, seed)
    times = tuple(t for t, _, _ in tape)
    sums = tuple(accumulate(times, initial=0.0))
    class_times = [tuple(t for t, c, _ in tape if c == i) for i in range(len(model.workloads))]
    pairs = DecisionPairs(model, rates, mu, budget=26)
    discovery = Discovery(pairs)
    replayed = _checked_run(model, tape, horizon, certificate, discovery)
    if any(json.loads(json.dumps(replayed[key])) != original[key] for key in ("metrics", "decomposition")):
        raise ValueError(f"lookahead discovery changed frozen development trajectory {(load, seed)}")
    print(json.dumps({"load": load, "seed": seed, "substitutions": len(discovery.switches)}), flush=True)
    cases, replay_frames = [], 0
    for index in discovery.switches:
        queues, _ = discovery.prefix[index]
        support, lookahead = pairs.get(queues)
        start = discovery.frames[index]["start_s"]
        audits = [action_audit(model, queues, a, start, horizon, rates, mu, times, sums, class_times)
                  for a in (support, lookahead)]
        if (abs(audits[1]["predicted_total_area"] - lookahead.lookahead_cost) > 1e-8
                or abs(audits[0]["predicted_total_area"] - lookahead.support_incumbent_lookahead_cost) > 1e-8):
            raise ValueError("prediction audit differs from the frozen selection objective")
        branches = []
        for first in (support, lookahead):
            recorder = PrefixReplay(discovery.prefix, index, first, pairs.support)
            branches.append(_checked_run(model, tape, horizon, certificate, recorder))
            replay_frames += len(recorder.frames)
        effect = paired_effect(*branches, audits, len(tape))
        if abs(effect["residual_identity_error_job_s"]) > 1e-5 or abs(effect["flow_component_identity_error_job_s"]) > 1e-5:
            raise ValueError("paired continuation area decomposition failed")
        cases.append({
            "frame_index": index, "start_s": start, "queue": queues,
            "phase": "drain" if start >= horizon else "crosses_cutoff" if start + max(support.duration, lookahead.duration) > horizon else "arrival_window",
            "v4_action": action_description(model, support), "lookahead_action": action_description(model, lookahead),
            "v4_audit": audits[0], "lookahead_audit": audits[1], "effect": effect,
        })
        if len(cases) % 25 == 0:
            print(json.dumps({"load": load, "seed": seed, "pairs_completed": len(cases)}), flush=True)
    work = dict(pairs.accounting(), action_audit_evaluations=2 * len(cases), replay_frames=replay_frames)
    return {"load": load, "seed": seed, "job_count": len(tape), "source_frame_count": len(discovery.frames),
            "source_reproduced": True, "substitution_count": len(discovery.switches),
            "paired_replay_count": len(cases), "cases": cases, "diagnostic_work": work}


def summarize(rows):
    effects = [case["effect"] for row in rows for case in row["cases"]]
    keys = ("predicted_total_delta_job_s", "realized_total_delta_job_s", "cutoff_correction_delta_job_s",
            "current_arrival_residual_delta_job_s", "next_queue_residual_delta_job_s", "continuation_residual_delta_job_s")
    return {
        "substitutions": len(effects),
        "improved": sum(e["outcome"] == "improved" for e in effects),
        "worsened": sum(e["outcome"] == "worsened" for e in effects),
        "unchanged": sum(e["outcome"] == "unchanged" for e in effects),
        "mean_deltas_job_s": {key: mean(e[key] for e in effects) if effects else None for key in keys},
        "median_realized_delta_job_s": median(e["realized_total_delta_job_s"] for e in effects) if effects else None,
        "mean_flow_component_deltas_job_s": {key: mean(e["flow_component_deltas_job_s"][key] for e in effects) if effects else None for key in COMPONENTS},
    }


def build_diagnosis(output=DEFAULT_OUTPUT):
    _check_frozen_sources()
    snapshot = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != snapshot:
        raise ValueError("confirmed lookahead source changed")
    development = json.loads(DEVELOPMENT.read_text())
    base = development["protocol"]["base_protocol"]
    model = load_model()
    certificates = [capacity_certificate(model, load) for load in base["loads"]]
    if json.loads(json.dumps(model.snapshot())) != development["model"] or json.loads(json.dumps(certificates)) != development["certificates"]:
        raise ValueError("continuation diagnosis changed the registered model")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development mechanism diagnosis; no independent arrival seeds",
        "loads": base["loads"], "seeds": base["seeds"], "arrival_horizon_s": base["arrival_horizon_s"], "budget": 26,
        "terminal_service_rates": terminal_service_rates(model, certificates[0]),
        "source_policy": "frozen always-lookahead with weight 1 and 1% support cap",
        "state_roster": "every source frame where v4 and lookahead differ in departures, duration or sorted class completion offsets, including drain",
        "intervention": "same source prefix; force v4 versus lookahead once; both continue with v4 K26 to drain the identical original population",
        "prediction": "frozen J=A+EV; all delta signs lookahead-first minus v4-first, units job-seconds",
        "realized_target": "N times paired mean-flow difference; identical prefix contributions cancel",
        "area_partition": "current/future use each branch's own first-frame boundary, which may differ in absolute time",
        "cutoff": "h=min(tau,max(H-start,0)); truncated current arrival area Lambda*(tau*h-h*h/2); terminal Poisson mean lambda*h",
        "residual_identity": "delta F-delta J = cutoff correction + realized current arrival residual + realized next-queue residual + future area minus realized terminal value",
        "outcome_zero_tolerance_job_s": AREA_TOLERANCE,
        "diagnostic_work": "offline queue memoization: search and prediction counters cover selector cache misses; additional action audits and replay frames counted separately; forced replays are not new budgeted schedulers",
        "stopping_rule": "all substitutions on all nine source trajectories, once; no state filtering based on outcomes or policy tuning",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    originals = {(r["load"], r["seed"]): r for r in development["runs"] if r["budget"] == 26 and r["policy"] == "lookahead_cap_1pct"}
    tasks = [(model, cert, load, seed, protocol, originals[(load, seed)])
             for load, cert in zip(base["loads"], certificates, strict=True) for seed in base["seeds"]]
    with ProcessPoolExecutor(max_workers=protocol["execution_workers"]) as pool:
        runs = list(pool.map(_run_seed, tasks))
    checks = {
        "nine_frozen_source_trajectories_reproduced": all(r["source_reproduced"] for r in runs),
        "every_substitution_replayed": all(r["substitution_count"] == r["paired_replay_count"] for r in runs),
        "all_replays_pass_population_budget_drift_and_flow_checks": True,
        "paired_area_and_residual_identities_hold": True,
        "frozen_sources_unchanged": _source_texts() == snapshot,
    }
    report = {
        "experiment": "aligned_hardware_lookahead_single_substitution_continuation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "protocol": protocol, "checks": checks,
        "runs": runs,
        "summary_by_load": [{"load": load, **summarize([r for r in runs if r["load"] == load])} for load in base["loads"]],
        "summary_by_seed": [{"load": r["load"], "seed": r["seed"], **summarize([r])} for r in runs],
        "limitations": [
            "Interventions share development tapes and overlapping futures; cases are not independent replicates and their effects cannot be summed to infer whole-policy improvement.",
            "The continuation target includes subsequent arrivals and v4 decisions; its residual against V is not a pure terminal-function bias estimate.",
            "Only states visited by the frozen lookahead policy are studied; results need not apply to v4-only states or other continuation policies.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_diagnosis(args.output)
    print(json.dumps({"status": result["status"], "paired_cases": sum(r["paired_replay_count"] for r in result["runs"]), "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
