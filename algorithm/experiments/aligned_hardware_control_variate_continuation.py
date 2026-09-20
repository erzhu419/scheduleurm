"""Diagnose actual raw/CV disagreements with fixed-action paired continuations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from algorithm.control_variate_rollout_frame_candidates import select_control_variate_rollout_frame
from algorithm.experiments.aligned_hardware_control_variate_diagnostic import (
    _WORK_FIELDS, _empty_work, _sample_summary,
)
from algorithm.experiments.aligned_hardware_frame_mechanism import action_description
from algorithm.experiments.aligned_hardware_frame_model import capacity_certificate, load_model
from algorithm.experiments.aligned_hardware_lookahead import _check_frozen_sources, terminal_service_rates
from algorithm.experiments.aligned_hardware_lookahead_confirmation import _source_texts
from algorithm.experiments.aligned_hardware_waiting_cost import ARTIFACTS


DEFAULT_OUTPUT = ARTIFACTS / "aligned_hardware_control_variate_continuation_development_20260910.json"
DEVELOPMENT = ARTIFACTS / "aligned_hardware_control_variate_development_20260910.json"
SOURCE_POLICIES = ("poisson_rollout_cap_1pct", "control_variate_rollout_cap_1pct")
SAMPLES = 64
NAMESPACE = "aligned-cv-continuation:{sample_index}:{coordinate}"


def state_key(load, queues):
    return f"{load}:" + ",".join(map(str, queues))


def physical(description):
    return (tuple(description["departures"]), description["duration_s"],
            tuple(tuple(v) for v in description["completion_offsets"]))


def sample_state(model, certificate, load, mu, case):
    """Fresh paths for two pre-sampling fixed contenders; no real tape access."""
    queues = case["queue"]
    allowed = {physical(case[name]) for name in ("incumbent_action", "challenger_action")}
    work, samples = dict(_empty_work(), action_descriptions=0, action_description_completion_offset_reads=0), []
    for sample_index in range(SAMPLES):
        started = perf_counter()
        rngs = tuple(random.Random(NAMESPACE.format(sample_index=sample_index, coordinate=i))
                     for i in range(len(queues)))
        work["rng_setup_wall_s"] += perf_counter() - started
        started = perf_counter()
        selection = select_control_variate_rollout_frame(
            model.options, queues, model.lanes, 26,
            arrival_rates=certificate["arrival_rates"], terminal_service_rates=mu,
            arrival_rngs=rngs,
        )
        work["selector_wall_s"] += perf_counter() - started
        work["selector_calls"] += 1
        work["rng_initializations"] += len(rngs)
        for name, attribute in _WORK_FIELDS.items():
            work[name] += getattr(selection, attribute)
        description = action_description(model, selection)
        work["action_descriptions"] += 1
        work["action_description_completion_offset_reads"] += sum(map(len, description["completion_offsets"]))
        if not selection.rollout_used or physical(description) not in allowed:
            raise ValueError("repeated sample changed the fixed contender pair")
        raw, corrected = selection.raw_rollout_margin, selection.corrected_rollout_margin
        control = selection.analytic_challenger_sampled_control_value - selection.incumbent_sampled_control_value
        analytic = selection.root_lookahead_cost - selection.support_incumbent_lookahead_cost
        error = corrected - (raw - control + analytic)
        if (abs(error) > 1e-8 or selection.attempts > 26
                or selection.rollout_random_draws != selection.rollout_arrival_event_generations + sum(r > 0 for r in certificate["arrival_rates"])):
            raise ValueError("fresh diagnostic violated cost identity, budget or draw accounting")
        samples.append({"sample_index": sample_index, "raw_margin": raw,
                        "corrected_margin": corrected, "identity_error": error})
    summary = _sample_summary(samples)
    for label in ("raw", "corrected"):
        summary[f"{label}_margin_mc_se"] = sqrt(summary[f"{label}_margin_variance"] / SAMPLES)
    return {"state_id": state_key(load, queues), "load": load, "queue": queues,
            "incumbent_action": case["incumbent_action"], "challenger_action": case["challenger_action"],
            **summary, "samples": samples, "work": work}


def compare_estimates(case, state):
    """Oriented observed residual = fresh-mean minus sample + target minus mean."""
    orientation = case["orientation_cv_minus_raw"]
    realized = case["effect"]["realized_total_delta_job_s"]
    results = {}
    for label, sample_key in (("raw", "raw_rollout_margin"), ("corrected", "corrected_rollout_margin")):
        sample = orientation * case[sample_key]
        proxy = orientation * state[f"{label}_margin_mean"]
        se = state[f"{label}_margin_mc_se"]
        sample_adjustment, target_mismatch = proxy - sample, realized - proxy
        error = sample_adjustment + target_mismatch - (realized - sample)
        if abs(error) > 1e-6:
            raise ValueError("sample/proxy/realized residual partition failed")
        results[label] = {
            "discovery_sample_delta_job_s": sample, "fresh_proxy_mean_delta_job_s": proxy,
            "fresh_proxy_mean_mc_se_job_s": se, "fresh_mean_minus_discovery_sample_job_s": sample_adjustment,
            "realized_minus_fresh_proxy_mean_job_s": target_mismatch,
            "residual_identity_error_job_s": error,
            "proxy_preference": "unresolved" if abs(proxy) <= 2 * se else "cv" if proxy < 0 else "raw",
            "proxy_point_estimate_disagrees_with_realized": proxy * realized < 0,
        }
    return results


def summarize(rows):
    cases = [c for r in rows for c in r["cases"]]
    output = {"source_count": len(rows), "disagreements": len(cases),
              "unique_states": len({c["state_id"] for c in cases}),
              **{outcome: sum(c["effect"]["outcome"] == outcome for c in cases)
                 for outcome in ("improved", "worsened", "unchanged")},
              "mean_realized_delta_job_s": mean(c["effect"]["realized_total_delta_job_s"] for c in cases) if cases else None}
    for label in ("raw", "corrected"):
        group = [c["estimate_comparison"][label] for c in cases]
        output[label] = {
            "proxy_preference_counts": {name: sum(g["proxy_preference"] == name for g in group)
                                        for name in ("cv", "raw", "unresolved")},
            "proxy_point_estimate_disagrees_with_realized": sum(g["proxy_point_estimate_disagrees_with_realized"] for g in group),
            "mean_absolute_fresh_mean_minus_discovery_sample_job_s": mean(abs(g["fresh_mean_minus_discovery_sample_job_s"]) for g in group) if group else None,
            "mean_absolute_realized_minus_fresh_proxy_mean_job_s": mean(abs(g["realized_minus_fresh_proxy_mean_job_s"]) for g in group) if group else None,
        }
    return output


def build_diagnosis(output=DEFAULT_OUTPUT):
    from algorithm.experiments.aligned_control_variate_continuation_replay import run_source

    _check_frozen_sources()
    frozen = json.loads((ARTIFACTS / "aligned_hardware_lookahead_confirmation_20260910/source_snapshot.json").read_text())
    if _source_texts() != frozen:
        raise ValueError("confirmed sources changed")
    previous = json.loads(DEVELOPMENT.read_text())
    base = previous["protocol"]["base_protocol"]
    model = load_model()
    certificates = {load: capacity_certificate(model, load) for load in base["loads"]}
    mu = terminal_service_rates(model, certificates[base["loads"][0]])
    if (json.loads(json.dumps(model.snapshot())) != previous["model"]
            or json.loads(json.dumps(list(certificates.values()))) != previous["certificates"]
            or list(mu) != previous["protocol"]["terminal_service_rates"]):
        raise ValueError("diagnostic changed the frozen model or rates")
    protocol = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "development mechanism diagnosis using consumed seeds",
        "loads": base["loads"], "seeds": base["seeds"], "planning_seeds": [0, 1, 2],
        "source_policies": SOURCE_POLICIES, "budget": 26,
        "arrival_horizon_s": base["arrival_horizon_s"], "terminal_service_rates": mu,
        "source_comparison": "54 original trajectories; native RNG and counters unchanged; on active frames compare opposite selector with cloned pre-call RNG states",
        "state_roster": "all physical action disagreements in every source frame, including drain; same service/duration/sorted class offsets means same action",
        "intervention": "identical prefix, raw-first versus CV-first once, then common frozen v4 K26 on the original arrival tape until all original jobs complete",
        "sign": "all intervention deltas CV-first minus raw-first; fixed sampled margins challenger minus incumbent before orientation",
        "area_partition": "current/future use each action's own frame boundary",
        "repeated_sampling": {"samples_per_unique_load_queue_pair": SAMPLES, "rng_namespace": NAMESPACE,
                              "reuse": "one fixed contender-pair estimate per load/queue, shared by occurrences; sample ids crossed across states",
                              "estimate": "raw and corrected fixed-contender means, variances and Monte Carlo standard errors; no selected-minimum margin",
                              "descriptive_preference": "unresolved when absolute mean <= two MC standard errors; no performance gate"},
        "residual_partition": "realized minus discovery estimate = fresh proxy mean minus discovery estimate + realized minus fresh proxy mean",
        "diagnostic_cost": "native source work, shadow searches and RNG copies, cached continuation searches, replay frames, and every fresh-sample search/draw reported separately; none added to online K",
        "stopping_rule": "all 54 sources and all disagreements once, then 64 fresh samples for every unique load/queue; no outcome-based filtering, tuning or extra seeds",
        "execution_workers": 3,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    originals = {(r["load"], r["seed"], r["planning_seed"], r["policy"]): r
                 for r in previous["runs"] if r["budget"] == 26 and r["policy"] in SOURCE_POLICIES}
    tasks = [(model, certificates[load], load, seed, planning_seed, policy, protocol,
              originals[(load, seed, planning_seed, policy)])
             for load in base["loads"] for seed in base["seeds"] for planning_seed in (0, 1, 2) for policy in SOURCE_POLICIES]
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        runs = list(pool.map(run_source, tasks))
    states = {}
    for row in runs:
        for case in row["cases"]:
            key = state_key(row["load"], case["queue"])
            if key not in states:
                states[key] = sample_state(model, certificates[row["load"]], row["load"], mu, case)
            else:
                for name in ("incumbent_action", "challenger_action"):
                    if physical(case[name]) != physical(states[key][name]):
                        raise ValueError("one load/queue produced different pre-sampling contenders")
            case["state_id"] = key
            case["estimate_comparison"] = compare_estimates(case, states[key])
    checks = {
        "all_54_frozen_sources_reproduced": len(runs) == 54 and all(r["source_reproduced"] for r in runs),
        "all_actual_disagreements_replayed": all(r["substitution_count"] == r["paired_replay_count"] for r in runs),
        "population_budget_drift_flow_and_rng_pairing_hold": True,
        "fresh_sample_and_residual_identities_hold": True,
        "every_unique_state_has_64_fresh_samples": all(s["sample_count"] == SAMPLES for s in states.values()),
    }
    work = {key: sum(state["work"][key] for state in states.values())
            for key in (*_empty_work(), "action_descriptions", "action_description_completion_offset_reads")}
    report = {
        "experiment": "aligned_hardware_control_variate_paired_continuation_v1",
        "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protocol": protocol,
        "runs": runs, "states": list(states.values()), "repeated_sampling_work": work,
        "summary": summarize(runs),
        "summary_by_load": [{"load": load, **summarize([r for r in runs if r["load"] == load])} for load in base["loads"]],
        "summary_by_load_and_source": [{"load": load, "source_policy": policy,
                                        **summarize([r for r in runs if r["load"] == load and r["source_policy"] == policy])}
                                       for load in base["loads"] for policy in SOURCE_POLICIES],
        "wall_s": perf_counter() - started,
        "limitations": [
            "Disagreement selection conditions discovery margins. Fresh samples estimate the proxy expectation only on these visited states; 64 samples may leave its sign unresolved.",
            "Realized minus proxy mean mixes arrival-tape noise, horizon cutoff, and two-step/fluid versus v4 continuation. It is not an identified pure model bias.",
            "Sources reuse development tapes and cases have overlapping futures. Counts and means are descriptive, not independent replicates or additive whole-policy gains.",
            "Both raw and corrected estimates have the same fixed-candidate expectation; neither variance reduction nor agreement with continuation is required for execution PASS.",
        ],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_diagnosis(args.output)
    print(json.dumps({"status": result["status"], "summary": result["summary"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
