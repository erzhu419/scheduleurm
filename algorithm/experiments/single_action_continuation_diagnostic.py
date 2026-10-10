"""Same 64 root pairs, actual arrivals, but both suffixes continue with MW."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json

from algorithm.experiments.aligned_hardware_single_action_counterfactual import (
    ROOT, BASE as FIRST, path, saved_runs, make_planner, collect_work, paired_outcomes, PairedSelector, FEES,
)
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory
from algorithm.experiments.aligned_hardware_performance_necessity import load_models, replay_matches

BASE = ROOT / "md/experiment_artifacts/single_action_continuation_diagnostic_20260930"
SOURCE = "algorithm/experiments/single_action_continuation_diagnostic.py"


class MWContinuation:
    def __init__(self, planner, original_first):
        self.planner, self.original_first = planner, original_first
        self.calls, self.work, self.proposal_index = 0, Counter(), None

    def __call__(self, *context):
        if self.calls == 0:
            proposed = self.planner(*context)
            self.work.update({key: getattr(proposed, key) for key in FEES})
            self.work["proposal_calls"] += 1
            self.proposal_index = proposed.selection.option_indices[0]
            if self.original_first:
                self.calls += 1
                return proposed
        control = self.planner.maxweight(*context)
        self.work["root_option_probes"] += control.root_option_probes
        self.work["MW_calls"] += 1
        self.calls += 1
        return control


def build_protocol():
    prior = json.loads(path(FIRST, "_protocol.json").read_text())
    result = json.loads(FIRST.with_suffix(".json").read_text())
    if result["decision"] != "SINGLE_ACTION_DIAGNOSTIC_COMPLETE":
        raise ValueError("first paired diagnostic must be complete")
    for name, source in prior["execution_sources"].items():
        if (ROOT / name).read_text() != source:
            raise ValueError("frozen source changed: " + name)
    return {**prior, "experiment": "single_action_MW_continuation_diagnostic_v1",
        "first_registered_at_utc": prior["registered_at_utc"],
        "method": "Reuse all64 registered root targets, prefixes and literal arrivals. For each pair, original proposed first action versus same-state MW first action, both followed exclusively by MW. First proposal consumes the same captured RNG. Compare the sign of restricted mean differences with the original candidate-policy continuation diagnostic. No target selection or tuning by outcome.",
        "accounting": "Three factual replays plus128 MW suffixes, including128 first proposals. Prior phase costs remain separate and are not discarded. No extra null controls: previous resume engine and the three actual null controls remain frozen.",
        "execution_sources": {**prior["execution_sources"], SOURCE: (ROOT / SOURCE).read_text()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    expected.pop("registered_at_utc")
    if args.register:
        with path(BASE, "_protocol.json").open("x") as out:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()}, out, separators=(",", ":"), allow_nan=False)
        print("registered64 root pairs,128 MW suffixes", flush=True)
        return
    protocol = json.loads(path(BASE, "_protocol.json").read_text())
    if {key: value for key, value in protocol.items() if key != "registered_at_utc"} != expected:
        raise ValueError("registered sources or targets changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve previous outcome")
    first = json.loads(FIRST.with_suffix(".json").read_text())
    closed = {(r["tape_id"], r["frame_index"]): r for r in first["records"]}
    old = saved_runs()
    models = {m.node: m for m in load_models(protocol["models"])}
    records, work = [], Counter()
    for tape in protocol["tapes"]:
        model = models[tape["node"]]
        targets = [t for t in protocol["targets"] if t["tape_id"] == tape["tape_id"]]
        capture = PairedSelector(make_planner(model, tape, protocol), capture=[t["frame_index"] for t in targets])
        factual = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], protocol["environment_arrival_end"], protocol["observation_end"], capture, policy="arrival_rollout")
        if not replay_matches(factual, old[tape["tape_id"]]):
            raise ValueError("factual prefix reproduction failed")
        work.update(collect_work(capture, factual))
        for target in targets:
            index = target["frame_index"]
            checkpoint = prepare_checkpoint(factual, index)
            work.update({key: checkpoint[key] for key in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
            arms = []
            for original in (True, False):
                selector = MWContinuation(make_planner(model, tape, protocol), original)
                selector.planner.rng.setstate(capture.states[index])
                work["checkpoint_rng_words_restored"] += len(capture.states[index][1])
                arm = resume_single_action_trajectory(model, protocol["environment_arrival_end"], protocol["observation_end"], selector, checkpoint, policy="MW_suffix")
                if selector.proposal_index != target["factual_option_index"]:
                    raise ValueError("root context changed")
                work.update(collect_work(selector, arm))
                arms.append(arm)
            effects = paired_outcomes(*arms, protocol, model.workloads.index("gpu_llm_distilgpt2"))
            earlier = closed[target["tape_id"], index]
            records.append({**target, **effects,
                "candidate_continuation_delta_T_mean": earlier["delta_T_mean"],
                "candidate_continuation_delta_LLM_T_mean": earlier["delta_LLM_T_mean"],
                "mean_ranking_flipped": effects["delta_T_mean"] * earlier["delta_T_mean"] < 0,
                "LLM_mean_ranking_flipped": effects["delta_LLM_T_mean"] * earlier["delta_LLM_T_mean"] < 0})
    if len(records) != len(protocol["targets"]):
        raise ValueError("all64 pairs required")
    result = {"decision": "CONTINUATION_DIAGNOSTIC_COMPLETE", "registered_at_utc": protocol["registered_at_utc"],
        "scope": "Retrospective paired continuation mechanism, actual future in offline harness only. Dependent interventions, no deployable rule or independent confirmation.",
        "records": records, "pairs": len(records), "factual_replays": 3, "suffixes": 2 * len(records),
        "mean_ranking_flips": sum(r["mean_ranking_flipped"] for r in records),
        "LLM_mean_ranking_flips": sum(r["LLM_mean_ranking_flipped"] for r in records),
        "actual_work": dict(work), "prior_phase_actual_work": first["actual_work"]}
    with BASE.with_suffix(".json").open("x") as out:
        json.dump(result, out, separators=(",", ":"), allow_nan=False)
    print(json.dumps({k: result[k] for k in ("decision", "pairs", "mean_ranking_flips", "LLM_mean_ranking_flips")}), flush=True)


if __name__ == "__main__":
    main()
