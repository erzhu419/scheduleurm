"""Label both unmatched native roots using the frozen OLD-current suffix."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import json
from math import inf, nextafter
from pathlib import Path
from statistics import mean
from time import perf_counter

from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepBGIncumbentOriginalRiskLaneRollout
from algorithm.experiments import aligned_hardware_current_policy_continuation_alignment as alignment
from algorithm.experiments import aligned_hardware_two_step_current_policy_root_ranking as ranking
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_bg_incumbent_counterfactual import CONTEXT, same
from algorithm.experiments.aligned_hardware_two_step_bg_incumbent_original_risk_development import check_rollout
from algorithm.experiments.single_action_counterfactual_replay import prepare_checkpoint, resume_single_action_trajectory
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    TwoStepContinuationCrossSelector, collect_work, reconstruct_prefix_states, restore_planner_state,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_current_policy_third_root_labels_20261002"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def dump(file, data):
    file.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def register(directory):
    prior = alignment.load_protocol(directory)
    result = json.loads(alignment.path(directory, "_summary.json").read_text())
    records = json.loads(alignment.path(directory, "_records_summary.json").read_text())
    if (not result["execution_checks_passed"] or result["summary"]["available_pairs"] != 22
            or result["registered_at_utc"] != prior["registered_at_utc"]
            or records["registered_at_utc"] != prior["registered_at_utc"]):
        raise ValueError("completed registered22 continuation alignment required")
    rows = records["records"]
    if [(r["tape_id"], r["frame_index"]) for r in rows] != [
            (r["tape_id"], r["frame_index"]) for r in prior["anchors"]]:
        raise ValueError("keep every frozen anchor in its original order")
    targets = [r for r in rows if r["native_choice_kind"] == "third"]
    if len(targets) != 2 or any(r["baseline_pair_scores"]["pair_preference"] != "A" for r in targets):
        raise ValueError("both actual unmatched roots must retain their frozen A baseline")
    factual = ranking.saved_runs(prior)
    clocks = {(factual[r["tape_id"]]["arrival_horizon"], factual[r["tape_id"]]["observation_end"])
        for r in targets}
    if len(clocks) != 1:
        raise ValueError("both cached factual clocks must match")
    environment_end, observation_end = clocks.pop()
    protocol = {key: prior[key] for key in ("models", "tapes", "anchors", "conditional_records",
        "previous_trajectories", "planning_samples", "planning_event_cap", "birth_cohort", "full24_parent_decision")}
    protocol.update(environment_arrival_end=environment_end,
        arrival_horizon=prior["birth_cohort"][1], observation_end=observation_end)
    protocol.update(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        alignment_registered_at_utc=prior["registered_at_utc"], targets=targets,
        selections=[{key: r[key] for key in ("tape_id", "frame_index", "node", "load", "seed",
            "native_choice_kind", "new_native_option_index", "baseline_pair_scores")} for r in rows],
        expected_new_suffix_executions=4,
        method="Only both actual third roots. Reconstruct cached causal prefix state. Execute one null and one forced-third suffix per target, each consuming the complete OLD-current first proposal. Force only the first physical action, freely replan with OLD-current afterwards. Null must reproduce full jobs/actions/clock/rollout and final RNG/counter. Use exact reuse for the same native decisions and charge its logical work; save only labels, not suffix trajectories.",
        interpretation="22 conditional actual-selection effects on reused development states, not an additive closed-loop gain. Mean and tails share the full fixed birth cohort. Cached20 P90 uses prior four-arm common IDs; two new third labels use their paired common IDs. Report support separately; these are averages of conditional P90 differences, not pooled P90. Retain full24 HOLD and sealed confirmation.",
        execution_sources={**prior["execution_sources"],
            "algorithm/experiments/aligned_hardware_background_root_counterfactual.py":
                (ROOT / "algorithm/experiments/aligned_hardware_background_root_counterfactual.py").read_text(),
            SOURCE: (ROOT / SOURCE).read_text()})
    dump(path(directory, "_protocol.json"), protocol)
    print(json.dumps({"targets": len(targets), "new_suffix_executions": 4,
        "registered_at_utc": protocol["registered_at_utc"]}), flush=True)


def load_protocol(directory):
    protocol = json.loads(path(directory, "_protocol.json").read_text())
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered source changed: " + name)
    return protocol


class FirstRootSelector(TwoStepContinuationCrossSelector):
    """Retain native proposal metadata; the offline first action has no forecast."""
    def __init__(self, planner, root=None):
        super().__init__(planner, first_root="original", suffix="current")
        self.root = root

    def __call__(self, *context):
        proposed = super().__call__(*context)
        if self.calls != 1 or self.root is None:
            return proposed
        root = self.root
        queue = context[1]
        if any(d > q for d, q in zip(self.planner.requirements[root], queue, strict=True)):
            raise ValueError("registered third root is infeasible")
        control = self.planner.maxweight(*context)
        self.work["override_root_option_probes"] += control.root_option_probes
        self.work["override_calls"] += 1
        mw, = control.selection.option_indices
        support = Fraction(sum(q * d for q, d in zip(queue, self.planner.requirements[root], strict=True))) / self.planner.durations[root]
        best = Fraction(sum(q * d for q, d in zip(queue, control.selection.service, strict=True))) / self.planner.durations[mw]
        exact = support == best
        upper = float(support) if exact else control.selection.upper_bound
        if not exact and Fraction.from_float(upper) < best:
            upper = nextafter(upper, inf)
        selected = replace(control.selection, option_counts=tuple(int(i == root) for i in range(len(self.planner.options))),
            option_indices=(root,), service=self.planner.requirements[root], duration=float(self.planner.durations[root]),
            score=float(support), upper_bound=upper, gap=max(0., upper - float(support)),
            exact=False, score_exact=exact, canonical_exact=False)
        self.intervention.update(actual_applied_option_index=root, replacement=root, forced_single_action=True,
            retained_feasibility_metadata_scope="native OLD-current proposal; forced action has no risk/forecast claim")
        applied = {"changed": root != mw, "improved_incumbent": root != proposed.incumbent_root_index,
            "risk_changed_unconstrained_root": root != proposed.unconstrained_root_index}
        for key, value in applied.items():
            self.status["applied_" + key] += int(value) - int(getattr(proposed, key))
        return replace(proposed, selection=selected, changed=root != mw,
            improved_incumbent=root != proposed.incumbent_root_index,
            risk_changed_unconstrained_root=applied["risk_changed_unconstrained_root"],
            chosen_predicted_completion_s=None, chosen_predicted_squared_flow_s2=None,
            incumbent_actual_predicted_squared_flow_s2=None)


def make_planner(model, tape, protocol):
    return ExactReuseTwoStepBGIncumbentOriginalRiskLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online", protocol["planning_samples"], protocol["planning_event_cap"])


def run_shard(index, directory):
    started = perf_counter()
    protocol = load_protocol(directory)
    target = protocol["targets"][index]
    key = target["tape_id"]
    tape = next(t for t in protocol["tapes"] if t["tape_id"] == key)
    model = next(m for m in ranking.load_models(protocol["models"]) if m.node == tape["node"])
    factual = ranking.saved_runs(protocol)[key]
    anchor = next(a for a in protocol["anchors"] if a["tape_id"] == key)
    frame_index = target["frame_index"]
    frame = factual["frames"][frame_index]
    planner = make_planner(model, tape, protocol)
    work = Counter(controller_constructors=1, recourse_controller_constructors=1)
    work.update({"preprocessing_" + k: v for k, v in planner.preprocessing_work.items()})
    states = reconstruct_prefix_states(planner, factual, [frame_index], work)
    final_rng, final_counter = planner.rng.getstate(), planner.decision_counter
    work["reconstruction_rng_words_copied"] += len(final_rng[1])
    work["reconstruction_counter_reads"] += 1
    checkpoint = prepare_checkpoint(factual, frame_index)
    work.update({k: checkpoint[k] for k in ("checkpoint_job_reads", "checkpoint_prefix_service_reads")})
    checks, status, reuse = {}, Counter(), Counter()
    for arm, root in (("null", None), ("third", target["new_native_option_index"])):
        selector = FirstRootSelector(make_planner(model, tape, protocol), root)
        work.update(restore_planner_state(selector.planner, states[frame_index]))

        def select(*context):
            decision = selector(*context)
            if selector.calls % 100 == 0:
                print(key, arm, "suffix dispatch", selector.calls, "service time", context[3], flush=True)
            return decision

        resumed = resume_single_action_trajectory(model, protocol["environment_arrival_end"],
            protocol["observation_end"], select, checkpoint, policy=arm)
        proposed = selector.first_proposal
        if (not same(proposed["rollout"], frame["rollout"])
                or proposed["original"] != anchor["factual_option_index"]
                or proposed["replacement"] != anchor["replacement_option_index"]
                or not all(same(resumed["frames"][0][k], frame[k]) for k in CONTEXT)
                or resumed["frames"][0]["option_indices"][0] != (root if root is not None else anchor["factual_option_index"])
                or selector.work["override_calls"] != int(root is not None)
                or not check_rollout({**resumed, "frames": resumed["frames"][1:]}, model, protocol, selector.planner)):
            raise ValueError("first proposal/context/action or native suffix differs: " + key + ":" + arm)
        if arm == "null":
            combined = {**resumed, "frames": factual["frames"][:frame_index] + resumed["frames"]}
            if (not ranking.same(combined["jobs"], factual["jobs"])
                    or not ranking.same(combined["frames"], factual["frames"])
                    or combined["clock"] != factual["clock"] or combined["last_frame_end"] != factual["last_frame_end"]
                    or selector.planner.rng.getstate() != final_rng or selector.planner.decision_counter != final_counter):
                raise ValueError("null jobs/actions/clock/rollout/RNG/counter differ: " + key)
        else:
            outcomes = paired_outcomes(factual, resumed, protocol, model.workloads)
        work.update(collect_work(selector, resumed))
        status.update(selector.status)
        reuse.update({"mw_" + k: v for k, v in selector.planner._mw_reuse.work.items()})
        reuse.update({"inner_BG_kernel_" + k: v for k, v in selector.planner.inner_planner.kernel_work.items()})
        checks[arm] = {"first_proposal_context_match_passed": True, "native_suffix_checks_passed": True,
            "null_reproduction_passed": arm == "null", "calls": selector.calls}
        print(key, arm, "complete", flush=True)
    record = {k: target[k] for k in ("tape_id", "frame_index", "node", "load", "seed", "new_native_option_index")}
    record.update(outcomes=outcomes, checks=checks)
    row = dict(index=index, registered_at_utc=protocol["registered_at_utc"], record=record,
        work=dict(work), status=dict(status), executed_reuse_work=dict(reuse), new_suffix_executions=2,
        wall_s=perf_counter() - started)
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps({"index": index, "checks_passed": True, "wall_s": row["wall_s"]}), flush=True)


def summarize(records):
    return {"states": len(records), "metrics": {metric: {
        "available": sum(r["conditional_metrics"][metric] is not None for r in records),
        "mean": mean(values) if (values := [r["conditional_metrics"][metric] for r in records
            if r["conditional_metrics"][metric] is not None]) else None,
        "helped": sum(v < 0 for v in values), "harmed": sum(v > 0 for v in values),
        "unchanged": sum(v == 0 for v in values)} for metric in METRICS}}


def merge(directory):
    protocol = load_protocol(directory)
    shards = [json.loads(path(directory, f"_shard_{i:02d}.json").read_text()) for i in range(2)]
    third, work = {}, Counter()
    for i, shard in enumerate(shards):
        if (shard["index"] != i or shard["registered_at_utc"] != protocol["registered_at_utc"]
                or any(shard["record"][k] != protocol["targets"][i][k] for k in ("tape_id", "frame_index", "new_native_option_index"))):
            raise ValueError("registered target shard differs")
        third[shard["record"]["tape_id"]] = shard["record"]
        work.update(shard["work"])
    records = []
    for selection, old in zip(protocol["selections"], protocol["conditional_records"], strict=True):
        if (selection["tape_id"], selection["frame_index"]) != (old["tape_id"], old["frame_index"]):
            raise ValueError("cached22 label order differs")
        if selection["native_choice_kind"] == "third":
            outcome = third[selection["tape_id"]]["outcomes"]
            metrics = {k: outcome[k] for k in METRICS}
            supports = outcome["snapshots"]["observation_end"]
            support = {k: {f: v[f] for f in ("arrivals", "common_jobs", "unfinished_jobs")} for k, v in supports.items()}
        else:
            factor = int(selection["native_choice_kind"] == "A") - int(selection["baseline_pair_scores"]["pair_preference"] == "A")
            metrics = {k: (v * factor if v is not None else None)
                for k, v in old["outcomes"]["root_effects"]["current"].items() if k in METRICS}
            supports = old["outcomes"]["snapshots"]["observation_end"]
            support = {k: {f: supports[k][f] for f in ("arrivals", "common_jobs", "unfinished_jobs")}
                for k in ("original_current", "BG_current")}
        records.append({**selection, "conditional_metrics": metrics,
            "p90_support": support, "p90_common_population_is_full_cohort": all(v["arrivals"] == v["common_jobs"] for v in support.values())})
    result = dict(decision="CURRENT_POLICY_THIRD_ROOT_LABELS_COMPLETE", execution_checks_passed=True,
        performance_gain_established=False, registered_at_utc=protocol["registered_at_utc"],
        full24_parent_decision=protocol["full24_parent_decision"], new_suffix_executions=sum(s["new_suffix_executions"] for s in shards),
        interpretation=protocol["interpretation"], summary=summarize(records),
        seeds={str(seed): summarize([r for r in records if r["seed"] == seed]) for seed in (41, 42, 43)},
        full_cohort_P90_support_states=sum(r["p90_common_population_is_full_cohort"] for r in records),
        native_third_roots_without_labels=0, third_root_records=list(third.values()), records=records, work=dict(work))
    dump(path(directory, ".json"), result)
    dump(path(directory, "_summary.json"), {k: v for k, v in result.items() if k not in ("records", "work")})
    dump(path(directory, "_records_summary.json"), {"registered_at_utc": protocol["registered_at_utc"], "records": records})
    print(json.dumps(result["summary"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--run-shard", type=int, choices=(0, 1))
    mode.add_argument("--merge", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir)
    elif args.run_shard is not None:
        run_shard(args.run_shard, args.artifact_dir)
    else:
        merge(args.artifact_dir)


if __name__ == "__main__":
    main()
