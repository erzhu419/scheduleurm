"""Rank the frozen A/BG pairs with current-policy continuation scores."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from algorithm.experiments.bg_incumbent_counterfactual_replay import FEES, restore_planner_state
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.two_step_bg_incumbent_continuation_cross_replay import (
    reconstruct_prefix_states, registered_anchors,
)
from algorithm.two_step_bg_incumbent_background_redecision_lane_rollout import TWO_STEP_WORK
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    INNER_CURRENT_WORK, TwoStepCurrentPolicyBGIncumbentLaneRollout,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_current_policy_root_ranking_20261001"
CROSS = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_bg_incumbent_continuation_cross_20261001"
CANDIDATE = ROOT / "md/experiment_artifacts/aligned_hardware_two_step_current_policy_bg_incumbent_development_20261001"
SOURCE = "algorithm/experiments/aligned_hardware_two_step_current_policy_root_ranking.py"
OLD_PREDICTIONS = (
    "incumbent_mw_predicted_completion_s", "incumbent_predicted_completion_s",
    "incumbent_mw_predicted_squared_flow_s2", "incumbent_predicted_squared_flow_s2",
)
SECONDS_TOLERANCE = 1e-9


def same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def artifact(base, suffix):
    return base.with_name(base.name + suffix)


def saved_runs(protocol):
    with gzip.open(ROOT / protocol["previous_trajectories"], "rt") as source:
        return {row["tape_id"]: row["policies"]["arrival_rollout"]
                for row in json.load(source)["runs"]}


def build_protocol():
    cross_protocol = json.loads(artifact(CROSS, "_protocol.json").read_text())
    cross = json.loads(CROSS.with_suffix(".json").read_text())
    candidate = json.loads(artifact(CANDIDATE, "_protocol.json").read_text())
    if (cross["decision"] != "TWO_STEP_BG_INCUMBENT_CONTINUATION_CROSS_COMPLETE"
            or not cross["execution_checks_passed"]
            or cross["registered_at_utc"] != cross_protocol["registered_at_utc"]
            or cross["anchors"] != 22
            or not same(cross["missing_tape_ids"], cross_protocol["missing_tape_ids"])):
        raise ValueError("matching completed frozen22-anchor continuation diagnostic required")
    sources = {**cross_protocol["execution_sources"], **candidate["execution_sources"]}
    for snapshots in (cross_protocol["execution_sources"], candidate["execution_sources"]):
        for name, source in snapshots.items():
            if (ROOT / name).read_text() != source:
                raise ValueError("frozen source changed: " + name)
    for key in ("models", "tapes", "planning_samples", "planning_event_cap", "planning_seed_rule", "birth_cohort"):
        if not same(cross_protocol[key], candidate[key]):
            raise ValueError("candidate and cached diagnostic inputs differ: " + key)
    selected = registered_anchors(cross_protocol["tapes"], saved_runs(cross_protocol))
    if (not same(selected["anchors"], cross_protocol["anchors"])
            or not same(selected["missing_tape_ids"], cross_protocol["missing_tape_ids"])):
        raise ValueError("retain the exact22 anchors and two missing tapes")
    by_key = {(row["tape_id"], row["frame_index"]): row for row in cross["records"]}
    records = [by_key[(anchor["tape_id"], anchor["frame_index"])] for anchor in selected["anchors"]]
    if len(by_key) != len(records) or any(
            not same({key: row[key] for key in anchor}, anchor)
            for row, anchor in zip(records, selected["anchors"], strict=True)):
        raise ValueError("one matching cached conditional label per anchor required")
    return {
        "experiment": "two_step_current_policy_fixed_pair_ranking_v1",
        "cross_protocol": str(artifact(CROSS, "_protocol.json").relative_to(ROOT)),
        "cross_result": str(CROSS.with_suffix(".json").relative_to(ROOT)),
        "cross_registered_at_utc": cross_protocol["registered_at_utc"],
        "candidate_protocol": str(artifact(CANDIDATE, "_protocol.json").relative_to(ROOT)),
        "candidate_registered_at_utc": candidate["registered_at_utc"],
        **{key: cross_protocol[key] for key in (
            "models", "tapes", "planning_samples", "planning_event_cap", "planning_seed_rule",
            "birth_cohort", "clock", "previous_trajectories")},
        **selected, "conditional_records": records, "workers": 8,
        "expected_native_proposals": 22, "expected_new_trajectories": 0,
        "near_zero_seconds_descriptive": SECONDS_TOLERANCE,
        "method": "Reuse all22 frozen original/BG action pairs and both missing tapes. Construct one recording new-candidate planner per24 tapes; reconstruct only outer prefix draws through every cached frame while verifying released counts, rates and canonical FIFO ages. Capture each anchor's pre-call state and the following cached call's pre-state as the expected post-prefix RNG. Restore the anchor state and execute one complete new native proposal. Record only the returned new scoring population. Require its original ordinary-BG incumbent/MW roots, four original-stage predictions and original-risk feasible counts to match cache, and its post-prefix RNG to match reconstructed prefix state. Compare native exact-Fraction S1(A)-S1(BG), with strict A improvement/BG ties, against saved full-cohort A-BG restricted mean under the frozen old-current suffix. Preserve new S2 and every native chosen root; third roots receive no borrowed A/BG label. Overflow scores stay unavailable.",
        "budget": "Exactly22 complete native proposals, no action execution or new trajectories. Per native planner retain8 samples/cap128. Charge all24 outer/inner-current/nested-BG constructions and preprocessing, all prefix draws/context/FIFO/state work, each proposal's main/incumbent/recourse/TWO primitive fields and inner-current ledger once. No duplicate selector aliases or ordinary-BG comparison fees at the current-policy layer.",
        "scope": "Fixed-pair ranking alignment on reused development states and frozen OLD-current continuation outcomes. Tagged-prefix S1 and full-cohort empirical mean are different targets. New native third-root choices and its ongoing closed loop have no labels here. This diagnostic does not stop or adopt the parallel full24 policy experiment; keep its complete MW/BG performance gates, all prior HOLDs and sealed confirmation unchanged.",
        "execution_sources": {**sources, SOURCE: (ROOT / SOURCE).read_text()},
    }


class RecordingPolicy(TwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Capture the returned native scores without changing their computation."""
    def __init__(self, *args):
        super().__init__(*args)
        self.recorded_scores = self.recorded_post_prefix_state = None
        self.recording_work = Counter()

    def __call__(self, *context):
        self.recorded_scores = self.recorded_post_prefix_state = None
        return super().__call__(*context)

    def score_background_roots(self, *context):
        scored = super().score_background_roots(*context)
        self.recorded_scores = scored
        self.recorded_post_prefix_state = {
            "rng_state": self.rng.getstate(), "decision_counter": self.decision_counter}
        self.recording_work["recorded_post_prefix_rng_words_copied"] += len(
            self.recorded_post_prefix_state["rng_state"][1])
        self.recording_work["recorded_post_prefix_counter_reads"] += 1
        return scored


def proposal_work(decision):
    work = Counter({key: getattr(decision, key) for key in (*FEES, *TWO_STEP_WORK)})
    work.update({"inner_current_" + key: decision.inner_current_work[key] for key in INNER_CURRENT_WORK})
    return work


def score_anchor(planner, frame, anchor, old_record, state, expected_post_rng, work):
    work.update(restore_planner_state(planner, state))
    ages = tuple(tuple(group) for group in frame["waiting_ages"])
    work["proposal_context_age_reads"] += sum(map(len, ages))
    work["proposal_context_coordinate_reads"] += len(frame["observed_arrival_counts"])
    work["proposal_context_lane_reads"] += len(frame["available_before"])
    decision = planner(planner.options, tuple(frame["queue"]), tuple(frame["available_before"]),
        frame["start"], frame["lane_ids"][0], ages, tuple(frame["observed_arrival_counts"]))
    work.update(proposal_work(decision))
    work.update(planner.recording_work)
    planner.recording_work.clear()
    post_rng = planner.rng.getstate()
    post_counter = planner.decision_counter
    work["proposal_post_rng_words_copied"] += len(post_rng[1])
    work["proposal_post_counter_reads"] += 1
    info = frame["rollout"]
    if (decision.incumbent_root_index != anchor["replacement_option_index"]
            or decision.incumbent_maxweight_root_index != anchor["maxweight_option_index"]
            or any(getattr(decision, key) != info[key] for key in OLD_PREDICTIONS)
            or decision.incumbent_candidate_count != info["incumbent_candidate_count"]
            or decision.incumbent_risk_feasible_candidates != info["incumbent_risk_feasible_candidates"]
            or decision.incumbent_risk_filtered_candidates != info["incumbent_risk_filtered_candidates"]
            or info["incumbent_actual_predicted_squared_flow_s2"] > info["incumbent_mw_predicted_squared_flow_s2"]
            or post_counter != state["decision_counter"] + 1
            or post_rng != expected_post_rng
            or (planner.recorded_post_prefix_state is not None and (
                planner.recorded_post_prefix_state["rng_state"] != expected_post_rng
                or planner.recorded_post_prefix_state["decision_counter"] != state["decision_counter"]))):
        raise ValueError("native original stage or post-prefix state differs: " + anchor["tape_id"])
    a, bg = anchor["factual_option_index"], anchor["replacement_option_index"]
    chosen, = decision.selection.option_indices
    scored = planner.recorded_scores
    overflow = bool(scored is not None and scored.background_rollout_overflow)
    available = (scored is not None and not overflow and a in scored.costs and bg in scored.costs)
    pair = None
    if available:
        a_cost, bg_cost = scored.costs[a], scored.costs[bg]
        delta = a_cost - bg_cost
        sign = int(delta > 0) - int(delta < 0)
        work["diagnostic_pair_score_reads"] += 4
        work["diagnostic_pair_ranking_comparisons"] += 2
        truth = old_record["outcomes"]["root_effects"]["current"]["delta_T_mean"]
        pair = {
            "A_S1": float(a_cost / decision.sample_count),
            "BG_S1": float(bg_cost / decision.sample_count),
            "A_S2": float(scored.squared_costs[a] / decision.sample_count),
            "BG_S2": float(scored.squared_costs[bg] / decision.sample_count),
            "S1_A_minus_BG": float(delta / decision.sample_count),
            "exact_S1_sign": sign, "strict_A_preferred": sign < 0,
            "pair_preference": "A" if sign < 0 else "BG",
            "old_current_mean_A_minus_BG": truth,
            "pair_choice_correct": (sign < 0) == (truth < 0),
            "raw_sign_matches": sign == int(truth > 0) - int(truth < 0),
            "changed_from_old_A_preference": sign >= 0,
        }
    return {**anchor, **{key: old_record[key] for key in ("node", "lanes", "load", "seed")},
        "original_stage_source_match_passed": True, "post_prefix_state_match_passed": True,
        "old_risk_mask_origin": "frozen ordinary-BG original-stage S2<=same-state MW",
        "cache_A_original_stage_S2": info["incumbent_actual_predicted_squared_flow_s2"],
        "native_original_stage_MW_S2": decision.incumbent_mw_predicted_squared_flow_s2,
        "pair_available": available, "new_background_overflow": overflow,
        "unavailable_reason": None if available else "background_overflow" if overflow else "scores_unavailable",
        "new_native_option_index": chosen,
        "native_choice_kind": "A" if chosen == a else "BG" if chosen == bg else "third",
        "third_selected_without_label": chosen not in (a, bg), "new_pair_scores": pair,
        "cached_current_labels": old_record["outcomes"]["root_effects"]["current"],
        "cached_BG_labels": old_record["outcomes"]["root_effects"]["BG"],
        "inner_current_fallbacks": decision.inner_current_fallbacks}


def run_tape(model, tape, protocol, factual, anchors, old_records):
    planner = RecordingPolicy(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online",
                              protocol["planning_samples"], protocol["planning_event_cap"])
    work = Counter(controller_constructors=1, recourse_controller_constructors=1,
                   nested_recourse_controller_constructors=1)
    work.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    wanted = {anchor["frame_index"] for anchor in anchors}
    wanted.update(anchor["frame_index"] + 1 for anchor in anchors
                  if anchor["frame_index"] + 1 < len(factual["frames"]))
    states = reconstruct_prefix_states(planner, factual, wanted, work)
    final_rng = planner.rng.getstate()
    work["reconstructed_final_rng_words_copied"] += len(final_rng[1])
    records = []
    for anchor in anchors:
        index = anchor["frame_index"]
        expected_rng = states[index + 1]["rng_state"] if index + 1 in states else final_rng
        records.append(score_anchor(planner, factual["frames"][index], anchor,
            old_records[(tape["tape_id"], index)], states[index], expected_rng, work))
    print(tape["tape_id"], "cached prefix matched; native proposals", len(records), flush=True)
    return {"tape_id": tape["tape_id"], "prefix_reproduction_passed": True,
            "frames_processed": len(factual["frames"]), "records": records, "work": dict(work)}


def summarize(records):
    available = [row for row in records if row["pair_available"]]
    pairs = [row["new_pair_scores"] for row in available]
    actual_help = lambda row: row["cached_current_labels"]["delta_T_mean"] < 0
    return {
        "anchors": len(records), "available_pairs": len(available),
        "overflow_pairs": sum(row["new_background_overflow"] for row in records),
        "unavailable_pairs": len(records) - len(available),
        "baseline_old_A_preference_correct": sum(actual_help(row) for row in records),
        "baseline_old_A_preference_total": len(records),
        "new_pair_choice_correct": sum(pair["pair_choice_correct"] for pair in pairs),
        "new_pair_raw_sign_matches": sum(pair["raw_sign_matches"] for pair in pairs),
        "old_A_preference_correct_on_available": sum(actual_help(row) for row in available),
        "changed_pair_preferences": sum(pair["changed_from_old_A_preference"] for pair in pairs),
        "rejected_harmful_A": sum(not pair["strict_A_preferred"] and not actual_help(row)
                                   for row, pair in zip(available, pairs, strict=True)),
        "rejected_beneficial_A": sum(not pair["strict_A_preferred"] and actual_help(row)
                                     for row, pair in zip(available, pairs, strict=True)),
        "kept_beneficial_A": sum(pair["strict_A_preferred"] and actual_help(row)
                                 for row, pair in zip(available, pairs, strict=True)),
        "kept_harmful_A": sum(pair["strict_A_preferred"] and not actual_help(row)
                              for row, pair in zip(available, pairs, strict=True)),
        "exact_pair_ties": sum(pair["exact_S1_sign"] == 0 for pair in pairs),
        "near_zero_pair_delta_descriptive": sum(abs(pair["S1_A_minus_BG"]) <= SECONDS_TOLERANCE for pair in pairs),
        "native_choices": dict(Counter(row["native_choice_kind"] for row in records)),
        "third_selected_without_label": sum(row["third_selected_without_label"] for row in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    expected = build_protocol()
    if args.register:
        with artifact(BASE, "_protocol.json").open("x") as output:
            json.dump({**expected, "registered_at_utc": datetime.now(timezone.utc).isoformat()},
                      output, separators=(",", ":"), allow_nan=False)
        print("registered22 native proposals; new trajectories0", flush=True)
        return
    protocol = json.loads(artifact(BASE, "_protocol.json").read_text())
    if not same({key: value for key, value in protocol.items() if key != "registered_at_utc"}, expected):
        raise ValueError("registered sources, anchors, cached labels or inputs changed")
    if BASE.with_suffix(".json").exists():
        raise FileExistsError("preserve existing fixed-pair diagnostic result")
    models = {model.node: model for model in load_models(protocol["models"])}
    factual = saved_runs(protocol)
    old = {(row["tape_id"], row["frame_index"]): row for row in protocol["conditional_records"]}
    work, records, checks = Counter(), [], []
    with ProcessPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = [pool.submit(run_tape, models[tape["node"]], tape, protocol, factual[tape["tape_id"]],
            [anchor for anchor in protocol["anchors"] if anchor["tape_id"] == tape["tape_id"]], old)
            for tape in protocol["tapes"]]
        for future in as_completed(futures):
            row = future.result()
            work.update(row["work"])
            records.extend(row["records"])
            checks.append({key: row[key] for key in ("tape_id", "prefix_reproduction_passed", "frames_processed")})
    order = {(row["tape_id"], row["frame_index"]): index for index, row in enumerate(protocol["anchors"])}
    records.sort(key=lambda row: order[(row["tape_id"], row["frame_index"])])
    if len(records) != protocol["expected_native_proposals"]:
        raise ValueError("every frozen anchor requires one native proposal")
    primitive_probes = (
        "root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes",
        "inner_current_incumbent_root_option_probes", "inner_current_incumbent_rollout_option_probes",
        "inner_current_recourse_root_option_probes", "inner_current_recourse_rollout_option_probes",
    )
    work["total_option_probes"] = (sum(work[key] for key in primitive_probes)
        + sum(value for key, value in work.items() if key.startswith("preprocessing_") and key.endswith("option_probes")))
    result = {
        "decision": "TWO_STEP_CURRENT_POLICY_ROOT_RANKING_COMPLETE",
        "registered_at_utc": protocol["registered_at_utc"], "scope": protocol["scope"],
        "execution_checks_passed": True, "performance_gain_established": False,
        "registered_tapes": len(protocol["tapes"]), "native_proposals": len(records),
        "new_trajectory_executions": 0, "missing_tape_ids": protocol["missing_tape_ids"],
        "prefix_checks": checks, "summary": summarize(records), "records": records,
        "actual_work": dict(work),
        "strata": {"seed": [{"seed": seed, **summarize([row for row in records if row["seed"] == seed])}
                             for seed in sorted({row["seed"] for row in records})]},
    }
    with BASE.with_suffix(".json").open("x") as output:
        json.dump(result, output, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"decision": result["decision"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
