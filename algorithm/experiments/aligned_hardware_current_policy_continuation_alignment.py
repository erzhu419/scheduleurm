"""Frozen22 comparison with the same tagged jobs and persistent old-current suffix."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import inspect
import json
from pathlib import Path
import textwrap
from time import perf_counter, process_time

from algorithm.experiments import aligned_hardware_two_step_current_policy_population_accounting as population
from algorithm.experiments import aligned_hardware_two_step_current_policy_root_ranking as ranking
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_current_policy_continuation_alignment_20261002"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))


def matched_scorer():
    """Keep the native scorer, extending only its old-current continuation."""
    method = TwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Matched(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Tuple(self, node):
            self.generic_visit(node)
            if len(node.elts) == 2 and isinstance(node.elts[1], ast.Constant) and node.elts[1].value is True:
                node.elts[1] = ast.Constant(2 if isinstance(node.elts[0], ast.Name) else 1)
                self.changed["source_tags"] += 1
            return node

        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Name) and node.left.id == "recourse_steps"
                    and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant)):
                if isinstance(node.ops[0], ast.Lt) and node.comparators[0].value == 2:
                    self.changed["continuation_limits"] += 1
                    return ast.Constant(True)
                if isinstance(node.ops[0], ast.Eq) and node.comparators[0].value == 1:
                    self.changed["idle_counter_limit"] += 1
                    return ast.Compare(node.left, [ast.Gt()], [ast.Constant(0)])
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if (isinstance(target, ast.Tuple)
                    and [getattr(item, "id", None) for item in target.elts] == ["birth", "tagged"]):
                self.changed["admission_observer"] += 1
                return [node, ast.parse("self._population_admission(root, suffix.seed, coordinate, tagged, start + offset - birth)").body[0]]
            if isinstance(target, ast.Name) and target.id == "inner":
                self.changed["late_call_observer"] += 1
                return [node, ast.parse("self._observe_late_call(inner, recourse_steps)").body[0]]
            return node

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            target = node.target
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "two_step_work" and isinstance(target.slice, ast.Constant)):
                if target.slice.value == "recourse_second_calls":
                    self.changed["second_call_accounting"] += 1
                    return ast.If(ast.parse("recourse_steps == 1", mode="eval").body,
                        [node], ast.parse('self.continuation_work["late_current_calls"] += 1').body)
                if target.slice.value == "recourse_virtual_idle_counter_updates":
                    self.changed["late_idle_accounting"] += 1
                    return [node, ast.parse('if recourse_steps >= 2:\n    self.continuation_work["late_virtual_idle_counter_updates"] += 1').body[0]]
            return node

    transform = Matched()
    transform.visit(tree)
    expected = Counter(source_tags=2, continuation_limits=3, idle_counter_limit=1,
        admission_observer=1, late_call_observer=1, second_call_accounting=1, late_idle_accounting=1)
    if transform.changed != expected:
        raise ValueError("native continuation sites changed: " + str(transform.changed))
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<matched-current-continuation>", "exec"), namespace)
    return namespace[method.__name__]


MATCHED_SCORER = matched_scorer()


class MatchedContinuationPolicy(population.AccountingPolicy):
    """Same finite population; every nonempty virtual call uses frozen old-current."""
    def __init__(self, *args):
        super().__init__(*args)
        self.continuation_work = Counter()

    def __call__(self, *context):
        self.continuation_work.clear()
        return super().__call__(*context)

    def _observe_late_call(self, inner, step):
        if step >= 2:
            self.continuation_work["late_eligible_calls"] += inner.eligible
            self.continuation_work["late_planning_overflows"] += inner.planning_overflow
            self.continuation_work["late_background_overflows"] += inner.background_rollout_overflow

    def score_background_roots(self, *context):
        scored = MATCHED_SCORER(self, *context)
        self.recorded_scores = scored
        self.recorded_post_prefix_state = {"rng_state": self.rng.getstate(), "decision_counter": self.decision_counter}
        self.recording_work["recorded_post_prefix_rng_words_copied"] += len(self.recorded_post_prefix_state["rng_state"][1])
        self.recording_work["recorded_post_prefix_counter_reads"] += 1
        if self._recourse_work["recourse_calls"] != (self._two_step_work["recourse_first_calls"]
                + self._two_step_work["recourse_second_calls"] + self.continuation_work["late_current_calls"]):
            raise ValueError("first, second and late calls must reconstruct total current calls")
        return scored


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def register(directory):
    old_protocol = json.loads(population.path(directory, "_protocol.json").read_text())
    old_result = json.loads(population.path(directory, ".json").read_text())
    if (not old_result["execution_checks_passed"] or old_result["native_proposals"] != 22
            or old_protocol["registered_at_utc"] != old_result["registered_at_utc"]
            or old_result["full24_parent_decision"] != "TWO_STEP_CURRENT_POLICY_BG_INCUMBENT_DEVELOPMENT_HOLD"):
        raise ValueError("completed frozen22 population accounting with full24 HOLD required")
    sources = dict(old_protocol["execution_sources"])
    for name, literal in sources.items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("frozen source changed: " + name)
    sources[SOURCE] = (ROOT / SOURCE).read_text()
    baseline = []
    for row in old_result["records"]:
        counts = [{key: cell[key] for key in ("root", "sample_index", "source", "coordinate", "N")}
                  for cell in row["per_sample_cells"] if cell["source"] != "unscored_background"]
        baseline.append({key: value for key, value in row.items() if key != "per_sample_cells"}
                        | {"tagged_sample_counts": counts})
    if [(row["tape_id"], row["frame_index"]) for row in baseline] != [
            (row["tape_id"], row["frame_index"]) for row in old_protocol["anchors"]]:
        raise ValueError("retain all22 original anchors in order")
    result = {key: old_protocol[key] for key in ("models", "tapes", "anchors", "missing_tape_ids",
        "conditional_records", "planning_samples", "planning_event_cap", "planning_seed_rule", "birth_cohort",
        "previous_trajectories", "cached_native_work", "full24_parent_decision")}
    result.update(experiment="frozen22_same_population_old_current_continuation_v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(), population_registered_at_utc=old_protocol["registered_at_utc"],
        baseline_population_records=baseline, execution_sources=sources,
        method="Keep all22 fixedA/BG anchors and both missing tapes, all24 causal prefixes, original ordinaryBG S2<=MW risk mask and incumbent, strict S1/BG-first chooser, eight prefix samples/cap128, suffix seed rule and finite visible+tau_max tagged population. Replace virtual MW from the third nonempty call onward by the same frozen OLD-current policy, whose own inner planning remains twoBG thenMW. Restore first virtual RNG/counter once; persist thereafter, update counts for every released birth and advance all postfirst idle counters. No recursive new-candidate calls, altered window, class gate or new service trajectories.",
        accounting="Native recourse and nested-current ledgers charge every call including late fallbacks. recourse_second_calls counts only the second call; late_current_calls is separate, with recourse_calls=first+second+late. Store actual reuse and observer work independently. Per-layer samples/cap remain8/128; total execution work may increase and is not held equal to the two-call cache.",
        interpretation="Compare all22 exact fixed-pair S1 signs against cached old-current full-cohort mean labels. Report corrected and damaged choices, available denominators, same-source/class sign changes and every native third root without assigning it an A/B label. Whole-stage overflow is unavailable even when earlier partial pairs exist. No performance gain or new policy adoption follows from anchor correctness; preserve original MW/BG full24 HOLD and sealed confirmation. Tagged and empirical cohort targets still differ; residual errors do not isolate population omission.")
    population.dump(path(directory, "_protocol.json"), result)
    print(json.dumps({"registered_tapes": len(result["tapes"]), "registered_anchors": len(result["anchors"]),
        "new_service_trajectories": 0}), flush=True)


def load_protocol(directory):
    protocol = json.loads(path(directory, "_protocol.json").read_text())
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered source changed: " + name)
    return protocol


def fixed_pair_record(planner, native, old, baseline, model):
    checks = population.exact_reconstruction(planner)
    result = {key: native[key] for key in ("tape_id", "frame_index", "node", "load", "seed",
        "original_stage_source_match_passed", "post_prefix_state_match_passed", "new_native_option_index", "native_choice_kind")}
    result.update(baseline_group=baseline["group"], baseline_pair_scores=baseline["native_pair_scores"],
        old_current_T_mean_A_minus_BG_s=baseline["old_current_T_mean_A_minus_BG_s"],
        pair_available=native["pair_available"], new_background_overflow=native["new_background_overflow"],
        baseline_classes=baseline["classes"], accounting_checks=checks,
        continuation_work=dict(planner.continuation_work), inner_current_fallbacks=native["inner_current_fallbacks"])
    if not native["pair_available"]:
        result.update(new_pair_scores=None, new_classes=None, population_cells=None,
            fixed_pair_choice_correct=None, fixed_pair_preference=None, tagged_sample_counts_match_passed=None)
        return result
    # The preference of this fixed A/B pair can differ from a native third root.
    pair_native = dict(native, native_choice_kind=native["new_pair_scores"]["pair_preference"])
    attributed = population.pair_accounting(planner, pair_native, old, model)
    a, bg = attributed["A_option_index"], attributed["BG_option_index"]
    expected = {(cell["root"], cell["sample_index"], cell["source"], cell["coordinate"]): cell["N"]
                for cell in baseline["tagged_sample_counts"]}
    observed = {key: cell["n"] for key, cell in planner.population.items()
                if key[0] in (a, bg) and key[2] != "unscored_background"}
    if expected != observed:
        raise ValueError("complete scored population changed between continuations")
    result.update(new_pair_scores=native["new_pair_scores"], new_classes=attributed["classes"],
        population_cells=attributed["population_cells"], fixed_pair_choice_correct=native["new_pair_scores"]["pair_choice_correct"],
        fixed_pair_preference=native["new_pair_scores"]["pair_preference"], tagged_sample_counts_match_passed=True)
    return result


def run_shard(index, directory):
    wall, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    tape = protocol["tapes"][index]
    model = next(model for model in ranking.load_models(protocol["models"]) if model.node == tape["node"])
    factual = ranking.saved_runs(protocol)[tape["tape_id"]]
    anchors = [row for row in protocol["anchors"] if row["tape_id"] == tape["tape_id"]]
    planner = MatchedContinuationPolicy(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online",
        protocol["planning_samples"], protocol["planning_event_cap"])
    work = Counter(controller_constructors=1, recourse_controller_constructors=1, nested_recourse_controller_constructors=1)
    work.update({"preprocessing_" + key: value for key, value in planner.preprocessing_work.items()})
    wanted = {anchor["frame_index"] for anchor in anchors}
    wanted.update(anchor["frame_index"] + 1 for anchor in anchors if anchor["frame_index"] + 1 < len(factual["frames"]))
    states = ranking.reconstruct_prefix_states(planner, factual, wanted, work)
    final_rng = planner.rng.getstate()
    work["reconstructed_final_rng_words_copied"] += len(final_rng[1])
    records = []
    for anchor in anchors:
        frame_index = anchor["frame_index"]
        key = (tape["tape_id"], frame_index)
        old = next(row for row in protocol["conditional_records"] if (row["tape_id"], row["frame_index"]) == key)
        baseline = next(row for row in protocol["baseline_population_records"] if (row["tape_id"], row["frame_index"]) == key)
        rng = states[frame_index + 1]["rng_state"] if frame_index + 1 in states else final_rng
        native = ranking.score_anchor(planner, factual["frames"][frame_index], anchor, old, states[frame_index], rng, work)
        records.append(fixed_pair_record(planner, native, old, baseline, model))
    row = {"registered_at_utc": protocol["registered_at_utc"], "index": index, "tape_id": tape["tape_id"],
        "prefix_reproduction_passed": True, "records": records, "native_logical_work": dict(work),
        "executed_reuse_work": planner.reuse_work,
        "diagnostic_work": {"observed_admissions": sum(cell["n"] for cell in planner.population.values()),
            "additional_flow_evaluations": sum(cell["n"] for cell in planner.population.values())},
        "wall_s": perf_counter() - wall, "cpu_s": process_time() - cpu}
    population.dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps({key: row[key] for key in ("tape_id", "index", "prefix_reproduction_passed", "wall_s", "cpu_s")}
        | {"anchors": len(records)}, ensure_ascii=False), flush=True)


def summarize(records):
    available = [row for row in records if row["pair_available"]]
    before = lambda row: row["baseline_pair_scores"]["pair_choice_correct"]
    return {"anchors": len(records), "available_pairs": len(available),
        "unavailable_pairs": len(records) - len(available),
        "baseline_correct": sum(before(row) for row in records),
        "baseline_correct_on_available": sum(before(row) for row in available),
        "matched_correct_on_available": sum(row["fixed_pair_choice_correct"] for row in available),
        "corrected_pairs": sum(not before(row) and row["fixed_pair_choice_correct"] for row in available),
        "damaged_pairs": sum(before(row) and not row["fixed_pair_choice_correct"] for row in available),
        "native_choices": dict(Counter(row["native_choice_kind"] for row in records)),
        "native_third_roots_without_labels": sum(row["native_choice_kind"] == "third" for row in records),
        "classes": [{"coordinate": c, "workload": available[0]["new_classes"][c]["workload"],
            "baseline_benefit_actual_harm_on_available": sum(row["baseline_classes"][c]["predicted_benefit_actual_harm"] for row in available),
            "matched_benefit_actual_harm_on_available": sum(row["new_classes"][c]["predicted_benefit_actual_harm"] for row in available),
            "baseline_harm_actual_benefit_on_available": sum(row["baseline_classes"][c]["predicted_harm_actual_benefit"] for row in available),
            "matched_harm_actual_benefit_on_available": sum(row["new_classes"][c]["predicted_harm_actual_benefit"] for row in available)}
            for c in range(len(available[0]["new_classes"]))] if available else []}


def merge(directory):
    protocol = load_protocol(directory)
    shards = [json.loads(path(directory, f"_shard_{index:02d}.json").read_text()) for index in range(len(protocol["tapes"]))]
    work, late = Counter(), Counter()
    for index, (row, tape) in enumerate(zip(shards, protocol["tapes"], strict=True)):
        if (row["index"] != index or row["tape_id"] != tape["tape_id"] or not row["prefix_reproduction_passed"]
                or row["registered_at_utc"] != protocol["registered_at_utc"]):
            raise ValueError("registered causal tape shard differs")
        work.update(row["native_logical_work"])
        for record in row["records"]:
            late.update(record["continuation_work"])
    # Original BG scoring and causal prefix reconstruction remain unchanged.
    unchanged = [key for key in protocol["cached_native_work"] if key.startswith(("preprocessing_", "prefix_", "cached_"))]
    unchanged += ["incumbent_" + name for name in (
        "root_option_probes", "rollout_calls", "rollout_option_probes", "rollout_idle_calls", "waiting_age_reads",
        "flow_evaluations", "background_planning_events", "background_planning_random_draws", "background_planning_rate_reads",
        "background_seed_initializations", "admitted_job_reads", "tag_flag_reads", "background_admissions")]
    if any(work[key] != protocol["cached_native_work"][key] for key in unchanged):
        raise ValueError("original BG or causal prefix work changed")
    if work["recourse_calls"] != work["recourse_first_calls"] + work["recourse_second_calls"] + late["late_current_calls"]:
        raise ValueError("total continuation fee accounting differs")
    primitive = ("root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes",
        "inner_current_incumbent_root_option_probes", "inner_current_incumbent_rollout_option_probes",
        "inner_current_recourse_root_option_probes", "inner_current_recourse_rollout_option_probes")
    work["total_option_probes"] = sum(work[key] for key in primitive) + sum(value
        for key, value in work.items() if key.startswith("preprocessing_") and key.endswith("option_probes"))
    records = [record for shard in shards for record in shard["records"]]
    if [(row["tape_id"], row["frame_index"]) for row in records] != [(row["tape_id"], row["frame_index"]) for row in protocol["anchors"]]:
        raise ValueError("retain all22 frozen anchors")
    result = {"decision": "CURRENT_POLICY_CONTINUATION_ALIGNMENT_COMPLETE", "execution_checks_passed": True,
        "performance_gain_established": False, "registered_at_utc": protocol["registered_at_utc"],
        "native_proposals": len(records), "new_service_trajectory_executions": 0,
        "full24_parent_decision": protocol["full24_parent_decision"], "interpretation": protocol["interpretation"],
        "missing_tape_ids": protocol["missing_tape_ids"], "native_logical_work": dict(work), "continuation_work": dict(late),
        "summary": summarize(records),
        "baseline_groups": {group: summarize([row for row in records if row["baseline_group"] == group]) for group in population.GROUPS},
        "seeds": {str(seed): summarize([row for row in records if row["seed"] == seed]) for seed in (41, 42, 43)},
        "records": records, "execution": [{key: row[key] for key in ("index", "tape_id", "wall_s", "cpu_s", "diagnostic_work", "executed_reuse_work")} for row in shards]}
    population.dump(path(directory, ".json"), result)
    small = {key: value for key, value in result.items() if key not in ("records", "execution", "native_logical_work")}
    population.dump(path(directory, "_summary.json"), small)
    population.dump(path(directory, "_records_summary.json"), {"registered_at_utc": protocol["registered_at_utc"], "records": records})
    print(json.dumps(small, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--register", action="store_true")
    modes.add_argument("--run-shard", type=int)
    modes.add_argument("--merge", action="store_true")
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
