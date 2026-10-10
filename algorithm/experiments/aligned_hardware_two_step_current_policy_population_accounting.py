"""Attribute the unchanged frozen22 root scores by population and workload."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import inspect
import json
from pathlib import Path
from statistics import mean
import textwrap
from time import perf_counter, process_time

from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.experiments import aligned_hardware_two_step_current_policy_root_ranking as ranking
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import TwoStepCurrentPolicyBGIncumbentLaneRollout


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "md/experiment_artifacts"
STEM = "aligned_hardware_two_step_current_policy_population_accounting_20261001"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = ("visible_fifo", "tagged_prefix", "unscored_background")
GROUPS = ("kept_beneficial_A", "kept_harmful_A", "rejected_harmful_A", "rejected_beneficial_A")


def instrumented_scorer():
    """Insert an admission observer into the original method, keeping its logic."""
    method = TwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Observer(ast.NodeTransformer):
        def __init__(self):
            self.tags = self.hooks = 0

        def visit_Tuple(self, node):
            self.generic_visit(node)
            if len(node.elts) == 2 and isinstance(node.elts[1], ast.Constant) and node.elts[1].value is True:
                # Explicit source tags also distinguish a prefix birth at time0.
                node.elts[1] = ast.Constant(2 if isinstance(node.elts[0], ast.Name) else 1)
                self.tags += 1
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [getattr(item, "id", None) for item in node.targets[0].elts] == ["birth", "tagged"]):
                self.hooks += 1
                hook = ast.parse("self._population_admission(root, suffix.seed, coordinate, tagged, start + offset - birth)").body[0]
                return [node, hook]
            return node

    observer = Observer()
    observer.visit(tree)
    if (observer.tags, observer.hooks) != (2, 1):
        raise ValueError("native FIFO tag/admission sites changed")
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<population-observer>", "exec"), namespace)
    return namespace[method.__name__]


OBSERVED_SCORER = instrumented_scorer()


class AccountingPolicy(ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Observe only outer costs; the already verified inner reuse stays exact."""
    def __init__(self, *args):
        super().__init__(*args)
        self.recorded_scores = self.recorded_post_prefix_state = None
        self.recording_work = Counter()
        self.population = {}

    def __call__(self, *context):
        self.recorded_scores = self.recorded_post_prefix_state = None
        self.population.clear()
        return super().__call__(*context)

    def _population_admission(self, root, seed, coordinate, tag, flow):
        source = SOURCES[tag - 1] if tag else SOURCES[2]
        key = (root, int(seed.rsplit(":", 1)[1]), source, coordinate)
        cell = self.population.setdefault(key, {"n": 0, "S1": Fraction(0), "S2": Fraction(0)})
        cell["n"] += 1
        cell["S1"] += flow
        cell["S2"] += flow * flow

    def score_background_roots(self, *context):
        scored = OBSERVED_SCORER(self, *context)
        self.recorded_scores = scored
        self.recorded_post_prefix_state = {"rng_state": self.rng.getstate(), "decision_counter": self.decision_counter}
        self.recording_work["recorded_post_prefix_rng_words_copied"] += len(self.recorded_post_prefix_state["rng_state"][1])
        self.recording_work["recorded_post_prefix_counter_reads"] += 1
        return scored


def population_total(planner, root, source, coordinate):
    selected = [cell for (r, _, s, c), cell in planner.population.items()
                if (r, s, c) == (root, source, coordinate)]
    return (sum(cell["n"] for cell in selected),
        sum((cell["S1"] for cell in selected), Fraction(0)),
        sum((cell["S2"] for cell in selected), Fraction(0)))


def population_cells(planner, root):
    cells = []
    for source in SOURCES:
        for coordinate in range(len(planner.arrival_rates)):
            n, first, second = population_total(planner, root, source, coordinate)
            cells.append({"source": source, "coordinate": coordinate,
                "scored": source != "unscored_background", "mean_jobs_per_sample": n / planner.sample_count,
                "S1_sum_flow_s_per_sample": float(first / planner.sample_count),
                "S2_sum_squared_flow_s2_per_sample": float(second / planner.sample_count),
                "mean_admitted_flow_s": float(first / n) if n else None})
    return cells


def exact_reconstruction(planner):
    scored = planner.recorded_scores
    for root in scored.costs:
        cells = [cell for (r, _, source, _), cell in planner.population.items()
                 if r == root and source != "unscored_background"]
        if (sum((cell["S1"] for cell in cells), Fraction(0)) != scored.costs[root]
                or sum((cell["S2"] for cell in cells), Fraction(0)) != scored.squared_costs[root]):
            raise ValueError("population observer does not reconstruct the native Fraction scores")
    # Overflow retains only actually admitted partial costs; no full-population claim.
    return {"all_roots_exact_Fraction_reconstruction_passed": True,
            "complete_population": not scored.background_rollout_overflow}


def pair_accounting(planner, native, old_record, model):
    checks = exact_reconstruction(planner)
    a, bg = native["factual_option_index"], native["replacement_option_index"]
    left, right = population_cells(planner, a), population_cells(planner, bg)
    predicted_n = sum(cell["mean_jobs_per_sample"] for cell in left if cell["scored"])
    cells = []
    for ca, cb in zip(left, right, strict=True):
        if ca["scored"] and ca["mean_jobs_per_sample"] != cb["mean_jobs_per_sample"]:
            raise ValueError("root branches must score the identical complete tagged population")
        _, first_a, second_a = population_total(planner, a, ca["source"], ca["coordinate"])
        _, first_b, second_b = population_total(planner, bg, cb["source"], cb["coordinate"])
        cells.append({"source": ca["source"], "coordinate": ca["coordinate"],
            "workload": model.workloads[ca["coordinate"]], "scored": ca["scored"],
            "A": ca, "BG": cb,
            "S1_A_minus_BG_s": float((first_a - first_b) / planner.sample_count),
            "S2_A_minus_BG_s2": float((second_a - second_b) / planner.sample_count)})
    snapshots = old_record["outcomes"]["snapshots"]
    end, horizon = snapshots["observation_end"], snapshots["arrival_horizon"]
    ea, eb = end["original_current"], end["BG_current"]
    classes = []
    for coordinate, (ca, cb) in enumerate(zip(ea["per_coordinate"], eb["per_coordinate"], strict=True)):
        predicted = [cell for cell in cells if cell["scored"] and cell["coordinate"] == coordinate]
        exact_delta = sum((population_total(planner, a, source, coordinate)[1]
            - population_total(planner, bg, source, coordinate)[1] for source in SOURCES[:2]), Fraction(0))
        delta = float(exact_delta / planner.sample_count)
        actual = ca["restricted_mean_flow_s"] - cb["restricted_mean_flow_s"]
        classes.append({"coordinate": coordinate, "workload": model.workloads[coordinate],
            "predicted_mean_jobs_per_sample": sum(cell["A"]["mean_jobs_per_sample"] for cell in predicted),
            "predicted_S1_A_minus_BG_s": delta,
            "predicted_mean_contribution_s": delta / predicted_n,
            "full_cohort_N": ca["arrivals"], "old_current_class_T_mean_delta_s": actual,
            "old_current_weighted_T_mean_contribution_s": actual * ca["arrivals"] / ea["arrivals"],
            "old_current_class_900_unfinished_delta": horizon["original_current"]["per_coordinate"][coordinate]["unfinished_jobs"]
                - horizon["BG_current"]["per_coordinate"][coordinate]["unfinished_jobs"],
            "predicted_benefit_actual_harm": exact_delta < 0 and actual > 0,
            "predicted_harm_actual_benefit": exact_delta > 0 and actual < 0})
    truth = native["cached_current_labels"]["delta_T_mean"]
    group = ("kept_" if native["native_choice_kind"] == "A" else "rejected_") + ("beneficial_A" if truth < 0 else "harmful_A")
    checks["native_cached_record_match_passed"] = True
    return {"tape_id": native["tape_id"], "frame_index": native["frame_index"],
        "node": native["node"], "load": native["load"], "seed": native["seed"], "group": group,
        "A_option_index": a, "BG_option_index": bg, "native_pair_scores": native["new_pair_scores"],
        "mean_tagged_jobs_per_sample": predicted_n,
        "old_current_T_mean_A_minus_BG_s": truth, "accounting_checks": checks,
        "population_cells": cells, "classes": classes,
        "per_sample_cells": [{"root": key[0], "sample_index": key[1], "source": key[2], "coordinate": key[3],
            "N": cell["n"], "S1_exact": str(cell["S1"]), "S2_exact": str(cell["S2"])}
            for key, cell in planner.population.items() if key[0] in (a, bg)]}


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def dump(destination, value):
    with destination.open("x") as out:
        json.dump(value, out, separators=(",", ":"), allow_nan=False)


def register(directory):
    protocol = json.loads(ranking.artifact(ranking.BASE, "_protocol.json").read_text())
    cached = json.loads(ranking.BASE.with_suffix(".json").read_text())
    final = json.loads((ARTIFACTS / "aligned_hardware_parallel_root_sharded_20261001_result_summary.json").read_text())
    if (not cached["execution_checks_passed"] or cached["native_proposals"] != 22
            or cached["registered_at_utc"] != protocol["registered_at_utc"]
            or final["decision"] != "TWO_STEP_CURRENT_POLICY_BG_INCUMBENT_DEVELOPMENT_HOLD"):
        raise ValueError("completed frozen22 scores and full24 HOLD required")
    selected = ranking.registered_anchors(protocol["tapes"], ranking.saved_runs(protocol))
    if not ranking.same(selected, {name: protocol[name] for name in ("anchors", "missing_tape_ids")}):
        raise ValueError("retain all22 outcome-unfiltered anchors and both missing tapes")
    sources = dict(protocol["execution_sources"])
    for name, literal in sources.items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("frozen source changed: " + name)
    for name in (SOURCE, "algorithm/exact_reuse_lane_rollout.py", "algorithm/exact_reuse_inner_decision.py", "algorithm/dyadic_background_rollout.py"):
        sources[name] = (ROOT / name).read_text()
    result = {key: protocol[key] for key in ("models", "tapes", "anchors", "missing_tape_ids", "conditional_records",
        "planning_samples", "planning_event_cap", "planning_seed_rule", "birth_cohort", "previous_trajectories")}
    result.update(experiment="frozen22_population_class_accounting_v1",
        registered_at_utc=datetime.now(timezone.utc).isoformat(), ranking_registered_at_utc=protocol["registered_at_utc"],
        cached_native_records=cached["records"], cached_native_work=cached["actual_work"], execution_sources=sources,
        full24_parent_decision=final["decision"], expected_native_proposals=22, expected_new_service_trajectories=0,
        method="Only observe outer admissions, with truthy source tags1=visible FIFO,2=sampled prefix and0=unscored suffix. Attribute exact Fraction S1/S2 by source,class,sample. Reuse the verified exact inner executor; match every cached native record and aggregate logical fee. Reconstruct all24 causal prefixes including both missing tapes. No new action, risk mask, class gate, score weighting or continuation depth.",
        limitations="Sampled tagged S1 is a summed flow over visible FIFO and tau_max prefix; complete empirical means use the birth cohort[300,900] under cached OLD-current suffix. Normalized contributions compare signs/composition, not forecast errors or causally isolated population effects. Unscored background is admitted-only until each branch's last tagged admission, with root-dependent population; never add its observed costs to native scores. Cached cross arms have no per-job labels, so no empirical visible/prefix/later birth decomposition.22 states share3 seeds and are not independent performance replicates.")
    dump(path(directory, "_protocol.json"), result)
    print(json.dumps({"registered_tapes": len(result["tapes"]), "registered_anchors": len(result["anchors"]),
        "new_service_trajectories": 0}), flush=True)


def load_protocol(directory):
    protocol = json.loads(path(directory, "_protocol.json").read_text())
    for name, literal in protocol["execution_sources"].items():
        if (ROOT / name).read_text() != literal:
            raise ValueError("registered source changed: " + name)
    return protocol


def run_shard(index, directory):
    wall, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    tape = protocol["tapes"][index]
    model = next(model for model in ranking.load_models(protocol["models"]) if model.node == tape["node"])
    factual = ranking.saved_runs(protocol)[tape["tape_id"]]
    anchors = [row for row in protocol["anchors"] if row["tape_id"] == tape["tape_id"]]
    planner = AccountingPolicy(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online",
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
        expected = next(row for row in protocol["cached_native_records"] if (row["tape_id"], row["frame_index"]) == key)
        rng = states[frame_index + 1]["rng_state"] if frame_index + 1 in states else final_rng
        native = ranking.score_anchor(planner, factual["frames"][frame_index], anchor, old, states[frame_index], rng, work)
        if not ranking.same(native, expected):
            raise ValueError("admission observation changed the frozen complete native proposal: " + str(key))
        records.append(pair_accounting(planner, native, old, model))
    row = {"registered_at_utc": protocol["registered_at_utc"], "index": index, "tape_id": tape["tape_id"],
        "prefix_reproduction_passed": True, "records": records, "native_logical_work": dict(work),
        "executed_reuse_work": planner.reuse_work,
            "diagnostic_work": {"observed_admissions": sum(cell["n"] for cell in planner.population.values()),
            "additional_flow_evaluations": sum(cell["n"] for cell in planner.population.values()),
            "recorded_cells": len(planner.population)}, "wall_s": perf_counter() - wall, "cpu_s": process_time() - cpu}
    dump(path(directory, f"_shard_{index:02d}.json"), row)
    print(json.dumps({key: row[key] for key in ("tape_id", "index", "prefix_reproduction_passed", "wall_s", "cpu_s")}
        | {"anchors": len(records)}, ensure_ascii=False), flush=True)


def summarize(records):
    return {"anchors": len(records),
        "classes": [{"coordinate": c, "workload": records[0]["classes"][c]["workload"],
            "predicted_benefit_actual_harm": sum(row["classes"][c]["predicted_benefit_actual_harm"] for row in records),
            "predicted_harm_actual_benefit": sum(row["classes"][c]["predicted_harm_actual_benefit"] for row in records),
            "predicted_mean_contribution_s_descriptive": mean(row["classes"][c]["predicted_mean_contribution_s"] for row in records),
            "old_current_weighted_T_mean_contribution_s_descriptive": mean(row["classes"][c]["old_current_weighted_T_mean_contribution_s"] for row in records)}
            for c in range(len(records[0]["classes"]))],
        "population_source_S1_delta_s_descriptive": {source: mean(sum(cell["S1_A_minus_BG_s"]
            for cell in row["population_cells"] if cell["source"] == source) for row in records) for source in SOURCES}}


def merge(directory):
    protocol = load_protocol(directory)
    shards = [json.loads(path(directory, f"_shard_{index:02d}.json").read_text()) for index in range(len(protocol["tapes"]))]
    for index, (row, tape) in enumerate(zip(shards, protocol["tapes"], strict=True)):
        if (row["index"] != index or row["tape_id"] != tape["tape_id"] or not row["prefix_reproduction_passed"]
                or row["registered_at_utc"] != protocol["registered_at_utc"]):
            raise ValueError("registered causal tape shard differs")
    work = Counter()
    for row in shards:
        work.update(row["native_logical_work"])
    primitive_probes = ("root_option_probes", "rollout_option_probes", "incumbent_root_option_probes",
        "incumbent_rollout_option_probes", "recourse_root_option_probes", "recourse_rollout_option_probes",
        "inner_current_incumbent_root_option_probes", "inner_current_incumbent_rollout_option_probes",
        "inner_current_recourse_root_option_probes", "inner_current_recourse_rollout_option_probes")
    work["total_option_probes"] = sum(work[key] for key in primitive_probes) + sum(value
        for key, value in work.items() if key.startswith("preprocessing_") and key.endswith("option_probes"))
    if work != Counter(protocol["cached_native_work"]):
        raise ValueError("aggregate original logical fee accounting changed")
    records = [record for shard in shards for record in shard["records"]]
    if [(row["tape_id"], row["frame_index"]) for row in records] != [(row["tape_id"], row["frame_index"]) for row in protocol["anchors"]]:
        raise ValueError("all22 registered anchors must appear once in frozen order")
    result = {"decision": "TWO_STEP_CURRENT_POLICY_POPULATION_ACCOUNTING_COMPLETE", "execution_checks_passed": True,
        "performance_gain_established": False, "registered_at_utc": protocol["registered_at_utc"],
        "native_proposals": len(records), "new_service_trajectory_executions": 0,
        "full24_parent_decision": protocol["full24_parent_decision"], "limitations": protocol["limitations"],
        "missing_tape_ids": protocol["missing_tape_ids"], "native_logical_work": dict(work),
        "execution": [{key: row[key] for key in ("index", "tape_id", "wall_s", "cpu_s", "diagnostic_work", "executed_reuse_work")} for row in shards],
        "summary": summarize(records), "groups": {group: summarize([row for row in records if row["group"] == group]) for group in GROUPS},
        "records": records}
    dump(path(directory, ".json"), result)
    small = {key: value for key, value in result.items() if key not in ("records", "execution", "native_logical_work")}
    dump(path(directory, "_summary.json"), small)
    dump(path(directory, "_records_summary.json"), {"registered_at_utc": protocol["registered_at_utc"],
        "records": [{key: value for key, value in row.items() if key != "per_sample_cells"} for row in records]})
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
