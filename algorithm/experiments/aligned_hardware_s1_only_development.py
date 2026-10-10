"""Remove only the S2 root filter on all 24 frozen development arrival tapes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

from algorithm.continuous_arrival_s1_only_lane_rollout import ContinuousArrivalS1OnlyLaneRollout
from algorithm.experiments.aligned_hardware_bg_strong_controls import (
    TimedSelector, aggregate, births, read, require, same, save)
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes, sign_counts
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics
from algorithm.experiments.aligned_hardware_arrival_aware_lane_rollout import rollout_checks as population_checks
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.arrival_aware_lane_rollout_summary import summarize
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory

ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_S1_only_development_20261003"
OLD = {"no_background": "aligned_hardware_continuous_arrival_age_tail_development_20260930",
       "BG": "aligned_hardware_background_competition_continuous_development_20260930"}
EXTRA_SOURCES = (str(Path(__file__).resolve().relative_to(ROOT)),
    "algorithm/continuous_arrival_s1_only_lane_rollout.py",
    "algorithm/experiments/aligned_hardware_bg_strong_controls.py",
    "algorithm/experiments/bg_confirmation_block_intervals.py",
    "algorithm/experiments/aligned_hardware_background_root_counterfactual.py",
    "algorithm/experiments/single_action_counterfactual_replay.py")
CUTOFFS = ("arrival_horizon", "observation_end")
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def register(directory):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    protocols, results, caches = {}, {}, {}
    for label, stem in OLD.items():
        protocol = read(Path(directory) / (stem + "_protocol.json"))
        result = read(Path(directory) / (stem + ".json"))
        require(protocol["registered_at_utc"] == result["registered_at_utc"]
            == design["source_execution_epochs"][label] and result["execution_checks_passed"],
            "original development epoch/checks differ: " + label)
        for name, literal in protocol["execution_sources"].items():
            require((ROOT / name).read_text() == literal, "original frozen source changed: " + name)
        cache = read(Path(directory) / (stem + "_trajectories.json.gz"))
        # The original no-background driver saved only {'runs': ...}.
        if label == "BG":
            require(cache["registered_at_utc"] == protocol["registered_at_utc"], "BG cache epoch differs")
        ids = {tape["tape_id"] for tape in protocol["tapes"]}
        cached = {row["tape_id"]: row["policies"] for row in cache["runs"]}
        references = {row["tape_id"]: row for row in result["paired_runs"]}
        require(len(cache["runs"]) == len(result["paired_runs"]) == len(ids) == 24
            and set(cached) == set(references) == ids, "all24 cached development tapes required")
        for tape in protocol["tapes"]:
            policies, reference = cached[tape["tape_id"]], references[tape["tape_id"]]
            require(reference["baseline_reproduction_passed"] and reference["rollout_checks_passed"], "cached checks failed")
            require(births(policies["maxweight"]) == births(policies["arrival_rollout"])
                == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])], "cached births differ")
            for cutoff in CUTOFFS:
                require(same(cohort_metrics(policies, protocol[cutoff], protocol["birth_cohort"]),
                    reference["snapshots"][cutoff]), "original pair cohort metrics differ: " + label)
        protocols[label], results[label], caches[label] = protocol, result, cached
    base = protocols["no_background"]
    shared = ("models", "tapes", "loads", "arrival_seeds", "birth_cohort", "arrival_horizon",
              "environment_arrival_end", "observation_end", "planning_samples", "planning_event_cap",
              "planning_seed_rule", "clock", "performance")
    require(all(same(base[key], protocols["BG"][key]) for key in shared), "development models/tapes/budgets differ")
    expected = {(node, lanes, load, seed) for node, lanes in design["population"]["catalogue_lanes"]
                for load in design["population"]["loads"] for seed in design["population"]["arrival_seeds"]}
    require({tuple(tape[k] for k in ("node", "lanes", "load", "seed")) for tape in base["tapes"]} == expected
        and base["arrival_seeds"] == [41, 42, 43] and base["planning_samples"] == 8
        and base["planning_event_cap"] == 128 and base["birth_cohort"] == [300, 900]
        and base["arrival_horizon"] == 900 and base["environment_arrival_end"] == base["observation_end"] == 1200,
        "retain the entire frozen24 development population")
    epoch, units = datetime.now(timezone.utc).isoformat(), []
    models = {model["node"]: model for model in base["models"]}
    mw_references = {row["tape_id"]: row for row in results["no_background"]["paired_runs"]}
    for index, tape in enumerate(base["tapes"]):
        old, bg = (caches[label][tape["tape_id"]] for label in ("no_background", "BG"))
        require(same(old["maxweight"]["jobs"], bg["maxweight"]["jobs"]), "cached MW physical trajectories differ")
        cached = {"maxweight": old["maxweight"], "no_background": old["arrival_rollout"], "BG": bg["arrival_rollout"]}
        fixture = path(directory, f"_fixture_{index:02d}.json.gz")
        save(fixture, dict(registered_at_utc=epoch, model=models[tape["node"]], tape=tape,
            source_execution_epochs=design["source_execution_epochs"],
            cached_policies={label: {"jobs": arm["jobs"], "snapshots": arm["snapshots"]} for label, arm in cached.items()},
            cached_MW_work=mw_references[tape["tape_id"]]["work"]["maxweight"]))
        units.append(dict(unit_index=index, fixture_file=fixture.name,
            **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")}))
    sources = {**protocols["BG"]["execution_sources"], **{name: (ROOT / name).read_text() for name in EXTRA_SOURCES}}
    save(path(directory, "_protocol.json"), dict(experiment=STEM, registered_at_utc=epoch,
        design=design, source_execution_epochs=design["source_execution_epochs"],
        **{key: base[key] for key in shared}, units=units, execution_sources=sources,
        expected_new_trajectories=24, reused_BG_MW_no_background_trajectory_arms=72,
        new_bg_mw_or_no_background_simulations=0, new_shadow_references=0,
        cached_totals_work={label: result["totals_work"] for label, result in results.items()},
        original_decisions={label: result["decision"] for label, result in results.items()},
        limitations=design["statistics"],
        registration_work=dict(cached_trajectory_records_read=96, fixtures=24,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"registered":true,"scientific_units":24,"new_reference_simulations":0}', flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(protocol["design"] == read(path(directory, "_design.json")), "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def check_s1_only(result, model, protocol, planner):
    if not population_checks(result, model, protocol):
        return False
    for frame in result["frames"]:
        info, counts, now = frame["rollout"], tuple(frame["observed_arrival_counts"]), frame["start"]
        ages, n = frame["waiting_ages"], sum(frame["queue"])
        ranked = info["eligible"] and not info["planning_overflow"]
        if (tuple(map(len, ages)) != tuple(frame["queue"])
                or any(any(age < 0 for age in group) or list(group) != sorted(group, reverse=True) for group in ages)
                or tuple(info["observed_arrival_counts"]) != counts or info["rate_exposure_s"] != now
                or tuple(info["estimated_arrival_rates"]) != tuple(count / now if now else 0. for count in counts)
                or info["rate_estimation_coordinate_reads"] != len(counts)
                or info["rate_estimation_divisions"] != (len(counts) if now else 0)
                or info["eligible"] and info["window_s"] != planner.maximum_duration
                or info["risk_filtered_candidates"] or info["risk_changed_unconstrained_root"]):
            return False
        if ranked:
            if (info["risk_feasible_candidates"] != info["candidate_count"]
                    or info["flow_evaluations"] > info["candidate_count"] * 8 * (n + 128)
                    or info["diagnostic_S2_comparisons"] != 1
                    or info["root_ordering_comparisons"] != info["candidate_count"] - 1
                    or info["strict_improvement_comparisons"] != 1):
                return False
        elif (info["flow_evaluations"] or info["diagnostic_S2_comparisons"]
              or info["root_ordering_comparisons"] or info["strict_improvement_comparisons"]
              or info["chosen_exceeds_MW_S2"]):
            return False
    return True


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 24, "unit-index must be0..23")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:02d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed unit")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and fixture["source_execution_epochs"] == protocol["source_execution_epochs"]
        and fixture["tape"]["tape_id"] == unit["tape_id"], "fixture epoch/identity differs")
    model, = load_models([fixture["model"]])
    tape, cached = fixture["tape"], fixture["cached_policies"]
    planner = ContinuousArrivalS1OnlyLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online", 8, 128)
    selector = TimedSelector(planner)
    result = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], 1200, 1200, selector, policy="S1_only")
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(all(births(result) == births(arm) for arm in cached.values()), "new trajectory births differ")
    require(check_s1_only(result, model, protocol, planner), "S1-only causal/budget/selection checks failed")
    contrasts = {label: paired_outcomes(arm, result, protocol, model.workloads) for label, arm in cached.items()}
    work = work_totals(result, planner, 900)
    work["controller_constructors"] = 1
    for key in ("diagnostic_S2_comparisons", "root_ordering_comparisons", "strict_improvement_comparisons"):
        work[key] = sum(frame["rollout"][key] for frame in result["frames"])
    work["diagnostic_S2_exceeding_decisions"] = sum(frame["rollout"]["chosen_exceeds_MW_S2"] for frame in result["frames"])
    save(row_file, dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, contrasts=contrasts, work=work, cached_MW_work=fixture["cached_MW_work"],
        computation=dict(selector_calls=selector.calls, selector_wall_s=selector.wall_s,
            max_selector_wall_s=selector.max_wall_s, wall_s=perf_counter() - started, cpu_s=process_time() - cpu,
            computation_delays_service=False)))
    print('{"unit_index":' + str(index) + ',"execution_checks_passed":true}', flush=True)


def contrast_summary(rows, label):
    records = [{**row, "outcomes": row["contrasts"][label]} for row in rows]
    groups, seeds, classes = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        groups[row["node"], row["load"]].append(row)
        seeds[row["seed"]].append(row)
        for item in row["outcomes"]["per_coordinate_deltas"]:
            classes[row["node"], row["load"], item["workload"]].append(item)
    return dict(orientation="S1-only minus " + label, pooled=aggregate(records),
        absolute_snapshots={cutoff: {arm: {metric: (sum(values) if metric == "unfinished_jobs" else mean(values))
            for metric in ("restricted_mean_flow_s", "common_p90_flow_s", "unfinished_jobs")
            for values in [[row["outcomes"]["snapshots"][cutoff][arm][metric] for row in records]]}
            for arm in ("factual", "replacement")} for cutoff in CUTOFFS},
        groups=[dict(node=node, load=load, **aggregate(group)) for (node, load), group in sorted(groups.items())],
        seed_aggregates=[dict(seed=seed, paired_cells=len(group), **aggregate(group)) for seed, group in sorted(seeds.items())],
        class_results=[dict(node=node, load=load, workload=workload,
            **{metric: dict(mean=mean(values) if all(v is not None for v in values) else None,
                signs=sign_counts(values), total=sum(values) if "unfinished" in metric else None)
                for metric in METRICS for values in [[item[metric] for item in group]]})
            for (node, load, workload), group in sorted(classes.items())],
        paired_results=[dict(**{key: row[key] for key in ("unit_index", "tape_id", "node", "load", "seed")},
            **row["outcomes"]) for row in records])


def merge(directory):
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:02d}.json")) for unit in protocol["units"]]
    require(len(rows) == 24 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in ("unit_index", "tape_id", "node", "lanes", "load", "seed"))
        for row, unit in zip(rows, protocol["units"], strict=True)), "all24 registered units must pass")
    gate_rows = [{**{key: row[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
        "baseline_reproduction_passed": True, "rollout_checks_passed": row["execution_checks_passed"],
        "snapshots": {cutoff: {"maxweight": arms["factual"], "arrival_rollout": arms["replacement"]}
            for cutoff, arms in row["contrasts"]["maxweight"]["snapshots"].items()},
        "work": {"maxweight": row["cached_MW_work"], "arrival_rollout": row["work"]}} for row in rows]
    screen = summarize(gate_rows, protocol)
    totals = Counter()
    for row in rows:
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epochs=protocol["source_execution_epochs"], execution_checks_passed=True,
        completed_new_trajectories=24, reused_BG_MW_no_background_trajectory_arms=72,
        new_bg_mw_or_no_background_simulations=0, new_shadow_references=0,
        comparisons={label: contrast_summary(rows, label) for label in ("no_background", "BG", "maxweight")},
        MW_reference_screen=screen, totals_work=dict(totals), cached_totals_work=protocol["cached_totals_work"],
        statistical_scope=protocol["design"]["statistics"], BG_main_result_preserved=True,
        automatic_adoption=False, full_trajectories_retained_on_server=True))
    print('{"execution_checks_passed":true,"completed_new_trajectories":24}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "md/experiment_artifacts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--register", action="store_true")
    mode.add_argument("--unit-index", type=int)
    mode.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.register:
        register(args.artifact_dir)
    elif args.merge:
        merge(args.artifact_dir)
    else:
        run_unit(args.artifact_dir, args.unit_index)


if __name__ == "__main__":
    main()
