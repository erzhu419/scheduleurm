"""Fixed-tape delay control and no-background ablation of the adopted BG policy."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import socket
from statistics import mean
from time import perf_counter, process_time

from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.lane_refill_simple_controls import SingleLaneSelection, select_single_lane_control
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes, sign_counts
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import cohort_metrics, check_rollout
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.bg_confirmation_block_intervals import block_intervals
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory

ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_BG_strong_controls_20261003"
OLD = "aligned_hardware_background_competition_continuous_confirmation_20260930"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
EXTRA_SOURCES = (SOURCE, "algorithm/experiments/bg_confirmation_block_intervals.py",
    "algorithm/experiments/aligned_hardware_background_root_counterfactual.py",
    "algorithm/experiments/single_action_counterfactual_replay.py")
POLICIES = ("delay", "no_background")
METRICS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished", "delta_1200_unfinished")
CUTOFFS = ("arrival_horizon", "observation_end")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def read(file):
    if str(file).endswith(".gz"):
        with gzip.open(file, "rt") as source:
            return json.load(source)
    return json.loads(Path(file).read_text())


def save(file, value):
    """Finish a server artifact before making its final name visible."""
    file, temporary = Path(file), Path(str(file) + ".tmp")
    require(not file.exists(), "preserve existing artifact: " + str(file))
    opener = gzip.open if str(file).endswith(".gz") else open
    with opener(temporary, "xt") as output:
        json.dump(value, output, separators=(",", ":"), allow_nan=False)
    temporary.rename(file)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def births(result):
    return [(job["coordinate"], job["index"], job["arrival"]) for job in result["jobs"]]


@dataclass(frozen=True)
class SelectionDecision:
    selection: SingleLaneSelection
    root_option_probes: int
    candidate_count: int
    rollout_calls: int = 0
    rollout_option_probes: int = 0
    eligible: bool = False
    changed: bool = False


def delay_selector(options, queues, available, now, lane, waiting_ages, observed_arrival_counts):
    selected = select_single_lane_control(options, queues, 1, 0, policy="delay")
    return SelectionDecision(selected, selected.option_probes, selected.candidate_evaluations)


class TimedSelector:
    """Record call latency without changing the frozen service clock or decision."""
    def __init__(self, selector):
        self.selector, self.calls, self.wall_s, self.max_wall_s = selector, 0, 0.0, 0.0

    def __call__(self, *context):
        started = perf_counter()
        decision = self.selector(*context)
        elapsed = perf_counter() - started
        self.calls += 1
        self.wall_s += elapsed
        self.max_wall_s = max(self.max_wall_s, elapsed)
        return decision


def register(directory):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    old = read(Path(directory) / (OLD + "_protocol.json"))
    completed = read(Path(directory) / (OLD + ".json"))
    require(old["registered_at_utc"] == completed["registered_at_utc"]
        == design["source_execution_epoch"] and completed["execution_checks_passed"]
        and completed["decision"] == "BACKGROUND_COMPETITION_CONTINUOUS_CONFIRMATION_PASS",
        "matching adopted BG confirmation required")
    require(len(old["tapes"]) == len(completed["paired_runs"]) == 96
        and old["planning_samples"] == 8 and old["planning_event_cap"] == 128
        and old["birth_cohort"] == [300, 900] and old["arrival_horizon"] == 900
        and old["environment_arrival_end"] == old["observation_end"] == 1200,
        "retain the original 96 tapes, budgets and cohort clock")
    for name, literal in old["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "original frozen source changed: " + name)
    cached = read(Path(directory) / (OLD + "_trajectories.json.gz"))
    require(cached["registered_at_utc"] == old["registered_at_utc"], "cached epoch differs")
    originals = {row["tape_id"]: row for row in cached["runs"]}
    references = {row["tape_id"]: row for row in completed["paired_runs"]}
    require(len(originals) == len(references) == 96
        and set(originals) == set(references) == {t["tape_id"] for t in old["tapes"]},
        "all original confirmation tapes must appear once")
    models = {model["node"]: model for model in old["models"]}
    epoch, units = datetime.now(timezone.utc).isoformat(), []
    for index, tape in enumerate(old["tapes"]):
        original, reference = originals[tape["tape_id"]]["policies"], references[tape["tape_id"]]
        require(reference["baseline_reproduction_passed"] and reference["rollout_checks_passed"], "cached checks failed")
        require(births(original["maxweight"]) == births(original["arrival_rollout"]) ==
            [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])], "cached births differ")
        for cutoff in CUTOFFS:
            require(same(cohort_metrics(original, old[cutoff], old["birth_cohort"]), reference["snapshots"][cutoff]),
                "cached cohort snapshots differ")
        fixture = path(directory, f"_fixture_{index:03d}.json.gz")
        save(fixture, dict(registered_at_utc=epoch, source_execution_epoch=old["registered_at_utc"],
            model=models[tape["node"]], tape=tape, cached_policies={
                label: {"jobs": result["jobs"], "snapshots": result["snapshots"]}
                for label, result in original.items()}, reference_snapshots=reference["snapshots"]))
        for policy_index, policy in enumerate(POLICIES):
            units.append(dict(unit_index=policy_index * 96 + index, policy=policy,
                tape_index=index, fixture_file=fixture.name,
                **{key: tape[key] for key in ("tape_id", "node", "lanes", "load", "seed")}))
    body = dict(experiment=STEM, registered_at_utc=epoch, design=design,
        source_execution_epoch=old["registered_at_utc"], original_BG_decision=completed["decision"],
        units=sorted(units, key=lambda unit: unit["unit_index"]),
        **{key: old[key] for key in ("models", "loads", "arrival_seeds", "birth_cohort", "arrival_horizon",
            "environment_arrival_end", "observation_end", "planning_samples", "planning_event_cap", "clock")},
        execution_sources={**old["execution_sources"], **{name: (ROOT / name).read_text() for name in EXTRA_SOURCES}},
        expected_new_trajectories=192, new_BG_or_MW_simulations=0, new_shadow_references=0,
        statistical_scope="Reused original confirmation arrival tapes, 12 paired seed blocks conditional on fixed planning streams; no new independent arrival confirmation.",
        ablation="ContinuousArrivalAgeTailLaneRollout: remove only sampled post-tau_max background competition and its combined-event overflow/work; retain tagged FIFO S1/S2, MW continuation, busy commitments, count/now, 8 prefix samples/cap128 and outer planning seed.",
        comparison_orientation="new control minus cached BG; negative means control is better",
        cached_totals_work=completed["totals_work"],
        registration_work=dict(cached_full_trajectories_read=192, fixture_files=96,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu))
    save(path(directory, "_protocol.json"), body)
    print(json.dumps(dict(registered=True, scientific_units=192, new_BG_or_MW_simulations=0)), flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(protocol["design"] == read(path(directory, "_design.json")), "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 192, "unit-index must be 0..191")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:03d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed unit artifacts")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and fixture["source_execution_epoch"] == protocol["source_execution_epoch"]
        and fixture["tape"]["tape_id"] == unit["tape_id"], "fixture identity/epoch differs")
    model, = load_models([fixture["model"]])
    tape, cached = fixture["tape"], fixture["cached_policies"]
    planner = (ContinuousArrivalAgeTailLaneRollout(model.options,
        f"arrival-plan-v1:{model.node}:{tape['load']}:online", 8, 128)
        if unit["policy"] == "no_background" else None)
    selector = TimedSelector(planner if planner is not None else delay_selector)
    result = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"],
        protocol["environment_arrival_end"], protocol["observation_end"], selector, policy=unit["policy"])
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(births(result) == births(cached["arrival_rollout"]) == births(cached["maxweight"]), "new trajectory births differ")
    if planner is not None:
        require(check_rollout(result, model, protocol, planner), "no-background rate/risk/budget checks failed")
    for cutoff in CUTOFFS:
        require(same(cohort_metrics(cached, protocol[cutoff], protocol["birth_cohort"]), fixture["reference_snapshots"][cutoff]),
            "original cached metrics changed")
    outcomes = paired_outcomes(cached["arrival_rollout"], result, protocol, model.workloads)
    work = (work_totals(result, planner, protocol["arrival_horizon"]) if planner is not None
        else {**result["rollout_work"], "total_option_probes": result["rollout_work"]["root_option_probes"]})
    work["controller_constructors"] = int(planner is not None)
    row = dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, outcomes=outcomes,
        snapshots={cutoff: {"BG": arms["factual"], "control": arms["replacement"]}
            for cutoff, arms in outcomes["snapshots"].items()},
        versus_MW=paired_outcomes(cached["maxweight"], result, protocol, model.workloads), work=work,
        computation=dict(host=socket.gethostname(), selector_calls=selector.calls,
            selector_wall_s=selector.wall_s, max_selector_wall_s=selector.max_wall_s,
            mean_selector_wall_s=selector.wall_s / selector.calls,
            unit_wall_s=perf_counter() - started, unit_cpu_s=process_time() - cpu,
            computation_delays_service=False))
    save(row_file, row)
    print(json.dumps(dict(unit_index=index, policy=unit["policy"], execution_checks_passed=True)), flush=True)


def aggregate(rows):
    return {metric: dict(mean=mean(values) if all(v is not None for v in values) else None,
        total=sum(values) if "unfinished" in metric and all(v is not None for v in values) else None, signs=sign_counts(values))
        for metric in METRICS for values in [[row["outcomes"][metric] for row in rows]]}


def merge(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    require(len(rows) == 192 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in ("unit_index", "policy", "tape_id", "node", "lanes", "load", "seed"))
        for row, unit in zip(rows, protocol["units"], strict=True)), "all 192 registered units must pass once")
    policies = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        groups, classes, seeds, work = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
        for row in selected:
            groups[row["node"], row["load"]].append(row)
            seeds[row["seed"]].append(row)
            for item in row["outcomes"]["per_coordinate_deltas"]:
                classes[row["node"], row["load"], item["workload"]].append(item)
            for key, value in row["work"].items():
                work[key] = max(work[key], value) if key in ("max_rollout_calls", "maximum_rollout_calls") else work[key] + value
        policies[policy] = dict(tapes=len(selected), pooled=aggregate(selected),
            block_intervals={cutoff: block_intervals(selected, labels=("BG", "control"), cutoff=cutoff)
                for cutoff in CUTOFFS},
            groups=[dict(node=node, load=load, **aggregate(group)) for (node, load), group in sorted(groups.items())],
            seed_aggregates=[dict(seed=seed, paired_cells=len(group), **aggregate(group)) for seed, group in sorted(seeds.items())],
            class_results=[dict(node=node, load=load, workload=workload,
                **{metric: dict(mean=mean(values) if all(v is not None for v in values) else None,
                    total=sum(values) if "unfinished" in metric else None, signs=sign_counts(values))
                    for metric in METRICS for values in [[item[metric] for item in group]]})
                for (node, load, workload), group in sorted(classes.items())], totals_work=dict(work),
            class_tails=[dict(node=node, load=load, coordinate=coordinate,
                counts={cutoff: {arm: {metric: sum(row["snapshots"][cutoff][arm]["per_coordinate"][coordinate][metric]
                    for row in group) for metric in ("queued_jobs", "inflight_jobs", "unfinished_jobs")}
                    for arm in ("BG", "control")} for cutoff in CUTOFFS})
                for (node, load), group in sorted(groups.items()) for coordinate in range(3)],
            computation=dict(selector_wall_s=sum(row["computation"]["selector_wall_s"] for row in selected),
                worker_cpu_s=sum(row["computation"]["unit_cpu_s"] for row in selected),
                max_selector_wall_s=max(row["computation"]["max_selector_wall_s"] for row in selected)),
            paired_results=[{**{key: row[key] for key in ("unit_index", "tape_id", "node", "load", "seed")},
                **{key: value for key, value in row["outcomes"].items() if key != "snapshots"},
                "versus_MW": {key: row["versus_MW"][key] for key in METRICS},
                "tails": {cutoff: {arm: {key: snapshot[key] for key in ("arrivals", "queued_jobs", "inflight_jobs", "unfinished_jobs", "common_jobs")}
                    for arm, snapshot in row["snapshots"][cutoff].items()} for cutoff in CUTOFFS}}
                for row in selected])
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epoch=protocol["source_execution_epoch"], execution_checks_passed=True,
        completed_new_trajectories=192, reused_BG_MW_trajectories=192, new_BG_or_MW_simulations=0,
        new_shadow_references=0, comparison_orientation=protocol["comparison_orientation"],
        statistical_scope=protocol["statistical_scope"], policies=policies,
        cached_totals_work=protocol["cached_totals_work"], registration_work=protocol["registration_work"],
        merge_work=dict(unit_rows_read=192, block_interval_calls=4,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu),
        BG_main_result_preserved=True, automatic_adoption=False, performance_gain_established=False))
    print(json.dumps(dict(execution_checks_passed=True, completed_new_trajectories=192)), flush=True)


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
