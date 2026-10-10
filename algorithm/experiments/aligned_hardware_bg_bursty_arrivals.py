"""New-seed BG performance under matched steady and periodic arrival rates."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter, process_time

from algorithm.background_competition_continuous_lane_rollout import BackgroundCompetitionContinuousLaneRollout
from algorithm.continuous_arrival_age_tail_lane_rollout import ContinuousArrivalAgeTailLaneRollout
from algorithm.continuous_arrival_s1_only_lane_rollout import ContinuousArrivalS1OnlyLaneRollout
from algorithm.experiments.aligned_hardware_bg_strong_controls import births, read, require, save
from algorithm.experiments.aligned_hardware_background_competition_continuous import background_work, check_rollout as check_bg
from algorithm.experiments.aligned_hardware_background_root_counterfactual import paired_outcomes
from algorithm.experiments.aligned_hardware_continuous_arrival_age_tail import check_rollout as check_no_background
from algorithm.experiments.aligned_hardware_frame_experiment import poisson_arrivals
from algorithm.experiments.aligned_hardware_observed_rate_age_tail_development import work_totals
from algorithm.experiments.aligned_hardware_performance_necessity import load_models
from algorithm.experiments.aligned_hardware_s1_only_development import check_s1_only
from algorithm.experiments.aligned_hardware_s1_only_supplemental import ProgressSelector
from algorithm.experiments.bg_bursty_statistics import analyse_bursty, ARRIVAL_SEEDS, CELLS, CONTRASTS, PROFILES
from algorithm.experiments.matched_bursty_arrivals import modulate_arrival_tape, modulation_high_first
from algorithm.experiments.observed_rate_contextual_lane_refill_replay import simulate_observed_rate_contextual_trajectory

ROOT = Path(__file__).resolve().parents[2]
STEM = "aligned_hardware_BG_bursty_arrivals_20261004"
OLD = "aligned_hardware_background_competition_continuous_confirmation_20260930"
S1 = "aligned_hardware_S1_only_supplemental_20261004"
POLICIES = ("BG", "maxweight", "no_background", "S1_only")
IDENTITY = ("tape_id", "node", "lanes", "load", "seed")
CUTOFFS = ("arrival_horizon", "observation_end")
EXTRA_SOURCES = (str(Path(__file__).resolve().relative_to(ROOT)),
    "algorithm/experiments/matched_bursty_arrivals.py", "algorithm/experiments/bg_bursty_statistics.py")


def path(directory, suffix):
    return Path(directory) / (STEM + suffix)


def register(directory):
    started, cpu = perf_counter(), process_time()
    design = read(path(directory, "_design.json"))
    old = read(Path(directory) / (OLD + "_protocol.json"))
    completed = read(Path(directory) / (OLD + ".json"))
    s1 = read(Path(directory) / (S1 + "_protocol.json"))
    s1_result = read(Path(directory) / (S1 + "_summary.json"))
    require(old["registered_at_utc"] == completed["registered_at_utc"] == design["source_execution_epochs"]["BG"]
        and s1["registered_at_utc"] == s1_result["registered_at_utc"] == design["source_execution_epochs"]["S1_supplemental"]
        and completed["execution_checks_passed"] and s1_result["execution_checks_passed"], "frozen source epochs/checks differ")
    sources = {}
    for protocol in (old, s1):
        for name, literal in protocol["execution_sources"].items():
            require((ROOT / name).read_text() == literal, "frozen dependency changed: " + name)
            sources[name] = literal
    require(old["models"] == s1["models"] and old["planning_samples"] == 8 and old["planning_event_cap"] == 128
        and old["birth_cohort"] == [300, 900] and old["arrival_horizon"] == 900
        and old["environment_arrival_end"] == old["observation_end"] == 1200
        and list(ARRIVAL_SEEDS) == design["population"]["arrival_seeds"]
        and not set(ARRIVAL_SEEDS) & set(old["arrival_seeds"]), "new seeds and original model/budget required")
    epoch = datetime.now(timezone.utc).isoformat()
    models = {model["node"]: model for model in old["models"]}
    tapes, units, paired_tape_checks = [], [], []
    for node, lanes, load in CELLS:
        model = models[node]
        require(model["lanes"] == lanes, "catalogue lane identity differs")
        rates = next(row["arrival_rates"] for row in old["capacities"] if row["node"] == node and row["load"] == load)
        for seed in ARRIVAL_SEEDS:
            steady = poisson_arrivals(rates, 1200., seed)
            bursty = modulate_arrival_tape(steady, seed)
            for cutoff in (300, 900, 1200):
                require({(c, i) for t, c, i in steady if t <= cutoff}
                    == {(c, i) for t, c, i in bursty if t <= cutoff}, "matched cutoff births differ")
            require({(c, i) for t, c, i in steady if 300 <= t <= 900}
                == {(c, i) for t, c, i in bursty if 300 <= t <= 900}, "matched cohort IDs differ")
            paired_tape_checks.append(dict(node=node, lanes=lanes, load=load, seed=seed,
                total_jobs=len(steady), cohort_jobs=sum(300 <= t <= 900 for t, _, _ in steady),
                high_first=modulation_high_first(seed)))
            for profile, arrivals in zip(PROFILES, (steady, bursty), strict=True):
                index = len(tapes)
                tape = dict(tape_id=f"{profile}:{node}:R{lanes}:load{load}:seed{seed}",
                    profile=profile, node=node, lanes=lanes, load=load, seed=seed, arrivals=arrivals)
                fixture = path(directory, f"_fixture_{index:03d}.json.gz")
                save(fixture, dict(registered_at_utc=epoch, model=model, tape=tape))
                tapes.append(tape)
                for policy in POLICIES:
                    units.append(dict(unit_index=len(units), tape_index=index, fixture_file=fixture.name,
                        profile=profile, policy=policy, **{key: tape[key] for key in IDENTITY}))
    require(len(tapes) == 192 and len(units) == 768, "complete two-profile four-policy roster required")
    sources.update({name: (ROOT / name).read_text() for name in EXTRA_SOURCES})
    save(path(directory, "_protocol.json"), dict(experiment=STEM, registered_at_utc=epoch, design=design,
        source_execution_epochs=design["source_execution_epochs"],
        **{key: old[key] for key in ("models", "capacities", "loads", "performance", "planning_samples",
            "planning_event_cap", "planning_seed_rule", "clock", "birth_cohort", "arrival_horizon",
            "environment_arrival_end", "observation_end")},
        arrival_seeds=ARRIVAL_SEEDS, profiles=PROFILES, policies=POLICIES, tapes=tapes, units=units,
        execution_sources=sources, paired_tape_checks=paired_tape_checks,
        expected_new_trajectories=768, new_shadow_references=0, reused_trajectory_arms=0,
        statistical_scope=design["statistics"], limitations=design["limitations"],
        registration_work=dict(literal_steady_tapes_generated=96, modulated_tapes_generated=96,
            matched_boundary_and_cohort_checks=96, fixtures=192,
            wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"registered":true,"scientific_units":768,"new_shadow_references":0}', flush=True)


def load_protocol(directory):
    protocol = read(path(directory, "_protocol.json"))
    require(protocol["design"] == read(path(directory, "_design.json")), "registered design changed")
    for name, literal in protocol["execution_sources"].items():
        require((ROOT / name).read_text() == literal, "registered source changed: " + name)
    return protocol


def run_unit(directory, index):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    require(0 <= index < 768, "unit-index must be0..767")
    unit = protocol["units"][index]
    row_file, trajectory_file = (path(directory, f"_unit_{index:03d}{suffix}")
        for suffix in (".json", "_trajectory.json.gz"))
    require(not row_file.exists() and not trajectory_file.exists(), "preserve completed trajectory")
    fixture = read(Path(directory) / unit["fixture_file"])
    require(fixture["registered_at_utc"] == protocol["registered_at_utc"]
        and all(fixture["tape"][key] == unit[key] for key in (*IDENTITY, "profile")), "fixture identity/epoch differs")
    model, = load_models([fixture["model"]])
    tape, policy = fixture["tape"], unit["policy"]
    constructor = {"BG": BackgroundCompetitionContinuousLaneRollout,
        "maxweight": ContinuousArrivalAgeTailLaneRollout, "no_background": ContinuousArrivalAgeTailLaneRollout,
        "S1_only": ContinuousArrivalS1OnlyLaneRollout}[policy]
    planner = constructor(model.options, f"arrival-plan-v1:{model.node}:{tape['load']}:online", 8, 128)
    selector = ProgressSelector(planner.maxweight if policy == "maxweight" else planner)
    result = simulate_observed_rate_contextual_trajectory(model, tape["arrivals"], 1200, 1200, selector, policy=policy)
    save(trajectory_file, dict(registered_at_utc=protocol["registered_at_utc"], unit=unit, result=result))
    require(births(result) == [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])], "actual births differ")
    if policy != "maxweight":
        checker = {"BG": check_bg, "no_background": check_no_background, "S1_only": check_s1_only}[policy]
        require(checker(result, model, protocol, planner), "frozen causal/budget/selection checks failed")
    work = (background_work if policy == "BG" else work_totals)(result, planner, 900)
    work["controller_constructors"] = 1
    if policy == "S1_only":
        for key in ("diagnostic_S2_comparisons", "root_ordering_comparisons", "strict_improvement_comparisons"):
            work[key] = sum(frame["rollout"][key] for frame in result["frames"])
        work["diagnostic_S2_exceeding_decisions"] = sum(frame["rollout"]["chosen_exceeds_MW_S2"] for frame in result["frames"])
    save(row_file, dict(**unit, registered_at_utc=protocol["registered_at_utc"], execution_checks_passed=True,
        trajectory_file=trajectory_file.name, work=work,
        computation=dict(selector_calls=selector.calls, selector_wall_s=selector.wall_s,
            max_selector_wall_s=selector.max_wall_s, wall_s=perf_counter() - started, cpu_s=process_time() - cpu,
            computation_delays_service=False)))
    print('{"unit_index":' + str(index) + ',"execution_checks_passed":true}', flush=True)


def performance_screen(primary, performance):
    """Apply original numerical gates to new BG/MW pairs, without a shadow claim."""
    cutoff = primary["block_intervals"]["observation_end"]["metrics"]
    flow_gain = cutoff["restricted_mean_flow_s"]["relative_change"]["estimate"] <= -performance["min_gain"]
    load_checks = {str(load): mean(group["absolute_snapshots"]["observation_end"]["replacement"]["restricted_mean_flow_s"]
        - group["absolute_snapshots"]["observation_end"]["factual"]["restricted_mean_flow_s"]
        for group in primary["groups"] if group["load"] == load) <= 0 for load in (0.8, 0.95)}
    pooled_tails = {name: item["metrics"]["unfinished_jobs"]["paired_difference"]["estimate"] <= 0
        for name, item in primary["block_intervals"].items()}
    failures = []
    for group in primary["groups"]:
        for name, pair in group["absolute_snapshots"].items():
            failed = []
            for metric in ("common_p90_flow_s", "unfinished_jobs"):
                baseline, candidate = (pair[arm][metric] for arm in ("factual", "replacement"))
                if baseline is None or candidate is None or candidate > (1 + performance["group_secondary_tolerance"]) * baseline:
                    failed.append(metric)
            if failed:
                failures.append(dict(node=group["node"], lanes=group["lanes"], load=group["load"], cutoff=name,
                    failed_metrics=failed))
    passed = flow_gain and all(load_checks.values()) and all(pooled_tails.values()) and not failures
    return dict(decision="BG_EXTERNAL_PERFORMANCE_" + ("PASS" if passed else "HOLD"),
        performance_gain_established=passed, flow_gain_threshold_passed=flow_gain,
        pooled_mean_improvement=-cutoff["restricted_mean_flow_s"]["relative_change"]["estimate"],
        load_mean_nonregression_checks=load_checks, pooled_unfinished_checks=pooled_tails,
        failed_secondary_groups=failures, group_secondary_passed=not failures,
        independent_shadow_reproduction_performed=False,
        validation="Unchanged frozen MW engine; actual literal-birth identity, feasible support and FIFO execution checks on each new trajectory.",
        scope="Original1% mean/5% group tolerance and nonincreasing pooled tails; descriptive per-profile screen, no parameter selection or automatic adoption.")


def merge(directory):
    started, cpu = perf_counter(), process_time()
    protocol = load_protocol(directory)
    rows = [read(path(directory, f"_unit_{unit['unit_index']:03d}.json")) for unit in protocol["units"]]
    require(len(rows) == 768 and all(row["execution_checks_passed"]
        and row["registered_at_utc"] == protocol["registered_at_utc"]
        and all(row[key] == unit[key] for key in ("unit_index", "policy", "profile", *IDENTITY))
        for row, unit in zip(rows, protocol["units"], strict=True)), "all768 registered units must pass")
    indexed = {(row["tape_id"], row["policy"]): row for row in rows}
    comparisons = {profile: {contrast: [] for contrast in CONTRASTS} for profile in PROFILES}
    models = {model["node"]: model for model in protocol["models"]}
    for tape in protocol["tapes"]:
        arms = {}
        for policy in POLICIES:
            row = indexed[tape["tape_id"], policy]
            saved = read(Path(directory) / row["trajectory_file"])
            require(saved["registered_at_utc"] == protocol["registered_at_utc"]
                and all(saved["unit"][key] == row[key] for key in ("unit_index", "policy", "profile", *IDENTITY)),
                "saved trajectory identity/epoch differs")
            arms[policy] = {"jobs": saved["result"]["jobs"]}
        expected = [(int(c), int(i), float(t)) for t, c, i in sorted(tape["arrivals"])]
        require(all(births(arm) == expected for arm in arms.values()), "four-policy actual births differ")
        for contrast, (baseline, candidate) in CONTRASTS.items():
            outcomes = paired_outcomes(arms[baseline], arms[candidate], protocol, models[tape["node"]]["workloads"])
            comparisons[tape["profile"]][contrast].append(dict(**{key: tape[key] for key in IDENTITY},
                profile=tape["profile"], outcomes=outcomes))
    statistics = analyse_bursty(comparisons)
    work = {profile: {policy: Counter() for policy in POLICIES} for profile in PROFILES}
    for row in rows:
        totals = work[row["profile"]][row["policy"]]
        for key, value in row["work"].items():
            totals[key] = max(totals[key], value) if key == "max_rollout_calls" else totals[key] + value
    profiles = statistics["profile_summaries"]
    save(path(directory, "_summary.json"), dict(experiment=STEM, registered_at_utc=protocol["registered_at_utc"],
        source_execution_epochs=protocol["source_execution_epochs"], execution_checks_passed=True,
        completed_new_trajectories=768, literal_tapes=192, reused_trajectory_arms=0, new_shadow_references=0,
        statistics=statistics, performance_screens={profile: performance_screen(profiles[profile]["BG_minus_maxweight"],
            protocol["performance"]) for profile in PROFILES},
        totals_work=work, paired_tape_checks=protocol["paired_tape_checks"],
        statistical_scope=protocol["statistical_scope"], limitations=protocol["limitations"],
        BG_main_result_preserved=True, automatic_adoption=False, full_trajectories_retained_on_server=True,
        merge_work=dict(unit_rows_read=768, full_trajectory_records_read=768,
            separate_profile_policy_pairs=576, wall_s=perf_counter() - started, cpu_s=process_time() - cpu)))
    print('{"execution_checks_passed":true,"completed_new_trajectories":768}', flush=True)


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
