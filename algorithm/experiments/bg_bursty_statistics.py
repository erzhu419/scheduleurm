"""Matched seed-block uncertainty for BG's steady/bursty arrival experiment.

Each profile uses its own pairwise shared-completion populations. Difference of
differences compares the two profiles' effects, never transfers their job IDs or
P90 populations. Resampling keeps both profiles, all eight cells and all policy
contrasts together in each of the twelve new actual-arrival seed blocks.
"""
from collections import defaultdict
from math import fsum, isfinite
from random import Random

from algorithm.experiments.bg_confirmation_block_intervals import _percentile_interval
from algorithm.experiments.aligned_hardware_background_root_counterfactual import sign_counts


ARRIVAL_SEEDS = tuple(range(200001, 200013))
CELLS = tuple((node, lanes, load)
              for node, lanes in (("cpu_a", 2), ("node007", 4),
                                  ("gpu_a", 2), ("gpu_b", 2))
              for load in (0.8, 0.95))
PROFILES = ("steady", "bursty")
CONTRASTS = {
    "BG_minus_maxweight": ("maxweight", "BG"),
    "no_background_minus_BG": ("BG", "no_background"),
    "S1_only_minus_no_background": ("no_background", "S1_only"),
}
CUTOFF_TIMES = {"arrival_horizon": 900.0, "observation_end": 1200.0}
METRICS = ("restricted_mean_flow_s", "common_p90_flow_s", "unfinished_jobs")
EFFECTS = ("delta_T_mean", "delta_T_common_p90", "delta_900_unfinished",
           "delta_1200_unfinished")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20261004


def _identity(row):
    return row["node"], row["lanes"], row["load"], row["seed"]


def _blocks(rows):
    expected = {(*cell, seed) for cell in CELLS for seed in ARRIVAL_SEEDS}
    identities = [_identity(row) for row in rows]
    tape_ids = [row["tape_id"] for row in rows]
    if (len(rows) != 96 or len(set(identities)) != 96 or set(identities) != expected
            or len(set(tape_ids)) != 96):
        raise ValueError("require all twelve new seed blocks and all eight cells exactly once")
    blocks = defaultdict(list)
    for row in rows:
        for cutoff, time in CUTOFF_TIMES.items():
            pair = row["outcomes"]["snapshots"][cutoff]
            baseline, candidate = (pair[arm] for arm in ("factual", "replacement"))
            if (baseline["time"] != time or candidate["time"] != time
                    or baseline["arrivals"] != candidate["arrivals"]
                    or baseline["common_jobs"] != candidate["common_jobs"]):
                raise ValueError("each policy pair must share cutoff, birth cohort and common-ID count")
        blocks[row["seed"]].append(row)
    return [sorted(blocks[seed], key=_identity) for seed in ARRIVAL_SEEDS]


def _metric_blocks(blocks, cutoff, metric):
    values, missing = [], 0
    for block in blocks:
        pair = []
        for arm in ("factual", "replacement"):
            cells = [row["outcomes"]["snapshots"][cutoff][arm][metric] for row in block]
            missing += sum(value is None for value in cells)
            if any(value is not None and (not isfinite(value) or value < 0) for value in cells):
                raise ValueError("flow metrics and unfinished counts must be finite and nonnegative")
            pair.append(None if any(value is None for value in cells) else
                        fsum(cells) / (1 if metric == "unfinished_jobs" else len(CELLS)))
        values.append(pair)
    if missing and metric != "common_p90_flow_s":
        raise ValueError("restricted means and unfinished counts must exist for every registered cell")
    return values, missing


def _unavailable(missing):
    return dict(estimable=False, missing_policy_cell_values=missing,
                reason="missing pairwise common-ID statistic; every seed and cell retained")


def _profile_intervals(blocks, cutoff, draws):
    result = {}
    for metric in METRICS:
        values, missing = _metric_blocks(blocks, cutoff, metric)
        if missing:
            result[metric] = _unavailable(missing)
            continue
        scale = 1 if metric == "unfinished_jobs" else 1 / len(ARRIVAL_SEEDS)
        point = [fsum(value[arm] for value in values) * scale for arm in (0, 1)]
        differences, ratios = [], []
        undefined_ratios = 0
        for indices in draws:
            baseline, candidate = (fsum(values[index][arm] for index in indices) * scale
                                   for arm in (0, 1))
            differences.append(candidate - baseline)
            if metric != "unfinished_jobs":
                if baseline:
                    ratios.append(candidate / baseline - 1)
                else:
                    undefined_ratios += 1
        entry = dict(estimable=True, baseline=point[0], candidate=point[1],
                     aggregation="total_jobs" if metric == "unfinished_jobs" else "equal_tape_mean",
                     paired_difference=dict(estimate=point[1] - point[0],
                                            ci95=_percentile_interval(differences)))
        if metric != "unfinished_jobs":
            entry["relative_change"] = dict(estimate=point[1] / point[0] - 1 if point[0] else None,
                ci95=None if undefined_ratios else _percentile_interval(ratios),
                undefined_bootstrap_replicates=undefined_ratios,
                definition="candidate_mean / baseline_mean - 1")
        result[metric] = entry
    return dict(time=CUTOFF_TIMES[cutoff], metrics=result)


def _difference_of_differences(steady, bursty, cutoff, draws):
    result = {}
    for metric in METRICS:
        profile_values = [_metric_blocks(blocks, cutoff, metric) for blocks in (steady, bursty)]
        missing = sum(item[1] for item in profile_values)
        if missing:
            result[metric] = _unavailable(missing)
            continue
        scale = 1 if metric == "unfinished_jobs" else 1 / len(ARRIVAL_SEEDS)
        effects = [[candidate - baseline for baseline, candidate in item[0]]
                   for item in profile_values]
        differences = [bursty_effect - steady_effect
                       for steady_effect, bursty_effect in zip(*effects)]
        bootstrap = [fsum(differences[index] for index in indices) * scale for indices in draws]
        result[metric] = dict(estimable=True,
            aggregation="difference_of_total_jobs" if metric == "unfinished_jobs" else "equal_tape_effect_change",
            steady_effect=fsum(effects[0]) * scale, bursty_effect=fsum(effects[1]) * scale,
            paired_difference=dict(estimate=fsum(differences) * scale,
                                   ci95=_percentile_interval(bootstrap)))
    return dict(time=CUTOFF_TIMES[cutoff], metrics=result)


def _aggregate(records):
    return {metric: dict(mean=fsum(values) / len(values) if all(v is not None for v in values) else None,
        total=fsum(values) if "unfinished" in metric and all(v is not None for v in values) else None,
        signs=sign_counts(values))
        for metric in EFFECTS for values in [[record[metric] for record in records]]}


def _absolute_snapshots(records, coordinate=None):
    counts = ("arrivals", "completed_jobs", "queued_jobs", "inflight_jobs", "unfinished_jobs",
              "common_jobs" if coordinate is None else "common_class_jobs")
    flows = (("restricted_mean_flow_s", "common_p90_flow_s") if coordinate is None else
             ("restricted_mean_flow_s", "completed_p90_flow_s", "common_class_mean_flow_s",
              "common_class_p90_flow_s"))
    snapshots = {}
    for cutoff in CUTOFF_TIMES:
        snapshots[cutoff] = {}
        for arm in ("factual", "replacement"):
            groups = [record["snapshots"][cutoff][arm] for record in records]
            if coordinate is not None:
                groups = [group["per_coordinate"][coordinate] for group in groups]
            snapshots[cutoff][arm] = {metric: fsum(group[metric] for group in groups) for metric in counts}
            for metric in flows:
                values = [group[metric] for group in groups]
                snapshots[cutoff][arm][metric] = (fsum(values) / len(values)
                                                 if all(v is not None for v in values) else None)
    return snapshots


def _descriptive(rows):
    groups, seeds, classes = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["node"], row["lanes"], row["load"]].append(row["outcomes"])
        seeds[row["seed"]].append(row["outcomes"])
        for item in row["outcomes"]["per_coordinate_deltas"]:
            classes[row["node"], row["load"], item["workload"], item["coordinate"]].append(row["outcomes"])
    return dict(pooled=_aggregate([row["outcomes"] for row in rows]),
        groups=[dict(node=node, lanes=lanes, load=load, paired_tapes=len(records), **_aggregate(records),
                     absolute_snapshots=_absolute_snapshots(records))
                for (node, lanes, load), records in sorted(groups.items())],
        seed_aggregates=[dict(seed=seed, paired_cells=len(records), **_aggregate(records))
                         for seed, records in sorted(seeds.items())],
        class_results=[dict(node=node, load=load, workload=workload, coordinate=coordinate,
            paired_tapes=len(records), **_aggregate([record["per_coordinate_deltas"][coordinate] for record in records]),
            absolute_snapshots=_absolute_snapshots(records, coordinate))
            for (node, load, workload, coordinate), records in sorted(classes.items())],
        paired_results=[dict(**{key: row[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
                             **{metric: row["outcomes"][metric] for metric in EFFECTS}) for row in rows])


def analyse_bursty(comparisons_by_profile):
    """Analyse the fixed two-profile, three-contrast, twelve-seed experiment.

    Input: profile -> contrast -> 96 rows. Each row has tape identity and
    outcomes from paired_outcomes(factual, replacement, protocol, workloads).
    The factual/replacement order must follow CONTRASTS' baseline/candidate.
    """
    if set(comparisons_by_profile) != set(PROFILES):
        raise ValueError("require both registered steady and bursty profiles")
    blocks = {}
    for profile in PROFILES:
        if set(comparisons_by_profile[profile]) != set(CONTRASTS):
            raise ValueError("require all three registered policy contrasts")
        identities = None
        for contrast in CONTRASTS:
            rows = comparisons_by_profile[profile][contrast]
            blocks[profile, contrast] = _blocks(rows)
            actual = {row["tape_id"]: _identity(row) for row in rows}
            if identities is not None and actual != identities:
                raise ValueError("policy contrasts within a profile must use identical literal tapes")
            identities = actual
    rng = Random(BOOTSTRAP_SEED)
    draws = [tuple(rng.randrange(len(ARRIVAL_SEEDS)) for _ in ARRIVAL_SEEDS)
             for _ in range(BOOTSTRAP_REPLICATES)]
    profiles = {profile: {contrast: dict(
        orientation=f"{labels[1]} minus {labels[0]}",
        role="primary" if contrast == "BG_minus_maxweight" else "mechanism",
        **_descriptive(comparisons_by_profile[profile][contrast]),
        block_intervals={cutoff: _profile_intervals(blocks[profile, contrast], cutoff, draws)
                         for cutoff in CUTOFF_TIMES})
        for contrast, labels in CONTRASTS.items()} for profile in PROFILES}
    differences = {contrast: dict(orientation="bursty policy effect minus steady policy effect",
        policy_effect=f"{labels[1]} minus {labels[0]}",
        snapshots={cutoff: _difference_of_differences(
            blocks["steady", contrast], blocks["bursty", contrast], cutoff, draws)
            for cutoff in CUTOFF_TIMES}) for contrast, labels in CONTRASTS.items()}
    return dict(analysis="BG_steady_bursty_new_seed_block_uncertainty",
        arrival_seeds=ARRIVAL_SEEDS, arrival_seed_blocks=len(ARRIVAL_SEEDS),
        cells_per_profile_block=len(CELLS), paired_tapes_per_profile=96,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_generator="Python random.Random.randrange; twelve matched whole seed blocks with replacement",
        interval="pointwise percentile 95%; linear interpolation at 0.025/0.975",
        statistical_scope="New actual-arrival seed blocks, conditional on fixed planning streams; pointwise intervals without multiplicity adjustment or a new adoption gate.",
        common_p90_definition="equal-tape mean of each profile and policy pair's shared-completion P90s",
        profile_summaries=profiles, difference_of_differences=differences,
        automatic_adoption=False)
