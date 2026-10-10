"""Descriptive BG/MW comparisons under mean and empirical service execution.

Keep three complete arrival/service-seed blocks and all eight cells. Each
condition retains its own pairwise shared-completion P90 population; matched
effect changes compare statistics, not pooled jobs or independent trajectories.
"""
from collections import defaultdict
from math import fsum

from algorithm.experiments.bg_bursty_statistics import (
    CELLS, CUTOFF_TIMES, EFFECTS, _absolute_snapshots, _aggregate, _descriptive,
)
from algorithm.experiments.aligned_hardware_background_root_counterfactual import sign_counts


CONDITIONS = ("mean", "empirical")
ARRIVAL_SEEDS = (300001, 300002, 300003)
SERVICE_SEEDS = (400001, 400002, 400003)
SEED_PAIRS = dict(zip(ARRIVAL_SEEDS, SERVICE_SEEDS))
SNAPSHOT_EFFECTS = ("restricted_mean_flow_s", "common_p90_flow_s",
                    "queued_jobs", "inflight_jobs", "unfinished_jobs")


def _identity(row):
    return row["node"], row["lanes"], row["load"], row["seed"], row["service_seed"]


def _registered_rows(rows, condition, seed_pairs):
    expected = {(*cell, arrival, service) for cell in CELLS for arrival, service in seed_pairs.items()}
    identities = [_identity(row) for row in rows]
    expected_count = len(expected)
    if (len(rows) != expected_count or set(identities) != expected or len(set(identities)) != expected_count
            or len({row["tape_id"] for row in rows}) != expected_count
            or any(row["condition"] != condition for row in rows)):
        raise ValueError("require all registered cells and zipped arrival/service seed blocks")
    for row in rows:
        for cutoff, time in CUTOFF_TIMES.items():
            pair = row["outcomes"]["snapshots"][cutoff]
            baseline, candidate = (pair[arm] for arm in ("factual", "replacement"))
            if (baseline["time"] != time or candidate["time"] != time
                    or baseline["arrivals"] != candidate["arrivals"]
                    or baseline["common_jobs"] != candidate["common_jobs"]):
                raise ValueError("each policy pair must share cutoff, birth cohort and common-ID count")
    return sorted(rows, key=_identity)


def _summary(rows, seed_pairs):
    summary = _descriptive(rows)
    summary["orientation"] = "BG minus maxweight"
    summary["pooled"]["absolute_snapshots"] = _absolute_snapshots([row["outcomes"] for row in rows])
    summary["seed_blocks"] = summary.pop("seed_aggregates")
    for block in summary["seed_blocks"]:
        block["service_seed"] = seed_pairs[block["seed"]]
        block["absolute_snapshots"] = _absolute_snapshots(
            [row["outcomes"] for row in rows if row["seed"] == block["seed"]])
    for record, row in zip(summary["paired_results"], rows):
        record.update(service_seed=row["service_seed"], condition=row["condition"],
                      absolute_snapshots=_absolute_snapshots([row["outcomes"]]))
    summary["aggregation"] = "Equal-tape flow/P90 means; total cohort and common-completion task counts."
    summary["common_p90_definition"] = "Equal-tape mean of each condition's separately paired shared-completion P90s."
    return summary


def _subtract(candidate, baseline):
    return candidate - baseline if candidate is not None and baseline is not None else None


def _effect_change(mean, empirical):
    result = {key: empirical[key] for key in
              ("tape_id", "node", "lanes", "load", "seed", "service_seed")}
    result.update({metric: _subtract(empirical["outcomes"][metric], mean["outcomes"][metric])
                   for metric in EFFECTS})
    result["snapshot_effect_changes"] = {}
    for cutoff in CUTOFF_TIMES:
        changes = {}
        for metric in SNAPSHOT_EFFECTS:
            effects = []
            for row in (mean, empirical):
                pair = row["outcomes"]["snapshots"][cutoff]
                effects.append(_subtract(pair["replacement"][metric], pair["factual"][metric]))
            changes[metric] = _subtract(effects[1], effects[0])
        result["snapshot_effect_changes"][cutoff] = changes
    return result


def _change_summary(records):
    snapshots = {}
    for cutoff in CUTOFF_TIMES:
        metrics = {}
        for metric in SNAPSHOT_EFFECTS:
            values = [record["snapshot_effect_changes"][cutoff][metric] for record in records]
            complete = all(value is not None for value in values)
            count = metric.endswith("_jobs")
            metrics[metric] = dict(mean=fsum(values) / len(values) if complete else None,
                                  total=fsum(values) if count and complete else None,
                                  signs=sign_counts(values),
                                  aggregation="total_jobs" if count else "equal_tape_effect_change")
        snapshots[cutoff] = dict(time=CUTOFF_TIMES[cutoff], metrics=metrics)
    return dict(**_aggregate(records), snapshots=snapshots)


def analyse_empirical_development(comparisons_by_condition, *, arrival_seeds=None,
                                  service_seeds=None):
    """Analyse paired BG/MW rows under mean and empirical service, without intervals.

    Input condition -> one row per registered cell and seed block with tape_id/node/lanes/load/seed/service_seed/
    condition and outcomes from paired_outcomes(MW, BG, protocol, workloads).
    Both conditions must retain exactly the same arrival/service identities.
    """
    if set(comparisons_by_condition) != set(CONDITIONS):
        raise ValueError("require both registered mean and empirical execution conditions")
    arrival_seeds = tuple(SEED_PAIRS) if arrival_seeds is None else tuple(arrival_seeds)
    service_seeds = tuple(SERVICE_SEEDS) if service_seeds is None else tuple(service_seeds)
    if len(arrival_seeds) != len(service_seeds) or not arrival_seeds:
        raise ValueError("arrival and service seed rosters must be nonempty and same length")
    seed_pairs = dict(zip(arrival_seeds, service_seeds, strict=True))
    rows = {condition: _registered_rows(comparisons_by_condition[condition], condition, seed_pairs)
            for condition in CONDITIONS}
    expected = len(CELLS) * len(seed_pairs)
    for condition in CONDITIONS:
        condition_rows = rows[condition]
        identities = [_identity(row) for row in condition_rows]
        expected_identities = {(*cell, arrival, service)
                               for cell in CELLS for arrival, service in seed_pairs.items()}
        if (len(condition_rows) != expected or set(identities) != expected_identities
                or len(set(identities)) != expected):
            raise ValueError("registered cell and seed roster differs")
        for row in condition_rows:
            if row["service_seed"] != seed_pairs[row["seed"]]:
                raise ValueError("zipped arrival/service seed blocks differ")
    if ({row["tape_id"]: _identity(row) for row in rows["mean"]}
            != {row["tape_id"]: _identity(row) for row in rows["empirical"]}):
        raise ValueError("execution conditions must use identical literal tapes and seed pairs")
    changes = [_effect_change(mean, empirical)
               for mean, empirical in zip(rows["mean"], rows["empirical"], strict=True)]
    groups, blocks = defaultdict(list), defaultdict(list)
    for record in changes:
        groups[record["node"], record["lanes"], record["load"]].append(record)
        blocks[record["seed"]].append(record)
    return dict(analysis="BG_empirical_service_descriptive_development",
        arrival_seeds=arrival_seeds, service_seeds=service_seeds,
        joint_seed_blocks=len(seed_pairs), cells_per_block=len(CELLS),
        paired_tapes_per_condition=expected,
        condition_summaries={condition: _summary(rows[condition], seed_pairs) for condition in CONDITIONS},
        matched_effect_changes=dict(
            orientation="empirical BG-minus-MW effect minus mean BG-minus-MW effect",
            pooled=_change_summary(changes),
            groups=[dict(node=node, lanes=lanes, load=load, paired_tapes=len(records),
                         **_change_summary(records))
                    for (node, lanes, load), records in sorted(groups.items())],
            seed_blocks=[dict(seed=seed, service_seed=seed_pairs[seed], paired_cells=len(records),
                              **_change_summary(records)) for seed, records in sorted(blocks.items())],
            paired_results=changes),
        statistical_scope=(f"Descriptive comparison on {len(seed_pairs)} complete joint arrival/service-seed blocks, "
            f"conditional on the fixed fitted phase-derived service bank and planning rule; not{expected} independent "
            "observations, physical parameter uncertainty or a new independent BG confirmation."),
        confidence_intervals_computed=False, automatic_adoption=False,
        seed_or_service_sample_selection_performed=False)


def performance_screen(summary, performance):
    """Apply the original numerical gates separately to one execution condition."""
    snapshots = summary["pooled"]["absolute_snapshots"]
    final = snapshots["observation_end"]
    baseline, candidate = (final[arm]["restricted_mean_flow_s"] for arm in ("factual", "replacement"))
    relative = candidate / baseline - 1 if baseline and candidate is not None else None
    flow_gain = relative is not None and relative <= -performance["min_gain"]
    load_checks = {str(load): fsum(
        group["absolute_snapshots"]["observation_end"]["replacement"]["restricted_mean_flow_s"]
        - group["absolute_snapshots"]["observation_end"]["factual"]["restricted_mean_flow_s"]
        for group in summary["groups"] if group["load"] == load) <= 0 for load in (0.8, 0.95)}
    pooled_tails = {cutoff: pair["replacement"]["unfinished_jobs"] <= pair["factual"]["unfinished_jobs"]
                    for cutoff, pair in snapshots.items()}
    failures = []
    for group in summary["groups"]:
        for cutoff, pair in group["absolute_snapshots"].items():
            failed = []
            for metric in ("common_p90_flow_s", "unfinished_jobs"):
                baseline, candidate = (pair[arm][metric] for arm in ("factual", "replacement"))
                if (baseline is None or candidate is None
                        or candidate > (1 + performance["group_secondary_tolerance"]) * baseline):
                    failed.append(metric)
            if failed:
                failures.append(dict(node=group["node"], lanes=group["lanes"], load=group["load"],
                                     cutoff=cutoff, failed_metrics=failed))
    passed = flow_gain and all(load_checks.values()) and all(pooled_tails.values()) and not failures
    return dict(decision="BG_DEVELOPMENT_PERFORMANCE_" + ("PASS" if passed else "HOLD"),
        numerical_gates_passed=passed, flow_gain_threshold_passed=flow_gain,
        pooled_mean_improvement=-relative if relative is not None else None,
        load_mean_nonregression_checks=load_checks, pooled_unfinished_checks=pooled_tails,
        failed_secondary_groups=failures, group_secondary_passed=not failures,
        independent_confirmation_performed=False, automatic_adoption=False,
        scope="Original1% mean/5% group tolerance and nonincreasing pooled tails; descriptive development only.")
