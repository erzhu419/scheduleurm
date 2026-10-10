"""Three fixed planning-stream replications of BG on matched arrival tapes.

Average the three pair-specific statistics inside each arrival cell first, then
resample twelve whole actual-arrival seed blocks. MW's raw cohort statistics are
reused once; its shared-completion statistics may differ against each BG stream.
No synthetic pooled trajectory or planning-stream random-effects inference is
constructed.
"""
from math import fsum
from random import Random

from algorithm.experiments.bg_bursty_statistics import (
    ARRIVAL_SEEDS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CUTOFF_TIMES, EFFECTS,
    PROFILES, _blocks, _descriptive, _difference_of_differences, _identity,
    _profile_intervals,
)


PLANNING_REPLICATIONS = (1, 2, 3)
COUNTS = ("arrivals", "completed_jobs", "queued_jobs", "inflight_jobs", "unfinished_jobs")
OWN_COHORT_METRICS = (*COUNTS, "restricted_mean_flow_s")
CLASS_FLOWS = ("restricted_mean_flow_s", "completed_p90_flow_s",
               "common_class_mean_flow_s", "common_class_p90_flow_s")


def _mean(values):
    return fsum(values) / len(values) if all(value is not None for value in values) else None


def _average_snapshot(samples, coordinate=None):
    if coordinate is not None:
        samples = [sample["per_coordinate"][coordinate] for sample in samples]
    fields = (*COUNTS, *(CLASS_FLOWS if coordinate is not None else
                         ("restricted_mean_flow_s", "common_p90_flow_s")),
              "common_class_jobs" if coordinate is not None else "common_jobs")
    result = {field: _mean([sample[field] for sample in samples]) for field in fields}
    if coordinate is None:
        result["time"] = samples[0]["time"]
        result["per_coordinate"] = [_average_snapshot(samples, coordinate)
                                     for coordinate in range(len(samples[0]["per_coordinate"]))]
    return result


def _validate_reused_MW(rows):
    """Only full-cohort statistics are invariant; common populations are paired."""
    for cutoff in CUTOFF_TIMES:
        snapshots = [row["outcomes"]["snapshots"][cutoff]["factual"] for row in rows]
        for coordinate in (None, *range(len(snapshots[0]["per_coordinate"]))):
            samples = snapshots if coordinate is None else [s["per_coordinate"][coordinate] for s in snapshots]
            if any(any(sample[field] != samples[0][field] for field in OWN_COHORT_METRICS)
                   for sample in samples[1:]):
                raise ValueError("the reused MW full-cohort counts and means must match across planning streams")


def _average_cell(rows):
    _validate_reused_MW(rows)
    outcomes = [row["outcomes"] for row in rows]
    classes = [outcome["per_coordinate_deltas"] for outcome in outcomes]
    labels = [[(item["coordinate"], item["workload"]) for item in group] for group in classes]
    if any(group != labels[0] for group in labels[1:]):
        raise ValueError("planning replications must preserve class identities and ordering")
    averaged = {metric: _mean([outcome[metric] for outcome in outcomes]) for metric in EFFECTS}
    averaged["snapshots"] = {cutoff: {arm: _average_snapshot(
        [outcome["snapshots"][cutoff][arm] for outcome in outcomes])
        for arm in ("factual", "replacement")} for cutoff in CUTOFF_TIMES}
    averaged["per_coordinate_deltas"] = [dict(coordinate=coordinate, workload=workload,
        **{metric: _mean([group[index][metric] for group in classes]) for metric in EFFECTS})
        for index, (coordinate, workload) in enumerate(labels[0])]
    return dict(**{key: rows[0][key] for key in ("tape_id", "node", "lanes", "load", "seed", "profile")},
                outcomes=averaged)


def _summary(rows, blocks, draws, averaged):
    result = dict(orientation="BG minus maxweight", role="primary", **_descriptive(rows),
        block_intervals={cutoff: _profile_intervals(blocks, cutoff, draws) for cutoff in CUTOFF_TIMES})
    if averaged:
        result["planning_replications"] = PLANNING_REPLICATIONS
        result["aggregation"] = (
            "Arithmetic mean of three pair-specific statistics within each identical arrival cell; "
            "equal-cell flow means and mean total task counts over the three planning replications.")
        result["common_p90_definition"] = (
            "Mean of three separately paired shared-completion P90s within each cell, then equal-tape mean; "
            "no pooling of completion populations.")
        result["count_definition"] = (
            "Every displayed cohort or common-completion count is averaged over the three streams before "
            "summing cells; reused MW cohort counts are counted once, not tripled.")
        for interval in result["block_intervals"].values():
            interval["metrics"]["unfinished_jobs"]["aggregation"] = "mean_total_jobs_over_three_planning_replications"
    else:
        replication = rows[0]["planning_replication"]
        result["planning_replication"] = replication
        for output, row in zip(result["paired_results"], rows):
            output["planning_replication"] = replication
            if "planning_seed" in row:
                output["planning_seed"] = row["planning_seed"]
    return result


def analyse_planning_streams(rows_by_profile):
    """Return averaged and individual fixed-stream summaries on all192 tapes.

    Input profile -> 288 rows with planning_replication=1,2,3 and outcomes from
    paired_outcomes(MW, BG, protocol, workloads). Each replication contains the
    same96 tape identities; each profile has the twelve registered arrival seeds.
    """
    if set(rows_by_profile) != set(PROFILES):
        raise ValueError("require both registered steady and bursty profiles")
    stream_rows, stream_blocks, averaged_rows, averaged_blocks = {}, {}, {}, {}
    for profile in PROFILES:
        rows = rows_by_profile[profile]
        if (len(rows) != 288 or any(row["profile"] != profile for row in rows)
                or {row["planning_replication"] for row in rows} != set(PLANNING_REPLICATIONS)):
            raise ValueError("require all288 paired rows and all three fixed streams in each profile")
        reference = None
        by_cell = {}
        for replication in PLANNING_REPLICATIONS:
            selected = [row for row in rows if row["planning_replication"] == replication]
            stream_rows[profile, replication] = selected
            stream_blocks[profile, replication] = _blocks(selected)
            identities = {row["tape_id"]: _identity(row) for row in selected}
            if reference is not None and identities != reference:
                raise ValueError("planning streams must use the same literal tapes within each profile")
            reference = identities
            for row in selected:
                by_cell.setdefault(_identity(row), []).append(row)
        averaged_rows[profile] = [_average_cell(by_cell[identity]) for identity in sorted(by_cell)]
        averaged_blocks[profile] = _blocks(averaged_rows[profile])
    rng = Random(BOOTSTRAP_SEED)
    draws = [tuple(rng.randrange(len(ARRIVAL_SEEDS)) for _ in ARRIVAL_SEEDS)
             for _ in range(BOOTSTRAP_REPLICATES)]
    profiles = {profile: dict(
        averaged=_summary(averaged_rows[profile], averaged_blocks[profile], draws, True),
        replications={str(replication): _summary(stream_rows[profile, replication],
            stream_blocks[profile, replication], draws, False) for replication in PLANNING_REPLICATIONS})
        for profile in PROFILES}
    differences = dict(averaged=dict(
        orientation="bursty minus steady of the within-cell three-stream mean BG-minus-MW effect",
        snapshots={cutoff: _difference_of_differences(averaged_blocks["steady"],
            averaged_blocks["bursty"], cutoff, draws) for cutoff in CUTOFF_TIMES}),
        replications={str(replication): dict(orientation="bursty minus steady BG-minus-MW effect",
            snapshots={cutoff: _difference_of_differences(stream_blocks["steady", replication],
                stream_blocks["bursty", replication], cutoff, draws) for cutoff in CUTOFF_TIMES})
            for replication in PLANNING_REPLICATIONS})
    for interval in differences["averaged"]["snapshots"].values():
        interval["metrics"]["unfinished_jobs"]["aggregation"] = "difference_of_mean_total_jobs_over_three_planning_replications"
    return dict(analysis="BG_three_fixed_planning_streams_sensitivity",
        arrival_seeds=ARRIVAL_SEEDS, arrival_seed_blocks=len(ARRIVAL_SEEDS),
        cells_per_profile_block=8, paired_tapes_per_profile=96,
        planning_replications=PLANNING_REPLICATIONS, paired_runs_per_profile=288,
        bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_generator="Python random.Random.randrange; twelve matched whole arrival-seed blocks with replacement",
        interval="pointwise percentile95%; linear interpolation at0.025/0.975",
        statistical_scope=("Three prespecified fixed planning streams on the existing192 arrival tapes; "
            "average within each cell before resampling twelve actual-arrival seed blocks, retaining both profiles "
            "and all streams together. Conditional on these three streams; no stream selection, no576-sample "
            "independence assumption, no planning-stream random-effects inference or new independent confirmation."),
        profile_summaries=profiles, difference_of_differences=differences,
        automatic_adoption=False, stream_selection_performed=False)
