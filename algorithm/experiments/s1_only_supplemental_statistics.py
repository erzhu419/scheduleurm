"""Paired seed-block intervals for the post-confirmation S1-only supplement.

Use each comparator's own shared-completion snapshots. In particular, a single
S1-only trajectory can have different common P90 values against different arms.
The original BG confirmation and its gates remain unchanged.
"""
from algorithm.experiments.bg_confirmation_block_intervals import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CUTOFF_TIMES, block_intervals,
)


COMPARATORS = ("no_background", "BG", "maxweight")


def supplemental_statistics(rows):
    """Analyse all 96 unit rows using 12 complete paired arrival-seed blocks.

    Rows use the S1-only development runner's identity and contrasts schema.
    Differences and relative changes are S1-only minus each comparator; the
    denominator of a relative change is the resampled aggregate comparator.
    """
    rows = list(rows)
    comparisons = {}
    for comparator in COMPARATORS:
        paired_rows = [dict(
            **{key: row[key] for key in ("tape_id", "node", "lanes", "load", "seed")},
            snapshots={cutoff: {
                comparator: pair["factual"], "S1_only": pair["replacement"],
            } for cutoff, pair in row["contrasts"][comparator]["snapshots"].items()},
        ) for row in rows]
        comparisons[comparator] = dict(
            role="primary" if comparator == "no_background" else "descriptive",
            orientation="S1-only minus " + comparator,
            block_intervals={cutoff: block_intervals(
                paired_rows, labels=(comparator, "S1_only"), cutoff=cutoff,
                bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            ) for cutoff in CUTOFF_TIMES},
        )
    return dict(
        analysis="post_confirmation_S1_only_supplemental_paired_seed_blocks",
        primary_comparator="no_background", comparisons=comparisons,
        statistical_scope=("Post-confirmation supplemental comparison on the original 96 tapes; "
            "12 whole actual-arrival seed blocks, conditional on fixed planning streams. "
            "Pointwise percentile 95% intervals without multiplicity adjustment; "
            "no new independent BG confirmation and no new PASS gate."),
        common_p90_definition="equal-tape mean of each comparator pair's shared-completion P90s",
        BG_main_result_preserved=True, automatic_adoption=False,
    )
