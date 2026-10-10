"""Recover option multisets from cached simultaneous job admissions.

The replay stores completion as native ``start + offset``. Match that value
directly, retaining all matches; jobs alone do not identify idle calls, lane
identities, or the order of admissions at a shared timestamp.
"""
from collections import Counter, defaultdict
from itertools import combinations_with_replacement


def reconstruct(jobs, options, lanes):
    """Return physical admission groups and supported option combinations."""
    by_start = defaultdict(list)
    for job in jobs:
        if job["start"] is not None:
            by_start[job["start"]].append(job)
    combinations = [indices for width in range(1, lanes + 1)
        for indices in combinations_with_replacement(range(len(options)), width)]
    groups, action_counts = [], Counter()
    for start, admitted in sorted(by_start.items()):
        actual = Counter((job["coordinate"], job["completion"]) for job in admitted)
        matches = []
        for indices in combinations:
            expected = Counter((coordinate, start + offset)
                for index in indices
                for coordinate, offsets in enumerate(options[index].completion_offsets)
                for offset in offsets)
            if expected == actual:
                matches.append(indices)
        if not matches:
            raise ValueError(f"No catalogue option combination matches admissions at {start}")
        unique = matches[0] if len(matches) == 1 else None
        if unique is not None:
            action_counts.update(unique)
        dimension = len(options[0].requirements)
        groups.append(dict(start=start,
            job_ids=sorted((job["coordinate"], job["index"]) for job in admitted),
            jobs_per_coordinate=[sum(job["coordinate"] == c for job in admitted)
                for c in range(dimension)],
            completion_counts=[dict(coordinate=c, completion=end, jobs=count)
                for (c, end), count in sorted(actual.items())],
            option_combinations=matches,
            unique_option_indices=unique,
            unique_action_ids=[options[i].action_id for i in unique] if unique is not None else None,
            reserved_lane_ends=sorted(start + options[i].duration for i in unique)
                if unique is not None else None))
    return dict(group_count=len(groups),
        admitted_job_count=sum(len(group["job_ids"]) for group in groups),
        ambiguous_group_count=sum(group["unique_option_indices"] is None for group in groups),
        uniquely_reconstructed_action_counts=dict(sorted(action_counts.items())), groups=groups)
