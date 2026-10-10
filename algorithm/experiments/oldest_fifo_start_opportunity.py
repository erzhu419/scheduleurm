"""Describe the next free lane for visible FIFO heads after one root action.

This is an admission opportunity, not a prediction of the next dispatched root.
No arrivals, planner choices, or future job outcomes enter the calculation.
"""


def start_opportunity(requirements, duration, available, now, lane, waiting_ages):
    heads = [group[0] if group else None for group in waiting_ages]
    oldest = max((age for age in heads if age is not None), default=None)
    coordinates = tuple(c for c, age in enumerate(heads) if age is not None and age == oldest)
    ends = [max(now, end) for end in available]
    ends[lane] = now + duration
    next_free = min(ends)
    starts = tuple(None if age is None else now if requirements[c] else next_free
        for c, age in enumerate(heads))
    return dict(oldest_coordinates=coordinates, oldest_age=oldest,
        earliest_start_per_coordinate=starts,
        oldest_start_sum=sum(starts[c] - now for c in coordinates))
