"""Price later-arrival waiting using cached H FIFO clearance clocks.

For each class, C is the last admission of its jobs waiting at H, or H when
that queue is empty.  The proxy is lambda * (C-H)^2 / 2.  Rates are the
anchor's original estimates; no actual future births or labels enter it.

Limitations: clearance depends on later arrival competition, so this is a
fixed-clock externality proxy, not a proven expected waiting lower bound.
It omits later jobs' service costs and represents lane coupling only through
the cached class clearance clocks.
"""
from fractions import Fraction


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (list, tuple)):
        return Fraction(*value)
    return Fraction.from_float(float(value))


def clearance_externality(jobs, waiting_ids, rates, horizon):
    """Return exact class contributions from a completed native job cache.

    ``waiting_ids`` comes from the original H terminal_state, never from a
    queue reconstructed using later completions.  Missing or unfinished H
    waiting jobs make the proxy unavailable and raise ValueError; callers
    should report that outcome rather than run a new continuation.
    """
    horizon = _fraction(horizon)
    if len(waiting_ids) != len(rates):
        raise ValueError("H FIFO coordinates differ from anchor rate coordinates")
    by_id = {tuple(job["id"]): job for job in jobs}
    contributions = []
    for coordinate, identities in enumerate(waiting_ids):
        rate = _fraction(rates[coordinate])
        if rate < 0:
            raise ValueError("anchor estimated arrival rate is negative")
        clearance = horizon
        for identity in identities:
            job = by_id.get(tuple(identity))
            if job is None or job["start"] is None or job["completion"] is None:
                raise ValueError("H waiting job has no complete cached admission")
            start, completion = _fraction(job["start"]), _fraction(job["completion"])
            if job["coordinate"] != coordinate or start < horizon or completion < start:
                raise ValueError("cached admission differs from the original H FIFO")
            clearance = max(clearance, start)
        delay = clearance - horizon
        contributions.append(dict(coordinate=coordinate,
            waiting_jobs_at_H=len(identities), rate=rate, clearance=clearance,
            delay=delay, externality=rate * delay * delay / 2))
    return dict(externality=sum((row["externality"] for row in contributions), Fraction(0)),
        per_coordinate=tuple(contributions))
