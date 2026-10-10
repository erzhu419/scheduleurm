"""Finite FIFO replay with causal selection and censored observation snapshots.

The simulator clock contains measured frame durations only. Selector and
reference-audit runtimes are diagnostics, never extra simulated processing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from math import floor, isfinite
from time import perf_counter

from algorithm.canonical_count_space_bnb_frame_candidates import select_canonical_count_space_bnb_frame
from algorithm.experiments.aligned_hardware_shared_threshold_scaling_screen import (
    COUNTERS, _metrics, _record, check_selection, reference_key,
)


class ReferenceAudit:
    """Share reference results between policies, never policy selections.

    A full-reference policy can seed its own calls through ``observe``. Those
    calls remain charged to that policy; only cache misses in ``lookup`` are
    charged to the separate audit workload.
    """

    def __init__(self, model, budget=4096):
        self.model = model
        self.budget = budget
        self.by_queue = {}
        self.lookup_calls = self.cache_hits = self.observed_policy_calls = 0
        self.metrics = _metrics()

    def observe(self, queue, selected):
        if selected.lp_node_budget != self.budget:
            raise ValueError("reference-policy call must use the registered reference budget")
        self.observed_policy_calls += 1
        self.by_queue.setdefault(tuple(queue), selected)
        return selected

    def lookup(self, queue):
        queue = tuple(queue)
        self.lookup_calls += 1
        if queue in self.by_queue:
            self.cache_hits += 1
            return self.by_queue[queue]
        started = perf_counter()
        selected = select_canonical_count_space_bnb_frame(
            self.model.options, queue, self.model.lanes, self.budget,
        )
        _record(self.metrics, selected, self.model, queue, self.model.lanes,
                self.budget, reference_key(selected, queue, len(self.model.options)),
                elapsed_s=perf_counter() - started)
        self.by_queue[queue] = selected
        return selected

    def summary(self):
        complete = sum(reference_key(result, queue, len(self.model.options)) is not None
                       for queue, result in self.by_queue.items())
        return {
            "cache_size": len(self.by_queue), "lookup_calls": self.lookup_calls,
            "cache_hits": self.cache_hits, "observed_policy_calls": self.observed_policy_calls,
            "actual_calls": self.metrics["calls"], "calls": self.metrics["calls"],
            "search_work": dict(self.metrics["search_work"]),
            "operational_elapsed_s": self.metrics["operational_elapsed_s"],
            "certificate_failures": self.metrics["failures"],
            "failures": self.metrics["failures"],
            "first_failure": self.metrics["first_failure"],
            "complete_states": complete, "incomplete_count": len(self.by_queue) - complete,
        }


def _snapshot_group(jobs, cutoff):
    released = [job for job in jobs if job["arrival"] <= cutoff]
    completed = [job for job in released
                 if job["completion"] is not None and job["completion"] <= cutoff]
    queued = sum(job["start"] is None or job["start"] > cutoff for job in released)
    flows = sorted(job["completion"] - job["arrival"] for job in completed)
    percentile = None
    if flows:
        position = 0.9 * (len(flows) - 1)
        left = floor(position)
        right = min(left + 1, len(flows) - 1)
        percentile = flows[left] + (position - left) * (flows[right] - flows[left])
    restricted = sum(min(cutoff, job["completion"] if job["completion"] is not None else cutoff)
                     - job["arrival"] for job in released)
    return {
        "time": cutoff, "arrivals": len(released), "completed_jobs": len(completed),
        "queued_jobs": queued, "inflight_jobs": len(released) - len(completed) - queued,
        "unfinished_jobs": len(released) - len(completed),
        "restricted_mean_flow_s": restricted / len(released) if released else None,
        "completed_mean_flow_s": sum(flows) / len(flows) if flows else None,
        "completed_p90_flow_s": percentile,
        "completion_rate": len(completed) / len(released) if released else None,
    }


def trajectory_snapshot(jobs, cutoff, coordinates):
    """All released jobs contribute; later scheduled completions are censored."""
    snapshot = _snapshot_group(jobs, cutoff)
    snapshot["per_coordinate"] = [
        _snapshot_group([job for job in jobs if job["coordinate"] == coordinate], cutoff)
        for coordinate in range(coordinates)
    ]
    return snapshot


def simulate_trajectory(model, tape, arrival_horizon, observation_end, budget, selector,
                        reference_cache=None, audit_limit=4, policy="candidate"):
    """Run a finite arrival tape until its jobs drain or the observation ends.

    ``tape`` consists of ``(arrival_time, coordinate, within_class_index)``
    triples. The selector receives only the static catalogue and released queue
    sizes. A frame that crosses T finishes in the event record, while snapshots
    count completion only when its actual offset is at or before the cutoff.
    """
    if not (0 <= arrival_horizon <= observation_end and isfinite(observation_end)):
        raise ValueError("require finite 0 <= arrival_horizon <= observation_end")
    coordinates = len(model.workloads)
    events = tuple(sorted(tape))
    if any(not isfinite(time) or not 0 <= time <= arrival_horizon or
           not 0 <= coordinate < coordinates for time, coordinate, _ in events):
        raise ValueError("arrival tape is outside the registered horizon or coordinates")
    if len({(coordinate, index) for _, coordinate, index in events}) != len(events):
        raise ValueError("arrival identities must be unique within coordinates")
    jobs = [{"coordinate": coordinate, "index": index, "arrival": float(time),
             "start": None, "completion": None} for time, coordinate, index in events]
    queues = [deque() for _ in range(coordinates)]
    reference = reference_cache if reference_cache is not None else ReferenceAudit(model)
    audit_before = reference.summary()
    metrics = _metrics()
    metrics["capacity_propagation"] = {}
    invariants = dict.fromkeys(("resource_feasible", "completion_offsets_valid",
                                "service_matches_departures", "queue_recursion"), True)
    frames, witnesses = [], []
    first_uncertified = first_incomplete_reference = None
    complete_references = incomplete_references = cursor = 0
    now = 0.0

    def release(until):
        nonlocal cursor
        counts = [0] * coordinates
        while cursor < len(jobs) and jobs[cursor]["arrival"] <= until:
            coordinate = jobs[cursor]["coordinate"]
            queues[coordinate].append(cursor)
            counts[coordinate] += 1
            cursor += 1
        return tuple(counts)

    release(now)
    while now < observation_end and (cursor < len(jobs) or any(queues)):
        queue = tuple(len(waiting) for waiting in queues)
        started = perf_counter()
        result = selector(model.options, queue, model.lanes, budget)
        elapsed = perf_counter() - started
        selected = getattr(result, "selection", result)
        ref = (reference.observe(queue, selected) if policy == "full_reference"
               else reference.lookup(queue))
        expected = reference_key(ref, queue, len(model.options))
        complete_references += expected is not None
        incomplete_references += expected is None
        if expected is None and first_incomplete_reference is None:
            first_incomplete_reference = {"frame": len(frames), "start": now,
                                          "queue": queue, "selection": asdict(ref)}
        preprocessing = asdict(result.preprocessing) if hasattr(result, "preprocessing") else None
        _record(metrics, selected, model, queue, model.lanes, budget, expected,
                preprocessing, ref, elapsed)
        if hasattr(result, "capacity_propagation"):
            for key, value in asdict(result.capacity_propagation).items():
                metrics["capacity_propagation"][key] = metrics["capacity_propagation"].get(key, 0) + value
        checks = check_selection(selected, model.options, queue, model.lanes, budget, expected, ref)
        if any(checks[key] for key in ("invalid_counts", "index_count_mismatch", "service_count_mismatch",
                                       "infeasible_service", "duration_count_mismatch")):
            raise ValueError(f"infeasible selected frame at {now}: {checks}")
        if not isfinite(selected.duration) or selected.duration <= 0:
            raise ValueError("selected frame must advance time")
        if not selected.canonical_exact and first_uncertified is None:
            first_uncertified = {"frame": len(frames), "start": now,
                                 "queue": queue, "selection": asdict(selected)}
        departures = [0] * coordinates
        class_offsets = [[] for _ in range(coordinates)]
        for option_index in selected.option_indices:
            if option_index is None:
                continue
            option = model.options[option_index]
            if len(option.completion_offsets) != coordinates or any(
                len(offsets) != option.requirements[c] or any(
                    not isfinite(offset) or not 0 <= offset <= option.duration for offset in offsets)
                for c, offsets in enumerate(option.completion_offsets)
            ):
                raise ValueError(f"invalid completion offsets for {option.action_id}")
            for coordinate, offsets in enumerate(option.completion_offsets):
                class_offsets[coordinate].extend(offsets)
        # Jobs within a class are interchangeable. Assign the oldest admitted
        # jobs to the earliest completion slots of this simultaneous frame.
        for coordinate, offsets in enumerate(class_offsets):
            for offset in sorted(offsets):
                job = jobs[queues[coordinate].popleft()]
                job["start"], job["completion"] = now, now + offset
                departures[coordinate] += 1
        if tuple(departures) != selected.service:
            raise ValueError("selected service differs from executed departures")
        end = now + selected.duration
        arrivals_during = release(min(end, observation_end))
        after = tuple(len(waiting) for waiting in queues)
        if any(q1 != q0 - done + added for q0, done, added, q1 in
               zip(queue, departures, arrivals_during, after, strict=True)):
            raise ValueError("queue recursion failed")
        frame = {
            "start": now, "duration": selected.duration, "queue": queue,
            "option_indices": selected.option_indices, "service": selected.service,
            "score": selected.score, "score_exact": selected.score_exact,
            "canonical_exact": selected.canonical_exact, "gap": selected.gap,
            "search_work": {counter: getattr(selected, counter) for counter in COUNTERS},
        }
        frames.append(frame)
        if len(witnesses) < audit_limit and (any(checks.values()) or not selected.canonical_exact):
            witnesses.append({**frame, "checks": {key: value for key, value in checks.items() if value},
                              "reference_complete": expected is not None,
                              "reference_indices": ref.option_indices})
        now = end
    audit_after = reference.summary()
    audit_delta = {key: audit_after[key] - audit_before[key] for key in (
        "lookup_calls", "cache_hits", "observed_policy_calls", "actual_calls",
        "operational_elapsed_s", "certificate_failures",
    )}
    audit_delta.update({
        "search_work": {key: audit_after["search_work"][key] - audit_before["search_work"][key]
                        for key in COUNTERS},
        "cache_size": audit_after["cache_size"],
        "complete_frame_count": complete_references, "incomplete_frame_count": incomplete_references,
        "calls": audit_after["calls"] - audit_before["calls"],
        "failures": audit_after["failures"] - audit_before["failures"],
        "incomplete_count": incomplete_references,
        "first_incomplete": first_incomplete_reference,
    })
    snapshots = {label: trajectory_snapshot(jobs, cutoff, coordinates) for label, cutoff in (
        ("arrival_horizon", arrival_horizon), ("observation_end", observation_end),
    )}
    return {
        "policy": policy, "budget": budget, "arrival_horizon": arrival_horizon,
        "observation_end": observation_end, "frame_count": len(frames),
        "last_frame_end": now, "frames": frames, "jobs": jobs, "snapshots": snapshots,
        "selector_metrics": metrics, "reference_audit": audit_delta, "invariants": invariants,
        "first_uncertified_frame": first_uncertified, "first_witnesses": witnesses,
    }
