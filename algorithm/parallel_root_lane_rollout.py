"""Parallel exact root/two-sample chunks of the frozen current-policy scorer."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import multiprocessing
from time import perf_counter, process_time

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.background_competition_continuous_lane_rollout import (
    BackgroundRootScores, _LazyBackgroundArrivals,
)
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.one_step_background_redecision_lane_rollout import INNER_WORK
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import INNER_CURRENT_WORK


_worker_config = _worker_planner = None


def _semantic_state(planner):
    return (planner.rng.getstate(), planner.decision_counter, tuple(planner.arrival_rates),
            tuple(planner._outer_observed_counts)
            if hasattr(planner, "_outer_observed_counts") else None)


def _restore_semantic(planner, state):
    rng, counter, rates, counts = state
    planner.rng.setstate(rng)
    planner.decision_counter, planner.arrival_rates = counter, rates
    if counts is not None:
        planner._outer_observed_counts = counts


def _accumulate(target, addition):
    for name, value in addition.items():
        target[name] = max(target.get(name, 0), value) if "_peak_" in name else target.get(name, 0) + value


def _initialize_worker(options, seed, samples, cap):
    global _worker_config, _worker_planner
    _worker_config = options, seed, samples, cap
    _worker_planner = None


def _run_root(payload):
    """One physical chunk; all computed work is returned, even if discarded."""
    global _worker_planner
    cpu, wall = process_time(), perf_counter()
    constructor_cpu = constructor_wall = 0.0
    preprocessing = {}
    if _worker_planner is None:
        begin_cpu, begin_wall = process_time(), perf_counter()
        _worker_planner = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout(*_worker_config)
        constructor_cpu, constructor_wall = process_time() - begin_cpu, perf_counter() - begin_wall
        preprocessing = dict(_worker_planner.preprocessing_work)
    planner = _worker_planner
    before = planner.reuse_work
    _restore_semantic(planner, payload["outer_state"])
    _restore_semantic(planner.inner_planner, payload["input_states"][0])
    _restore_semantic(planner.inner_planner.inner_planner, payload["input_states"][1])
    planner._recourse_work, planner._two_step_work = Counter(), Counter()
    planner._inner_current_work, planner._inner_current_fallbacks = Counter(), Counter()
    planner._mw_reuse.begin_outer_decision()
    planner._inner_decision_reuse.begin_outer_decision()
    result = score_single_root(planner, payload)
    planner._mw_reuse.finish_outer_decision()
    scope = planner._inner_decision_reuse.work
    _accumulate(planner._inner_decision_reuse_work, scope)
    after = planner.reuse_work
    result["actual_reuse_work"] = {
        group: {name: (scope.get(name, 0) if group == "inner_decision_reuse_work" and "_peak_" in name
                       else value - before[group].get(name, 0))
                for name, value in values.items()}
        for group, values in after.items()}
    result.update(worker_cpu_s=process_time() - cpu, worker_wall_s=perf_counter() - wall,
        constructor_cpu_s=constructor_cpu, constructor_wall_s=constructor_wall,
        preprocessing=preprocessing)
    return result


def score_single_root(planner, payload):
    """Score one explicit root and a contiguous pair of globally indexed paths."""
    queues, available, now, lane, waiting_ages, paths = (
        payload[name] for name in ("queue", "available", "now", "lane", "ages", "paths"))
    queue = tuple(queues)
    if len(waiting_ages) != len(queue) or any(
            len(group) != count for group, count in zip(waiting_ages, queue, strict=True)):
        raise ValueError("visible FIFO age counts differ from the waiting queue")
    visible = tuple(tuple((-Fraction.from_float(float(age)), True) for age in group)
                    for group in waiting_ages)
    roots = (payload["root"],)
    absolute_origin = Fraction.from_float(float(now))
    relative = tuple(max(Fraction(0), Fraction.from_float(float(end)) - absolute_origin)
                     for end in available)
    prepared_paths, suffixes = [], []
    for sample_index, path in enumerate(paths):
        if isinstance(path, PlanningArrivals):
            if path.overflow:
                raise ValueError("an overflowing planning population cannot be scored")
            path = path.events
        events = tuple((Fraction.from_float(float(event[0])), event[1]) for event in path)
        prepared_paths.append(events)
        suffixes.append(_LazyBackgroundArrivals(
            planner.arrival_rates, planner.maximum_duration,
                f"{planner.planning_seed}:background:{planner.decision_counter}:{payload['sample_start'] + sample_index}",
            planner.event_cap, len(events)))
    work, two_step_work = planner._recourse_work, planner._two_step_work
    post_prefix_state = planner.rng.getstate()
    outer_counter = planner.decision_counter
    costs = dict.fromkeys(roots, Fraction(0))
    squared = dict.fromkeys(roots, Fraction(0))
    calls = idle_calls = evaluations = admitted_reads = tag_reads = background_admissions = 0

    def scores(overflow=False):
        return {
            "root": payload["root"], "sample_start": payload["sample_start"],
            "cost": costs[payload["root"]], "squared": squared[payload["root"]],
            "calls": calls, "idle_calls": idle_calls, "evaluations": evaluations,
            "admitted_reads": admitted_reads, "tag_reads": tag_reads,
            "background_admissions": background_admissions, "overflow": overflow,
            "suffix_work": [(len(suffix.events), suffix.random_draws, suffix.rate_reads, suffix.seed_initializations)
                            for suffix in suffixes],
            "recourse_work": dict(work), "two_step_work": dict(two_step_work),
            "inner_current_work": dict(planner._inner_current_work),
            "inner_current_fallbacks": dict(planner._inner_current_fallbacks),
            "current_touched": work["recourse_calls"] > 0,
            "BG_touched": planner._inner_current_work["recourse_calls"] > 0,
            "states": (_semantic_state(planner.inner_planner),
                       _semantic_state(planner.inner_planner.inner_planner)),
        }

    def admit(index, start, waiting):
        nonlocal evaluations, admitted_reads, tag_reads, background_admissions, tagged_remaining
        first = second = Fraction(0)
        for coordinate, offsets in enumerate(planner.sorted_completion_offsets[index]):
            for offset in offsets:
                birth, tagged = waiting[coordinate].popleft()
                admitted_reads += 1
                tag_reads += 1
                if tagged:
                    flow = start + offset - birth
                    first += flow
                    second += flow * flow
                    evaluations += 1
                    tagged_remaining -= 1
                else:
                    background_admissions += 1
        return first, second

    for root in roots:
        for events, suffix in zip(prepared_paths, suffixes, strict=True):
            waiting = [deque(group) for group in visible]
            lane_ends = list(relative)
            lane_ends[lane] = planner.durations[root]
            counts = list(planner._outer_observed_counts)
            work["recourse_count_state_reads"] += len(counts)
            tagged_remaining = sum(queue) + len(events)
            first, second = admit(root, Fraction(0), waiting)
            cursor = background_cursor = prior_idle_calls = recourse_steps = 0
            while tagged_remaining:
                next_lane = min(range(len(lane_ends)), key=lambda i: (lane_ends[i], i))
                start = lane_ends[next_lane]
                while cursor < len(events) and events[cursor][0] <= start:
                    birth, coordinate = events[cursor]
                    waiting[coordinate].append((birth, True))
                    if recourse_steps < 2:
                        counts[coordinate] += 1
                        work["recourse_arrival_counter_updates"] += 1
                    cursor += 1
                if not suffix.release_through(start):
                    costs[root] += first
                    squared[root] += second
                    return scores(True)
                while background_cursor < len(suffix.events):
                    birth, coordinate = suffix.events[background_cursor]
                    if birth > start:
                        break
                    waiting[coordinate].append((birth, False))
                    if recourse_steps < 2:
                        counts[coordinate] += 1
                        work["recourse_arrival_counter_updates"] += 1
                    background_cursor += 1
                current_queue = tuple(len(group) for group in waiting)
                chosen, _ = planner._best_index(current_queue)
                calls += 1
                if chosen is None:
                    idle_calls += 1
                    if recourse_steps == 0:
                        prior_idle_calls += 1
                    elif recourse_steps == 1:
                        planner.inner_planner.decision_counter += 1
                        work["recourse_counter_reads"] += 1
                        work["recourse_counter_writes"] += 1
                        two_step_work["recourse_virtual_idle_counter_updates"] += 1
                    lane_ends[next_lane] = start + 1
                    continue
                if recourse_steps < 2:
                    if recourse_steps == 0:
                        planner.inner_planner.rng.setstate(post_prefix_state)
                        planner.inner_planner.decision_counter = outer_counter + 1 + prior_idle_calls
                        work["recourse_rng_words_restored"] += len(post_prefix_state[1])
                        work["recourse_counter_writes"] += 1
                        two_step_work["recourse_first_calls"] += 1
                    else:
                        two_step_work["recourse_second_calls"] += 1
                    absolute_now = float(absolute_origin + start)
                    absolute_ends = tuple(float(absolute_origin + end) for end in lane_ends)
                    ages = tuple(tuple(float(start - birth) for birth, _ in group) for group in waiting)
                    work["recourse_context_age_reads"] += sum(current_queue)
                    work["recourse_context_coordinate_reads"] += len(counts)
                    work["recourse_context_lane_reads"] += len(lane_ends)
                    inner = planner.inner_planner(planner.options, current_queue, absolute_ends,
                                               absolute_now, next_lane, ages, tuple(counts))
                    work["recourse_calls"] += 1
                    work["recourse_eligible_calls"] += inner.eligible
                    work["recourse_changed_calls"] += inner.changed
                    work["recourse_planning_overflows"] += inner.planning_overflow
                    work["recourse_background_overflows"] += inner.background_rollout_overflow
                    work["recourse_risk_changed_unconstrained_calls"] += inner.risk_changed_unconstrained_root
                    for name in INNER_WORK:
                        work["recourse_" + name] += getattr(inner, name)
                    for name in INNER_CURRENT_WORK:
                        planner._inner_current_work[name] += getattr(inner, name)
                    if inner.planning_overflow:
                        if inner.incumbent_background_rollout_overflow:
                            stage = "original_stage_overflows"
                        elif inner.background_rollout_overflow:
                            stage = "second_stage_overflows"
                        else:
                            stage = "prefix_overflows"
                        planner._inner_current_fallbacks[stage] += 1
                        fallback = "MW_fallbacks" if inner.incumbent_root_index is None else "BG_fallbacks"
                        planner._inner_current_fallbacks[fallback] += 1
                    chosen, = inner.selection.option_indices
                    recourse_steps += 1
                additional_first, additional_second = admit(chosen, start, waiting)
                first += additional_first
                second += additional_second
                lane_ends[next_lane] = start + planner.durations[chosen]
            costs[root] += first
            squared[root] += second
    return scores()


class ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout(
        ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Keep the native chooser while distributing fixed two-sample chunks.

    Prefix draws and the ordinary-BG incumbent stage remain in the parent.
    Each worker retains both causal virtual calls and their serial descendants.
    Results are included in root-major/chunk order, ending at the first native
    overflow. Shared suffix fees use per-sample prefix maxima. All physically
    computed chunks, including discarded ones, contribute actual work.
    Pool startup/import CPU is outside worker function timings; score wall time
    includes startup on the first call. Worker constructor timings are already
    included in worker_cpu_s/worker_wall_s.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128, workers=22):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.workers = workers
        self._pool = None
        self.parallel_work = Counter()
        self._worker_reuse_work = {name: {} for name in super().reuse_work}
        self._scope_worker_peaks = Counter()

    @property
    def reuse_work(self):
        result = super().reuse_work
        for group, values in self._worker_reuse_work.items():
            _accumulate(result[group], values)
        result["parallel_root_work"] = dict(self.parallel_work)
        return result

    def _choose(self, *context):
        self._scope_worker_peaks = Counter()
        result = super()._choose(*context)
        for name in ("current_cache_peak_entries", "BG_cache_peak_entries"):
            self.last_reuse_work["inner_decision_reuse_work"][name] = self._scope_worker_peaks[name]
        return result

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        if len(waiting_ages) != len(queues) or any(
                len(group) != count for group, count in zip(waiting_ages, queues, strict=True)):
            raise ValueError("visible FIFO age counts differ from the waiting queue")
        paths = tuple(path.events if isinstance(path, PlanningArrivals) and not path.overflow
                      else path for path in paths)
        if any(isinstance(path, PlanningArrivals) for path in paths):
            raise ValueError("an overflowing planning population cannot be scored")
        roots = tuple(index for index, counts in enumerate(self.requirements)
            if all(need <= count for need, count in zip(counts, queues, strict=True)))
        work = self._recourse_work
        state = _semantic_state(self)
        work["recourse_rng_words_copied"] += len(state[0][1])
        work["recourse_counter_reads"] += 1
        base = {"queue": tuple(queues), "available": tuple(available), "now": now,
            "lane": lane, "ages": tuple(tuple(group) for group in waiting_ages),
            "outer_state": state, "input_states": (
                _semantic_state(self.inner_planner), _semantic_state(self.inner_planner.inner_planner))}
        payloads = [{**base, "root": root, "sample_start": start, "paths": paths[start:start + 2]}
                    for root in roots for start in range(0, len(paths), 2)]
        begin = perf_counter()
        self.parallel_work["score_calls"] += 1
        self.parallel_work["submitted_roots"] += len(roots)
        self.parallel_work["submitted_chunks"] += len(payloads)
        if payloads and self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker,
                initargs=(self.options, self.planning_seed, self.sample_count, self.event_cap))
        futures = [self._pool.submit(_run_root, payload) for payload in payloads]
        results = [future.result() for future in futures]
        self.parallel_work["completed_chunks"] += len(results)
        for result in results:
            self.parallel_work["worker_cpu_s"] += result["worker_cpu_s"]
            self.parallel_work["worker_wall_s"] += result["worker_wall_s"]
            self.parallel_work["worker_constructor_cpu_s"] += result["constructor_cpu_s"]
            self.parallel_work["worker_constructor_wall_s"] += result["constructor_wall_s"]
            if result["preprocessing"]:
                self.parallel_work["worker_planner_constructors"] += 3
            for name, value in result["preprocessing"].items():
                self.parallel_work["preprocessing_" + name] += value
            for group, values in result["actual_reuse_work"].items():
                _accumulate(self._worker_reuse_work[group], values)
                if group == "inner_decision_reuse_work":
                    for name, value in values.items():
                        if "_peak_" in name:
                            self._scope_worker_peaks[name] = max(self._scope_worker_peaks[name], value)
        costs, squared = dict.fromkeys(roots, Fraction(0)), dict.fromkeys(roots, Fraction(0))
        suffix_work = [[0] * 4 for _ in paths]
        totals = Counter()
        overflow = False
        included = 0
        for result in results:
            included += 1
            costs[result["root"]] += result["cost"]
            squared[result["root"]] += result["squared"]
            for name in ("calls", "idle_calls", "evaluations", "admitted_reads", "tag_reads", "background_admissions"):
                totals[name] += result[name]
            for destination, name in ((work, "recourse_work"),
                    (self._two_step_work, "two_step_work"), (self._inner_current_work, "inner_current_work"),
                    (self._inner_current_fallbacks, "inner_current_fallbacks")):
                destination.update(result[name])
            for local, fees in enumerate(result["suffix_work"]):
                slot = suffix_work[result["sample_start"] + local]
                for index, value in enumerate(fees):
                    slot[index] = max(slot[index], value)
            if result["current_touched"]:
                _restore_semantic(self.inner_planner, result["states"][0])
            if result["BG_touched"]:
                _restore_semantic(self.inner_planner.inner_planner, result["states"][1])
            if result["overflow"]:
                overflow = True
                break
        self.parallel_work["included_chunks"] += included
        self.parallel_work["discarded_chunks"] += len(results) - included
        self.parallel_work["score_wall_s"] += perf_counter() - begin
        suffix_totals = tuple(sum(row[index] for row in suffix_work) for index in range(4))
        return BackgroundRootScores(costs, squared, len(roots), totals["calls"],
            totals["calls"] * len(self.options), len(self.options), totals["idle_calls"],
            len(paths), sum(queues), totals["evaluations"], *suffix_totals,
            totals["admitted_reads"], totals["tag_reads"], totals["background_admissions"], overflow)
