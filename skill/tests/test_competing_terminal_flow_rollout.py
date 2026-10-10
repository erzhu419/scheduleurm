"""Independent terminal-cost examples and the real cached-JSON entry point."""
from copy import deepcopy
from fractions import Fraction
import json

from algorithm.arrival_aware_lane_rollout import ArrivalAwareLaneRollout
from algorithm.background_competition_continuous_lane_rollout import _LazyBackgroundArrivals
from algorithm.experiments.aligned_hardware_frame_model import FrameOption
from algorithm.experiments.competing_terminal_flow_rollout import _simulate_terminal, evaluate_cached


F = Fraction


class NativeMWFixture:
    _best_index = ArrivalAwareLaneRollout._best_index

    def __init__(self, options):
        self.options = tuple(options)
        self.requirements = tuple(option.requirements for option in options)
        self.durations = tuple(F(option.duration) for option in options)
        self.duration_ratios = tuple((value.numerator, value.denominator) for value in self.durations)
        self.sorted_completion_offsets = tuple(tuple(tuple(sorted(map(F, group)))
            for group in option.completion_offsets) for option in options)

    def __call__(self, *args):
        raise AssertionError("the cached window must not consume another finite proposal")


class FixedSuffix:
    def __init__(self):
        self.future = ((F(3), 1), (F(41, 10), 1), (F(21, 5), 1), (F(59, 10), 1))
        self.events = [self.future[0]]
        self.random_draws = self.rate_reads = self.seed_initializations = 0
        self.overflow = False

    def release_through(self, start):
        self.events = [event for event in self.future if event[0] <= start]
        return True


def job(identity, coordinate, birth, source, start, completion):
    return dict(id=identity, coordinate=coordinate, birth=F(birth), source=source,
        start=F(start), completion=F(completion))


def test_recovered_waiting_jobs_compete_with_unscored_births_and_keep_committed_clocks():
    options = (FrameOption("A", (1, 0), 2., ((.5,), ())),
        FrameOption("B", (0, 1), 2., ((), (1.5,))))
    jobs = [job(("visible", 0, 0), 0, -1, "visible", 0, F(1, 2)),
        job(("prefix", 0), 0, 1, "prefix", 5, F(11, 2)),
        job(("prefix", 1), 1, 2, "prefix", 4, F(11, 2)),
        job(("background", 0), 1, 3, "background", 6, F(15, 2)),
        job(("visible", 1, 0), 1, 0, "visible", 3, F(9, 2))]
    saved = deepcopy(jobs)
    state = dict(queue=(1, 2), waiting_ids=((jobs[1]["id"],), (jobs[2]["id"], jobs[3]["id"])),
        lane_ends=(F(4), F(5)), counts=(2, 3))
    result = _simulate_terminal(NativeMWFixture(options), jobs, state, FixedSuffix(),
        horizon=F(4), common_area=F(21, 2), waiting_area=F(9), inflight_area=F(3, 2))
    assert jobs == saved
    assert result["complete"] and not result["overflow"]
    # B0@4, B1@5, unscored B@6/@7, A@8. The last A finishes at 8.5.
    # Fixed waiting residuals are 1.5+2.5+4.5; the committed job adds .5.
    assert result["terminal_cost"] == 9
    assert result["cost"] == F(39, 2)
    assert result["common_area"] == F(21, 2)
    assert result["calls"] == 0 and result["terminal_calls"] == 5
    assert result["incremental_tail_work"]["competing_admissions"] == 2
    assert result["final_queue"] == (0, 1)
    assert result["population"]["total"] == 5
    assert result["population"]["remaining"] == 0
    restored = {entry["id"]: entry for entry in result["population"]["jobs"]}
    assert restored[jobs[1]["id"]]["start"] == 8
    assert restored[jobs[1]["id"]]["completion"] == F(17, 2)
    assert restored[jobs[3]["id"]]["start"] == 5
    assert restored[jobs[3]["id"]]["completion"] == F(13, 2)
    assert restored[jobs[0]["id"]] == jobs[0]
    assert restored[jobs[4]["id"]] == jobs[4]
    assert set(restored) == {entry["id"] for entry in jobs}


def json_value(value):
    if isinstance(value, F):
        return [value.numerator, value.denominator]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def test_public_json_entry_reproduces_native_rng_prefix_and_rejects_changed_cache():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    seed, cap, horizon = "terminal-replay:background:7:3", 128, F(4)
    suffix = _LazyBackgroundArrivals((1.,), 2., seed, cap, 0)
    assert suffix.release_through(horizon)
    events = tuple(suffix.events)
    births = (F(-1),) + tuple(birth for birth, _ in events)
    jobs = [job(("visible", 0, 0) if index == 0 else ("background", index - 1),
        0, birth, "visible" if index == 0 else "background",
        4 + 2 * (index // 2), F(9, 2) + 2 * (index // 2)) for index, birth in enumerate(births)]
    common = sum((4 - max(F(0), birth) for birth in births), F(0))
    terminal = sum((F(1, 2) + 2 * (index // 2) for index in range(len(jobs))), F(0))
    cached = dict(root=0, sample_index=3,
        environment=dict(event_cap=cap, horizon=horizon, window_s=2., rates=(1.,),
            seed=seed, prefix_events=(), sample_index=3, anchor_counter=7),
        flow=dict(complete=True, overflow=False, horizon=horizon,
            background_events=events, suffix_work=(len(events), suffix.random_draws,
                suffix.rate_reads, suffix.seed_initializations), population=dict(jobs=jobs),
            terminal_state=dict(queue=(len(jobs),), waiting_ids=(tuple(entry["id"] for entry in jobs),),
                lane_ends=(F(4), F(4)), counts=(len(jobs),)), common_area=common,
            waiting_area=common, inflight_area=F(0), past_flow_constant=F(1),
            calls=2, idle_calls=1, logical_work={"retained": 3}, proposal_status={}))
    cached = json.loads(json.dumps(json_value(cached)))
    saved = deepcopy(cached)
    result = evaluate_cached(options, cached, event_cap=cap)
    assert cached == saved
    assert result["root"] == 0 and result["sample_index"] == 3
    assert result["environment"] == cached["environment"]
    assert result["complete"]
    flow = result["flow"]
    # In one FIFO class, later births cannot overtake any H-waiting job.
    assert flow["common_area"] == common
    assert flow["terminal_cost"] == terminal
    assert flow["cost"] == common + terminal
    assert flow["population"]["total"] == len(jobs)
    assert flow["reused_window_work"]["new_finite_controller_calls"] == 0
    assert flow["reused_window_work"]["logical_work"] == {"retained": 3}
    assert flow["regeneration_work"]["background_events"] == len(events)
    last_dispatch = F(4 + 2 * ((len(jobs) - 1) // 2))
    assert suffix.release_through(last_dispatch)
    assert flow["background_events"] == tuple(suffix.events)
    assert flow["suffix_work"] == (len(suffix.events), suffix.random_draws,
        suffix.rate_reads, suffix.seed_initializations)
    assert len(result["scored_population"]) == len(jobs)
    for field in ("background_events", "suffix_work"):
        changed = deepcopy(cached)
        if field == "background_events":
            changed["flow"][field].append([[4, 1], 0])
        else:
            changed["flow"][field][1] += 1
        try:
            evaluate_cached(options, changed, event_cap=cap)
        except ValueError as error:
            assert "cached H background" in str(error)
        else:
            raise AssertionError("changed native H stream/work must stop terminal replay")


def test_same_cap_overflow_keeps_fixed_jobs_and_partial_terminal_accrual():
    options = (FrameOption("single", (1,), 2., ((.5,),)),)
    suffix = _LazyBackgroundArrivals((1.,), 2., 0, 1, 0)
    assert suffix.release_through(F(4)) and len(suffix.events) == 1
    birth = suffix.events[0][0]
    jobs = [job(("visible", 0, 0), 0, 0, "visible", 5, F(11, 2)),
        job(("background", 0), 0, birth, "background", F(11, 2), 6),
        job(("visible", 0, 1), 0, -1, "visible", 3, F(7, 2)),
        job(("visible", 0, 2), 0, -2, "visible", F(7, 2), 4)]
    state = dict(queue=(2,), waiting_ids=((jobs[0]["id"], jobs[1]["id"]),),
        lane_ends=(F(5), F(11, 2)), counts=(4,))
    common = 4 + (4 - birth) + F(7, 2) + 4
    waiting = 4 + (4 - birth) + 3 + F(7, 2)
    result = _simulate_terminal(NativeMWFixture(options), jobs, state, suffix,
        horizon=F(4), common_area=common, waiting_area=waiting, inflight_area=F(1))
    assert result["overflow"] and not result["complete"]
    assert suffix.overflow and len(suffix.events) == 2
    crossing = suffix.events[-1][0]
    assert 5 < crossing < F(11, 2)
    assert result["partial_terminal_observation_end"] == crossing
    assert result["population"]["total"] == 4
    assert result["population"]["remaining"] == 1
    assert result["population"]["known_completion_residual"] == F(3, 2)
    assert result["population"]["pending_waiting_accrual"] == crossing - 4
    assert result["terminal_cost"] == F(3, 2) + crossing - 4
    assert result["cost"] == common + F(3, 2) + crossing - 4
    assert result["terminal_calls"] == 1
    assert len(result["background_events"]) == 2


if __name__ == "__main__":
    test_recovered_waiting_jobs_compete_with_unscored_births_and_keep_committed_clocks()
    test_public_json_entry_reproduces_native_rng_prefix_and_rejects_changed_cache()
    test_same_cap_overflow_keeps_fixed_jobs_and_partial_terminal_accrual()
    print("3 passed", flush=True)
