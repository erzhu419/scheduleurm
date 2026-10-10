"""Record native exact scores for an offline finite-parent incumbent rule."""
import ast
from collections import Counter
import inspect
import textwrap

from algorithm.arrival_aware_lane_rollout import PlanningArrivals
from algorithm.parallel_root_lane_rollout import _semantic_state
from algorithm import tick_persistent_current_policy_lane_rollout as tick
from algorithm.tick_persistent_current_policy_lane_rollout import (
    ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout,
    TickPersistentCurrentPolicyBGIncumbentLaneRollout,
)
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import (
    TwoStepBGIncumbentOriginalRiskLaneRollout,
)


def choose_parent_incumbent(scored, original, parent_root, mw_root, durations):
    """Choose within both exact risk sets, requiring strict S1 gain over parent.

    Both score objects must contain complete native root scores on the same
    sampled population. This pure rule does not execute a production policy;
    the parent root is supplied by the cached finite policy at this state.
    """
    if original.background_rollout_overflow or scored.background_rollout_overflow:
        raise ValueError("parent-incumbent rule requires complete native scores")
    original_limit = original.squared_costs[mw_root]
    if original.squared_costs[parent_root] > original_limit:
        raise ValueError("cached parent root is outside the original risk set")
    parent_limit = scored.squared_costs[parent_root]
    permitted = [root for root in scored.costs
        if original.squared_costs[root] <= original_limit
        and scored.squared_costs[root] <= parent_limit]
    chosen = min(permitted, key=lambda root:
        (scored.costs[root], root != parent_root, durations[root], root))
    return chosen if scored.costs[chosen] < scored.costs[parent_root] else parent_root


def last_touched_frames(frames, before):
    """Locate each child's last native touch strictly before the anchor."""
    found = [None, None]
    for index in range(before - 1, -1, -1):
        info = frames[index]["rollout"]
        for level, calls in enumerate((info["recourse_calls"], info["inner_current_work"]["recourse_calls"])):
            if found[level] is None and calls:
                found[level] = index
        if None not in found:
            break
    return tuple(found)


def recover_frame_child_states(planner, frame, paths, outer_state, needed=(0, 1), input_states=None):
    """Recover last-included touched states with native root/two-sample chunks.

    ``outer_state`` is the native semantic tuple at this frame's post-prefix
    point. First current/BG calls reset their RNG/counter, and their wrappers
    overwrite rates/counts, so touched outputs are independent of incoming
    child states. Untouched outputs are never returned. For complete frames
    inspect chunks backwards; for main-stage overflow replay the included
    prefix forwards. The caller restores each level's last touched frame and
    retains constructor state only when that level has never been touched.
    """
    work = Counter()
    paths = tuple(path.events if isinstance(path, PlanningArrivals) and not path.overflow
        else path for path in paths)
    if any(isinstance(path, PlanningArrivals) for path in paths):
        raise ValueError("child-state recovery requires completed outer prefix samples")
    if input_states is None:
        input_states = (_semantic_state(planner.inner_planner),
            _semantic_state(planner.inner_planner.inner_planner))
        work["prefix_recovery_input_planner_state_reads"] += 2
        work["prefix_recovery_input_rng_words_copied"] += sum(len(state[0][1]) for state in input_states)
        work["prefix_recovery_input_counter_reads"] += 2
        work["prefix_recovery_input_rate_reads"] += sum(len(state[2]) for state in input_states)
        work["prefix_recovery_input_observed_count_reads"] += sum(len(state[3])
            for state in input_states if state[3] is not None)
    queue = tuple(frame["queue"])
    roots = tuple(root for root, requirements in enumerate(planner.requirements)
        if all(need <= count for need, count in zip(requirements, queue, strict=True)))
    base = dict(queue=queue, available=tuple(frame["available_before"]), now=frame["start"],
        lane=frame["lane_ids"][0], ages=tuple(tuple(group) for group in frame["waiting_ages"]),
        outer_state=outer_state, input_states=input_states)
    payloads = [{**base, "root": root, "sample_start": start, "paths": paths[start:start + 2]}
        for root in roots for start in range(0, len(paths), 2)]
    overflow = frame["rollout"]["background_rollout_overflow"]
    tick._initialize_worker(planner.options, planner.planning_seed, planner.sample_count, planner.event_cap)
    states, encountered_overflow = {}, False
    for payload in payloads if overflow else reversed(payloads):
        result = tick._run_root(payload)
        work["prefix_recovery_chunks"] += 1
        for name in ("worker_cpu_s", "worker_wall_s", "constructor_cpu_s", "constructor_wall_s"):
            work["prefix_recovery_" + name] += result[name]
        for name in ("calls", "idle_calls", "evaluations", "admitted_reads", "tag_reads", "background_admissions"):
            work["prefix_recovery_" + name] += result[name]
        for index, name in enumerate(("background_planning_events", "background_planning_random_draws",
                "background_planning_rate_reads", "background_seed_initializations")):
            work["prefix_recovery_" + name] += sum(row[index] for row in result["suffix_work"])
        if result["preprocessing"]:
            work["prefix_recovery_planner_constructors"] += 3
        groups = {"preprocessing": result["preprocessing"], **result["actual_reuse_work"],
            **{name: result[name] for name in ("recourse_work", "two_step_work", "inner_current_work",
                "inner_current_fallbacks", "continuation_work")}}
        for group, fees in groups.items():
            for name, fee in fees.items():
                key = "prefix_recovery_" + group + "_" + name
                work[key] = max(work[key], fee) if "_peak_" in name else work[key] + fee
        for level, touched in enumerate((result["current_touched"], result["BG_touched"])):
            if level in needed and touched and (overflow or level not in states):
                states[level] = result["states"][level]
        if result["overflow"]:
            if not overflow:
                raise ValueError("cached complete frame has an overflowing native chunk")
            encountered_overflow = True
            break
        if not overflow and all(level in states for level in needed):
            break
    if (overflow and not encountered_overflow) or any(level not in states for level in needed):
        raise ValueError("cached touched-child metadata did not reproduce")
    return states, dict(work)


def _recording_original_choose():
    """Insert one reference assignment after the existing original BG score."""
    method = TwoStepBGIncumbentOriginalRiskLaneRollout._choose
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Record(ast.NodeTransformer):
        def __init__(self):
            self.sites = 0

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "original"):
                self.sites += 1
                return [node, ast.parse("self.recorded_original_scores = original").body[0]]
            return node

    record = Record()
    record.visit(tree)
    if record.sites != 1:
        raise ValueError("native original BG score assignment changed")
    namespace = {**method.__globals__, "__name__": __name__}
    exec(compile(ast.fix_missing_locations(tree), "<record-original-BG-scores>", "exec"), namespace)
    return namespace[method.__name__]


class _RecordingOriginalRisk(TwoStepBGIncumbentOriginalRiskLaneRollout):
    # In the recording classes' MRO this replaces only the original-risk stage;
    # all native outer/reuse/current/parallel wrappers still execute normally.
    _choose = _recording_original_choose()


class _RecordingScores:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording_work = Counter()
        self._reset_recorded_scores()

    def _reset_recorded_scores(self):
        self.recorded_original_scores = self.recorded_scores = None
        self.recorded_post_prefix_state = None

    def __call__(self, *context):
        self._reset_recorded_scores()
        return super().__call__(*context)

    def score_background_roots(self, *context):
        self.recorded_post_prefix_state = dict(
            rng_state=self.rng.getstate(), decision_counter=self.decision_counter)
        self.recording_work["recorded_post_prefix_rng_words_copied"] += len(
            self.recorded_post_prefix_state["rng_state"][1])
        self.recording_work["recorded_post_prefix_counter_reads"] += 1
        self.recorded_scores = super().score_background_roots(*context)
        return self.recorded_scores


class RecordingPersistentCurrentPolicyScores(_RecordingScores,
        TickPersistentCurrentPolicyBGIncumbentLaneRollout, _RecordingOriginalRisk):
    """Serial Tick native proposal with original/main exact scores attached.

    Records reset before each real call. A skipped stage remains None, and an
    overflowing stage retains its actual partial scores. Reference assignments
    add no scoring calls or native fees; one diagnostic RNG snapshot copies the
    outer state when the main stage is entered. Full inner states stay native.
    """


class RecordingParallelPersistentCurrentPolicyScores(_RecordingScores,
        ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout, _RecordingOriginalRisk):
    """Same instrumentation around native parallel Tick root aggregation.

    Worker construction, root/sample indices, included/discarded fees and final
    parent inner-state restoration are inherited without alteration. The pool
    lifecycle and workers argument are the native parallel API.
    """
