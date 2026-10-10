"""Persistent-current execution with an integer-clock BG suffix kernel."""
from types import FunctionType, MethodType

from algorithm import parallel_persistent_current_policy_lane_rollout as native
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import (
    PersistentCurrentPolicyBGIncumbentLaneRollout,
)
from algorithm.tick_background_rollout import score_background_roots


class TickPersistentCurrentPolicyBGIncumbentLaneRollout(
        PersistentCurrentPolicyBGIncumbentLaneRollout):
    """Change only the deepest BG scorer; keep the frozen chooser and state."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        background = self.inner_planner.inner_planner
        background.score_background_roots = MethodType(score_background_roots, background)


# Reuse the existing worker/aggregation construction with this leaf kernel.
# The returned functions share one worker namespace and are module globals so
# the original spawn pool can pickle them by their qualified names.
_build_functions = FunctionType(native.parallel_functions.__code__, {
    **native.parallel_functions.__globals__, "__name__": __name__,
    "PersistentCurrentPolicyBGIncumbentLaneRollout":
        TickPersistentCurrentPolicyBGIncumbentLaneRollout,
})
score_single_root, _run_root, _initialize_worker, PARALLEL_SCORER = _build_functions()


class ParallelTickPersistentCurrentPolicyBGIncumbentLaneRollout(
        TickPersistentCurrentPolicyBGIncumbentLaneRollout,
        native.ParallelPersistentCurrentPolicyBGIncumbentLaneRollout):
    """Use the same root/two-sample pool with tick-kernel worker planners."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128,
                 workers=13):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.workers = workers

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        scored = PARALLEL_SCORER(self, queues, available, now, lane, waiting_ages, paths)
        if self._recourse_work["recourse_calls"] != (self._two_step_work["recourse_first_calls"]
                + self._two_step_work["recourse_second_calls"]
                + self.continuation_work["late_current_calls"]):
            raise ValueError("first, second and late calls must reconstruct total current calls")
        return scored
