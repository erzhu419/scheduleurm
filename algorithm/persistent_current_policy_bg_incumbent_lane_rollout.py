"""Value finite tagged jobs with persistent frozen-current continuation."""
import ast
from collections import Counter
from dataclasses import dataclass, fields, replace
import inspect
import textwrap

from algorithm.exact_reuse_lane_rollout import (
    ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout,
)
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    TwoStepCurrentPolicyBGIncumbentDecision,
    TwoStepCurrentPolicyBGIncumbentLaneRollout,
)


LATE_CURRENT_WORK = (
    "late_current_calls", "late_eligible_calls", "late_planning_overflows",
    "late_background_overflows", "late_virtual_idle_counter_updates",
)


@dataclass(frozen=True)
class PersistentCurrentPolicyBGIncumbentDecision(TwoStepCurrentPolicyBGIncumbentDecision):
    late_current_calls: int = 0
    late_eligible_calls: int = 0
    late_planning_overflows: int = 0
    late_background_overflows: int = 0
    late_virtual_idle_counter_updates: int = 0


def persistent_scorer():
    """Extend the native two-current scorer without admission observers."""
    method = TwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Persistent(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Name) and node.left.id == "recourse_steps"
                    and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant)):
                if isinstance(node.ops[0], ast.Lt) and node.comparators[0].value == 2:
                    self.changed["continuation_limits"] += 1
                    return ast.Constant(True)
                if isinstance(node.ops[0], ast.Eq) and node.comparators[0].value == 1:
                    self.changed["idle_counter_limit"] += 1
                    return ast.Compare(node.left, [ast.Gt()], [ast.Constant(0)])
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "inner":
                self.changed["late_call_accounting"] += 1
                return [node, ast.parse(
                    'if recourse_steps >= 2:\n'
                    '    self.continuation_work["late_eligible_calls"] += inner.eligible\n'
                    '    self.continuation_work["late_planning_overflows"] += inner.planning_overflow\n'
                    '    self.continuation_work["late_background_overflows"] += inner.background_rollout_overflow'
                ).body[0]]
            return node

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            target = node.target
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == "two_step_work" and isinstance(target.slice, ast.Constant)):
                if target.slice.value == "recourse_second_calls":
                    self.changed["second_call_accounting"] += 1
                    return ast.If(ast.parse("recourse_steps == 1", mode="eval").body,
                        [node], ast.parse('self.continuation_work["late_current_calls"] += 1').body)
                if target.slice.value == "recourse_virtual_idle_counter_updates":
                    self.changed["late_idle_accounting"] += 1
                    return [node, ast.parse(
                        'if recourse_steps >= 2:\n'
                        '    self.continuation_work["late_virtual_idle_counter_updates"] += 1'
                    ).body[0]]
            return node

    transform = Persistent()
    transform.visit(tree)
    expected = Counter(continuation_limits=3, idle_counter_limit=1,
        late_call_accounting=1, second_call_accounting=1, late_idle_accounting=1)
    if transform.changed != expected:
        raise ValueError("native continuation sites changed: " + str(transform.changed))
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<persistent-current-continuation>", "exec"), namespace)
    return namespace[method.__name__]


PERSISTENT_SCORER = persistent_scorer()


class PersistentCurrentPolicyBGIncumbentLaneRollout(
        ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Keep frozen OLD-current active until the finite tagged cohort clears.

    Ordinary BG still supplies the S2<=MW feasible roots, the incumbent and
    BG-first S1 ties. Every nonempty virtual call uses the same persistent
    OLD-current child, whose own continuation remains two BG calls then MW.
    The first call restores post-prefix RNG/counter once per root/sample;
    subsequent births and idle calls advance that child state. Real decisions
    replan with this candidate. Its default population/window and8/cap128 are
    unchanged, and all native overflow fallbacks and exact reuse remain active.

    Existing recourse and nested-current ledgers include every later call.
    The late fields are subsets of those fees, with second counting only the
    second call and recourse_calls=first+second+late. They reset on each real
    decision and are included directly in the returned decision dataclass.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.continuation_work = Counter()

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return PersistentCurrentPolicyBGIncumbentDecision(
            **{attribute.name: getattr(decision, attribute.name)
               for attribute in fields(TwoStepCurrentPolicyBGIncumbentDecision)})

    def _choose(self, options, queues, available, now, lane, waiting_ages):
        self.continuation_work.clear()
        decision = super()._choose(options, queues, available, now, lane, waiting_ages)
        return replace(decision,
            **{name: self.continuation_work[name] for name in LATE_CURRENT_WORK})

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        scored = PERSISTENT_SCORER(self, queues, available, now, lane, waiting_ages, paths)
        if self._recourse_work["recourse_calls"] != (self._two_step_work["recourse_first_calls"]
                + self._two_step_work["recourse_second_calls"] + self.continuation_work["late_current_calls"]):
            raise ValueError("first, second and late calls must reconstruct total current calls")
        return scored
