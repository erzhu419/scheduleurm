"""Finite-current root ordering with a fixed FIFO waiting-square penalty."""
import ast
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from fractions import Fraction
import inspect
import textwrap

from algorithm.background_competition_continuous_lane_rollout import BackgroundRootScores
from algorithm.exact_reuse_lane_rollout import ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout
from algorithm.two_step_bg_incumbent_original_risk_lane_rollout import TwoStepBGIncumbentOriginalRiskLaneRollout
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    INNER_CURRENT_FALLBACKS, INNER_CURRENT_WORK,
    TwoStepCurrentPolicyBGIncumbentDecision, TwoStepCurrentPolicyBGIncumbentLaneRollout,
)


WAITING_WORK = (
    "tagged_waiting_subtractions", "tagged_waiting_squares", "tagged_tau_divisions",
    "tagged_penalty_additions", "augmented_score_additions",
)


def waiting_penalty(start, birth, maximum_duration):
    """Return exact W²/τ; completion offsets have no role in waiting cost."""
    waiting = Fraction(start) - Fraction(birth)
    return waiting * waiting / Fraction(maximum_duration)


@dataclass(frozen=True)
class FIFOBackgroundRootScores(BackgroundRootScores):
    waiting_penalties: dict
    augmented_costs: dict


@dataclass(frozen=True)
class FIFOWaitingPenaltyDecision(TwoStepCurrentPolicyBGIncumbentDecision):
    baseline_predicted_waiting_penalty_s: float | None = None
    chosen_predicted_waiting_penalty_s: float | None = None
    baseline_predicted_augmented_score_s: float | None = None
    chosen_predicted_augmented_score_s: float | None = None
    fifo_waiting_work: dict = field(default_factory=dict)


def _augmented_chooser():
    """Change only second-stage ordering and its strict-improvement test."""
    method = TwoStepBGIncumbentOriginalRiskLaneRollout._choose
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Ordering(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_FunctionDef(self, node):
            if node.name == "ordering":
                for expression in ast.walk(node):
                    if (isinstance(expression, ast.Attribute) and expression.attr == "costs"
                            and isinstance(expression.value, ast.Name) and expression.value.id == "scored"):
                        expression.attr = "augmented_costs"
                        self.changed["ordering"] += 1
                return node
            return self.generic_visit(node)

        def visit_Assign(self, node):
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "improved":
                for expression in ast.walk(node.value):
                    if isinstance(expression, ast.Attribute) and expression.attr == "costs":
                        expression.attr = "augmented_costs"
                        self.changed["strict_improvement"] += 1
            return self.generic_visit(node)

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Name) and node.func.id == "replace"
                    and any(item.arg == "improved_incumbent" for item in node.keywords)):
                for name, expression in (
                    ("baseline_predicted_waiting_penalty_s", "scored.waiting_penalties[incumbent]"),
                    ("chosen_predicted_waiting_penalty_s", "scored.waiting_penalties[actual]"),
                    ("baseline_predicted_augmented_score_s", "scored.augmented_costs[incumbent]"),
                    ("chosen_predicted_augmented_score_s", "scored.augmented_costs[actual]"),
                ):
                    node.keywords.append(ast.keyword(name, ast.parse(
                        "float(" + expression + " / len(paths))", mode="eval").body))
                self.changed["score_metadata"] += 1
            return node

    transform = Ordering()
    tree = transform.visit(tree)
    if transform.changed != Counter(ordering=1, strict_improvement=2, score_metadata=1):
        raise ValueError("finite root-ordering sites changed: " + str(transform.changed))
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[method.__name__]


def _waiting_scorer():
    """Attach exact waiting penalties at the existing tagged admissions."""
    method = TwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Waiting(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "squared":
                self.changed["penalty_initialization"] += 1
                return [node, ast.parse("waiting_penalties = dict.fromkeys(roots, Fraction(0))").body[0]]
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            if isinstance(node.test, ast.Name) and node.test.id == "tagged":
                self.changed["tagged_admission"] += 1
                node.body.insert(0, ast.parse(
                    "waiting_penalties[root] += self._waiting_penalty(start, birth)").body[0])
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "BackgroundRootScores":
                self.changed["scored_return"] += 1
                node.func.id = "FIFOBackgroundRootScores"
                node.args.extend((ast.Name("waiting_penalties", ast.Load()),
                    ast.parse("self._augmented_costs(costs, waiting_penalties)", mode="eval").body))
            return node

    transform = Waiting()
    tree = transform.visit(tree)
    if transform.changed != Counter(penalty_initialization=1, tagged_admission=1, scored_return=1):
        raise ValueError("finite FIFO admission sites changed: " + str(transform.changed))
    namespace = {**method.__globals__, "FIFOBackgroundRootScores": FIFOBackgroundRootScores}
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[method.__name__]


def _exact_reuse_chooser():
    """Keep the existing exact-reuse scope, replacing only its native call."""
    method = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout._choose
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class NativeCall(ast.NodeTransformer):
        changed = 0

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "decision":
                node.value.func = ast.Attribute(ast.Name("self", ast.Load()), "_choose_augmented", ast.Load())
                self.changed += 1
            return node

    transform = NativeCall()
    tree = transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("exact-reuse native chooser site changed")
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace[method.__name__]


AUGMENTED_CHOOSER = _augmented_chooser()
WAITING_SCORER = _waiting_scorer()
EXACT_REUSE_CHOOSER = _exact_reuse_chooser()


class FIFOWaitingPenaltyCurrentPolicyLaneRollout(ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Rank the finite parent's existing roots by S1 plus tagged W²/τ_max.

    Coefficient one and catalogue τ_max are fixed. The ordinary BG incumbent,
    original S2<=MW feasible set, eight independent populations, cap128,
    two frozen-current virtual calls and all fallback semantics are unchanged.
    Genuine S1/S2 remain separate from the augmented ordering score. The
    planner adds arithmetic only, with no new draws or controller calls.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.fifo_waiting_work = Counter(dict.fromkeys(WAITING_WORK, 0))
        self._fifo_waiting_actual_work = Counter(dict.fromkeys(WAITING_WORK, 0))

    @property
    def reuse_work(self):
        result = ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout.reuse_work.fget(self)
        result["fifo_waiting_actual_work"] = dict(self._fifo_waiting_actual_work)
        return result

    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return FIFOWaitingPenaltyDecision(
            **{attribute.name: getattr(decision, attribute.name)
               for attribute in fields(TwoStepCurrentPolicyBGIncumbentDecision)})

    def _waiting_penalty(self, start, birth):
        for name in WAITING_WORK[:4]:
            self.fifo_waiting_work[name] += 1
            self._fifo_waiting_actual_work[name] += 1
        return waiting_penalty(start, birth, self.maximum_duration)

    def _augmented_costs(self, costs, penalties):
        self.fifo_waiting_work["augmented_score_additions"] += len(costs)
        self._fifo_waiting_actual_work["augmented_score_additions"] += len(costs)
        return {root: cost + penalties[root] for root, cost in costs.items()}

    def _choose(self, *context):
        self.fifo_waiting_work.clear()
        self.fifo_waiting_work.update(dict.fromkeys(WAITING_WORK, 0))
        return EXACT_REUSE_CHOOSER(self, *context)

    def _choose_augmented(self, *context):
        self._inner_current_work, self._inner_current_fallbacks = Counter(), Counter()
        decision = AUGMENTED_CHOOSER(self, *context)
        return replace(decision,
            inner_current_work={name: self._inner_current_work[name] for name in INNER_CURRENT_WORK},
            inner_current_fallbacks={name: self._inner_current_fallbacks[name] for name in INNER_CURRENT_FALLBACKS},
            fifo_waiting_work=dict(self.fifo_waiting_work))

    score_background_roots = WAITING_SCORER
