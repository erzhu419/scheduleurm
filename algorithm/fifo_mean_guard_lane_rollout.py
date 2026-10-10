"""Add a genuine-flow admissibility condition to the frozen FIFO root score."""
import ast
from collections import Counter
from dataclasses import dataclass, fields, replace
import inspect
import textwrap

from algorithm import fifo_waiting_penalty_lane_rollout as fifo
from algorithm.parallel_fifo_waiting_penalty_lane_rollout import (
    ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout,
)
from algorithm.two_step_current_policy_bg_incumbent_lane_rollout import (
    INNER_CURRENT_FALLBACKS, INNER_CURRENT_WORK,
)


@dataclass(frozen=True)
class FIFOMeanGuardDecision(fifo.FIFOWaitingPenaltyDecision):
    mean_guard_comparisons: int = 0
    mean_guard_allowed: int = 0
    mean_guard_filtered: int = 0


def _add_mean_guard(tree):
    """Filter only original-risk roots; keep their risk metadata unchanged."""
    class MeanGuard(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_For(self, node):
            self.generic_visit(node)
            if (isinstance(node.target, ast.Name) and node.target.id == "index"
                    and isinstance(node.iter, ast.Attribute) and node.iter.attr == "costs"
                    and isinstance(node.iter.value, ast.Name) and node.iter.value.id == "scored"):
                self.changed["mean_condition"] += 1
                node.body.insert(-1, ast.parse(
                    "if scored.costs[index] > scored.costs[incumbent]:\n"
                    "    continue").body[0])
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Name) and node.func.id == "replace"
                    and any(item.arg == "improved_incumbent" for item in node.keywords)):
                self.changed["completed_metadata"] += 1
                for item in node.keywords:
                    if item.arg in ("risk_feasible_candidates", "risk_filtered_candidates"):
                        self.changed["risk_population_metadata"] += 1
                        expression = ("len(original_permitted)" if item.arg == "risk_feasible_candidates"
                                      else "scored.candidate_count - len(original_permitted)")
                        item.value = ast.parse(expression, mode="eval").body
                for name, expression in (
                    ("mean_guard_comparisons", "len(original_permitted)"),
                    ("mean_guard_allowed", "len(permitted)"),
                    ("mean_guard_filtered", "len(original_permitted) - len(permitted)"),
                ):
                    node.keywords.append(ast.keyword(name, ast.parse(expression, mode="eval").body))
            return node

    transform = MeanGuard()
    tree = transform.visit(tree)
    if transform.changed != Counter(mean_condition=1, completed_metadata=1, risk_population_metadata=2):
        raise ValueError("FIFO root mean-guard sites changed: " + str(transform.changed))
    return tree


def _mean_guard_chooser():
    """Reuse the existing J chooser generator, inserting only the mean guard."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fifo._augmented_chooser)))

    class Factory(ast.NodeTransformer):
        changed = 0

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if (isinstance(target, ast.Name) and target.id == "tree"
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "visit" and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "transform"):
                self.changed += 1
                return [node, ast.parse("tree = _add_mean_guard(tree)").body[0]]
            return node

    transform = Factory()
    tree = transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("FIFO augmented chooser generation site changed")
    namespace = {**fifo.__dict__, "_add_mean_guard": _add_mean_guard}
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return namespace["_augmented_chooser"]()


MEAN_GUARD_CHOOSER = _mean_guard_chooser()


class FIFOMeanGuardLaneRollout(fifo.FIFOWaitingPenaltyCurrentPolicyLaneRollout):
    """Keep J ordering among original-risk roots whose genuine S1<=BG S1.

    BG is always mean-feasible and wins J ties. Risk feasible/filtered fields
    continue describing the original BG S2<=MW set; mean filtering has its own
    counts. Existing ordering fees use the actual mean-allowed population.
    Every fallback retains zero mean-guard work and the original partial FIFO
    work. All sampled paths, native child controllers and scores are unchanged.
    """
    def maxweight(self, options, queues, available, now, lane, waiting_ages,
                  observed_arrival_counts=None):
        decision = super().maxweight(options, queues, available, now, lane, waiting_ages)
        return FIFOMeanGuardDecision(
            **{attribute.name: getattr(decision, attribute.name)
               for attribute in fields(fifo.FIFOWaitingPenaltyDecision)})

    def _choose_augmented(self, *context):
        self._inner_current_work, self._inner_current_fallbacks = Counter(), Counter()
        decision = MEAN_GUARD_CHOOSER(self, *context)
        return replace(decision,
            inner_current_work={name: self._inner_current_work[name] for name in INNER_CURRENT_WORK},
            inner_current_fallbacks={name: self._inner_current_fallbacks[name] for name in INNER_CURRENT_FALLBACKS},
            fifo_waiting_work=dict(self.fifo_waiting_work))


class ParallelFIFOMeanGuardLaneRollout(
        FIFOMeanGuardLaneRollout, ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout):
    """Reuse the tested FIFO workers; the mean guard changes caller choice only."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128, workers=13):
        super().__init__(options, planning_seed, sample_count, event_cap, workers=workers)
