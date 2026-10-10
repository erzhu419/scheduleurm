"""Retain finite native checks and check the declared augmented ordering."""
import ast
import inspect
from math import isclose, isfinite
import textwrap

from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as finite


FIFO_WORK = (
    "tagged_waiting_subtractions", "tagged_waiting_squares",
    "tagged_tau_divisions", "tagged_penalty_additions", "augmented_score_additions",
)
FIFO_SCORES = (
    "baseline_predicted_waiting_penalty_s", "chosen_predicted_waiting_penalty_s",
    "baseline_predicted_augmented_score_s", "chosen_predicted_augmented_score_s",
)


def _augmented_order_checker():
    """Replace only the outer S1 ordering predicate, keeping genuine S1/S2."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(finite.check_rollout)))

    class OuterOrdering(ast.NodeTransformer):
        changed = 0

        def visit_Compare(self, node):
            self.generic_visit(node)
            expected = ast.parse(
                'info["chosen_predicted_completion_s"] > info["baseline_predicted_completion_s"]',
                mode="eval").body
            if ast.dump(node) == ast.dump(expected):
                self.changed += 1
                return ast.parse(
                    'info["chosen_predicted_augmented_score_s"] > '
                    'info["baseline_predicted_augmented_score_s"]', mode="eval").body
            return node

    transform = OuterOrdering()
    transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("finite outer S1 ordering site changed")
    namespace = dict(finite.check_rollout.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<fifo-waiting-checker>", "exec"), namespace)
    return namespace["check_rollout"]


_NATIVE_CHECKS = _augmented_order_checker()


def check_fifo_scores(info):
    """Check additive scores and the fees of all consumed tagged admissions."""
    work = info["fifo_waiting_work"]
    if set(work) != set(FIFO_WORK) or any(value < 0 for value in work.values()):
        return False
    if (any(work[name] != info["flow_evaluations"] for name in FIFO_WORK[:-1])
            or work["augmented_score_additions"] != info["candidate_count"]):
        return False
    complete = info["candidate_count"] > 0 and not info["planning_overflow"]
    if not complete:
        return all(info[name] is None for name in FIFO_SCORES)
    if any(info[name] is None or not isfinite(info[name]) for name in FIFO_SCORES):
        return False
    for prefix in ("baseline", "chosen"):
        penalty = info[prefix + "_predicted_waiting_penalty_s"]
        genuine = info[prefix + "_predicted_completion_s"]
        augmented = info[prefix + "_predicted_augmented_score_s"]
        if penalty < 0 or not isclose(augmented, genuine + penalty, rel_tol=1e-12, abs_tol=1e-10):
            return False
    if info["chosen_predicted_augmented_score_s"] > info["baseline_predicted_augmented_score_s"]:
        return False
    if not info["improved_incumbent"] and any(
            info["chosen_predicted_" + field] != info["baseline_predicted_" + field]
            for field in ("waiting_penalty_s", "augmented_score_s")):
        return False
    return True


def check_rollout(result, model, protocol, planner):
    return (_NATIVE_CHECKS(result, model, protocol, planner)
            and all(check_fifo_scores(frame["rollout"]) for frame in result["frames"]))
