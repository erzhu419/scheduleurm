"""Check fixed FIFO J ordering within the same-model BG mean guard."""
import ast
import inspect
import textwrap

from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as finite
from algorithm.experiments.fifo_waiting_penalty_checks import check_fifo_scores

MEAN_GUARD_WORK = ("mean_guard_comparisons", "mean_guard_allowed", "mean_guard_filtered")


def _mean_guard_native_checker():
    """Retain native genuine S1<=BG; adjust only the paid outer ordering count."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(finite.check_rollout)))

    class OrderingFee(ast.NodeTransformer):
        changed = 0

        def visit_Compare(self, node):
            self.generic_visit(node)
            expected = ast.parse(
                'info["incumbent_lookahead_ordering_comparisons"] != k + p - 2', mode="eval").body
            if ast.dump(node) == ast.dump(expected):
                self.changed += 1
                return ast.parse('info["incumbent_lookahead_ordering_comparisons"] '
                    '!= k + info["mean_guard_allowed"] - 2', mode="eval").body
            return node

    transform = OrderingFee()
    transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("finite outer ordering-fee site changed")
    namespace = dict(finite.check_rollout.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<fifo-mean-guard-checker>", "exec"), namespace)
    return namespace["check_rollout"]


_NATIVE_CHECKS = _mean_guard_native_checker()


def check_mean_guard_scores(info):
    if not check_fifo_scores(info):
        return False
    comparisons, allowed, filtered = (info[name] for name in MEAN_GUARD_WORK)
    if info["candidate_count"] == 0 or info["planning_overflow"]:
        return comparisons == allowed == filtered == 0
    return (comparisons == info["risk_feasible_candidates"]
        and 1 <= allowed <= comparisons and filtered == comparisons - allowed
        and info["incumbent_lookahead_ordering_comparisons"] == info["candidate_count"] + allowed - 2
        and info["chosen_predicted_completion_s"] <= info["baseline_predicted_completion_s"])


def check_rollout(result, model, protocol, planner):
    return (_NATIVE_CHECKS(result, model, protocol, planner)
            and all(check_mean_guard_scores(frame["rollout"]) for frame in result["frames"]))
