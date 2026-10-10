"""Retain the frozen service checker, accounting for every late current call."""
import ast
import inspect
import textwrap
from types import FunctionType

from algorithm.experiments import aligned_hardware_two_step_current_policy_bg_incumbent_development as frozen
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import LATE_CURRENT_WORK


def _persistent_recourse_checker():
    method = frozen.check_current_recourse
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class ExtendCalls(ast.NodeTransformer):
        def __init__(self):
            self.changed = 0

        def visit_Compare(self, node):
            self.generic_visit(node)
            if (isinstance(node.left, ast.Name) and node.left.id == "calls"
                    and len(node.ops) == 1 and isinstance(node.ops[0], ast.NotEq)
                    and len(node.comparators) == 1
                    and ast.dump(node.comparators[0]) == ast.dump(
                        ast.parse("first + second", mode="eval").body)):
                node.comparators[0] = ast.parse(
                    'first + second + info["late_current_calls"]', mode="eval").body
                self.changed += 1
            return node

    transform = ExtendCalls()
    transform.visit(tree)
    if transform.changed != 1:
        raise ValueError("frozen outer first/second call identity changed")
    namespace = dict(method.__globals__)
    exec(compile(ast.fix_missing_locations(tree), "<persistent-current-checker>", "exec"), namespace)
    return namespace[method.__name__]


_FROZEN_RECOURSE_WITH_LATE_CALLS = _persistent_recourse_checker()


def check_current_recourse(info, *context):
    if any(info[name] < 0 for name in LATE_CURRENT_WORK):
        return False
    calls, eligible = info["recourse_calls"], info["recourse_eligible_calls"]
    late, late_eligible = info["late_current_calls"], info["late_eligible_calls"]
    overflow, background = info["recourse_planning_overflows"], info["recourse_background_overflows"]
    late_overflow, late_background = info["late_planning_overflows"], info["late_background_overflows"]
    if (calls != info["rollout_calls"] - info["rollout_idle_calls"]
            or not 0 <= late_eligible <= late <= calls
            or not 0 <= eligible - late_eligible <= info["recourse_first_calls"] + info["recourse_second_calls"]
            or not 0 <= late_background <= late_overflow <= late_eligible
            or not 0 <= background - late_background <= overflow - late_overflow <= eligible - late_eligible
            or info["late_virtual_idle_counter_updates"] > info["recourse_virtual_idle_counter_updates"]):
        return False
    # All old bounds, idle reads/writes, original BG, and nested two-BG fees stay.
    return _FROZEN_RECOURSE_WITH_LATE_CALLS(info, *context)


_namespace = dict(frozen.check_rollout.__globals__)
_namespace["check_current_recourse"] = check_current_recourse
check_rollout = FunctionType(frozen.check_rollout.__code__, _namespace,
                            "check_rollout", frozen.check_rollout.__defaults__)
