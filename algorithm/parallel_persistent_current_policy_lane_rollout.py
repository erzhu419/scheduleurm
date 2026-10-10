"""Parallel exact chunks of the persistent frozen-current continuation."""
import ast
from collections import Counter
import inspect
import textwrap
from types import FunctionType

from algorithm import parallel_root_lane_rollout as parallel
from algorithm.persistent_current_policy_bg_incumbent_lane_rollout import (
    PersistentCurrentPolicyBGIncumbentLaneRollout,
)


def parallel_functions():
    """Adapt the existing chunk, worker and ordered aggregation methods."""
    namespace = dict(parallel.__dict__)
    namespace.update(__name__=__name__, _worker_config=None, _worker_planner=None,
        PersistentCurrentPolicyBGIncumbentLaneRollout=PersistentCurrentPolicyBGIncumbentLaneRollout)
    tree = ast.parse(textwrap.dedent(inspect.getsource(parallel.score_single_root)))

    class PersistentChunk(ast.NodeTransformer):
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
                    '    planner.continuation_work["late_eligible_calls"] += inner.eligible\n'
                    '    planner.continuation_work["late_planning_overflows"] += inner.planning_overflow\n'
                    '    planner.continuation_work["late_background_overflows"] += inner.background_rollout_overflow'
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
                        [node], ast.parse('planner.continuation_work["late_current_calls"] += 1').body)
                if target.slice.value == "recourse_virtual_idle_counter_updates":
                    self.changed["late_idle_accounting"] += 1
                    return [node, ast.parse(
                        'if recourse_steps >= 2:\n'
                        '    planner.continuation_work["late_virtual_idle_counter_updates"] += 1'
                    ).body[0]]
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            if any(isinstance(key, ast.Constant) and key.value == "recourse_work" for key in node.keys):
                self.changed["chunk_continuation_work"] += 1
                node.keys.append(ast.Constant("continuation_work"))
                node.values.append(ast.parse("dict(planner.continuation_work)", mode="eval").body)
            return node

    chunk = PersistentChunk()
    chunk.visit(tree)
    expected = Counter(continuation_limits=3, idle_counter_limit=1,
        late_call_accounting=1, second_call_accounting=1, late_idle_accounting=1,
        chunk_continuation_work=1)
    if chunk.changed != expected:
        raise ValueError("parallel native continuation sites changed: " + str(chunk.changed))
    exec(compile(ast.fix_missing_locations(tree), "<parallel-persistent-chunk>", "exec"), namespace)

    tree = ast.parse(textwrap.dedent(inspect.getsource(parallel._run_root)))

    class PersistentWorker(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Name(self, node):
            if node.id == "ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout":
                self.changed["constructor"] += 1
                node.id = "PersistentCurrentPolicyBGIncumbentLaneRollout"
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "planner":
                self.changed["continuation_reset"] += 1
                return [node, ast.parse("planner.continuation_work.clear()").body[0]]
            return node

    worker = PersistentWorker()
    worker.visit(tree)
    if worker.changed != Counter(constructor=1, continuation_reset=1):
        raise ValueError("parallel native worker sites changed: " + str(worker.changed))
    exec(compile(ast.fix_missing_locations(tree), "<parallel-persistent-worker>", "exec"), namespace)
    namespace["_initialize_worker"] = FunctionType(
        parallel._initialize_worker.__code__, namespace, "_initialize_worker")

    method = parallel.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class PersistentAggregation(ast.NodeTransformer):
        def __init__(self):
            self.changed = 0

        def visit_For(self, node):
            self.generic_visit(node)
            if (isinstance(node.target, ast.Tuple)
                    and [getattr(item, "id", None) for item in node.target.elts] == ["destination", "name"]):
                self.changed += 1
                node.iter.elts.append(ast.parse(
                    '(self.continuation_work, "continuation_work")', mode="eval").body)
            return node

    aggregation = PersistentAggregation()
    aggregation.visit(tree)
    if aggregation.changed != 1:
        raise ValueError("parallel native aggregation site changed")
    exec(compile(ast.fix_missing_locations(tree), "<parallel-persistent-aggregation>", "exec"), namespace)
    return namespace["score_single_root"], namespace["_run_root"], namespace["_initialize_worker"], namespace[method.__name__]


score_single_root, _run_root, _initialize_worker, PARALLEL_SCORER = parallel_functions()


class ParallelPersistentCurrentPolicyBGIncumbentLaneRollout(
        PersistentCurrentPolicyBGIncumbentLaneRollout,
        parallel.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Preserve serial persistent decisions while distributing two-sample chunks.

    The inherited pool lifecycle and actual reuse ledger charge every computed
    chunk. Native scores, child states and late logical fees include only the
    root-major prefix ending at the first native overflow, using global sample
    indices and the existing per-sample suffix maxima. The worker uses the
    same frozen OLD-current child as serial persistent execution.
    """
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128, workers=13):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.workers = workers

    def score_background_roots(self, queues, available, now, lane, waiting_ages, paths):
        scored = PARALLEL_SCORER(self, queues, available, now, lane, waiting_ages, paths)
        if self._recourse_work["recourse_calls"] != (self._two_step_work["recourse_first_calls"]
                + self._two_step_work["recourse_second_calls"] + self.continuation_work["late_current_calls"]):
            raise ValueError("first, second and late calls must reconstruct total current calls")
        return scored
