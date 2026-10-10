"""Exact parallel chunks for the finite FIFO waiting-penalty root score."""
import ast
from collections import Counter
import inspect
import textwrap
from types import FunctionType

from algorithm import parallel_root_lane_rollout as native
from algorithm.fifo_waiting_penalty_lane_rollout import (
    FIFOBackgroundRootScores, FIFOWaitingPenaltyCurrentPolicyLaneRollout, WAITING_WORK,
)


def _parallel_functions():
    """Keep finite native chunks, child-state restoration and overflow order."""
    namespace = {**native.__dict__, "__name__": __name__,
        "_worker_config": None, "_worker_planner": None,
        "FIFOWaitingPenaltyCurrentPolicyLaneRollout": FIFOWaitingPenaltyCurrentPolicyLaneRollout,
        "FIFOBackgroundRootScores": FIFOBackgroundRootScores, "WAITING_WORK": WAITING_WORK}
    tree = ast.parse(textwrap.dedent(inspect.getsource(native.score_single_root)))

    class Chunk(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "squared":
                self.changed["initialization"] += 1
                return [node, ast.parse("waiting_penalties = dict.fromkeys(roots, Fraction(0))").body[0]]
            return node

        def visit_If(self, node):
            self.generic_visit(node)
            if isinstance(node.test, ast.Name) and node.test.id == "tagged":
                self.changed["admission"] += 1
                node.body.insert(0, ast.parse(
                    "waiting_penalties[root] += planner._waiting_penalty(start, birth)").body[0])
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == "scores":
                self.changed["score_addition"] += 1
                node.body.insert(0, ast.parse(
                    "augmented = planner._augmented_costs(costs, waiting_penalties)").body[0])
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            if any(isinstance(key, ast.Constant) and key.value == "recourse_work" for key in node.keys):
                self.changed["result"] += 1
                for name, expression in (
                    ("waiting_penalty", "waiting_penalties[payload['root']]"),
                    ("augmented_cost", "augmented[payload['root']]"),
                    ("fifo_waiting_work", "dict(planner.fifo_waiting_work)"),
                ):
                    node.keys.append(ast.Constant(name))
                    node.values.append(ast.parse(expression, mode="eval").body)
            return node

    chunk = Chunk()
    tree = chunk.visit(tree)
    if chunk.changed != Counter(initialization=1, admission=1, score_addition=1, result=1):
        raise ValueError("parallel finite waiting sites changed: " + str(chunk.changed))
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)

    tree = ast.parse(textwrap.dedent(inspect.getsource(native._run_root)))

    class Worker(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Name(self, node):
            if node.id == "ExactReuseTwoStepCurrentPolicyBGIncumbentLaneRollout":
                self.changed["constructor"] += 1
                node.id = "FIFOWaitingPenaltyCurrentPolicyLaneRollout"
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            target = node.targets[0] if len(node.targets) == 1 else None
            if isinstance(target, ast.Name) and target.id == "planner":
                self.changed["reset"] += 1
                return [node, ast.parse(
                    "planner.fifo_waiting_work = Counter(dict.fromkeys(WAITING_WORK, 0))").body[0]]
            return node

    worker = Worker()
    tree = worker.visit(tree)
    if worker.changed != Counter(constructor=1, reset=1):
        raise ValueError("parallel finite worker sites changed: " + str(worker.changed))
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    namespace["_initialize_worker"] = FunctionType(native._initialize_worker.__code__, namespace, "_initialize_worker")

    method = native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout.score_background_roots
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Aggregation(ast.NodeTransformer):
        def __init__(self):
            self.changed = Counter()

        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
                    and [getattr(item, "id", None) for item in node.targets[0].elts] == ["costs", "squared"]):
                self.changed["initialization"] += 1
                return [node, ast.parse("waiting_penalties = dict.fromkeys(roots, Fraction(0))").body[0]]
            return node

        def visit_AugAssign(self, node):
            self.generic_visit(node)
            if (isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "squared"):
                self.changed["included_penalty"] += 1
                return [node, *ast.parse(
                    "waiting_penalties[result['root']] += result['waiting_penalty']\n"
                    "self.fifo_waiting_work.update({name: result['fifo_waiting_work'][name] for name in WAITING_WORK[:4]})"
                ).body]
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "BackgroundRootScores":
                self.changed["scored_return"] += 1
                node.func.id = "FIFOBackgroundRootScores"
                node.args.extend((ast.Name("waiting_penalties", ast.Load()),
                    ast.parse("self._augmented_costs(costs, waiting_penalties)", mode="eval").body))
            return node

    aggregation = Aggregation()
    tree = aggregation.visit(tree)
    if aggregation.changed != Counter(initialization=1, included_penalty=1, scored_return=1):
        raise ValueError("parallel finite aggregation sites changed: " + str(aggregation.changed))
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    return (namespace["score_single_root"], namespace["_run_root"], namespace["_initialize_worker"],
            namespace[method.__name__])


score_single_root, _run_root, _initialize_worker, PARALLEL_SCORER = _parallel_functions()


class ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout(
        FIFOWaitingPenaltyCurrentPolicyLaneRollout,
        native.ParallelRootTwoStepCurrentPolicyBGIncumbentLaneRollout):
    """Same two-sample/global-index pool; charge discarded waiting work too."""
    def __init__(self, options, planning_seed, sample_count=8, event_cap=128, workers=22):
        super().__init__(options, planning_seed, sample_count, event_cap)
        self.workers = workers
        self._worker_reuse_work["fifo_waiting_actual_work"] = {}

    @property
    def reuse_work(self):
        result = FIFOWaitingPenaltyCurrentPolicyLaneRollout.reuse_work.fget(self)
        for group, values in self._worker_reuse_work.items():
            native._accumulate(result[group], values)
        result["parallel_root_work"] = dict(self.parallel_work)
        return result

    def _choose(self, *context):
        self._scope_worker_peaks = Counter()
        decision = super()._choose(*context)
        for name in ("current_cache_peak_entries", "BG_cache_peak_entries"):
            self.last_reuse_work["inner_decision_reuse_work"][name] = self._scope_worker_peaks[name]
        return decision

    score_background_roots = PARALLEL_SCORER
