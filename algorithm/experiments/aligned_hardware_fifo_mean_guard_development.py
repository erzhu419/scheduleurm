"""Full24 FIFO J evaluation with a fixed same-model BG mean guard."""
import ast
import gzip
import inspect
import json
from pathlib import Path
import textwrap
from types import FunctionType

from algorithm.fifo_mean_guard_lane_rollout import ParallelFIFOMeanGuardLaneRollout
from algorithm.experiments import aligned_hardware_fifo_waiting_penalty_development as fifo
from algorithm.experiments.fifo_mean_guard_checks import MEAN_GUARD_WORK, check_rollout

ROOT, ARTIFACTS = fifo.ROOT, fifo.ARTIFACTS
STEM = "aligned_hardware_fifo_mean_guard_development_20261003"
SOURCE = str(Path(__file__).resolve().relative_to(ROOT))
SOURCES = (*fifo.SOURCES, SOURCE, "algorithm/fifo_mean_guard_lane_rollout.py",
    "algorithm/experiments/fifo_mean_guard_checks.py",
    "md/experiment_artifacts/" + STEM + "_design.json")
COMPARISONS = (*fifo.COMPARISONS, "unconstrained_FIFO")

_namespace = {**fifo.__dict__, "__name__": __name__, "STEM": STEM, "SOURCE": SOURCE,
    "SOURCES": SOURCES, "COMPARISONS": COMPARISONS, "MEAN_GUARD_WORK": MEAN_GUARD_WORK,
    "check_rollout": check_rollout,
    "ParallelFIFOWaitingPenaltyCurrentPolicyLaneRollout": ParallelFIFOMeanGuardLaneRollout}


def _bind(function):
    return FunctionType(function.__code__, _namespace, function.__name__, function.__defaults__)


def _adapt(function, transform):
    tree = transform.visit(ast.parse(textwrap.dedent(inspect.getsource(function))))
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), _namespace)
    return _namespace[function.__name__]


for _name in ("path", "read", "dump", "load_protocol", "snapshots", "action_comparison",
              "parent_increment", "compact", "main"):
    _namespace[_name] = _bind(getattr(fifo, _name))
_finite_references = _bind(fifo.references)


def _unconstrained_references(directory, science):
    """Certify the completed FIFO24 cache, used only for descriptive contrasts."""
    prior = fifo.load_protocol(directory)
    completed = fifo.read(fifo.path(directory, "_summary.json"))
    if (not completed["execution_checks_passed"] or completed["completed_shards"] != 24
            or completed["registered_at_utc"] != prior["registered_at_utc"]
            or prior["models"] != science["models"] or prior["tapes"] != science["tapes"]
            or prior["planning_samples"] != 8 or prior["planning_event_cap"] != 128):
        raise ValueError("completed same-tape unconstrained FIFO24 diagnostic cache required")
    refs = []
    for index, tape in enumerate(science["tapes"]):
        doc = fifo.read(fifo.path(directory, f"_shard_{index:02d}.json"))
        if (doc["index"] != index or doc["registered_at_utc"] != prior["registered_at_utc"]
                or doc["row"]["tape_id"] != tape["tape_id"]
                or not doc["row"]["rollout_checks_passed"] or not Path(doc["trajectory_file"]).is_file()):
            raise ValueError("certified FIFO diagnostic trajectory required")
        refs.append({"tape_id": tape["tape_id"], "trajectory_file": doc["trajectory_file"]})
    return {"unconstrained_FIFO_registered_at_utc": prior["registered_at_utc"],
        "unconstrained_FIFO_references": refs, "unconstrained_FIFO_prior_decision": completed["decision"]}


def references(protocol, index):
    controls = _finite_references(protocol, index)
    ref = protocol["unconstrained_FIFO_references"][index]
    with gzip.open(ref["trajectory_file"], "rt") as source:
        saved = json.load(source)
    if saved["registered_at_utc"] != protocol["unconstrained_FIFO_registered_at_utc"]:
        raise ValueError("registered unconstrained FIFO diagnostic trajectory differs")
    controls["unconstrained_FIFO"] = next(r["policies"]["arrival_rollout"] for r in saved["runs"]
                                          if r["tape_id"] == ref["tape_id"])
    return controls


_namespace.update(references=references, _unconstrained_references=_unconstrained_references)


class _Registration(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "protocol" and node.func.attr == "update"):
            replacements = {
                "experiment": "fifo_mean_guard_finite_full24_development_v1",
                "method": "Add only the same-model genuine S1_tag<=BG S1_tag admissibility test to fixed FIFO J=S1_tag+sum(W_tag^2)/tau_max, coefficient1. Preserve ordinary BG S2<=MW roots and metadata, BG-first J ties, two OLD-current calls then MW, original8 independent paths/cap128, eligibility and fallbacks. This sampled mean guard does not guarantee actual trajectory mean improvement.",
                "budget": "Preserve native finite/FIFO logical fees and partial overflow work. After a complete new stage, test each original S2-feasible root once against BG genuine S1; record mean_guard_comparisons, allowed and filtered counts separately. Outer J ordering compares candidate_count+mean_guard_allowed-2 pairs. Parallel workers and virtual children keep frozen finite/FIFO scoring; only the real caller applies the mean guard. Charge all physical chunks and CPU including discarded chunks; constructor CPU is already within worker CPU.",
            }
            for keyword in node.keywords:
                if keyword.arg in replacements:
                    keyword.value = ast.Constant(replacements[keyword.arg])
                    self.changed += 1
            node.keywords.append(ast.keyword(None, ast.parse(
                "_unconstrained_references(directory, science)", mode="eval").body))
        return node


_registration = _Registration()
register = _adapt(fifo.register, _registration)
if _registration.changed != 3:
    raise ValueError("FIFO registration metadata sites changed")


class _Work(ast.NodeTransformer):
    changed = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "executed":
            self.changed += 1
            return [ast.parse("work.update({name: sum(f['rollout'][name] for f in candidate['frames']) "
                "for name in MEAN_GUARD_WORK})").body[0], node]
        return node


_work = _Work()
run_shard = _adapt(fifo.run_shard, _work)
if _work.changed != 1:
    raise ValueError("FIFO candidate work site changed")


class _Summary(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Constant(self, node):
        if node.value == "FIFO_WAITING_PENALTY_DEVELOPMENT_":
            self.changed += 1
            return ast.copy_location(ast.Constant("FIFO_MEAN_GUARD_DEVELOPMENT_"), node)
        if node.value == "fifo_waiting_penalty_candidate":
            return ast.copy_location(ast.Constant("fifo_mean_guard_candidate"), node)
        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        if node.arg == "performance_gate_applied":
            node.value = ast.parse("role in ('MW', 'BG')", mode="eval").body
        if node.arg == "interpretation":
            node.value = ast.Constant("Same24 development closed loops and three shared seeds. Original MW/BG joint gate and unchanged finite-parent increment decide promotion. Cached unconstrained FIFO24 contrasts are diagnostic only. Sampled S1<=BG is an admissibility rule, not a guarantee of actual mean nonregression; genuine S1/S2 and failed outcomes remain reported. Confirmation is untouched.")
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "dict" and any(k.arg == "decision" for k in node.keywords):
            node.keywords.extend([
                ast.keyword("diagnostic_reference_trajectories", ast.Constant(24)),
                ast.keyword("mean_guard_work", ast.parse(
                    "{name: sum(r['work']['arrival_rollout'][name] for r in rows) for name in MEAN_GUARD_WORK}",
                    mode="eval").body),
            ])
        return node


_summary = _Summary()
merge = _adapt(fifo.merge, _summary)
if _summary.changed != 1:
    raise ValueError("FIFO decision-prefix site changed")
globals().update({name: _namespace[name] for name in (
    "path", "read", "dump", "load_protocol", "snapshots", "action_comparison",
    "parent_increment", "compact", "main")})


if __name__ == "__main__":
    main()
