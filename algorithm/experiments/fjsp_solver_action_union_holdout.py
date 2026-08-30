"""Prospective exact-ledger holdout for the FJSP solver-action union."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import POLICY_ORDER, build_schedule
from algorithm.experiments.fjsp_exact_ledger_confirmation_v2 import (
    exact_completion_ledger,
    exact_ledger_relation,
)
from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_solver_action_union import (
    FROZEN_CONFIG,
    POLICY,
    build_fjsp_solver_action_union,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "scheduleurm.fjsp_solver_action_union_holdout.v1"
PREREGISTRATION_SCHEMA = "scheduleurm.fjsp_solver_action_union_preregistration.v1"
DEFAULT_SUITE_ROOT = REPO_ROOT / "tests/data/fjsp_solver_action_union_holdout_v1"
DEFAULT_PREREGISTRATION = DEFAULT_SUITE_ROOT / "preregistration.json"
DEFAULT_FULL = REPO_ROOT / "md/experiment_artifacts/fjsp_solver_action_union_holdout_v1_full.json.gz"
DEFAULT_COMPACT = REPO_ROOT / "md/experiment_artifacts/fjsp_solver_action_union_holdout_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md/fjsp_solver_action_union_holdout_v1.md"


class FJSPSolverActionUnionHoldoutError(ValueError):
    """Raised when a frozen FJSP holdout contract or ledger is invalid."""


def verify_preregistration(path: str | Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    prereg_path = Path(path)
    prereg = _load_json(prereg_path)
    if prereg.get("schema_version") != PREREGISTRATION_SCHEMA:
        raise FJSPSolverActionUnionHoldoutError("unexpected preregistration schema")
    claimed = str(prereg.get("preregistration_sha256_excluding_self") or "")
    payload = dict(prereg)
    payload.pop("preregistration_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise FJSPSolverActionUnionHoldoutError("preregistration digest mismatch")
    if prereg.get("holdout_outcomes_observed_before_freeze") is not False:
        raise FJSPSolverActionUnionHoldoutError("pre-outcome freeze is not certified")
    if prereg.get("solver_action_config") != FROZEN_CONFIG.snapshot():
        raise FJSPSolverActionUnionHoldoutError("solver action config changed after freeze")
    if tuple(prereg.get("baseline_policies") or ()) != tuple(POLICY_ORDER):
        raise FJSPSolverActionUnionHoldoutError("fixed-policy baseline family changed")
    for relative, expected_hash in sorted(
        (prereg.get("implementation_file_sha256") or {}).items()
    ):
        source = REPO_ROOT / relative
        if not source.is_file() or _file_sha256(source) != expected_hash:
            raise FJSPSolverActionUnionHoldoutError(
                f"implementation changed after preregistration: {relative}"
            )

    selected = list(prereg.get("selected_instances") or ())
    if not selected:
        raise FJSPSolverActionUnionHoldoutError("no holdout instances are registered")
    verified = []
    for row in selected:
        source = REPO_ROOT / str(row["local_path"])
        if not source.is_file() or _file_sha256(source) != str(row["file_sha256"]):
            raise FJSPSolverActionUnionHoldoutError(
                f"registered source changed: {row.get('local_path')}"
            )
        instance = load_fjsp_instance(source)
        if instance.source_sha256 != str(row["canonical_source_sha256"]):
            raise FJSPSolverActionUnionHoldoutError(
                f"canonical source changed: {row.get('local_path')}"
            )
        verified.append({"registration": row, "instance": instance})
    return {
        "ready": True,
        "path": str(prereg_path),
        "file_sha256": _file_sha256(prereg_path),
        "content_sha256_excluding_self": claimed,
        "instance_count": len(verified),
        "verified_instances": verified,
        "holdout_feedback_used_for_policy_revision": False,
    }


def build_holdout_report(
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    freeze = verify_preregistration(preregistration_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for verified in freeze["verified_instances"]:
        registration = verified["registration"]
        try:
            rows.append(_evaluate_instance(registration, verified["instance"]))
        except Exception as exc:
            failures.append(
                {
                    "relative_path": registration.get("upstream_relative_path"),
                    "family": registration.get("family"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    registered_count = int(freeze["instance_count"])
    protocol_pass = bool(
        registered_count > 0
        and len(rows) + len(failures) == registered_count
        and not failures
        and all(row["union_exact_argmax"] for row in rows)
        and all(row["selected_feasible"] for row in rows)
        and all(row["solver_candidate_feasible"] for row in rows)
    )
    nondominated_count = sum(
        row["solver_relation"] != "comparator_dominates"
        and row["base_family_relation"] != "comparator_dominates"
        and not row["fixed_policy_dominators"]
        for row in rows
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": {
            "pass": protocol_pass,
            "status": (
                "FJSP_SOLVER_ACTION_UNION_HOLDOUT_PASS"
                if protocol_pass
                else "FJSP_SOLVER_ACTION_UNION_HOLDOUT_FAIL"
            ),
            "registered_instance_count": registered_count,
            "completed_instance_count": len(rows),
            "failure_count": len(failures),
            "complete_solver_action_coverage_ready": bool(
                protocol_pass and all(row["solver_candidate_feasible"] for row in rows)
            ),
            "performance_nondominance_ready": bool(
                protocol_pass and nondominated_count == registered_count
            ),
            "global_fjsp_optimality_claim_ready": False,
        },
        "freeze": {
            key: freeze[key]
            for key in (
                "path",
                "file_sha256",
                "content_sha256_excluding_self",
                "instance_count",
                "holdout_feedback_used_for_policy_revision",
            )
        },
        "policy": POLICY,
        "solver_action_config": FROZEN_CONFIG.snapshot(),
        "rows": rows,
        "failures": failures,
        "aggregate": {
            "exact_union_argmax_count": sum(row["union_exact_argmax"] for row in rows),
            "solver_feasible_count": sum(row["solver_candidate_feasible"] for row in rows),
            "selected_nondominated_count": nondominated_count,
            "selected_origin_counts": dict(
                sorted(
                    Counter(
                        origin
                        for row in rows
                        for origin in row["selected_origins"]
                    ).items()
                )
            ),
            "solver_relation_counts": dict(
                sorted(Counter(row["solver_relation"] for row in rows).items())
            ),
            "base_family_relation_counts": dict(
                sorted(Counter(row["base_family_relation"] for row in rows).items())
            ),
            "fixed_policy_dominator_count": sum(
                bool(row["fixed_policy_dominators"]) for row in rows
            ),
            "geomean_selected_over_base": _metric_ratios(
                rows, "selected_exact_ledger", "base_family_exact_ledger"
            ),
            "geomean_selected_over_solver": _metric_ratios(
                rows, "selected_exact_ledger", "solver_exact_ledger"
            ),
        },
        "theorem_mapping": {
            "candidate_action": "complete feasible FJSP trajectory",
            "finite_family": "frozen heuristic family union fixed-budget CP-SAT action",
            "selector": "exact renewal-frame score argmax over the finite union",
            "verification": "integer precedence and machine nonoverlap completion ledger",
        },
        "claim_boundary": {
            "supports": [
                "post-freeze exact-ledger evaluation on registered unused Hurink instances",
                "fixed-budget solver trajectory admitted into the finite action family",
                "exact argmax only over that registered finite family",
            ],
            "does_not_support": [
                "global or arbitrary-instance FJSP optimality",
                "multiobjective optimality of a makespan-only CP-SAT incumbent",
                "free solver runtime or production deployment",
                "automatic stochastic-recurrence transfer",
            ],
        },
    }
    report["scientific_content_sha256"] = _digest(report)
    return report


def _evaluate_instance(registration: Mapping[str, Any], instance: Any) -> dict[str, Any]:
    run = build_fjsp_solver_action_union(instance)
    selected = exact_completion_ledger(instance, run["schedule"])
    base = exact_completion_ledger(
        instance, run["base_family"]["selected_schedule"]
    )
    solver_feasible = bool(run["solver_candidate"]["feasible"])
    solver = (
        exact_completion_ledger(instance, run["solver_candidate"]["schedule"])
        if solver_feasible
        else None
    )
    fixed_ledgers = {}
    fixed_relations = {}
    for policy in POLICY_ORDER:
        baseline = build_schedule(instance, policy)
        ledger = exact_completion_ledger(instance, baseline["schedule"])
        fixed_ledgers[policy] = ledger
        fixed_relations[policy] = exact_ledger_relation(selected, ledger)
    solver_relation = (
        exact_ledger_relation(selected, solver) if solver is not None else "no_solver_action"
    )
    base_relation = exact_ledger_relation(selected, base)
    selected_feasible = bool(
        run["selection"]["exact_union_argmax"]
        and selected["operation_ledger"]
        and run["gate"]["solver_action_admitted_only_after_feasibility_audit"]
    )
    return {
        "upstream_relative_path": registration["upstream_relative_path"],
        "local_path": registration["local_path"],
        "family": registration["family"],
        "file_sha256": registration["file_sha256"],
        "canonical_source_sha256": registration["canonical_source_sha256"],
        "job_count": instance.job_count,
        "machine_count": instance.machine_count,
        "operation_count": instance.operation_count,
        "selected_feasible": selected_feasible,
        "union_exact_argmax": bool(run["selection"]["exact_union_argmax"]),
        "effective_union_action_count": int(
            run["selection"]["effective_union_action_count"]
        ),
        "selected_origins": list(run["selection"]["selected_origins"]),
        "selected_renewal_score": run["selection"]["renewal_score"],
        "selected_exact_ledger": selected,
        "base_family_exact_ledger": base,
        "base_family_relation": base_relation,
        "solver_candidate_feasible": solver_feasible,
        "solver_status": run["solver_candidate"]["status"],
        "solver_optimality_proved_for_makespan": run["solver_candidate"][
            "optimality_proved_for_makespan"
        ],
        "solver_wall_time_s": run["solver_candidate"]["wall_time_s"],
        "solver_exact_ledger": solver,
        "solver_relation": solver_relation,
        "fixed_policy_exact_ledgers": fixed_ledgers,
        "fixed_policy_relations": fixed_relations,
        "fixed_policy_dominators": sorted(
            policy
            for policy, relation in fixed_relations.items()
            if relation == "comparator_dominates"
        ),
        "selection_audit": run["selection"],
    }


def _metric_ratios(
    rows: Sequence[Mapping[str, Any]], ours_key: str, comparator_key: str
) -> dict[str, float | None]:
    if not rows or any(row.get(comparator_key) is None for row in rows):
        return {"makespan": None, "sum_completion": None}
    return {
        "makespan": _geomean(
            float(row[ours_key]["makespan_ticks"])
            / float(row[comparator_key]["makespan_ticks"])
            for row in rows
        ),
        "sum_completion": _geomean(
            float(row[ours_key]["sum_completion_ticks"])
            / float(row[comparator_key]["sum_completion_ticks"])
            for row in rows
        ),
    }


def compact_certificate(
    report: Mapping[str, Any], full_path: Path, full_file_hash: str
) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}.compact.v1",
        "gate": report["gate"],
        "full_artifact_path": str(full_path.relative_to(REPO_ROOT)),
        "full_artifact_gzip_sha256": full_file_hash,
        "full_scientific_content_sha256": report["scientific_content_sha256"],
        "freeze": report["freeze"],
        "policy": report["policy"],
        "solver_action_config": report["solver_action_config"],
        "aggregate": report["aggregate"],
        "rows": [
            {
                key: row[key]
                for key in (
                    "upstream_relative_path",
                    "family",
                    "file_sha256",
                    "job_count",
                    "machine_count",
                    "operation_count",
                    "selected_feasible",
                    "union_exact_argmax",
                    "effective_union_action_count",
                    "selected_origins",
                    "selected_exact_ledger",
                    "base_family_exact_ledger",
                    "base_family_relation",
                    "solver_candidate_feasible",
                    "solver_status",
                    "solver_optimality_proved_for_makespan",
                    "solver_wall_time_s",
                    "solver_exact_ledger",
                    "solver_relation",
                    "fixed_policy_relations",
                    "fixed_policy_dominators",
                )
            }
            for row in report["rows"]
        ],
        "claim_boundary": report["claim_boundary"],
    }
    compact["compact_content_sha256_excluding_self"] = _digest(compact)
    return compact


def write_artifacts(
    report: Mapping[str, Any],
    *,
    full_path: str | Path = DEFAULT_FULL,
    compact_path: str | Path = DEFAULT_COMPACT,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    full = _repo_path(full_path)
    compact = _repo_path(compact_path)
    markdown = _repo_path(markdown_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    full_hash = _file_sha256(full)
    certificate = compact_certificate(report, full, full_hash)
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(markdown_report(certificate), encoding="utf-8")
    return {
        "full": str(full),
        "full_file_sha256": full_hash,
        "compact": str(compact),
        "compact_file_sha256": _file_sha256(compact),
        "markdown": str(markdown),
    }


def verify_artifact_pair(compact_path: str | Path = DEFAULT_COMPACT) -> dict[str, Any]:
    compact_path = Path(compact_path)
    compact = _load_json(compact_path)
    claimed = compact.get("compact_content_sha256_excluding_self")
    payload = dict(compact)
    payload.pop("compact_content_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise FJSPSolverActionUnionHoldoutError("compact digest mismatch")
    full_path = REPO_ROOT / str(compact["full_artifact_path"])
    if _file_sha256(full_path) != compact["full_artifact_gzip_sha256"]:
        raise FJSPSolverActionUnionHoldoutError("full gzip hash mismatch")
    with gzip.open(full_path, "rt", encoding="ascii") as stream:
        full = json.load(stream)
    full_payload = dict(full)
    full_claim = full_payload.pop("scientific_content_sha256", None)
    if _digest(full_payload) != full_claim:
        raise FJSPSolverActionUnionHoldoutError("full scientific digest mismatch")
    if full_claim != compact["full_scientific_content_sha256"]:
        raise FJSPSolverActionUnionHoldoutError("compact/full scientific digest mismatch")
    return {
        "ready": True,
        "registered_instance_count": full["gate"]["registered_instance_count"],
        "compact_file_sha256": _file_sha256(compact_path),
        "full_file_sha256": _file_sha256(full_path),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    gate = compact["gate"]
    aggregate = compact["aggregate"]
    return "\n".join(
        [
            "# FJSP solver-action union prospective holdout",
            "",
            f"- Status: `{gate['status']}`",
            f"- Instances: `{gate['completed_instance_count']}/{gate['registered_instance_count']}`.",
            f"- Solver-action coverage: `{aggregate['solver_feasible_count']}/{gate['registered_instance_count']}`.",
            f"- Exact union argmax: `{aggregate['exact_union_argmax_count']}/{gate['registered_instance_count']}`.",
            f"- Pareto-nondominated against solver/base/fixed policies: `{aggregate['selected_nondominated_count']}/{gate['registered_instance_count']}`.",
            f"- Geomean selected/base makespan: `{aggregate['geomean_selected_over_base']['makespan']}`.",
            f"- Geomean selected/base sum completion: `{aggregate['geomean_selected_over_base']['sum_completion']}`.",
            f"- Geomean selected/solver makespan: `{aggregate['geomean_selected_over_solver']['makespan']}`.",
            f"- Geomean selected/solver sum completion: `{aggregate['geomean_selected_over_solver']['sum_completion']}`.",
            "",
            "The result is exact only over the preregistered finite union. A feasible",
            "time-bounded CP-SAT action is not an unrestricted or multiobjective optimum.",
            "",
        ]
    )


def _geomean(values: Sequence[float] | Any) -> float:
    rows = [float(value) for value in values]
    if not rows or any(value <= 0 or not math.isfinite(value) for value in rows):
        raise FJSPSolverActionUnionHoldoutError("geometric mean inputs are invalid")
    return round(math.exp(statistics.fmean(math.log(value) for value in rows)), 12)


def _repo_path(path: str | Path) -> Path:
    output = Path(path)
    return output if output.is_absolute() else REPO_ROOT / output


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FJSPSolverActionUnionHoldoutError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(raw.encode("ascii")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_artifact_pair(args.compact), indent=2, sort_keys=True))
        return 0
    report = build_holdout_report(args.preregistration)
    outputs = write_artifacts(
        report,
        full_path=args.full,
        compact_path=args.compact,
        markdown_path=args.markdown,
    )
    print(json.dumps({"gate": report["gate"], "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
