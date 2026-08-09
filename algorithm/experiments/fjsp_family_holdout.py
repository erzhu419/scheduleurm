"""Prospective family-level FJSP holdout for the renewal-frame selector."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import POLICY_ORDER, build_schedule
from algorithm.experiments.fjsp_instances import FJSPInstance, load_fjsp_instance
from algorithm.experiments.fjsp_renewal_frame_upgrade import (
    POLICY,
    build_fjsp_renewal_frame_schedule,
)
from algorithm.experiments.or_exact_reference import solve_fjsp_cp_sat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "tests" / "data" / "fjsp_family_external_holdout"
DEFAULT_PREREGISTRATION = DEFAULT_ROOT / "preregistration.json"
DEFAULT_MANIFEST = DEFAULT_ROOT / "source_manifest.json"
DEFAULT_JSON = DEFAULT_ROOT / "fjsp_family_external_holdout_gate.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "fjsp_family_external_holdout_gate.md"
MATCHED_REFERENCE_FLOOR_S = 0.05


class FJSPFamilyHoldoutError(ValueError):
    """Raised when the prospective family-level contract fails closed."""


def validate_frozen_suite(
    *,
    suite_root: str | Path = DEFAULT_ROOT,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(suite_root)
    preregistration_file = Path(preregistration_path)
    manifest_file = Path(manifest_path)
    preregistration = _load_json(preregistration_file)
    manifest = _load_json(manifest_file)
    if preregistration.get("schema_version") != (
        "scheduleurm.fjsp_family_holdout.preregistration.v1"
    ):
        raise FJSPFamilyHoldoutError("unexpected preregistration schema")
    if manifest.get("schema_version") != "scheduleurm.fjsp_family_holdout.source.v1":
        raise FJSPFamilyHoldoutError("unexpected source manifest schema")
    prereg_hash = _file_sha256(preregistration_file)
    if manifest.get("preregistration_sha256") != prereg_hash:
        raise FJSPFamilyHoldoutError("source manifest does not bind preregistration")

    freeze = preregistration.get("implementation_freeze") or {}
    commit = str(freeze.get("repository_commit") or "")
    if len(commit) != 40:
        raise FJSPFamilyHoldoutError("invalid implementation freeze commit")
    _git_check(["cat-file", "-e", f"{commit}^{{commit}}"])
    implementation_files = freeze.get("implementation_sha256") or {}
    if not implementation_files:
        raise FJSPFamilyHoldoutError("implementation freeze is empty")
    verified_implementation = []
    for relative, expected in sorted(implementation_files.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FJSPFamilyHoldoutError(f"unsafe implementation path: {relative}")
        current = sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        committed = sha256(
            _git_output(["show", f"{commit}:{relative_path.as_posix()}"])
        ).hexdigest()
        if current != expected or committed != expected:
            raise FJSPFamilyHoldoutError(
                f"implementation drift after preregistration: {relative}"
            )
        verified_implementation.append({"path": str(relative), "sha256": expected})

    selected = tuple(str(row) for row in preregistration["selection"]["members"])
    rows = manifest.get("instances")
    if not isinstance(rows, list) or tuple(str(row.get("relative_path")) for row in rows) != selected:
        raise FJSPFamilyHoldoutError("manifest rows differ from preregistered selection")
    if len(selected) != len(set(selected)) or not selected:
        raise FJSPFamilyHoldoutError("selected instance paths must be nonempty and unique")
    excluded_families = set(preregistration["selection"]["excluded_development_families"])
    verified = []
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise FJSPFamilyHoldoutError(f"unsafe source path: {relative}")
        source = root / relative
        if _file_sha256(source) != str(row["sha256"]):
            raise FJSPFamilyHoldoutError(f"source hash mismatch: {relative}")
        instance = load_fjsp_instance(source)
        shape = {
            "job_count": instance.job_count,
            "machine_count": instance.machine_count,
            "operation_count": instance.operation_count,
        }
        if shape != {key: int(row[key]) for key in shape}:
            raise FJSPFamilyHoldoutError(f"source shape mismatch: {relative}")
        if str(row["family"]) in excluded_families:
            raise FJSPFamilyHoldoutError(f"development family leaked into holdout: {relative}")
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        if lower <= 0 or upper < lower:
            raise FJSPFamilyHoldoutError(f"invalid public bound: {relative}")
        verified.append({**row, "source": source, "instance": instance})
    return {
        "pass": True,
        "preregistration": preregistration,
        "preregistration_sha256": prereg_hash,
        "manifest_sha256": _file_sha256(manifest_file),
        "implementation": verified_implementation,
        "verified_instances": verified,
    }


def evaluate_instance(
    instance: FJSPInstance,
    *,
    lower_bound: float,
    upper_bound: float,
    run_cp_sat: bool,
    fixed_reference_budget_s: float,
) -> dict[str, Any]:
    started = perf_counter()
    ours = build_fjsp_renewal_frame_schedule(instance)
    ours_runtime = perf_counter() - started
    baselines = {policy: build_schedule(instance, policy) for policy in POLICY_ORDER}
    policies = {POLICY: ours["metrics"], **{key: value["metrics"] for key, value in baselines.items()}}
    comparisons = {
        policy: _relation(ours["metrics"], run["metrics"])
        for policy, run in baselines.items()
    }
    exact_reference = None
    if run_cp_sat:
        exact_reference = {
            "ours_runtime_matched_solver_budget": solve_fjsp_cp_sat(
                instance,
                time_limit_s=max(MATCHED_REFERENCE_FLOOR_S, ours_runtime),
                workers=1,
                incumbent_schedule=ours["schedule"],
            ),
            "fixed_budget": solve_fjsp_cp_sat(
                instance,
                time_limit_s=float(fixed_reference_budget_s),
                workers=1,
                incumbent_schedule=ours["schedule"],
            ),
        }
    reference_ready = bool(
        exact_reference is None
        or all(
            row["feasible_solution"] is True
            and row["verifier"]["pass"] is True
            for row in exact_reference.values()
        )
    )
    return {
        "instance": instance.snapshot(),
        "public_bounds": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "status": "closed" if lower_bound == upper_bound else "open",
        },
        "ours": ours,
        "ours_observed_runtime_s": round(ours_runtime, 9),
        "baselines": baselines,
        "policy_metrics": policies,
        "baseline_relations": comparisons,
        "ours_pareto_nondominated": not any(
            relation == "baseline_dominates" for relation in comparisons.values()
        ),
        "ours_strictly_dominates_every_baseline": all(
            relation == "ours_dominates" for relation in comparisons.values()
        ),
        "ours_makespan_over_public_upper_bound": (
            float(ours["metrics"]["makespan"]) / upper_bound
        ),
        "exact_reference": exact_reference,
        "reference_ready": reference_ready,
    }


def build_holdout_report(
    *,
    suite_root: str | Path = DEFAULT_ROOT,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    source = validate_frozen_suite(
        suite_root=suite_root,
        preregistration_path=preregistration_path,
        manifest_path=manifest_path,
    )
    protocol = source["preregistration"]["protocol"]
    if float(protocol["matched_reference_floor_s"]) != MATCHED_REFERENCE_FLOOR_S:
        raise FJSPFamilyHoldoutError("matched-reference floor differs from frozen runner")
    rows = []
    for registered in source["verified_instances"]:
        rows.append(
            {
                "relative_path": registered["relative_path"],
                "family": registered["family"],
                "result": evaluate_instance(
                    registered["instance"],
                    lower_bound=float(registered["lower_bound"]),
                    upper_bound=float(registered["upper_bound"]),
                    run_cp_sat=bool(protocol["run_cp_sat_reference"]),
                    fixed_reference_budget_s=float(protocol["fixed_reference_budget_s"]),
                ),
            }
        )
    protocol_pass = bool(
        rows
        and all(row["result"]["ours_pareto_nondominated"] for row in rows)
        and all(row["result"]["reference_ready"] for row in rows)
        and all(
            row["result"]["ours"]["selection"]["exact_generated_family_argmax"]
            for row in rows
        )
    )
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
    return {
        "schema_version": "scheduleurm.fjsp_family_holdout.v1",
        "source": {
            "preregistration_sha256": source["preregistration_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "implementation": source["implementation"],
        },
        "protocol": protocol,
        "rows": rows,
        "aggregate": {
            "instance_count": len(rows),
            "family_counts": family_counts,
            "ours_pareto_nondominated_count": sum(
                row["result"]["ours_pareto_nondominated"] for row in rows
            ),
            "ours_strict_every_baseline_count": sum(
                row["result"]["ours_strictly_dominates_every_baseline"] for row in rows
            ),
            "geomean_ours_makespan_over_public_upper_bound": _geomean(
                row["result"]["ours_makespan_over_public_upper_bound"] for row in rows
            ),
        },
        "gate": {
            "pass": protocol_pass,
            "status": "FJSP_FAMILY_HOLDOUT_PASS" if protocol_pass else "FJSP_FAMILY_HOLDOUT_FAIL",
            "protocol_and_feasibility_ready": protocol_pass,
            "performance_superiority_claim_ready": bool(
                rows
                and all(
                    row["result"]["ours_strictly_dominates_every_baseline"]
                    for row in rows
                )
            ),
            "global_fjsp_optimality_claim_ready": False,
            "stochastic_stability_transfer_claim_ready": False,
        },
        "claim_boundary": {
            "baseline_union_nondominance_is_construction_not_superiority": True,
            "cp_sat_reference_optimizes_makespan_only": True,
            "cp_sat_time_limit_excludes_model_construction": True,
            "scope": "registered public families and instances only",
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    aggregate = report["aggregate"]
    lines = [
        "# Prospective FJSP family holdout",
        "",
        f"- Protocol gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Registered instances: `{aggregate['instance_count']}`",
        f"- Families: `{json.dumps(aggregate['family_counts'], sort_keys=True)}`",
        f"- Ours Pareto-nondominated: `{aggregate['ours_pareto_nondominated_count']}/{aggregate['instance_count']}`",
        f"- Ours strictly dominates every registered baseline: `{aggregate['ours_strict_every_baseline_count']}/{aggregate['instance_count']}`",
        f"- Geometric mean makespan/public UB: `{aggregate['geomean_ours_makespan_over_public_upper_bound']:.6f}`",
        "",
        "| Family | Instance | Ours makespan/UB | Nondominated | Strict all-baseline dominance |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        result = row["result"]
        lines.append(
            f"| {row['family']} | {result['instance']['name']} | "
            f"{result['ours_makespan_over_public_upper_bound']:.6f} | "
            f"{str(result['ours_pareto_nondominated']).lower()} | "
            f"{str(result['ours_strictly_dominates_every_baseline']).lower()} |"
        )
    lines.extend(
        [
            "",
            "The candidate union contains the registered heuristic trajectories. Its",
            "nondominance guarantee is therefore a construction property, not evidence of",
            "superiority over arbitrary FJSP solvers. CP-SAT is a separate time-bounded",
            "makespan reference; mean flow remains descriptive.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> None:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown_report(report), encoding="utf-8")


def _relation(ours: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    keys = ("makespan", "mean_flow_time")
    ours_weak = all(float(ours[key]) <= float(baseline[key]) + 1e-12 for key in keys)
    ours_strict = any(float(ours[key]) < float(baseline[key]) - 1e-12 for key in keys)
    baseline_weak = all(float(baseline[key]) <= float(ours[key]) + 1e-12 for key in keys)
    baseline_strict = any(float(baseline[key]) < float(ours[key]) - 1e-12 for key in keys)
    if ours_weak and ours_strict:
        return "ours_dominates"
    if baseline_weak and baseline_strict:
        return "baseline_dominates"
    if not ours_strict and not baseline_strict:
        return "tie"
    return "tradeoff"


def _geomean(values: Iterable[float]) -> float:
    import math

    rows = [float(value) for value in values]
    if not rows or any(value <= 0 or not math.isfinite(value) for value in rows):
        raise FJSPFamilyHoldoutError("geometric mean inputs must be finite and positive")
    return math.exp(sum(math.log(value) for value in rows) / len(rows))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FJSPFamilyHoldoutError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _git_check(args: Sequence[str]) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True)


def _git_output(args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_holdout_report(
        suite_root=args.suite_root,
        preregistration_path=args.preregistration,
        manifest_path=args.manifest,
    )
    write_artifacts(report, json_path=args.output, markdown_path=args.markdown)
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
