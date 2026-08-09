"""Prospective multi-family PSPLIB holdout for the renewal-frame selector.

The source instances, implementation hashes, public optima, selection rule, and
solver budgets are committed before this runner is allowed to emit outcomes.
Protocol validity is deliberately separate from empirical superiority.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.mmrcpsp_benchmark import schedule_instance
from algorithm.experiments.mmrcpsp_instances import (
    MMRCPSPInstance,
    parse_psplib_mm,
)
from algorithm.experiments.mmrcpsp_renewal_frame_upgrade import (
    POLICY,
    build_mmrcpsp_renewal_frame_schedule,
)
from algorithm.experiments.mmrcpsp_trajectory_upgrade import BASELINE_POLICIES
from algorithm.experiments.or_exact_reference import solve_mmrcpsp_cp_sat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "tests" / "data" / "mmrcpsp_family_external_holdout"
DEFAULT_PREREGISTRATION = DEFAULT_ROOT / "preregistration.json"
DEFAULT_MANIFEST = DEFAULT_ROOT / "source_manifest.json"
DEFAULT_JSON = DEFAULT_ROOT / "mmrcpsp_family_external_holdout_gate.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "mmrcpsp_family_external_holdout_gate.md"
MATCHED_REFERENCE_FLOOR_S = 0.05


class MMRCPSPFamilyHoldoutError(ValueError):
    """Raised when the frozen family-level protocol fails closed."""


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
        "scheduleurm.mmrcpsp_family_holdout.preregistration.v1"
    ):
        raise MMRCPSPFamilyHoldoutError("unexpected preregistration schema")
    if manifest.get("schema_version") != (
        "scheduleurm.mmrcpsp_family_holdout.source.v1"
    ):
        raise MMRCPSPFamilyHoldoutError("unexpected source manifest schema")
    prereg_hash = _file_sha256(preregistration_file)
    if manifest.get("preregistration_sha256") != prereg_hash:
        raise MMRCPSPFamilyHoldoutError("manifest does not bind preregistration")

    prereg_relative = preregistration_file.resolve().relative_to(REPO_ROOT.resolve())
    prereg_commit = _git_output(
        ["log", "-1", "--format=%H", "--", prereg_relative.as_posix()]
    ).decode().strip()
    if len(prereg_commit) != 40:
        raise MMRCPSPFamilyHoldoutError("preregistration is not committed")
    committed_prereg = _git_output(
        ["show", f"{prereg_commit}:{prereg_relative.as_posix()}"]
    )
    if sha256(committed_prereg).hexdigest() != prereg_hash:
        raise MMRCPSPFamilyHoldoutError("committed preregistration differs from disk")
    for output in preregistration.get("outputs") or ():
        relative = Path(str(output))
        if relative.is_absolute() or ".." in relative.parts:
            raise MMRCPSPFamilyHoldoutError(f"unsafe output path: {output}")
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{prereg_commit}:{relative.as_posix()}"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if exists:
            raise MMRCPSPFamilyHoldoutError(
                f"result predates preregistration: {output}"
            )

    freeze = preregistration.get("implementation_freeze") or {}
    freeze_commit = str(freeze.get("repository_commit") or "")
    if len(freeze_commit) != 40:
        raise MMRCPSPFamilyHoldoutError("invalid implementation freeze commit")
    _git_check(["cat-file", "-e", f"{freeze_commit}^{{commit}}"])
    verified_implementation = []
    for relative, expected in sorted(
        (freeze.get("implementation_sha256") or {}).items()
    ):
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise MMRCPSPFamilyHoldoutError(f"unsafe implementation path: {relative}")
        current = _file_sha256(REPO_ROOT / path)
        committed = sha256(
            _git_output(["show", f"{freeze_commit}:{path.as_posix()}"])
        ).hexdigest()
        if current != expected or committed != expected:
            raise MMRCPSPFamilyHoldoutError(
                f"implementation drift after preregistration: {relative}"
            )
        verified_implementation.append({"path": str(relative), "sha256": expected})
    if not verified_implementation:
        raise MMRCPSPFamilyHoldoutError("empty implementation freeze")

    members = tuple(str(row) for row in preregistration["selection"]["members"])
    rows = manifest.get("instances")
    if not isinstance(rows, list) or tuple(
        str(row.get("relative_path")) for row in rows
    ) != members:
        raise MMRCPSPFamilyHoldoutError("manifest differs from registered selection")
    if not members or len(members) != len(set(members)):
        raise MMRCPSPFamilyHoldoutError("members must be nonempty and unique")
    verified = []
    for row in rows:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise MMRCPSPFamilyHoldoutError(f"unsafe source path: {relative}")
        source = root / relative
        if _file_sha256(source) != str(row["sha256"]):
            raise MMRCPSPFamilyHoldoutError(f"source hash mismatch: {relative}")
        instance = parse_psplib_mm(source)
        shape = instance_shape(instance)
        if shape != {key: int(row[key]) for key in shape}:
            raise MMRCPSPFamilyHoldoutError(f"source shape mismatch: {relative}")
        optimum = int(row["public_optimal_makespan"])
        if optimum <= 0 or optimum >= 16384:
            raise MMRCPSPFamilyHoldoutError(f"invalid public optimum: {relative}")
        verified.append({**row, "source": source, "instance": instance})
    return {
        "pass": True,
        "preregistration": preregistration,
        "preregistration_sha256": prereg_hash,
        "preregistration_commit": prereg_commit,
        "manifest_sha256": _file_sha256(manifest_file),
        "implementation": verified_implementation,
        "verified_instances": verified,
    }


def instance_shape(instance: MMRCPSPInstance) -> dict[str, int]:
    return {
        "declared_job_count": int(instance.declared_job_count),
        "real_job_count": len(instance.real_job_ids),
        "mode_count": sum(len(activity.modes) for activity in instance.activities),
        "precedence_arc_count": sum(
            len(activity.successors) for activity in instance.activities
        ),
        "renewable_resource_count": len(instance.renewable_capacities),
        "nonrenewable_resource_count": len(instance.nonrenewable_capacities),
        "horizon": int(instance.horizon),
    }


def evaluate_instance(
    instance: MMRCPSPInstance,
    *,
    public_optimal_makespan: int,
    run_cp_sat: bool,
    fixed_reference_budget_s: float,
) -> dict[str, Any]:
    started = perf_counter()
    ours = build_mmrcpsp_renewal_frame_schedule(instance)
    ours_runtime = perf_counter() - started
    baselines = {
        policy: schedule_instance(instance, policy).snapshot()
        for policy in BASELINE_POLICIES
    }
    if not all(row["feasible"] for row in baselines.values()):
        failed = [key for key, row in baselines.items() if not row["feasible"]]
        raise MMRCPSPFamilyHoldoutError(
            f"{instance.name}: frozen baseline infeasible: {failed}"
        )
    relations = {
        policy: _relation(ours["metrics"], row["metrics"])
        for policy, row in baselines.items()
    }
    exact_reference = None
    if run_cp_sat:
        exact_reference = {
            "ours_runtime_matched_solver_budget": solve_mmrcpsp_cp_sat(
                instance,
                time_limit_s=max(MATCHED_REFERENCE_FLOOR_S, ours_runtime),
                workers=1,
                incumbent_schedule=ours["schedule"],
            ),
            "fixed_budget": solve_mmrcpsp_cp_sat(
                instance,
                time_limit_s=float(fixed_reference_budget_s),
                workers=1,
                incumbent_schedule=ours["schedule"],
            ),
        }
    reference_ready = bool(
        exact_reference is None
        or all(
            row["feasible_solution"] is True and row["verifier"]["schedule_feasible"] is True
            for row in exact_reference.values()
        )
    )
    ours_makespan = float(ours["metrics"]["makespan"])
    optimum = float(public_optimal_makespan)
    if ours_makespan + 1e-12 < optimum:
        raise MMRCPSPFamilyHoldoutError(
            f"{instance.name}: ours beats registered exact optimum; source mismatch"
        )
    return {
        "instance": instance.snapshot(),
        "public_optimal_makespan": int(public_optimal_makespan),
        "ours": ours,
        "ours_observed_runtime_s": round(ours_runtime, 9),
        "baselines": baselines,
        "baseline_relations": relations,
        "ours_pareto_nondominated": not any(
            relation == "baseline_dominates" for relation in relations.values()
        ),
        "ours_strictly_dominates_every_baseline": all(
            relation == "ours_dominates" for relation in relations.values()
        ),
        "ours_makespan_over_optimum": ours_makespan / optimum,
        "ours_attains_public_optimum": abs(ours_makespan - optimum) <= 1e-12,
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
        raise MMRCPSPFamilyHoldoutError("matched reference floor drifted")
    rows = []
    for registered in source["verified_instances"]:
        rows.append(
            {
                "relative_path": registered["relative_path"],
                "family": registered["family"],
                "result": evaluate_instance(
                    registered["instance"],
                    public_optimal_makespan=int(registered["public_optimal_makespan"]),
                    run_cp_sat=bool(protocol["run_cp_sat_reference"]),
                    fixed_reference_budget_s=float(protocol["fixed_reference_budget_s"]),
                ),
            }
        )
    protocol_pass = bool(
        rows
        and all(row["result"]["reference_ready"] for row in rows)
        and all(
            row["result"]["ours"]["selection"]["exact_generated_family_argmax"]
            for row in rows
        )
    )
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
    strict_count = sum(
        row["result"]["ours_strictly_dominates_every_baseline"] for row in rows
    )
    return {
        "schema_version": "scheduleurm.mmrcpsp_family_holdout.v1",
        "source": {
            "preregistration_sha256": source["preregistration_sha256"],
            "preregistration_commit": source["preregistration_commit"],
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
            "ours_strict_every_baseline_count": strict_count,
            "ours_attains_public_optimum_count": sum(
                row["result"]["ours_attains_public_optimum"] for row in rows
            ),
            "geomean_ours_makespan_over_public_optimum": _geomean(
                row["result"]["ours_makespan_over_optimum"] for row in rows
            ),
        },
        "gate": {
            "pass": protocol_pass,
            "status": (
                "MMRCPSP_FAMILY_HOLDOUT_PASS"
                if protocol_pass
                else "MMRCPSP_FAMILY_HOLDOUT_FAIL"
            ),
            "protocol_and_feasibility_ready": protocol_pass,
            "performance_superiority_claim_ready": bool(
                rows and strict_count == len(rows)
            ),
            "global_mmrcpsp_optimality_claim_ready": False,
            "stochastic_stability_transfer_claim_ready": False,
        },
        "claim_boundary": {
            "baseline_union_nondominance_is_construction_not_superiority": True,
            "public_optimum_is_makespan_only": True,
            "cp_sat_reference_optimizes_makespan_only": True,
            "scope": "registered PSPLIB multi-mode families and instances only",
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    aggregate = report["aggregate"]
    gate = report["gate"]
    lines = [
        "# Prospective PSPLIB MMRCPSP family holdout",
        "",
        f"- Protocol gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Registered instances: `{aggregate['instance_count']}`",
        f"- Families: `{json.dumps(aggregate['family_counts'], sort_keys=True)}`",
        f"- Ours Pareto-nondominated: `{aggregate['ours_pareto_nondominated_count']}/{aggregate['instance_count']}`",
        f"- Ours strictly dominates every registered heuristic: `{aggregate['ours_strict_every_baseline_count']}/{aggregate['instance_count']}`",
        f"- Ours attains public optimal makespan: `{aggregate['ours_attains_public_optimum_count']}/{aggregate['instance_count']}`",
        f"- Geometric mean makespan/optimum: `{aggregate['geomean_ours_makespan_over_public_optimum']:.6f}`",
        "",
        "| Family | Instance | Makespan/optimum | Nondominated | Strict all-heuristic dominance |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        result = row["result"]
        lines.append(
            f"| {row['family']} | {result['instance']['name']} | "
            f"{result['ours_makespan_over_optimum']:.6f} | "
            f"{str(result['ours_pareto_nondominated']).lower()} | "
            f"{str(result['ours_strictly_dominates_every_baseline']).lower()} |"
        )
    lines.extend(
        [
            "",
            "Protocol validity does not imply performance superiority. The registered",
            "heuristic union is contained in the generated candidate family; its",
            "nondominance property is therefore constructional. Public optima and the",
            "time-bounded CP-SAT sidecar concern makespan only, while mean flow remains",
            "a descriptive second objective.",
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
    rows = [float(value) for value in values]
    if not rows or any(value <= 0 or not math.isfinite(value) for value in rows):
        raise MMRCPSPFamilyHoldoutError("geometric mean inputs must be positive")
    return math.exp(sum(math.log(value) for value in rows) / len(rows))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MMRCPSPFamilyHoldoutError(f"expected JSON object: {path}")
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
