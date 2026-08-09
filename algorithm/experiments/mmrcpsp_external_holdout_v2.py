"""Run the repaired frozen MMRCPSP policy on a disjoint PSPLIB holdout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_external_holdout import (
    ExternalHoldoutError,
    _aggregate,
    frozen_protocol_sha256,
    frozen_protocol_snapshot,
)
from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm
from algorithm.experiments.mmrcpsp_trajectory_upgrade import (
    TrajectoryUpgradeError,
    build_trajectory_upgrade,
    write_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = REPO_ROOT / "tests" / "data" / "mmrcpsp_psplib_external_holdout_v2"
DEFAULT_PREREGISTRATION = DEFAULT_SUITE_DIR / "preregistration.json"
DEFAULT_MANIFEST = DEFAULT_SUITE_DIR / "source_manifest.json"
DEFAULT_JSON = DEFAULT_SUITE_DIR / "mmrcpsp_psplib_external_holdout_v2_gate.json"
DEFAULT_MARKDOWN = DEFAULT_SUITE_DIR / "mmrcpsp_psplib_external_holdout_v2_gate.md"
V1_PREREGISTRATION = (
    REPO_ROOT / "tests" / "data" / "mmrcpsp_psplib_external_holdout" / "preregistration.json"
)

FROZEN_IMPLEMENTATION_COMMIT = "c62cddd198d15faaa3d9c2d36415142a42caf727"
PREREGISTRATION_COMMIT = "2856430e340906fb9f7f5470f481e322b6ee5d3c"
PREREGISTRATION_SHA256 = "a8738729005e23cad012bc2f97630a01a93955983430baffb6955f62f241f20c"
FROZEN_PROTOCOL_SHA256 = "999cc0c9ac667e3589364dc2a92ddb1f1803594d9971b0bd812eb84e1e739d11"
FROZEN_IMPLEMENTATION_SHA256 = {
    "algorithm/experiments/mmrcpsp_trajectory_upgrade.py": (
        "f0363c727cc85878bb8c14bfaab3bc76f2ae5befb1c91bf565f46e9533d21f5d"
    ),
    "algorithm/experiments/mmrcpsp_benchmark.py": (
        "d2998f8579e3030d20cff0de42d572dc1f09e8379276c49a976b7f56af63d754"
    ),
    "algorithm/experiments/mmrcpsp_instances.py": (
        "d2d6e0dd0ef0ec9f9d9ff1795024aded81d567dd54e1b6e6200fc306819f99b2"
    ),
    "algorithm/experiments/mmrcpsp_suite_benchmark.py": (
        "14a8164151307321d88541e6f5306c3863a51bc1e54ba88a137c7a2dbc957fed"
    ),
}


def verify_pre_analysis_freeze(
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    preregistration = Path(preregistration_path)
    relative = preregistration.resolve().relative_to(REPO_ROOT.resolve())
    _git_check(["cat-file", "-e", f"{FROZEN_IMPLEMENTATION_COMMIT}^{{commit}}"])
    _git_check(["cat-file", "-e", f"{PREREGISTRATION_COMMIT}^{{commit}}"])
    if frozen_protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise ExternalHoldoutError("current repaired protocol configuration drifted")
    implementation_files = []
    for path, expected in FROZEN_IMPLEMENTATION_SHA256.items():
        current = sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        frozen = sha256(
            _git_output(["show", f"{FROZEN_IMPLEMENTATION_COMMIT}:{path}"])
        ).hexdigest()
        if current != expected or frozen != expected:
            raise ExternalHoldoutError(f"repaired implementation drift: {path}")
        implementation_files.append({"path": path, "sha256": expected})
    preregistration_bytes = preregistration.read_bytes()
    if sha256(preregistration_bytes).hexdigest() != PREREGISTRATION_SHA256:
        raise ExternalHoldoutError("v2 preregistration bytes drifted")
    committed = _git_output(["show", f"{PREREGISTRATION_COMMIT}:{relative.as_posix()}"])
    if sha256(committed).hexdigest() != PREREGISTRATION_SHA256:
        raise ExternalHoldoutError("v2 preregistration commit blob mismatch")
    if _path_exists_in_commit(FROZEN_IMPLEMENTATION_COMMIT, relative):
        raise ExternalHoldoutError("v2 preregistration predates repaired policy freeze")
    if _commit_glob_has_mm_files(PREREGISTRATION_COMMIT, relative.parent):
        raise ExternalHoldoutError("v2 instance bytes were present at preregistration")
    return {
        "pass": True,
        "implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "protocol_config_sha256": FROZEN_PROTOCOL_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_files": implementation_files,
        "selection_registered_after_repair_freeze": True,
        "instance_bytes_absent_at_preregistration": True,
        "policy_or_config_updates_after_holdout": False,
    }


def validate_external_source(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    suite = Path(suite_dir)
    preregistration_file = Path(preregistration_path)
    manifest_file = Path(manifest_path)
    preregistration = _load_json(preregistration_file)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if preregistration.get("schema_version") != (
        "scheduleurm.mmrcpsp_external_holdout.preregistration.v2"
    ):
        raise ExternalHoldoutError("unexpected v2 preregistration schema")
    if manifest.get("schema_version") != (
        "scheduleurm.mmrcpsp_external_holdout.source.v2"
    ):
        raise ExternalHoldoutError("unexpected v2 source manifest schema")
    if manifest.get("suite") != preregistration.get("suite"):
        raise ExternalHoldoutError("v2 suite name differs from preregistration")
    if manifest.get("preregistration_sha256") != sha256(
        preregistration_file.read_bytes()
    ).hexdigest():
        raise ExternalHoldoutError("v2 source manifest does not bind preregistration")
    if manifest.get("source") != preregistration.get("source"):
        raise ExternalHoldoutError("v2 archive contract differs from preregistration")
    if manifest.get("pre_analysis_freeze") != preregistration.get(
        "pre_analysis_freeze"
    ):
        raise ExternalHoldoutError("v2 freeze contract differs from preregistration")

    selected = tuple(preregistration["selection"]["selected_members"])
    v1_selected = set(_load_json(V1_PREREGISTRATION)["selection"]["selected_members"])
    if len(selected) != 56 or len(set(selected)) != 56:
        raise ExternalHoldoutError("v2 holdout must contain 56 unique members")
    if set(selected) & v1_selected or "j1010_1.mm" in selected:
        raise ExternalHoldoutError("v2 holdout overlaps development or v1 rows")
    rows = manifest.get("instances")
    if not isinstance(rows, list) or tuple(str(row.get("file")) for row in rows) != selected:
        raise ExternalHoldoutError("v2 manifest differs from preregistered selection")
    actual = tuple(sorted(path.name for path in suite.glob("*.mm")))
    if actual != tuple(sorted(selected)):
        raise ExternalHoldoutError("v2 file set differs from preregistration")

    verified = []
    for row in rows:
        name = str(row["file"])
        if Path(name).name != name or not name.endswith(".mm"):
            raise ExternalHoldoutError(f"unsafe v2 source path: {name}")
        path = suite / name
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != str(row["sha256"]):
            raise ExternalHoldoutError(f"v2 source hash mismatch for {name}")
        if len(raw) != int(row["size_bytes"]):
            raise ExternalHoldoutError(f"v2 source size mismatch for {name}")
        instance = parse_psplib_mm(path)
        actual_semantics = {
            "jobs_including_dummies": instance.declared_job_count,
            "horizon": instance.horizon,
            "renewable_capacities": list(instance.renewable_capacities),
            "nonrenewable_capacities": list(instance.nonrenewable_capacities),
        }
        expected_semantics = {
            key: row[key]
            for key in (
                "jobs_including_dummies",
                "horizon",
                "renewable_capacities",
                "nonrenewable_capacities",
            )
        }
        if actual_semantics != expected_semantics:
            raise ExternalHoldoutError(f"v2 semantic metadata mismatch for {name}")
        verified.append({**row, "path": path})
    return {
        "pass": True,
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "verified_instances": verified,
        "disjoint_from_v1_and_development": True,
    }


def build_external_holdout_report(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    freeze = verify_pre_analysis_freeze(preregistration_path)
    source = validate_external_source(suite_dir, preregistration_path, manifest_path)
    config_before = frozen_protocol_sha256()
    paths = [Path(row["path"]) for row in source["verified_instances"]]
    if workers == 1:
        results = [_run_one(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as executor:
            results = list(executor.map(_run_one, paths))
    config_after = frozen_protocol_sha256()
    if config_before != config_after:
        raise ExternalHoldoutError("v2 frozen protocol changed during execution")

    rows: dict[str, Any] = {}
    failures = []
    theorem_mapping: Mapping[str, Any] = {}
    for result in results:
        if result["failure"] is not None:
            failures.append(result["failure"])
            continue
        rows[result["instance"]] = result["row"]
        theorem_mapping = result["theorem_mapping"]
    completed_all = not failures and len(rows) == len(paths)
    all_feasible = completed_all and all(
        row["gate"]["all_evaluated_schedules_feasible"] for row in rows.values()
    )
    exact = completed_all and all(
        row["oracle_audit"]["exact_over_candidate_family"] for row in rows.values()
    )
    zero_resource = completed_all and all(
        row["gate"]["zero_resource_violation"] for row in rows.values()
    )
    zero_precedence = completed_all and all(
        row["gate"]["zero_precedence_violation"] for row in rows.values()
    )
    aggregate = _aggregate(rows) if rows else {}
    nondominated = sum(row["pareto"]["selected_is_nondominated"] for row in rows.values())
    protocol_pass = bool(
        freeze["pass"]
        and source["pass"]
        and completed_all
        and all_feasible
        and exact
        and zero_resource
        and zero_precedence
    )
    return {
        "schema_version": "scheduleurm.mmrcpsp_external_holdout.v2",
        "suite": "PSPLIB j10 multi-mode disjoint post-repair holdout",
        "deterministic_algorithm": True,
        "parallel_execution_is_only_an_instance_level_runtime_optimization": True,
        "worker_count": min(workers, len(paths)),
        "source_manifest_sha256": source["manifest_sha256"],
        "pre_analysis_freeze": freeze,
        "frozen_protocol": {
            "config": frozen_protocol_snapshot(),
            "config_sha256_before": config_before,
            "config_sha256_after": config_after,
            "per_instance_tuning": False,
            "holdout_feedback_used_for_configuration": False,
        },
        "instances": rows,
        "failures": failures,
        "aggregate": aggregate,
        "theorem_mapping": theorem_mapping,
        "gate": {
            "pass": protocol_pass,
            "prospective_external_holdout_ready": protocol_pass,
            "registered_instance_count": len(paths),
            "completed_instance_count": len(rows),
            "completed_all_registered_instances": completed_all,
            "all_schedules_feasible": all_feasible,
            "zero_resource_violation": zero_resource,
            "zero_precedence_violation": zero_precedence,
            "exact_generated_family_argmax_on_every_instance": exact,
            "ours_instance_pareto_nondominated_count": nondominated,
            "performance_superiority_claim_ready": bool(
                aggregate
                and aggregate[
                    "ours_strictly_pareto_dominates_every_baseline_on_every_instance"
                ]
            ),
            "claim_scope": "disjoint_post_repair_psplib_j10_parameter_cell_holdout",
            "global_mmrcpsp_optimality_claim": False,
            "stochastic_throughput_optimality_claim": False,
        },
    }


def _run_one(path: Path) -> dict[str, Any]:
    try:
        report = build_trajectory_upgrade((path,))
    except TrajectoryUpgradeError as exc:
        return {
            "instance": path.stem,
            "row": None,
            "theorem_mapping": {},
            "failure": {
                "instance": path.stem,
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "fail_closed": True,
            },
        }
    return {
        "instance": path.stem,
        "row": report["instances"][path.stem],
        "theorem_mapping": report["theorem_mapping"],
        "failure": None,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# Disjoint post-repair PSPLIB MMRCPSP holdout",
        "",
        f"- Protocol gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Repair freeze: `{report['pre_analysis_freeze']['implementation_commit']}`",
        f"- Preregistration: `{report['pre_analysis_freeze']['preregistration_commit']}`",
        f"- Completed: `{gate['completed_instance_count']}/{gate['registered_instance_count']}`.",
        f"- Ours Pareto-nondominated: `{gate['ours_instance_pareto_nondominated_count']}/{gate['registered_instance_count']}`.",
        f"- Strict all-instance baseline dominance ready: `{str(gate['performance_superiority_claim_ready']).lower()}`.",
        "- The 56 instances are disjoint from the v1 holdout and development fixture.",
        "- No per-instance tuning or holdout feedback was used.",
        "",
    ]
    if report["failures"]:
        lines.extend(["## Failures", ""])
        for failure in report["failures"]:
            lines.append(
                f"- `{failure['instance']}`: `{failure['exception_type']}` / `{failure['reason']}`"
            )
        lines.append("")
    if report["aggregate"]:
        lines.extend(
            [
                "| Policy | Geo. makespan / instance best | Geo. mean flow / instance best | Pareto instances |",
                "|---|---:|---:|---:|",
            ]
        )
        for policy, row in report["aggregate"]["policies"].items():
            lines.append(
                f"| `{policy}` | {row['geomean_makespan_to_instance_best']:.6f} | "
                f"{row['geomean_mean_flow_to_instance_best']:.6f} | "
                f"{row['pareto_frontier_count']} |"
            )
        lines.append("")
    lines.extend(
        [
            "This is prospective evidence for the registered finite holdout only.",
            "It does not establish global MMRCPSP optimality, arbitrary-solver",
            "dominance, or stochastic-arrival stability.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    write_artifact(report, output_json)
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExternalHoldoutError(f"expected JSON object: {path}")
    return payload


def _path_exists_in_commit(commit: str, relative: Path) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _commit_glob_has_mm_files(commit: str, relative_dir: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, relative_dir.as_posix()],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return any(line.endswith(".mm") for line in result.stdout.splitlines())


def _git_check(args: list[str]) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True)


def _git_output(args: list[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    report = build_external_holdout_report(
        args.suite_dir,
        args.preregistration,
        args.manifest,
        workers=args.workers,
    )
    json_path, markdown_path = write_artifacts(report, args.json, args.markdown)
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "failures": report["failures"],
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
