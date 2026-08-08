"""Run the frozen MMRCPSP trajectory policy on a preregistered PSPLIB holdout.

The policy implementation was committed before the holdout selection was
registered, and the selection was committed before any selected instance was
materialized in the repository.  This module verifies both boundaries, checks
every source byte and semantic header, and then applies the unchanged finite
trajectory policy without per-instance tuning or outcome feedback.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm
from algorithm.experiments.mmrcpsp_trajectory_upgrade import (
    BASELINE_POLICIES,
    BEAM_SEEDS,
    BEAM_WIDTH,
    CONFIGURATION_PENALTY_WEIGHT,
    LOCAL_SEED_LIMIT,
    MODE_PENALTY_WEIGHT,
    PENALTY_BOUND,
    TRAJECTORY_POLICY,
    TrajectoryUpgradeError,
    build_trajectory_upgrade,
    write_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = REPO_ROOT / "tests" / "data" / "mmrcpsp_psplib_external_holdout"
DEFAULT_PREREGISTRATION = DEFAULT_SUITE_DIR / "preregistration.json"
DEFAULT_MANIFEST = DEFAULT_SUITE_DIR / "source_manifest.json"
DEFAULT_JSON = DEFAULT_SUITE_DIR / "mmrcpsp_psplib_external_holdout_gate.json"
DEFAULT_MARKDOWN = DEFAULT_SUITE_DIR / "mmrcpsp_psplib_external_holdout_gate.md"

FROZEN_IMPLEMENTATION_COMMIT = "61b835eb17216ccd2e5e950d2aceab09e05a3f27"
PREREGISTRATION_COMMIT = "5d806861d074b9957eac9878e5f78cb70f595ad8"
PREREGISTRATION_SHA256 = "86c89fb3f7582e5a786b870b78fc5c424566eaa6cfe998a59d415fea9f71b125"
FROZEN_PROTOCOL_SHA256 = "999cc0c9ac667e3589364dc2a92ddb1f1803594d9971b0bd812eb84e1e739d11"
FROZEN_IMPLEMENTATION_SHA256 = {
    "algorithm/experiments/mmrcpsp_trajectory_upgrade.py": (
        "f0363c727cc85878bb8c14bfaab3bc76f2ae5befb1c91bf565f46e9533d21f5d"
    ),
    "algorithm/experiments/mmrcpsp_benchmark.py": (
        "d2998f8579e3030d20cff0de42d572dc1f09e8379276c49a976b7f56af63d754"
    ),
    "algorithm/experiments/mmrcpsp_instances.py": (
        "8227b8854b391b9cd763b9411a68634ed54775095f306a3e1c324d79ad5e1bff"
    ),
    "algorithm/experiments/mmrcpsp_suite_benchmark.py": (
        "14a8164151307321d88541e6f5306c3863a51bc1e54ba88a137c7a2dbc957fed"
    ),
}


class ExternalHoldoutError(ValueError):
    """Raised when the post-freeze holdout contract is not auditable."""


def frozen_protocol_snapshot() -> dict[str, Any]:
    return {
        "baseline_policies": list(BASELINE_POLICIES),
        "beam_seeds": [[name, list(weights)] for name, weights in BEAM_SEEDS],
        "beam_width": BEAM_WIDTH,
        "configuration_penalty_weight": CONFIGURATION_PENALTY_WEIGHT,
        "local_seed_limit": LOCAL_SEED_LIMIT,
        "mode_penalty_weight": MODE_PENALTY_WEIGHT,
        "penalty_bound": PENALTY_BOUND,
    }


def frozen_protocol_sha256() -> str:
    payload = json.dumps(
        frozen_protocol_snapshot(), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return sha256(payload).hexdigest()


def verify_pre_analysis_freeze(
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    preregistration = Path(preregistration_path)
    relative_preregistration = preregistration.resolve().relative_to(REPO_ROOT.resolve())
    _git_check(["cat-file", "-e", f"{FROZEN_IMPLEMENTATION_COMMIT}^{{commit}}"])
    _git_check(["cat-file", "-e", f"{PREREGISTRATION_COMMIT}^{{commit}}"])
    if frozen_protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise ExternalHoldoutError("current MMRCPSP protocol configuration drifted")

    implementation_files = []
    for relative, expected in FROZEN_IMPLEMENTATION_SHA256.items():
        current = sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        if current != expected:
            raise ExternalHoldoutError(f"post-freeze implementation drift: {relative}")
        committed = sha256(
            _git_output(["show", f"{FROZEN_IMPLEMENTATION_COMMIT}:{relative}"])
        ).hexdigest()
        if committed != expected:
            raise ExternalHoldoutError(f"freeze commit blob mismatch: {relative}")
        implementation_files.append({"path": relative, "sha256": current})

    preregistration_bytes = preregistration.read_bytes()
    current_preregistration_sha = sha256(preregistration_bytes).hexdigest()
    if current_preregistration_sha != PREREGISTRATION_SHA256:
        raise ExternalHoldoutError("preregistration bytes drifted")
    committed_preregistration_sha = sha256(
        _git_output(
            ["show", f"{PREREGISTRATION_COMMIT}:{relative_preregistration.as_posix()}"]
        )
    ).hexdigest()
    if committed_preregistration_sha != PREREGISTRATION_SHA256:
        raise ExternalHoldoutError("preregistration commit blob mismatch")

    source_manifest_relative = relative_preregistration.parent / "source_manifest.json"
    if _path_exists_in_commit(FROZEN_IMPLEMENTATION_COMMIT, relative_preregistration):
        raise ExternalHoldoutError("holdout preregistration predates the policy freeze")
    if _path_exists_in_commit(PREREGISTRATION_COMMIT, source_manifest_relative):
        raise ExternalHoldoutError("source manifest was present at preregistration")
    if _commit_glob_has_mm_files(PREREGISTRATION_COMMIT, relative_preregistration.parent):
        raise ExternalHoldoutError("holdout instance bytes were present at preregistration")

    return {
        "pass": True,
        "implementation_commit": FROZEN_IMPLEMENTATION_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "protocol_config_sha256": FROZEN_PROTOCOL_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "implementation_files": implementation_files,
        "selection_registered_after_policy_freeze": True,
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
        "scheduleurm.mmrcpsp_external_holdout.preregistration.v1"
    ):
        raise ExternalHoldoutError("unexpected preregistration schema")
    if manifest.get("schema_version") != (
        "scheduleurm.mmrcpsp_external_holdout.source.v1"
    ):
        raise ExternalHoldoutError("unexpected source manifest schema")
    if manifest.get("suite") != preregistration.get("suite"):
        raise ExternalHoldoutError("suite name differs from preregistration")
    if manifest.get("preregistration_sha256") != sha256(
        preregistration_file.read_bytes()
    ).hexdigest():
        raise ExternalHoldoutError("source manifest does not bind preregistration")
    if manifest.get("source") != preregistration.get("source"):
        raise ExternalHoldoutError("source archive contract differs from preregistration")
    if manifest.get("pre_analysis_freeze") != preregistration.get(
        "pre_analysis_freeze"
    ):
        raise ExternalHoldoutError("freeze contract differs from preregistration")

    selected = tuple(preregistration["selection"]["selected_members"])
    if len(selected) != 56 or len(set(selected)) != len(selected):
        raise ExternalHoldoutError("preregistered holdout must contain 56 unique members")
    rows = manifest.get("instances")
    if not isinstance(rows, list):
        raise ExternalHoldoutError("source manifest instances must be a list")
    registered = tuple(str(row.get("file")) for row in rows)
    if registered != selected:
        raise ExternalHoldoutError("source manifest differs from preregistered selection")
    actual = tuple(sorted(path.name for path in suite.glob("*.mm")))
    if actual != tuple(sorted(selected)):
        raise ExternalHoldoutError("holdout file set differs from preregistration")

    verified = []
    for row in rows:
        file_name = str(row["file"])
        if Path(file_name).name != file_name or not file_name.endswith(".mm"):
            raise ExternalHoldoutError(f"unsafe source path: {file_name}")
        path = suite / file_name
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != str(row["sha256"]):
            raise ExternalHoldoutError(f"source hash mismatch for {file_name}")
        if len(raw) != int(row["size_bytes"]):
            raise ExternalHoldoutError(f"source size mismatch for {file_name}")
        instance = parse_psplib_mm(path)
        expected_semantics = {
            "jobs_including_dummies": int(row["jobs_including_dummies"]),
            "horizon": int(row["horizon"]),
            "renewable_capacities": list(row["renewable_capacities"]),
            "nonrenewable_capacities": list(row["nonrenewable_capacities"]),
        }
        actual_semantics = {
            "jobs_including_dummies": instance.declared_job_count,
            "horizon": instance.horizon,
            "renewable_capacities": list(instance.renewable_capacities),
            "nonrenewable_capacities": list(instance.nonrenewable_capacities),
        }
        if actual_semantics != expected_semantics:
            raise ExternalHoldoutError(f"semantic metadata mismatch for {file_name}")
        verified.append({**row, "path": path})
    return {
        "pass": True,
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "verified_instances": verified,
    }


def build_external_holdout_report(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    freeze = verify_pre_analysis_freeze(preregistration_path)
    source = validate_external_source(suite_dir, preregistration_path, manifest_path)
    config_before = frozen_protocol_sha256()
    rows: dict[str, Any] = {}
    failure: dict[str, Any] | None = None
    theorem_mapping: Mapping[str, Any] = {}
    for source_row in source["verified_instances"]:
        path = Path(source_row["path"])
        try:
            instance_report = build_trajectory_upgrade((path,))
        except TrajectoryUpgradeError as exc:
            failure = {
                "instance": path.stem,
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "fail_closed": True,
            }
            break
        rows.update(instance_report["instances"])
        theorem_mapping = instance_report["theorem_mapping"]
    config_after = frozen_protocol_sha256()
    if config_before != config_after:
        raise ExternalHoldoutError("frozen protocol changed during holdout execution")

    completed_all = failure is None and len(rows) == len(source["verified_instances"])
    all_feasible = completed_all and all(
        bool(row["gate"]["all_evaluated_schedules_feasible"])
        for row in rows.values()
    )
    exact = completed_all and all(
        bool(row["oracle_audit"]["exact_over_candidate_family"])
        for row in rows.values()
    )
    zero_resource = completed_all and all(
        bool(row["gate"]["zero_resource_violation"])
        for row in rows.values()
    )
    zero_precedence = completed_all and all(
        bool(row["gate"]["zero_precedence_violation"])
        for row in rows.values()
    )
    nondominated = sum(row["pareto"]["selected_is_nondominated"] for row in rows.values())
    protocol_pass = bool(
        freeze["pass"]
        and source["pass"]
        and all_feasible
        and exact
        and zero_resource
        and zero_precedence
    )
    aggregate = _aggregate(rows) if rows else _empty_aggregate()
    return {
        "schema_version": "scheduleurm.mmrcpsp_external_holdout.v1",
        "suite": "PSPLIB j10 multi-mode parameter-cell holdout",
        "deterministic_algorithm": True,
        "source_manifest_sha256": source["manifest_sha256"],
        "pre_analysis_freeze": freeze,
        "frozen_protocol": {
            "config": frozen_protocol_snapshot(),
            "config_sha256_before": config_before,
            "config_sha256_after": config_after,
            "per_instance_tuning": False,
            "holdout_feedback_used_for_configuration": False,
            "instance_bytes_absent_at_preregistration": True,
        },
        "instances": rows,
        "first_failure": failure,
        "aggregate": aggregate,
        "theorem_mapping": theorem_mapping,
        "gate": {
            "pass": protocol_pass,
            "prospective_external_holdout_ready": protocol_pass,
            "instance_count": len(rows),
            "registered_instance_count": len(source["verified_instances"]),
            "completed_all_registered_instances": completed_all,
            "all_schedules_feasible": all_feasible,
            "zero_resource_violation": zero_resource,
            "zero_precedence_violation": zero_precedence,
            "exact_generated_family_argmax_on_every_instance": exact,
            "ours_instance_pareto_nondominated_count": nondominated,
            "performance_superiority_claim_ready": bool(
                aggregate[
                    "ours_strictly_pareto_dominates_every_baseline_on_every_instance"
                ]
            ),
            "claim_scope": "post_freeze_psplib_j10_parameter_cell_holdout",
            "global_mmrcpsp_optimality_claim": False,
            "stochastic_throughput_optimality_claim": False,
        },
    }


def _aggregate(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    policies = (TRAJECTORY_POLICY, *BASELINE_POLICIES)
    metrics: dict[str, dict[str, list[float]]] = {
        policy: {"makespan": [], "mean_flow": []} for policy in policies
    }
    frontier_counts = {policy: 0 for policy in policies}
    strict_all_count = 0
    for row in rows.values():
        selected_metrics = row["selected_candidate"]["metrics"]
        per_policy = {TRAJECTORY_POLICY: selected_metrics}
        per_policy.update(
            {
                policy: result["metrics"]
                for policy, result in row["baseline_results"].items()
            }
        )
        best_makespan = min(float(item["makespan"]) for item in per_policy.values())
        best_flow = min(float(item["mean_flow_time"]) for item in per_policy.values())
        for policy, item in per_policy.items():
            metrics[policy]["makespan"].append(float(item["makespan"]) / best_makespan)
            metrics[policy]["mean_flow"].append(float(item["mean_flow_time"]) / best_flow)
        for policy in row["pareto"]["frontier"]:
            frontier_counts[policy] += 1
        strict_all_count += int(row["pareto"]["selected_strictly_dominates_every_baseline"])
    policy_rows = {}
    for policy in policies:
        policy_rows[policy] = {
            "geomean_makespan_to_instance_best": _geomean(metrics[policy]["makespan"]),
            "geomean_mean_flow_to_instance_best": _geomean(metrics[policy]["mean_flow"]),
            "pareto_frontier_count": frontier_counts[policy],
        }
    return {
        "policies": policy_rows,
        "ours_strictly_dominates_every_baseline_instance_count": strict_all_count,
        "ours_strictly_pareto_dominates_every_baseline_on_every_instance": (
            strict_all_count == len(rows)
        ),
    }


def _empty_aggregate() -> dict[str, Any]:
    return {
        "policies": {
            policy: {
                "geomean_makespan_to_instance_best": None,
                "geomean_mean_flow_to_instance_best": None,
                "pareto_frontier_count": 0,
            }
            for policy in (TRAJECTORY_POLICY, *BASELINE_POLICIES)
        },
        "ours_strictly_dominates_every_baseline_instance_count": 0,
        "ours_strictly_pareto_dominates_every_baseline_on_every_instance": False,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    aggregate = report["aggregate"]
    lines = [
        "# Post-freeze PSPLIB MMRCPSP holdout",
        "",
        f"- Protocol gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Frozen policy commit: `{report['pre_analysis_freeze']['implementation_commit']}`",
        f"- Preregistration commit: `{report['pre_analysis_freeze']['preregistration_commit']}`",
        f"- Registered official instances: `{gate['registered_instance_count']}`",
        "- The instance selection was outcome-blind and committed before materialization.",
        "- No per-instance tuning or holdout feedback was used.",
        f"- Completed before fail-closed stop: `{gate['instance_count']}/{gate['registered_instance_count']}`.",
        f"- Ours Pareto-nondominated among completed rows: `{gate['ours_instance_pareto_nondominated_count']}/{gate['instance_count']}`.",
        f"- Strict all-instance baseline dominance ready: `{str(gate['performance_superiority_claim_ready']).lower()}`.",
    ]
    if report["first_failure"] is not None:
        failure = report["first_failure"]
        lines.extend(
            [
                f"- First failure: `{failure['instance']}` / `{failure['exception_type']}`.",
                f"- Failure reason: `{failure['reason']}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Policy | Geo. makespan / instance best | Geo. mean flow / instance best | Pareto instances |",
            "|---|---:|---:|---:|",
        ]
    )
    for policy, row in aggregate["policies"].items():
        lines.append(
            f"| `{policy}` | {_format_optional(row['geomean_makespan_to_instance_best'])} | "
            f"{_format_optional(row['geomean_mean_flow_to_instance_best'])} | "
            f"{row['pareto_frontier_count']} |"
        )
    lines.extend(
        [
            "",
            "This gate establishes prospective behavior for the preregistered finite",
            "PSPLIB holdout. It does not claim global MMRCPSP optimality, dominance",
            "over arbitrary solvers or instances, or stochastic-arrival stability.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


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


def _geomean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ExternalHoldoutError("geometric mean requires positive finite values")
    return round(math.exp(sum(math.log(value) for value in values) / len(values)), 9)


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
    args = parser.parse_args()
    report = build_external_holdout_report(
        args.suite_dir, args.preregistration, args.manifest
    )
    json_path, markdown_path = write_artifacts(report, args.json, args.markdown)
    print(
        json.dumps(
            {"gate": report["gate"], "json": str(json_path), "markdown": str(markdown_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
