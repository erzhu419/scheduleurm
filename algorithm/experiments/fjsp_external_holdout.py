"""Run the repository-frozen FJSP trajectory policy on an external suite.

The selection implementation and configuration were committed before the
Kacem suite was copied into this repository.  This runner verifies that the
current selection files still match that commit, validates every source byte,
and then applies the frozen policy without tuning or feedback.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from algorithm.experiments.fjsp_instances import load_fjsp_instance
from algorithm.experiments.fjsp_trajectory_upgrade import (
    COMPARISON_ORDER,
    FROZEN_CONFIG,
    FROZEN_CONFIG_SHA256,
    TRAJECTORY_POLICY,
    _aggregate,
    _run_suite_instance,
    _theorem_mapping,
    frozen_config_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = REPO_ROOT / "tests" / "data" / "fjsp_kacem_external_holdout"
DEFAULT_MANIFEST = DEFAULT_SUITE_DIR / "source_manifest.json"
DEFAULT_JSON = DEFAULT_SUITE_DIR / "fjsp_kacem_external_holdout_gate.json"
DEFAULT_MARKDOWN = DEFAULT_SUITE_DIR / "fjsp_kacem_external_holdout_gate.md"
EXPECTED_INSTANCE_IDS = ("kacem1", "kacem2", "kacem3", "kacem4")
FROZEN_IMPLEMENTATION_COMMIT = "9a922be91161829b714e7c1c11b69d66e799f203"
FROZEN_IMPLEMENTATION_SHA256 = {
    "algorithm/experiments/fjsp_trajectory_upgrade.py": "c121539db26180b9d4c1129e1b7535f138d84201594f1c08478ac060ddc2d0d9",
    "algorithm/experiments/fjsp_benchmark.py": "dc00ff2ae33f4c9d1267952a9237318563785d14ac08f85eaa36b08622bce65e",
    "algorithm/experiments/fjsp_instances.py": "8542cc333fcee45c139dc8469f95551cd5967402d1b4d2106546a9abd4e9d2b1",
}


class ExternalHoldoutError(ValueError):
    """Raised when source or pre-analysis-freeze evidence is invalid."""


def validate_external_source(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    suite = Path(suite_dir)
    manifest_file = Path(manifest_path)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "scheduleurm.fjsp_external_holdout.source.v1":
        raise ExternalHoldoutError("unexpected external holdout manifest schema")
    if manifest.get("suite") != "Kacem-Hammadi-Borne 2002":
        raise ExternalHoldoutError("unexpected external holdout suite")
    freeze = manifest.get("pre_analysis_freeze", {})
    if freeze.get("repository_commit") != FROZEN_IMPLEMENTATION_COMMIT:
        raise ExternalHoldoutError("manifest does not bind the frozen implementation commit")
    if freeze.get("config_sha256") != FROZEN_CONFIG_SHA256:
        raise ExternalHoldoutError("manifest does not bind the frozen trajectory config")

    raw_rows = manifest.get("instances")
    if not isinstance(raw_rows, list):
        raise ExternalHoldoutError("manifest instances must be a list")
    ids = tuple(str(row.get("id")) for row in raw_rows)
    if ids != EXPECTED_INSTANCE_IDS or len(set(ids)) != len(ids):
        raise ExternalHoldoutError("external holdout must contain Kacem1-Kacem4 exactly")

    verified = []
    for row in raw_rows:
        source = suite / str(row["file"])
        raw_sha = sha256(source.read_bytes()).hexdigest()
        if raw_sha != str(row["sha256"]):
            raise ExternalHoldoutError(f"source hash mismatch for {row['id']}")
        instance = load_fjsp_instance(source)
        expected_shape = (
            int(row["job_count"]),
            int(row["machine_count"]),
            int(row["operation_count"]),
        )
        actual_shape = (instance.job_count, instance.machine_count, instance.operation_count)
        if actual_shape != expected_shape:
            raise ExternalHoldoutError(
                f"shape mismatch for {row['id']}: expected {expected_shape}, got {actual_shape}"
            )
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        if lower <= 0 or upper < lower or str(row["status"]) != "closed":
            raise ExternalHoldoutError(f"invalid bound metadata for {row['id']}")
        verified.append(
            {
                **row,
                "raw_sha256": raw_sha,
                "instance": instance,
            }
        )
    return {
        "pass": True,
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "source_repository_commit": str(manifest["source"]["commit"]),
        "verified_instances": verified,
    }


def build_external_holdout_report(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    freeze_audit = verify_pre_analysis_freeze()
    source_gate = validate_external_source(suite_dir, manifest_path)
    config_hash_before = frozen_config_sha256()
    rows = [
        _run_suite_instance(row, "post_freeze_external_holdout")
        for row in source_gate["verified_instances"]
    ]
    config_hash_after = frozen_config_sha256()
    if config_hash_before != config_hash_after:
        raise ExternalHoldoutError("frozen configuration changed during holdout execution")
    aggregate = _aggregate(rows)
    all_feasible = all(
        bool(result["feasible"])
        for row in rows
        for result in row["policies"].values()
    )
    exact = all(bool(row["trajectory_search"]["exact_family_argmax"]) for row in rows)
    zero_gap = all(
        abs(float(row["trajectory_search"]["oracle_gap_within_generated_family"])) <= 1e-12
        for row in rows
    )
    nondominated = sum(TRAJECTORY_POLICY in row["pareto_frontier_policies"] for row in rows)
    protocol_pass = bool(
        freeze_audit["pass"]
        and source_gate["pass"]
        and all_feasible
        and exact
        and zero_gap
        and nondominated == len(EXPECTED_INSTANCE_IDS)
    )
    return {
        "schema_version": "scheduleurm.fjsp_external_holdout.v1",
        "suite": "Kacem-Hammadi-Borne 2002",
        "deterministic_algorithm": True,
        "source_manifest_sha256": source_gate["manifest_sha256"],
        "source_repository_commit": source_gate["source_repository_commit"],
        "pre_analysis_freeze": freeze_audit,
        "frozen_protocol": {
            "config": FROZEN_CONFIG.snapshot(),
            "config_sha256_before": config_hash_before,
            "config_sha256_after": config_hash_after,
            "per_instance_tuning": False,
            "holdout_feedback_used_for_configuration": False,
            "external_suite_absent_from_freeze_commit": True,
        },
        "theorem_mapping": _theorem_mapping(),
        "instances": rows,
        "aggregate": aggregate,
        "gate": {
            "pass": protocol_pass,
            "prospective_external_holdout_ready": protocol_pass,
            "instance_count": len(rows),
            "all_schedules_feasible": all_feasible,
            "exact_generated_family_argmax_on_every_instance": exact,
            "zero_generated_family_oracle_gap_on_every_instance": zero_gap,
            "ours_instance_pareto_nondominated_count": nondominated,
            "performance_superiority_claim_ready": bool(
                aggregate["ours_strictly_pareto_dominates_every_classic_on_every_instance"]
            ),
            "claim_scope": "post_freeze_kacem_finite_trajectory_family",
            "global_fjsp_optimality_claim": False,
            "stochastic_throughput_optimality_claim": False,
        },
    }


def verify_pre_analysis_freeze() -> dict[str, Any]:
    _git_check(["cat-file", "-e", f"{FROZEN_IMPLEMENTATION_COMMIT}^{{commit}}"])
    if frozen_config_sha256() != FROZEN_CONFIG_SHA256:
        raise ExternalHoldoutError("current frozen config hash is invalid")
    files = []
    for relative, expected in FROZEN_IMPLEMENTATION_SHA256.items():
        actual = sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ExternalHoldoutError(f"post-freeze implementation drift: {relative}")
        committed = _git_output(["show", f"{FROZEN_IMPLEMENTATION_COMMIT}:{relative}"])
        committed_sha = sha256(committed).hexdigest()
        if committed_sha != expected:
            raise ExternalHoldoutError(f"freeze commit blob mismatch: {relative}")
        files.append({"path": relative, "sha256": actual})
    external_path = "tests/data/fjsp_kacem_external_holdout/source_manifest.json"
    absent = subprocess.run(
        ["git", "cat-file", "-e", f"{FROZEN_IMPLEMENTATION_COMMIT}:{external_path}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0
    if not absent:
        raise ExternalHoldoutError("external holdout was already present in the freeze commit")
    return {
        "pass": True,
        "repository_commit": FROZEN_IMPLEMENTATION_COMMIT,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "implementation_files": files,
        "external_suite_absent_from_freeze_commit": True,
        "policy_or_config_updates_after_holdout": False,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = report["gate"]
    aggregate = report["aggregate"]
    lines = [
        "# Post-freeze external FJSP holdout",
        "",
        f"- Protocol gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
        f"- Frozen repository commit: `{report['pre_analysis_freeze']['repository_commit']}`",
        f"- Frozen config: `{report['pre_analysis_freeze']['config_sha256']}`",
        "- External suite: complete Kacem1-Kacem4 collection, absent from the freeze commit.",
        "- No per-instance tuning or holdout feedback was used.",
        f"- Ours Pareto-nondominated: `{gate['ours_instance_pareto_nondominated_count']}/{gate['instance_count']}`.",
        f"- Strict all-instance classic-policy dominance ready: `{str(gate['performance_superiority_claim_ready']).lower()}`.",
        "",
        "| Policy | Geo. makespan/BKS | Geo. flow/best | Pareto instances |",
        "|---|---:|---:|---:|",
    ]
    for policy in COMPARISON_ORDER:
        row = aggregate["policies"][policy]
        lines.append(
            f"| {row['policy_label']} | {float(row['geomean_makespan_over_bks']):.6f} | "
            f"{float(row['geomean_mean_flow_to_instance_best']):.6f} | {row['pareto_frontier_count']} |"
        )
    lines.extend(
        [
            "",
            "| Instance | Ours makespan | Ours mean flow | BKS | Pareto frontier |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in report["instances"]:
        ours = row["policies"][TRAJECTORY_POLICY]
        lines.append(
            f"| {row['instance_id']} | {ours['makespan']} | {ours['mean_flow_time']} | "
            f"{row['bks']} | {', '.join(row['pareto_frontier_policies'])} |"
        )
    lines.extend(
        [
            "",
            "The gate establishes prospective post-freeze behavior for this external finite",
            "suite. It does not claim global FJSP optimality, arbitrary-instance dominance,",
            "or stochastic-arrival throughput optimality.",
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
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def _git_check(args: list[str]) -> None:
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True)


def _git_output(args: list[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_external_holdout_report(args.suite_dir, args.manifest)
    json_path, markdown_path = write_artifacts(report, args.json, args.markdown)
    print(json.dumps({"gate": report["gate"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))


if __name__ == "__main__":
    main()
