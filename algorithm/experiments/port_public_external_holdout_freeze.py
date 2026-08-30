"""Freeze a prospective BACASP-S R83/R84 public holdout.

The R89/R90 evaluation exposed an adapter error before all registered rows
could be evaluated.  Those outcomes remain immutable.  This module freezes a
disjoint replication pair after the adapter correction and before any policy
is run on the new inputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_public_external_holdout import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_DEADLINE,
    EXPECTED_Q,
    EXPECTED_REPOSITORY_COMMIT,
    EXPECTED_SPEED_SETUP,
    _digest,
)
from algorithm.experiments.port_scheduling_benchmark import BASELINE_POLICIES
from algorithm.experiments.port_trajectory_upgrade import (
    GLOBAL_TRAJECTORY_CONFIG,
    SOURCE_COST_METRICS,
    TrajectoryPlan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_REPLICATIONS = (83, 84)
REGISTERED_ON = "2026-08-30"
DEVELOPMENT_COMPACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_trajectory_upgrade_v2_compact_20260809.json"
)
DEVELOPMENT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_trajectory_upgrade_v2_full_20260809.json.gz"
)
PRIOR_MANIFESTS = (
    REPO_ROOT / "tests/data/port_bacasp_s_factor_holdout_r85_r86/source_manifest.json",
    REPO_ROOT / "tests/data/port_bacasp_s_external_holdout_r87_r88/source_manifest.json",
    REPO_ROOT / "tests/data/port_bacasp_s_external_holdout_r89_r90/source_manifest.json",
)
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/port_public_benchmark_adapter.py",
    "algorithm/experiments/port_public_external_holdout.py",
    "algorithm/experiments/port_public_external_holdout_freeze.py",
    "algorithm/experiments/port_scheduling_benchmark.py",
    "algorithm/experiments/port_scheduling_instances.py",
    "algorithm/experiments/port_trajectory_upgrade.py",
)
SOURCE_PATTERN = re.compile(
    r".*_(?P<rep>83|84)_Q(?P<q>5|10|15)_"
    r"S(?P<speed>160-0\.2|200-0\.15|240-0\.1)_"
    r"D(?P<deadline>1|1\.5|2)\.dat$"
)


def freeze_suite(*, source_repo: str | Path, output_root: str | Path) -> dict[str, Any]:
    source = Path(source_repo).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty holdout root: {output}")
    try:
        output_relative = output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("holdout output must be inside the Scheduleurm repository") from exc

    upstream_head = _git_output(source, ["rev-parse", "HEAD"]).decode().strip()
    if upstream_head != EXPECTED_REPOSITORY_COMMIT:
        raise ValueError(f"unexpected BACASP-S source commit: {upstream_head}")

    freeze_commit = _git_output(REPO_ROOT, ["rev-parse", "HEAD"]).decode().strip()
    implementation = _frozen_implementation(freeze_commit)
    development = _load_json(DEVELOPMENT_COMPACT)
    plan = TrajectoryPlan(tuple(development["selected_global_plan"]["policies"]))
    if plan.plan_id != development.get("selected_global_plan_id"):
        raise ValueError("development plan identity is inconsistent")
    if development.get("global_search_config_sha256") != GLOBAL_TRAJECTORY_CONFIG.digest:
        raise ValueError("development search configuration changed")

    selected = _select_sources(source / "instances" / "LargeMB")
    prior_paths, prior_hashes = _prior_source_identity()
    if any(row["upstream_path"] in prior_paths for row in selected):
        raise ValueError("prospective source path overlaps an earlier port suite")
    if any(row["sha256"] in prior_hashes for row in selected):
        raise ValueError("prospective source bytes overlap an earlier port suite")

    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for row in selected:
        source_path = row.pop("_source")
        shutil.copyfile(source_path, output / source_path.name)
        row["local_path"] = (output_relative / source_path.name).as_posix()
        manifest_rows.append(row)
    manifest = {
        "schema_version": "scheduleurm.port_bacasp_s_suite_manifest.v1",
        "dataset_name": "BACASP-S LargeMB",
        "repository_url": "https://github.com/juanfdata/bacasp_setup",
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "instances": manifest_rows,
    }
    manifest["manifest_content_sha256_excluding_self"] = _digest(manifest)
    manifest_path = output / "source_manifest.json"
    _write_json(manifest_path, manifest)

    compact_relative = DEVELOPMENT_COMPACT.relative_to(REPO_ROOT).as_posix()
    full_relative = DEVELOPMENT_FULL.relative_to(REPO_ROOT).as_posix()
    output_stem = "port_bacasp_s_external_holdout_r83_r84_20260830"
    preregistration: dict[str, Any] = {
        "schema_version": "scheduleurm.port_bacasp_s_preregistration.v1",
        "registered_on": REGISTERED_ON,
        "protocol": "post_freeze_disjoint_large_mb_r83_r84_after_adapter_correction",
        "upstream_repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "suite_manifest_path": (output_relative / "source_manifest.json").as_posix(),
        "suite_manifest_file_sha256": _file_sha256(manifest_path),
        "data_freeze_commit": freeze_commit,
        "implementation_freeze_commit": freeze_commit,
        "holdout_outcomes_observed_before_freeze": False,
        "split_rule": {
            "external_holdout_replications": list(EXPECTED_REPLICATIONS),
            "q_max_factors": list(EXPECTED_Q),
            "speed_setup_factors": list(EXPECTED_SPEED_SETUP),
            "deadline_factors": list(EXPECTED_DEADLINE),
            "registered_instance_count": 54,
            "all_registered_rows_remain_in_denominator": True,
        },
        "frozen_plan": plan.snapshot(),
        "frozen_search_config_sha256": GLOBAL_TRAJECTORY_CONFIG.digest,
        "development_candidate_family_size": int(development["trajectory_family_size"]),
        "development_compact_artifact": compact_relative,
        "development_compact_file_sha256": _file_sha256(DEVELOPMENT_COMPACT),
        "development_full_artifact": full_relative,
        "development_full_gzip_sha256": _file_sha256(DEVELOPMENT_FULL),
        "implementation_file_sha256": implementation,
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
        "bootstrap": {
            "method": "factor-cell cluster bootstrap; both R83/R84 rows retained per sampled cell",
            "confidence_level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "execution_budget": {
            "deployment_policy": "one frozen periodic trajectory",
            "holdout_rollouts_per_policy_per_instance": 1,
            "ours_holdout_candidate_search": False,
        },
        "failure_policy": {
            "parse_error": "retain as failure",
            "infeasible_or_deadlocked": "retain as failure",
            "baseline_dominance": "retain in performance denominator",
            "no_post_holdout_tuning": True,
        },
        "primary_protocol_gate": "all 54 rows authentic, parsed, feasible, and retained",
        "performance_diagnostic": "selected frozen plan Pareto-nondominance count and paired ratios",
        "score_boundary": {
            "development_outer_selector": "normalized terminal-cost virtual service minus bounded penalty",
            "low_level": "statewise Q^T physical lower_service minus action penalty",
            "holdout_outcomes_used_for_selection": False,
        },
        "claim_boundary": [
            "fixed-slot source-core projection only",
            "no BACASP-S mid-service migration",
            "no author-algorithm or unrestricted-optimum comparison",
            "no physical-port or stochastic-stability claim",
        ],
        "outputs": {
            "full_artifact": f"md/experiment_artifacts/{output_stem}_full.json.gz",
            "compact_artifact": f"md/experiment_artifacts/{output_stem}.json",
            "markdown": f"md/{output_stem}.md",
        },
    }
    preregistration["preregistration_sha256_excluding_self"] = _digest(preregistration)
    preregistration_path = output / "preregistration.json"
    _write_json(preregistration_path, preregistration)
    return {
        "instance_count": len(manifest_rows),
        "factor_cell_count": 27,
        "replications": list(EXPECTED_REPLICATIONS),
        "manifest_sha256": _file_sha256(manifest_path),
        "preregistration_sha256": _file_sha256(preregistration_path),
        "freeze_commit": freeze_commit,
    }


def _select_sources(root: Path) -> list[dict[str, Any]]:
    rows = []
    cell_replicates: dict[tuple[int, str, str], set[int]] = {}
    for path in sorted(root.glob("*.dat")):
        match = SOURCE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        raw = path.read_bytes()
        replication = int(match.group("rep"))
        q_max = int(match.group("q"))
        speed = match.group("speed")
        deadline = match.group("deadline")
        cell_replicates.setdefault((q_max, speed, deadline), set()).add(replication)
        rows.append(
            {
                "local_path": "",
                "upstream_path": path.relative_to(root.parents[1]).as_posix(),
                "sha256": sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "line_count": len(raw.splitlines()),
                "raw_bytes_modified": False,
                "split": "external_holdout",
                "replication": replication,
                "q_max_factor": q_max,
                "speed_setup_factor": speed,
                "deadline_factor": deadline,
                "_source": path,
            }
        )
    expected_cells = {
        (q, speed, deadline)
        for q in EXPECTED_Q
        for speed in EXPECTED_SPEED_SETUP
        for deadline in EXPECTED_DEADLINE
    }
    if len(rows) != 54 or set(cell_replicates) != expected_cells:
        raise ValueError("R83/R84 source suite must contain the complete 54-row factor grid")
    if any(values != set(EXPECTED_REPLICATIONS) for values in cell_replicates.values()):
        raise ValueError("each R83/R84 factor cell must contain both replications")
    if len({row["sha256"] for row in rows}) != 54:
        raise ValueError("R83/R84 source suite contains duplicate instance bytes")
    return rows


def _frozen_implementation(commit: str) -> dict[str, str]:
    rows = {}
    for relative in IMPLEMENTATION_PATHS:
        current = _file_sha256(REPO_ROOT / relative)
        committed = sha256(_git_output(REPO_ROOT, ["show", f"{commit}:{relative}"])).hexdigest()
        if current != committed:
            raise ValueError(f"implementation file is not frozen at HEAD: {relative}")
        rows[relative] = current
    return rows


def _prior_source_identity() -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for manifest_path in PRIOR_MANIFESTS:
        manifest = _load_json(manifest_path)
        for row in manifest.get("instances") or ():
            paths.add(str(row["upstream_path"]))
            hashes.add(str(row["sha256"]))
    return paths, hashes


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _git_output(repository: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True
    ).stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = freeze_suite(source_repo=args.source_repo, output_root=args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
