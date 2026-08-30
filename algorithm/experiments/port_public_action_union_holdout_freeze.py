"""Freeze the prospective BACASP-S R81/R82 action-union holdout."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_public_action_union import (
    OUTER_SCORE_CONFIG,
    frozen_action_family,
)
from algorithm.experiments.port_public_action_union_holdout import (
    EXPECTED_REPLICATIONS,
)
from algorithm.experiments.port_public_external_holdout import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_DEADLINE,
    EXPECTED_Q,
    EXPECTED_REPOSITORY_COMMIT,
    EXPECTED_SPEED_SETUP,
    REPO_ROOT,
    SOURCE_COST_METRICS,
    _digest,
)
from algorithm.experiments.port_scheduling_benchmark import BASELINE_POLICIES


REGISTERED_ON = "2026-08-30"
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/port_public_action_union.py",
    "algorithm/experiments/port_public_action_union_holdout.py",
    "algorithm/experiments/port_public_action_union_holdout_freeze.py",
    "algorithm/experiments/port_public_benchmark_adapter.py",
    "algorithm/experiments/port_public_external_holdout.py",
    "algorithm/experiments/port_scheduling_benchmark.py",
    "algorithm/experiments/port_scheduling_instances.py",
    "algorithm/experiments/port_trajectory_upgrade.py",
)
SOURCE_PATTERN = re.compile(
    r".*_(?P<rep>81|82)_Q(?P<q>5|10|15)_"
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
        raise ValueError("holdout output must be inside the repository") from exc
    upstream_head = _git_output(source, ["rev-parse", "HEAD"]).decode().strip()
    if upstream_head != EXPECTED_REPOSITORY_COMMIT:
        raise ValueError(f"unexpected BACASP-S source commit: {upstream_head}")

    freeze_commit = _git_output(REPO_ROOT, ["rev-parse", "HEAD"]).decode().strip()
    implementation = _frozen_implementation(freeze_commit)
    selected = _select_sources(source / "instances/LargeMB")
    prior_paths, prior_hashes = _prior_source_identity(output)
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

    family = [plan.snapshot() for plan in frozen_action_family()]
    preregistration: dict[str, Any] = {
        "schema_version": "scheduleurm.port_action_union_preregistration.v1",
        "registered_on": REGISTERED_ON,
        "protocol": "post_freeze_disjoint_large_mb_r81_r82_action_union",
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
        "frozen_action_family": family,
        "outer_score_config": OUTER_SCORE_CONFIG.snapshot(),
        "implementation_file_sha256": implementation,
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
        "bootstrap": {
            "method": "factor-cell cluster bootstrap with both replications per sampled cell",
            "confidence_level": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "execution_budget": {
            "candidate_count": len(family),
            "rollouts_per_candidate_per_instance": 1,
            "per_instance_tuning": False,
            "candidate_family_revision_after_holdout": False,
        },
        "failure_policy": {
            "parse_error": "retain as failure",
            "infeasible_or_deadlocked": "retain as failure",
            "baseline_dominance": "retain in performance denominator",
            "no_post_holdout_tuning": True,
        },
        "claim_boundary": [
            "exact only over the registered deterministic action family",
            "no unrestricted BACASP-S optimum or author-algorithm comparison",
            "no physical-port or stochastic-stability claim",
        ],
        "outputs": {
            "full_artifact": "md/experiment_artifacts/port_action_union_r81_r82_20260830_full.json.gz",
            "compact_artifact": "md/experiment_artifacts/port_action_union_r81_r82_20260830.json",
            "markdown": "md/port_action_union_r81_r82_20260830.md",
        },
    }
    preregistration["preregistration_sha256_excluding_self"] = _digest(preregistration)
    preregistration_path = output / "preregistration.json"
    _write_json(preregistration_path, preregistration)
    return {
        "instance_count": len(manifest_rows),
        "factor_cell_count": 27,
        "replications": list(EXPECTED_REPLICATIONS),
        "action_family_size": len(family),
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
        raise ValueError("R81/R82 source suite must contain the complete factor grid")
    if any(values != set(EXPECTED_REPLICATIONS) for values in cell_replicates.values()):
        raise ValueError("each factor cell must contain both replications")
    if len({row["sha256"] for row in rows}) != 54:
        raise ValueError("R81/R82 source suite contains duplicate bytes")
    return rows


def _prior_source_identity(output: Path) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for manifest_path in sorted((REPO_ROOT / "tests/data").glob("port_bacasp_s_*/source_manifest.json")):
        if output in manifest_path.parents:
            continue
        manifest = _load_json(manifest_path)
        for row in manifest.get("instances") or ():
            paths.add(str(row["upstream_path"]))
            hashes.add(str(row["sha256"]))
    return paths, hashes


def _frozen_implementation(commit: str) -> dict[str, str]:
    rows = {}
    for relative in IMPLEMENTATION_PATHS:
        current = _file_sha256(REPO_ROOT / relative)
        committed = sha256(_git_output(REPO_ROOT, ["show", f"{commit}:{relative}"])).hexdigest()
        if current != committed:
            raise ValueError(f"implementation file is not frozen at HEAD: {relative}")
        rows[relative] = current
    return rows


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
