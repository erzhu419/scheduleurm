"""Freeze unused Hurink instances for the FJSP solver-action union holdout."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.fjsp_exact_ledger_confirmation_v2 import (
    select_unused_instances,
)
from algorithm.experiments.fjsp_solver_action_union import FROZEN_CONFIG
from algorithm.experiments.fjsp_solver_action_union_holdout import (
    PREREGISTRATION_SCHEMA,
    REPO_ROOT,
    _digest,
)
from algorithm.experiments.fjsp_benchmark import POLICY_ORDER


EXPECTED_REPOSITORY_COMMIT = "e5e62a01023ff827f658255b793fd1b86e5c1707"
SOURCE_SUBDIRECTORY = "flexible-jobshop/instances/fjsp/HurinkJurischThole1994"
OLD_PREREGISTRATION = REPO_ROOT / "tests/data/fjsp_exact_ledger_confirmation_v2/preregistration.json"
REGISTERED_ON = "2026-08-30"
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/fjsp_benchmark.py",
    "algorithm/experiments/fjsp_exact_ledger_confirmation_v2.py",
    "algorithm/experiments/fjsp_instances.py",
    "algorithm/experiments/fjsp_renewal_frame_upgrade.py",
    "algorithm/experiments/fjsp_solver_action_union.py",
    "algorithm/experiments/fjsp_solver_action_union_holdout.py",
    "algorithm/experiments/fjsp_solver_action_union_holdout_freeze.py",
    "algorithm/experiments/fjsp_trajectory_upgrade.py",
    "algorithm/experiments/or_exact_reference.py",
    "algorithm/experiments/trajectory_renewal_frame_score.py",
)


def freeze_suite(*, source_repo: str | Path, output_root: str | Path) -> dict[str, Any]:
    repository = Path(source_repo).resolve()
    source_root = repository / SOURCE_SUBDIRECTORY
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty holdout root: {output}")
    try:
        output_relative = output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("holdout output must be inside the repository") from exc
    head = _git_output(repository, ["rev-parse", "HEAD"]).decode().strip()
    if head != EXPECTED_REPOSITORY_COMMIT:
        raise ValueError(f"unexpected FJSP benchmark commit: {head}")

    old = _load_json(OLD_PREREGISTRATION)["selection"]
    used_paths = set(old["used_registry"]["relative_paths"])
    used_hashes = set(old["used_registry"]["source_sha256"])
    for row in old["selected_instances"]:
        used_paths.add(str(row["relative_path"]))
        used_hashes.add(str(row["file_sha256"]))
        used_hashes.add(str(row["canonical_source_sha256"]))
    candidate_paths = sorted(path.relative_to(source_root) for path in source_root.glob("*/*.txt"))
    selection = select_unused_instances(
        source_root,
        candidate_paths,
        used_relative_paths=used_paths,
        used_source_sha256=used_hashes,
        per_family=5,
    )
    if selection["selected_count"] != 20:
        raise ValueError("prospective FJSP holdout must contain 20 instances")

    output.mkdir(parents=True, exist_ok=True)
    registered = []
    for row in selection["selected_instances"]:
        upstream = Path(str(row["relative_path"]))
        destination = output / upstream
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / upstream, destination)
        registered.append(
            {
                **row,
                "upstream_relative_path": upstream.as_posix(),
                "local_path": (output_relative / upstream).as_posix(),
            }
        )

    freeze_commit = _git_output(REPO_ROOT, ["rev-parse", "HEAD"]).decode().strip()
    implementation = _frozen_implementation(freeze_commit)
    preregistration: dict[str, Any] = {
        "schema_version": PREREGISTRATION_SCHEMA,
        "registered_on": REGISTERED_ON,
        "registration_label": "fjsp-solver-action-union-holdout-v1",
        "holdout_outcomes_observed_before_freeze": False,
        "source_identity": {
            "repository": "https://github.com/ScheduleOpt/benchmarks",
            "repository_commit": EXPECTED_REPOSITORY_COMMIT,
            "source_subdirectory": SOURCE_SUBDIRECTORY,
        },
        "selection_rule": selection["selection_rule"],
        "prior_used_registry": {
            "relative_path_count": len(used_paths),
            "source_hash_count": len(used_hashes),
            "old_preregistration": OLD_PREREGISTRATION.relative_to(REPO_ROOT).as_posix(),
            "old_preregistration_sha256": _file_sha256(OLD_PREREGISTRATION),
        },
        "selected_instances": registered,
        "implementation_freeze_commit": freeze_commit,
        "implementation_file_sha256": implementation,
        "solver_action_config": FROZEN_CONFIG.snapshot(),
        "baseline_policies": list(POLICY_ORDER),
        "comparison_coordinates": ["makespan_ticks", "sum_completion_ticks"],
        "failure_policy": {
            "parse_or_ledger_error": "retain as failure",
            "no_solver_incumbent": "retain as solver-action coverage failure",
            "baseline_dominance": "retain in performance denominator",
            "no_post_holdout_tuning": True,
        },
        "claim_boundary": [
            "exact only over the registered finite solver-action union",
            "no global FJSP or multiobjective optimality claim",
            "solver runtime reported separately",
        ],
        "outputs": {
            "full_artifact": "md/experiment_artifacts/fjsp_solver_action_union_holdout_v1_full.json.gz",
            "compact_artifact": "md/experiment_artifacts/fjsp_solver_action_union_holdout_v1.json",
            "markdown": "md/fjsp_solver_action_union_holdout_v1.md",
        },
    }
    preregistration["preregistration_sha256_excluding_self"] = _digest(preregistration)
    preregistration_path = output / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "selected_count": len(registered),
        "families": sorted({row["family"] for row in registered}),
        "unused_candidate_count_before_selection": selection["unused_candidate_count"],
        "freeze_commit": freeze_commit,
        "preregistration_sha256": _file_sha256(preregistration_path),
    }


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
