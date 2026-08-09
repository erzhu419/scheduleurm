"""Freeze a prospective size-stratified Hurink FJSP confirmation suite."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

from algorithm.experiments.fjsp_instances import load_fjsp_instance


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "e5e62a01023ff827f658255b793fd1b86e5c1707"
SUBFAMILIES = ("edata", "rdata", "sdata", "vdata")
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/fjsp_hurink_holdout.py",
    "algorithm/experiments/fjsp_hurink_holdout_freeze.py",
    "algorithm/experiments/fjsp_family_holdout.py",
    "algorithm/experiments/fjsp_renewal_frame_upgrade.py",
    "algorithm/experiments/trajectory_renewal_frame_score.py",
    "algorithm/experiments/fjsp_trajectory_upgrade.py",
    "algorithm/experiments/fjsp_benchmark.py",
    "algorithm/experiments/fjsp_instances.py",
    "algorithm/experiments/or_exact_reference.py",
)


def stratified_indices(count: int) -> tuple[int, ...]:
    if count < 5:
        raise ValueError("each Hurink subfamily must contain at least five instances")
    return tuple((multiplier * (count - 1)) // 4 for multiplier in range(5))


def freeze_suite(*, source_repo: str | Path, output_root: str | Path) -> dict[str, Any]:
    source = Path(source_repo).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty holdout root: {output}")
    upstream_head = _git_output(source, ["rev-parse", "HEAD"]).decode().strip()
    if upstream_head != UPSTREAM_COMMIT:
        raise ValueError(f"unexpected ScheduleOpt commit: {upstream_head}")
    bks_path = source / "flexible-jobshop" / "solutions" / "bks.json"
    bks_rows = json.loads(bks_path.read_text(encoding="utf-8"))
    bks_by_name = {str(row["instance"]): row for row in bks_rows}

    freeze_commit = _git_output(REPO_ROOT, ["rev-parse", "HEAD"]).decode().strip()
    implementation = {}
    for relative in IMPLEMENTATION_PATHS:
        current = _file_sha256(REPO_ROOT / relative)
        committed = sha256(
            _git_output(REPO_ROOT, ["show", f"{freeze_commit}:{relative}"])
        ).hexdigest()
        if current != committed:
            raise ValueError(f"implementation file is not frozen at HEAD: {relative}")
        implementation[relative] = current

    family_root = (
        source
        / "flexible-jobshop"
        / "instances"
        / "fjsp"
        / "HurinkJurischThole1994"
    )
    selected = []
    for subfamily in SUBFAMILIES:
        candidates = []
        for path in sorted((family_root / subfamily).glob("*.txt")):
            instance = load_fjsp_instance(path)
            candidates.append(
                (
                    instance.operation_count * instance.machine_count,
                    instance.job_count,
                    instance.machine_count,
                    instance.operation_count,
                    path.name,
                    path,
                    instance,
                )
            )
        candidates.sort(key=lambda row: row[:5])
        for rank in stratified_indices(len(candidates)):
            scale, jobs, machines, operations, name, path, instance = candidates[rank]
            bks = bks_by_name.get(f"{path.stem}_{subfamily}")
            if bks is None:
                raise ValueError(f"missing public bound: {subfamily}/{name}")
            relative = Path(subfamily) / name
            selected.append(
                {
                    "relative_path": relative.as_posix(),
                    "family": "HurinkJurischThole1994",
                    "subfamily": subfamily,
                    "source_path": path.relative_to(source).as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    "job_count": jobs,
                    "machine_count": machines,
                    "operation_count": operations,
                    "selection_scale": scale,
                    "lower_bound": bks["lower_bound"],
                    "upper_bound": bks["upper_bound"],
                    "bound_status": bks["status"],
                    "_source": path,
                }
            )
    members = [row["relative_path"] for row in selected]
    if len(members) != 20 or len(set(members)) != 20:
        raise ValueError("Hurink holdout must contain 20 unique instances")

    preregistration = {
        "schema_version": "scheduleurm.fjsp_hurink_holdout.preregistration.v1",
        "registered_on": "2026-08-09",
        "source": {
            "repository": "https://github.com/ScheduleOpt/benchmarks",
            "repository_commit": UPSTREAM_COMMIT,
            "bounds_path": "flexible-jobshop/solutions/bks.json",
            "bounds_sha256": _file_sha256(bks_path),
        },
        "implementation_freeze": {
            "repository_commit": freeze_commit,
            "implementation_sha256": implementation,
        },
        "selection": {
            "rule": (
                "Within each untouched Hurink data subfamily, sort by "
                "(operation_count*machine_count, job_count, machine_count, "
                "operation_count, filename) and take ranks floor(k(n-1)/4), k=0..4."
            ),
            "rule_uses_algorithm_outcomes": False,
            "subfamilies": list(SUBFAMILIES),
            "excluded_previously_observed_families": [
                "Brandimarte1993",
                "KacemHammadiBorne2002",
                "BehnkeGeiger2012",
                "ChambersBarnes1996",
                "DauzerePeresPaulli1994",
                "FattahiMehrabadJolai2007",
            ],
            "members": members,
        },
        "protocol": {
            "policy": "scheduleurm_fjsp_renewal_frame",
            "per_instance_tuning": False,
            "holdout_feedback_used_for_policy": False,
            "run_cp_sat_reference": True,
            "cp_sat_workers": 1,
            "matched_reference_floor_s": 0.05,
            "fixed_reference_budget_s": 5.0,
            "cp_sat_reference_coverage_is_not_protocol_pass_condition": True,
            "performance_is_not_protocol_pass_condition": True,
        },
        "claim_boundary": {
            "global_fjsp_optimality": False,
            "arbitrary_family_dominance": False,
            "stochastic_stability_transfer": False,
        },
        "outputs": [
            "md/experiment_artifacts/fjsp_hurink_holdout_20260809.json.gz",
            "md/experiment_artifacts/fjsp_hurink_holdout_20260809.json",
            "md/fjsp_hurink_holdout_20260809.md",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    preregistration_path = output / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prereg_hash = _file_sha256(preregistration_path)
    manifest_rows = []
    for row in selected:
        source_path = row.pop("_source")
        target = output / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
        manifest_rows.append(row)
    manifest = {
        "schema_version": "scheduleurm.fjsp_hurink_holdout.source.v1",
        "preregistration_sha256": prereg_hash,
        "source": preregistration["source"],
        "instances": manifest_rows,
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "instance_count": len(manifest_rows),
        "subfamilies": list(SUBFAMILIES),
        "preregistration_sha256": prereg_hash,
        "manifest_sha256": _file_sha256(manifest_path),
        "freeze_commit": freeze_commit,
    }


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
