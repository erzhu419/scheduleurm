"""Freeze an outcome-blind, size-stratified PSPLIB MMRCPSP family suite."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_family_holdout import instance_shape
from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("j12", "j14", "j16", "j18", "j20")
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/mmrcpsp_family_holdout.py",
    "algorithm/experiments/mmrcpsp_family_holdout_freeze.py",
    "algorithm/experiments/mmrcpsp_renewal_frame_upgrade.py",
    "algorithm/experiments/trajectory_renewal_frame_score.py",
    "algorithm/experiments/mmrcpsp_trajectory_upgrade.py",
    "algorithm/experiments/mmrcpsp_benchmark.py",
    "algorithm/experiments/mmrcpsp_instances.py",
    "algorithm/experiments/or_exact_reference.py",
)


def stratified_indices(count: int) -> tuple[int, ...]:
    if count < 5:
        raise ValueError("each PSPLIB family must contain at least five instances")
    return tuple((multiplier * (count - 1)) // 4 for multiplier in range(5))


def parse_optimum_table(path: str | Path) -> dict[tuple[int, int], int]:
    rows: dict[tuple[int, int], int] = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        parameter = int(fields[0])
        replicate = int(fields[1])
        makespan = int(fields[2])
        key = (parameter, replicate)
        if key in rows:
            raise ValueError(f"duplicate public optimum row: {key}")
        rows[key] = makespan
    if not rows:
        raise ValueError(f"empty public optimum table: {path}")
    return rows


def instance_identifier(family: str, stem: str) -> tuple[int, int]:
    match = re.fullmatch(rf"{re.escape(family)}(\d+)_(\d+)", stem)
    if match is None:
        raise ValueError(f"unexpected {family} instance name: {stem}")
    return int(match.group(1)), int(match.group(2))


def freeze_suite(*, source_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty holdout root: {output}")
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

    selected = []
    source_tables = {}
    for family in FAMILIES:
        family_root = source / family
        optimum_path = source / f"{family}.opt"
        optima = parse_optimum_table(optimum_path)
        source_tables[family] = {
            "instance_archive_sha256": _file_sha256(source / f"{family}.zip"),
            "public_optimum_sha256": _file_sha256(optimum_path),
            "download_url": (
                "https://www.om-db.wi.tum.de/psplib/"
                f"download_dataset.php?set={family}&mode=mm&format=zip"
            ),
            "optimum_url": (
                "https://www.om-db.wi.tum.de/psplib/"
                f"download_solution.php?set={family}&mode=mm&type=opt"
            ),
        }
        candidates = []
        for path in sorted(family_root.glob("*.mm")):
            instance = parse_psplib_mm(path)
            shape = instance_shape(instance)
            parameter, replicate = instance_identifier(family, path.stem)
            optimum = optima.get((parameter, replicate))
            if optimum is None or optimum >= 16384:
                raise ValueError(f"missing feasible public optimum: {path.name}")
            total_mode_duration = sum(
                mode.duration
                for activity in instance.activities
                for mode in activity.modes
            )
            structural_key = (
                shape["horizon"],
                shape["precedence_arc_count"],
                total_mode_duration,
                path.name,
            )
            candidates.append((structural_key, path, instance, shape, optimum))
        candidates.sort(key=lambda row: row[0])
        for rank in stratified_indices(len(candidates)):
            structural_key, path, instance, shape, optimum = candidates[rank]
            relative = Path(family) / path.name
            selected.append(
                {
                    "relative_path": relative.as_posix(),
                    "family": family,
                    "source_file": path.name,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                    **shape,
                    "selection_structural_key": list(structural_key[:-1]),
                    "public_optimal_makespan": optimum,
                    "_source": path,
                }
            )
    members = [row["relative_path"] for row in selected]
    if len(members) != 25 or len(set(members)) != 25:
        raise ValueError("family holdout must contain 25 unique instances")

    preregistration = {
        "schema_version": "scheduleurm.mmrcpsp_family_holdout.preregistration.v1",
        "registered_on": "2026-08-09",
        "source": {
            "collection": "PSPLIB multi-mode RCPSP",
            "landing_page": "https://www.om-db.wi.tum.de/psplib/data.php",
            "family_tables": source_tables,
        },
        "implementation_freeze": {
            "repository_commit": freeze_commit,
            "implementation_sha256": implementation,
        },
        "selection": {
            "rule": (
                "Within each registered feasible public family, sort by "
                "(horizon, precedence_arc_count, total_mode_duration, filename) "
                "and take ranks floor(k(n-1)/4), k=0..4. Public optimum values "
                "are attached only after structural selection."
            ),
            "rule_uses_scheduleurm_outcomes": False,
            "families": list(FAMILIES),
            "excluded_development_family": "j10",
            "members": members,
        },
        "protocol": {
            "policy": "scheduleurm_mmrcpsp_renewal_frame",
            "per_instance_tuning": False,
            "holdout_feedback_used_for_policy": False,
            "run_cp_sat_reference": True,
            "cp_sat_workers": 1,
            "matched_reference_floor_s": 0.05,
            "fixed_reference_budget_s": 5.0,
            "cp_sat_budget_excludes_model_construction": True,
            "performance_is_not_a_protocol_pass_condition": True,
        },
        "claim_boundary": {
            "global_mmrcpsp_optimality": False,
            "arbitrary_family_dominance": False,
            "stochastic_stability_transfer": False,
        },
        "outputs": [
            "tests/data/mmrcpsp_family_external_holdout/mmrcpsp_family_external_holdout_gate.json",
            "tests/data/mmrcpsp_family_external_holdout/mmrcpsp_family_external_holdout_gate.md",
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
        "schema_version": "scheduleurm.mmrcpsp_family_holdout.source.v1",
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
        "families": list(FAMILIES),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = freeze_suite(source_root=args.source_root, output_root=args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
