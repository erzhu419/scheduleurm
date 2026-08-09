"""Freeze untouched BACASP-S R85/R86 instances and the factor-policy contract."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_factor_policy_development import cell_key
from algorithm.experiments.port_public_external_holdout import _digest


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "e8eda86ec0435105ca313afe85987e4f4ab2cbd7"
EXPECTED_REPLICATIONS = (85, 86)
IMPLEMENTATION_PATHS = (
    "algorithm/experiments/port_factor_policy_development.py",
    "algorithm/experiments/port_factor_policy_holdout.py",
    "algorithm/experiments/port_factor_policy_holdout_freeze.py",
    "algorithm/experiments/port_public_benchmark_adapter.py",
    "algorithm/experiments/port_public_external_holdout.py",
    "algorithm/experiments/port_scheduling_benchmark.py",
    "algorithm/experiments/port_scheduling_instances.py",
    "algorithm/experiments/port_trajectory_upgrade.py",
)


def freeze_suite(
    *,
    source_repo: str | Path,
    development_compact: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_repo).resolve()
    development_path = Path(development_compact).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty holdout root: {output}")
    upstream_head = _git_output(source, ["rev-parse", "HEAD"]).decode().strip()
    if upstream_head != UPSTREAM_COMMIT:
        raise ValueError(f"unexpected BACASP-S source commit: {upstream_head}")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    mapping = development.get("factor_policy") or {}
    if len(mapping) != 27:
        raise ValueError("development factor policy must contain 27 cells")

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

    pattern = re.compile(
        r".*_(?P<rep>85|86)_Q(?P<q>5|10|15)_S(?P<speed>160-0\.2|200-0\.15|240-0\.1)_D(?P<deadline>1|1\.5|2)\.dat$"
    )
    selected = []
    for path in sorted((source / "instances" / "LargeMB").glob("*.dat")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        replication = int(match.group("rep"))
        q_max = int(match.group("q"))
        speed_setup = match.group("speed")
        deadline = match.group("deadline")
        key = cell_key(q_max, speed_setup, deadline)
        if key not in mapping:
            raise ValueError(f"factor policy misses source cell: {key}")
        relative = Path(path.name)
        raw = path.read_bytes()
        selected.append(
            {
                "local_path": (output.relative_to(REPO_ROOT) / relative).as_posix(),
                "upstream_path": path.relative_to(source).as_posix(),
                "sha256": sha256(raw).hexdigest(),
                "byte_count": len(raw),
                "line_count": len(raw.splitlines()),
                "raw_bytes_modified": False,
                "split": "external_holdout",
                "replication": replication,
                "q_max_factor": q_max,
                "speed_setup_factor": speed_setup,
                "deadline_factor": deadline,
                "factor_cell": key,
                "_source": path,
            }
        )
    if len(selected) != 54 or len({row["sha256"] for row in selected}) != 54:
        raise ValueError("R85/R86 confirmation must contain 54 unique instances")
    cell_replicates: dict[str, set[int]] = {}
    for row in selected:
        cell_replicates.setdefault(row["factor_cell"], set()).add(row["replication"])
    if len(cell_replicates) != 27 or any(
        values != set(EXPECTED_REPLICATIONS) for values in cell_replicates.values()
    ):
        raise ValueError("R85/R86 factor-cell coverage is incomplete")

    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for row in selected:
        source_path = row.pop("_source")
        target = output / source_path.name
        shutil.copyfile(source_path, target)
        manifest_rows.append(row)
    manifest = {
        "schema_version": "scheduleurm.port_bacasp_s_suite_manifest.v1",
        "dataset_name": "BACASP-S LargeMB",
        "repository_url": "https://github.com/juanfdata/bacasp_setup",
        "repository_commit": UPSTREAM_COMMIT,
        "instances": manifest_rows,
    }
    manifest["manifest_content_sha256_excluding_self"] = _digest(manifest)
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    preregistration = {
        "schema_version": "scheduleurm.port_factor_policy_holdout.preregistration.v1",
        "registered_on": "2026-08-09",
        "source": {
            "repository": "https://github.com/juanfdata/bacasp_setup",
            "repository_commit": UPSTREAM_COMMIT,
            "replications": list(EXPECTED_REPLICATIONS),
            "factor_cell_count": 27,
            "registered_instance_count": 54,
        },
        "suite_manifest_sha256": _file_sha256(manifest_path),
        "development_compact_path": str(development_path.relative_to(REPO_ROOT)),
        "development_compact_sha256": _file_sha256(development_path),
        "development_content_sha256": development[
            "compact_content_sha256_excluding_self"
        ],
        "factor_policy_sha256": _digest(mapping),
        "implementation_freeze": {
            "repository_commit": freeze_commit,
            "implementation_sha256": implementation,
        },
        "protocol": {
            "policy": "factor_conditioned_finite_trajectory",
            "conditioning_features": [
                "q_max_factor",
                "speed_setup_factor",
                "deadline_factor",
            ],
            "candidate_generation_on_holdout": False,
            "holdout_feedback_used_for_selection": False,
            "baseline_policies": [
                "fcfs_static",
                "spt_static",
                "edd_static",
                "reconfiguration_greedy",
            ],
            "evaluation_metrics": [
                "quay_makespan",
                "mean_quay_flow_time",
                "source_objective_cost",
            ],
        },
        "claim_boundary": {
            "unrestricted_port_optimality": False,
            "physical_terminal_deployment": False,
            "mid_service_migration": False,
            "stochastic_stability_transfer": False,
        },
        "outputs": [
            "md/experiment_artifacts/port_factor_policy_holdout_r85_r86_20260809.json.gz",
            "md/experiment_artifacts/port_factor_policy_holdout_r85_r86_20260809.json",
            "md/port_factor_policy_holdout_r85_r86_20260809.md",
        ],
    }
    preregistration_path = output / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "instance_count": len(manifest_rows),
        "factor_cell_count": len(cell_replicates),
        "manifest_sha256": _file_sha256(manifest_path),
        "preregistration_sha256": _file_sha256(preregistration_path),
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
    parser.add_argument("--development-compact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = freeze_suite(
        source_repo=args.source_repo,
        development_compact=args.development_compact,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
