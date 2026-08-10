"""Freeze an outcome-blind FJSP exact-ledger confirmation suite.

This command must be committed before it is run.  It verifies the fixed
ScheduleOpt source commit and every implementation byte at the repository
HEAD, excludes only previously registered sources and parser-invalid rows,
then copies a stratified unused Hurink suite and writes its preregistration.
It never executes a scheduling policy or CP-SAT.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Sequence

from algorithm.experiments.fjsp_exact_ledger_confirmation_v2 import (
    DEFAULT_IMPLEMENTATION_PATHS,
    build_preregistration_draft,
    select_unused_instances,
    write_preregistration_draft,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "e5e62a01023ff827f658255b793fd1b86e5c1707"
SOURCE_SUBDIR = Path(
    "flexible-jobshop/instances/fjsp/HurinkJurischThole1994"
)
FREEZE_IMPLEMENTATION_PATHS = (
    *DEFAULT_IMPLEMENTATION_PATHS,
    "algorithm/experiments/fjsp_exact_ledger_confirmation_v2_freeze.py",
)
DEFAULT_SOURCE_REPO = REPO_ROOT / "reference" / "repos" / "scheduleopt-benchmarks"
DEFAULT_USED_MANIFEST = (
    REPO_ROOT / "tests" / "data" / "fjsp_hurink_external_holdout" / "source_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "tests" / "data" / "fjsp_exact_ledger_confirmation_v2"
)


class FJSPExactLedgerFreezeError(ValueError):
    """Raised when a prospective freeze cannot be made auditable."""


def freeze_suite(
    *,
    source_repo: str | Path = DEFAULT_SOURCE_REPO,
    used_manifest_path: str | Path = DEFAULT_USED_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    per_family: int = 5,
) -> dict[str, Any]:
    source_repository = Path(source_repo).resolve()
    used_manifest_file = Path(used_manifest_path).resolve()
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FJSPExactLedgerFreezeError(
            f"refusing to overwrite nonempty holdout root: {output}"
        )
    upstream_head = _git_output(source_repository, "rev-parse", "HEAD").strip()
    if upstream_head != UPSTREAM_COMMIT:
        raise FJSPExactLedgerFreezeError(
            f"unexpected ScheduleOpt commit: {upstream_head}"
        )
    implementation_hashes = _verify_implementation_committed_at_head(
        FREEZE_IMPLEMENTATION_PATHS
    )

    used_manifest = _load_json(used_manifest_file)
    used_rows = used_manifest.get("instances")
    if not isinstance(used_rows, list) or not used_rows:
        raise FJSPExactLedgerFreezeError("used-source manifest is empty")
    used_paths = [str(row["relative_path"]) for row in used_rows]
    used_hashes = [str(row["sha256"]) for row in used_rows]

    source_root = source_repository / SOURCE_SUBDIR
    candidate_paths = [
        path.relative_to(source_root)
        for family in ("edata", "rdata", "sdata", "vdata")
        for path in sorted((source_root / family).glob("*.txt"))
    ]
    selection = select_unused_instances(
        source_root,
        candidate_paths,
        used_relative_paths=used_paths,
        used_source_sha256=used_hashes,
        per_family=per_family,
    )
    preregistration = build_preregistration_draft(
        selection,
        source_identity={
            "repository": "https://github.com/ScheduleOpt/benchmarks",
            "repository_commit": UPSTREAM_COMMIT,
            "source_subdirectory": SOURCE_SUBDIR.as_posix(),
            "used_manifest_path": used_manifest_file.relative_to(REPO_ROOT).as_posix(),
            "used_manifest_sha256": _file_sha256(used_manifest_file),
        },
        implementation_paths=FREEZE_IMPLEMENTATION_PATHS,
        registration_label="fjsp-exact-ledger-confirmation-v2-20260810",
        run_cp_sat_reference=True,
        cp_sat_time_limit_s=5.0,
        cp_sat_workers=1,
    )
    if preregistration["implementation_freeze"]["files"] != implementation_hashes:
        raise FJSPExactLedgerFreezeError("implementation freeze changed during selection")

    output.mkdir(parents=True, exist_ok=True)
    preregistration_path = output / "preregistration.json"
    frozen = write_preregistration_draft(preregistration, preregistration_path)
    copied = []
    for row in selection["selected_instances"]:
        relative = Path(str(row["relative_path"]))
        source = source_root / relative
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if _file_sha256(target) != row["file_sha256"]:
            raise FJSPExactLedgerFreezeError(
                f"copied source hash mismatch: {relative.as_posix()}"
            )
        copied.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": row["file_sha256"],
            }
        )
    manifest = {
        "schema_version": "scheduleurm.fjsp_exact_ledger_source.v2",
        "preregistration_sha256": frozen["sha256"],
        "source_repository_commit": UPSTREAM_COMMIT,
        "outcomes_observed_before_freeze": False,
        "instances": copied,
    }
    manifest_path = output / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "FJSP_EXACT_LEDGER_V2_FROZEN",
        "instance_count": len(copied),
        "family_count": len({Path(row["relative_path"]).parts[0] for row in copied}),
        "source_format_exclusion_count": len(selection["source_format_exclusions"]),
        "preregistration_sha256": frozen["sha256"],
        "source_manifest_sha256": _file_sha256(manifest_path),
        "repository_head": _git_output(REPO_ROOT, "rev-parse", "HEAD").strip(),
    }


def _verify_implementation_committed_at_head(
    paths: Sequence[str | Path],
) -> dict[str, str]:
    head = _git_output(REPO_ROOT, "rev-parse", "HEAD").strip()
    hashes: dict[str, str] = {}
    for value in paths:
        relative = Path(str(value))
        disk = REPO_ROOT / relative
        disk_hash = _file_sha256(disk)
        try:
            committed = subprocess.run(
                ["git", "show", f"{head}:{relative.as_posix()}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise FJSPExactLedgerFreezeError(
                f"implementation is not committed: {relative.as_posix()}"
            ) from exc
        committed_hash = sha256(committed).hexdigest()
        if committed_hash != disk_hash:
            raise FJSPExactLedgerFreezeError(
                f"implementation differs from HEAD: {relative.as_posix()}"
            )
        hashes[relative.as_posix()] = disk_hash
    return hashes


def _git_output(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout


def _file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FJSPExactLedgerFreezeError(f"missing file: {source}")
    return sha256(source.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FJSPExactLedgerFreezeError(f"expected JSON object: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--used-manifest", type=Path, default=DEFAULT_USED_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-family", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = freeze_suite(
        source_repo=args.source_repo,
        used_manifest_path=args.used_manifest,
        output_root=args.output_root,
        per_family=args.per_family,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
