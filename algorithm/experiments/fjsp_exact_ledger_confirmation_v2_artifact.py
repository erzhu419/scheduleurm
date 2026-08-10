"""Execute and serialize the frozen FJSP exact-ledger confirmation v2."""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from algorithm.experiments.fjsp_exact_ledger_confirmation_v2 import (
    run_exact_ledger_confirmation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "tests" / "data" / "fjsp_exact_ledger_confirmation_v2"
PREREGISTRATION = SUITE_ROOT / "preregistration.json"
PREREGISTRATION_SHA256 = "21a0ae639dda7715722412502b1e09c2000f1ac5d5160beaae662bd42316b92e"
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "fjsp_exact_ledger_confirmation_v2_20260810.json.gz"
)
DEFAULT_COMPACT = DEFAULT_FULL.with_suffix("").with_suffix(".json")
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "fjsp_exact_ledger_confirmation_v2_20260810.md"


class FJSPExactLedgerArtifactError(ValueError):
    """Raised when the execution wrapper is not auditable."""


def build_artifact() -> dict[str, Any]:
    prereg_commit = _committed_file_contract(
        PREREGISTRATION, PREREGISTRATION_SHA256
    )
    wrapper_relative = Path(__file__).resolve().relative_to(REPO_ROOT)
    wrapper_hash = _file_sha256(Path(__file__))
    wrapper_commit = _committed_file_contract(Path(__file__), wrapper_hash)
    report = run_exact_ledger_confirmation(
        PREREGISTRATION,
        expected_preregistration_sha256=PREREGISTRATION_SHA256,
        source_root=SUITE_ROOT,
    )
    return {
        **report,
        "execution_contract": {
            "preregistration_commit": prereg_commit,
            "execution_wrapper_path": wrapper_relative.as_posix(),
            "execution_wrapper_sha256": wrapper_hash,
            "execution_wrapper_commit": wrapper_commit,
            "results_observed_before_preregistration_commit": False,
        },
    }


def compact_certificate(report: Mapping[str, Any], full_sha256: str) -> dict[str, Any]:
    rows = []
    for row in report["rows"]:
        candidate = row["candidate_family_confirmation"]
        reference = row["cp_sat_reference_gap"]
        rows.append(
            {
                "relative_path": row["relative_path"],
                "family": row["family"],
                "source_sha256": row["source_sha256"],
                "selected_exact_ledger": candidate["selected_exact_ledger"],
                "exact_nondominated_by_fixed_policy_seeds": candidate[
                    "exact_pareto_nondominated_by_fixed_policy_seeds"
                ],
                "fixed_policy_dominators": candidate["fixed_policy_dominators"],
                "same_action_policies": candidate["same_action_policies"],
                "cp_sat_status": reference["status"],
                "cp_sat_reference_feasible": reference["reference_feasible"],
                "cp_sat_optimality_proved": reference[
                    "reference_optimality_proved"
                ],
                "cp_sat_exact_relation": reference.get("exact_relation"),
                "candidate_family_minus_reference": reference[
                    "candidate_family_minus_reference"
                ],
            }
        )
    compact = {
        "schema_version": "scheduleurm.fjsp_exact_ledger_confirmation.v2.compact.v1",
        "gate": report["gate"],
        "source_contract": report["source_contract"],
        "execution_contract": report["execution_contract"],
        "candidate_family_aggregate": report["candidate_family_aggregate"],
        "cp_sat_reference_aggregate": report["cp_sat_reference_aggregate"],
        "rows": rows,
        "claim_boundary": report["claim_boundary"],
        "full_artifact_gzip_sha256": full_sha256,
    }
    compact["artifact_sha256_excluding_self"] = _digest(compact)
    return compact


def write_outputs(
    report: Mapping[str, Any],
    *,
    full_path: str | Path = DEFAULT_FULL,
    compact_path: str | Path = DEFAULT_COMPACT,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    full = Path(full_path)
    compact_file = Path(compact_path)
    markdown = Path(markdown_path)
    for path in (full, compact_file, markdown):
        if path.exists():
            raise FJSPExactLedgerArtifactError(f"refusing to overwrite result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    full_hash = _file_sha256(full)
    compact = compact_certificate(report, full_hash)
    compact_file.write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(compact), encoding="utf-8")
    return {
        "full": str(full),
        "full_sha256": full_hash,
        "compact": str(compact_file),
        "compact_sha256": _file_sha256(compact_file),
        "markdown": str(markdown),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    candidate = compact["candidate_family_aggregate"]
    reference = compact["cp_sat_reference_aggregate"]
    instance_count = int(candidate["instance_count"])
    cp_sat_dominates = sum(
        row["cp_sat_exact_relation"] == "comparator_dominates"
        for row in compact["rows"]
    )
    lines = [
        "# Prospective FJSP exact-ledger confirmation v2",
        "",
        f"- Status: `{compact['gate']['status']}`",
        f"- Registered instances: `{instance_count}`",
        f"- Exact baseline-union nondominance: `{candidate['exact_nondominated_count']}/{instance_count}`",
        f"- Feasible fixed-budget CP-SAT references: `{reference['feasible_reference_count']}/{instance_count}`",
        f"- Fixed-budget CP-SAT dominates generated family: `{cp_sat_dominates}/{instance_count}`",
        "- Performance does not determine protocol validity.",
        "- A feasible or makespan-optimal CP-SAT row is not a proof of multiobjective global optimality.",
        "",
        "| Family | Instance | Baseline-union nondominated | CP-SAT relation | CP-SAT optimal |",
        "|---|---|---:|---|---:|",
    ]
    for row in compact["rows"]:
        lines.append(
            f"| {row['family']} | {row['relative_path']} | "
            f"{str(row['exact_nondominated_by_fixed_policy_seeds']).lower()} | "
            f"{row['cp_sat_exact_relation'] or 'unavailable'} | "
            f"{str(row['cp_sat_optimality_proved']).lower()} |"
        )
    lines.extend(["", str(compact["claim_boundary"]), ""])
    return "\n".join(lines)


def _committed_file_contract(path: Path, expected_sha256: str) -> str:
    source = path.resolve()
    relative = source.relative_to(REPO_ROOT)
    if _file_sha256(source) != expected_sha256:
        raise FJSPExactLedgerArtifactError(f"disk hash mismatch: {relative}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise FJSPExactLedgerArtifactError(f"file is not committed: {relative}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if sha256(committed).hexdigest() != expected_sha256:
        raise FJSPExactLedgerArtifactError(f"committed hash mismatch: {relative}")
    return commit


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_artifact()
    outputs = write_outputs(
        report,
        full_path=args.full,
        compact_path=args.compact,
        markdown_path=args.markdown,
    )
    print(json.dumps({"gate": report["gate"], "outputs": outputs}, indent=2))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
