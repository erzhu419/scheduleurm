"""Audit repeatability of two FJSP exact-ledger confirmation executions."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "fjsp_exact_ledger_confirmation_v2_20260810.json"
)
DEFAULT_REPLICATION = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "fjsp_exact_ledger_confirmation_v2_replication_20260810.json"
)
DEFAULT_JSON = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "fjsp_exact_ledger_replication_audit_20260810.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "fjsp_exact_ledger_replication_audit_20260810.md"


class FJSPReplicationAuditError(ValueError):
    """Raised when executions are incompatible or outputs would be overwritten."""


def build_replication_audit(
    primary: Mapping[str, Any], replication: Mapping[str, Any]
) -> dict[str, Any]:
    primary_rows = {str(row["relative_path"]): row for row in primary["rows"]}
    replication_rows = {
        str(row["relative_path"]): row for row in replication["rows"]
    }
    if set(primary_rows) != set(replication_rows) or not primary_rows:
        raise FJSPReplicationAuditError("execution instance sets differ")
    candidate_fields = (
        "selected_exact_ledger",
        "exact_nondominated_by_fixed_policy_seeds",
        "fixed_policy_dominators",
        "same_action_policies",
    )
    qualitative_reference_fields = (
        "cp_sat_status",
        "cp_sat_reference_feasible",
        "cp_sat_optimality_proved",
        "cp_sat_exact_relation",
    )
    differing_reference_gaps = []
    candidate_match_count = 0
    qualitative_reference_match_count = 0
    numeric_reference_gap_match_count = 0
    for relative_path in sorted(primary_rows):
        left = primary_rows[relative_path]
        right = replication_rows[relative_path]
        candidate_match_count += all(left[field] == right[field] for field in candidate_fields)
        qualitative_reference_match_count += all(
            left[field] == right[field] for field in qualitative_reference_fields
        )
        numeric_match = (
            left["candidate_family_minus_reference"]
            == right["candidate_family_minus_reference"]
        )
        numeric_reference_gap_match_count += numeric_match
        if not numeric_match:
            differing_reference_gaps.append(
                {
                    "relative_path": relative_path,
                    "primary_candidate_family_minus_reference": left[
                        "candidate_family_minus_reference"
                    ],
                    "replication_candidate_family_minus_reference": right[
                        "candidate_family_minus_reference"
                    ],
                }
            )
    count = len(primary_rows)
    candidate_ready = candidate_match_count == count
    qualitative_ready = qualitative_reference_match_count == count
    strict_full_ready = (
        candidate_ready
        and qualitative_ready
        and numeric_reference_gap_match_count == count
        and primary == replication
    )
    report = {
        "schema_version": "scheduleurm.fjsp_exact_ledger.replication_audit.v1",
        "execution_count": 2,
        "instance_count": count,
        "source_contract_identical": primary["source_contract"]
        == replication["source_contract"],
        "execution_contract_identical": primary["execution_contract"]
        == replication["execution_contract"],
        "candidate_family_aggregate_identical": primary[
            "candidate_family_aggregate"
        ]
        == replication["candidate_family_aggregate"],
        "cp_sat_reference_aggregate_identical": primary[
            "cp_sat_reference_aggregate"
        ]
        == replication["cp_sat_reference_aggregate"],
        "candidate_exact_ledger_match_count": candidate_match_count,
        "candidate_exact_ledger_reproducibility_ready": candidate_ready,
        "cp_sat_qualitative_disposition_match_count": qualitative_reference_match_count,
        "cp_sat_qualitative_disposition_reproducibility_ready": qualitative_ready,
        "cp_sat_numeric_gap_match_count": numeric_reference_gap_match_count,
        "cp_sat_wall_clock_numeric_reproducibility_ready": (
            numeric_reference_gap_match_count == count
        ),
        "strict_full_artifact_reproducibility_ready": strict_full_ready,
        "differing_cp_sat_numeric_gaps": differing_reference_gaps,
        "gate": {
            "pass": candidate_ready and qualitative_ready,
            "status": (
                "FJSP_REPLICATION_AUDIT_PASS_WITH_CP_SAT_VARIATION"
                if candidate_ready and qualitative_ready and not strict_full_ready
                else (
                    "FJSP_REPLICATION_AUDIT_STRICT_PASS"
                    if strict_full_ready
                    else "FJSP_REPLICATION_AUDIT_FAIL"
                )
            ),
        },
        "claim_boundary": {
            "supports": (
                "repeatability of the frozen candidate action ledgers and the "
                "qualitative fixed-budget comparator dispositions over these two runs"
            ),
            "does_not_support": (
                "bitwise reproducibility of a wall-clock-limited CP-SAT incumbent or "
                "global FJSP optimality"
            ),
            "interpretation": (
                "The 5-second single-worker CP-SAT reference preserves status and "
                "Pareto relation here, but three incumbent objective gaps vary."
            ),
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    output = Path(json_path)
    markdown = Path(markdown_path)
    for path in (output, markdown):
        if path.exists():
            raise FJSPReplicationAuditError(f"refusing to overwrite result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(report), encoding="utf-8")
    return {
        "json": str(output),
        "json_sha256": _file_sha256(output),
        "markdown": str(markdown),
        "markdown_sha256": _file_sha256(markdown),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# FJSP exact-ledger replication audit",
        "",
        f"- Status: `{report['gate']['status']}`",
        f"- Candidate ledger identity: `{report['candidate_exact_ledger_match_count']}/{report['instance_count']}`",
        f"- CP-SAT qualitative disposition identity: `{report['cp_sat_qualitative_disposition_match_count']}/{report['instance_count']}`",
        f"- CP-SAT numeric gap identity: `{report['cp_sat_numeric_gap_match_count']}/{report['instance_count']}`",
        f"- Strict full-artifact reproducibility: `{str(report['strict_full_artifact_reproducibility_ready']).lower()}`",
        "",
        "The frozen Scheduleurm candidate ledgers reproduce exactly. The 5-second "
        "single-worker CP-SAT comparator preserves feasibility, optimal-status counts, "
        "and qualitative Pareto relations, but its incumbent gaps are not bitwise stable.",
        "",
        "| Instance | Primary gap | Replication gap |",
        "|---|---|---|",
    ]
    for row in report["differing_cp_sat_numeric_gaps"]:
        lines.append(
            f"| {row['relative_path']} | "
            f"`{row['primary_candidate_family_minus_reference']}` | "
            f"`{row['replication_candidate_family_minus_reference']}` |"
        )
    lines.extend(["", "## Claim Boundary", "", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--replication", type=Path, default=DEFAULT_REPLICATION)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    replication = json.loads(args.replication.read_text(encoding="utf-8"))
    report = build_replication_audit(primary, replication)
    outputs = write_outputs(report, json_path=args.json, markdown_path=args.markdown)
    print(json.dumps({"gate": report["gate"], "outputs": outputs}, indent=2))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
