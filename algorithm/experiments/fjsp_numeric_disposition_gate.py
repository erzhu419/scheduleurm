"""Versioned disposition for legacy FJSP floating-point Pareto artifacts.

The two source holdouts are immutable prospective artifacts.  This gate does
not rerun or rewrite them.  It binds their bytes, reconstructs exact completion
ledgers from each stored schedule, and separates protocol integrity from the
performance relation that was previously evaluated on differently rounded
mean-flow values.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILY_ARTIFACT = (
    REPO_ROOT
    / "tests"
    / "data"
    / "fjsp_family_external_holdout"
    / "fjsp_family_external_holdout_gate.json"
)
HURINK_COMPACT = (
    REPO_ROOT / "md" / "experiment_artifacts" / "fjsp_hurink_holdout_20260809.json"
)
HURINK_FULL = HURINK_COMPACT.with_suffix(".json.gz")
DEFAULT_JSON = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "fjsp_numeric_disposition_gate_20260810.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "fjsp_numeric_disposition_gate_20260810.md"

EXPECTED_SHA256 = {
    "family": "6cc2c74f7f88765639fe701cd26c15e963bf0051a8f4aa38cb1dff6ba56c104f",
    "hurink_compact": "8c5c51e6f58a6813255c453b827842396846724ff056a42e162165e53cce4572",
    "hurink_full": "85cb7606fc689fb7db29829dbade1db00488787a97f76aa589ecaaebc27a87c3",
}


class FJSPNumericDispositionError(ValueError):
    """Raised when a frozen source or exact-ledger audit is inconsistent."""


def exact_completion_ledger(
    schedule: Sequence[Mapping[str, Any]], *, job_count: int
) -> dict[str, Any]:
    if int(job_count) <= 0:
        raise FJSPNumericDispositionError("job_count must be positive")
    completion = [Decimal(0) for _ in range(int(job_count))]
    canonical_schedule = []
    for row in schedule:
        job_id = int(row["job_id"])
        operation_id = int(row["operation_id"])
        machine_id = int(row["machine_id"])
        if not 0 <= job_id < int(job_count):
            raise FJSPNumericDispositionError(f"invalid job id {job_id}")
        start = _decimal(row["start_time"])
        end = _decimal(row["end_time"])
        if end < start:
            raise FJSPNumericDispositionError("operation ends before it starts")
        completion[job_id] = max(completion[job_id], end)
        canonical_schedule.append(
            (job_id, operation_id, machine_id, _decimal_text(start), _decimal_text(end))
        )
    canonical_schedule.sort()
    makespan = max(completion, default=Decimal(0))
    total_completion = sum(completion, Decimal(0))
    fingerprint = sha256(
        json.dumps(canonical_schedule, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "makespan_ticks": _decimal_text(makespan),
        "sum_completion_ticks": _decimal_text(total_completion),
        "job_count": int(job_count),
        "schedule_fingerprint": fingerprint,
    }


def exact_relation(
    ours: Mapping[str, Any], baseline: Mapping[str, Any]
) -> str:
    if ours["schedule_fingerprint"] == baseline["schedule_fingerprint"]:
        return "same_action"
    ours_cost = (
        _decimal(ours["makespan_ticks"]),
        _decimal(ours["sum_completion_ticks"]),
    )
    baseline_cost = (
        _decimal(baseline["makespan_ticks"]),
        _decimal(baseline["sum_completion_ticks"]),
    )
    ours_weak = all(left <= right for left, right in zip(ours_cost, baseline_cost))
    ours_strict = any(left < right for left, right in zip(ours_cost, baseline_cost))
    baseline_weak = all(
        right <= left for left, right in zip(ours_cost, baseline_cost)
    )
    baseline_strict = any(
        right < left for left, right in zip(ours_cost, baseline_cost)
    )
    if ours_weak and ours_strict:
        return "ours_dominates"
    if baseline_weak and baseline_strict:
        return "baseline_dominates"
    if ours_cost == baseline_cost:
        return "metric_tie_distinct_action"
    return "tradeoff"


def build_disposition(
    *,
    family_path: Path = FAMILY_ARTIFACT,
    hurink_compact_path: Path = HURINK_COMPACT,
    hurink_full_path: Path = HURINK_FULL,
) -> dict[str, Any]:
    source_paths = {
        "family": Path(family_path),
        "hurink_compact": Path(hurink_compact_path),
        "hurink_full": Path(hurink_full_path),
    }
    source_hashes = {name: _file_sha256(path) for name, path in source_paths.items()}
    expected = EXPECTED_SHA256 if source_paths == {
        "family": FAMILY_ARTIFACT,
        "hurink_compact": HURINK_COMPACT,
        "hurink_full": HURINK_FULL,
    } else source_hashes
    mismatches = {
        name: {"expected": expected[name], "actual": source_hashes[name]}
        for name in source_hashes
        if source_hashes[name] != expected[name]
    }
    if mismatches:
        raise FJSPNumericDispositionError(f"source artifact hash mismatch: {mismatches}")

    family = _load_json(source_paths["family"])
    hurink_compact = _load_json(source_paths["hurink_compact"])
    with gzip.open(source_paths["hurink_full"], "rt", encoding="utf-8") as handle:
        hurink_full = json.load(handle)
    _validate_source_artifacts(family, hurink_compact, hurink_full)

    groups = {
        "first_family_holdout": family["rows"],
        "hurink_confirmation": hurink_full["rows"],
    }
    reports = {}
    all_false_dominance = []
    for group, rows in groups.items():
        audited = [_audit_row(row) for row in rows]
        reports[group] = {
            "instance_count": len(audited),
            "old_pareto_nondominated_count": sum(
                bool(row["old_ours_pareto_nondominated"]) for row in audited
            ),
            "exact_ledger_pareto_nondominated_count": sum(
                bool(row["exact_ledger_pareto_nondominated"]) for row in audited
            ),
            "numeric_false_dominance_count": sum(
                bool(row["numeric_false_dominance"]) for row in audited
            ),
            "rows": audited,
        }
        all_false_dominance.extend(
            {"group": group, **row}
            for row in audited
            if row["numeric_false_dominance"]
        )

    ready = bool(
        len(all_false_dominance) == 2
        and reports["first_family_holdout"]["exact_ledger_pareto_nondominated_count"]
        == 20
        and reports["hurink_confirmation"]["exact_ledger_pareto_nondominated_count"]
        == 20
    )
    return {
        "schema_version": "scheduleurm.fjsp_numeric_disposition.v1",
        "gate": {
            "pass": ready,
            "status": "FJSP_NUMERIC_DISPOSITION_PASS" if ready else "FJSP_NUMERIC_DISPOSITION_FAIL",
            "source_protocol_artifacts_immutable": True,
            "protocol_integrity_separate_from_performance": True,
            "exact_completion_ledger_ready": ready,
            "global_fjsp_optimality_claim_ready": False,
        },
        "source_sha256": source_hashes,
        "groups": reports,
        "numeric_false_dominance_rows": all_false_dominance,
        "claim_boundary": (
            "This disposition corrects only two baseline-union Pareto relations "
            "whose stored schedules are identical but whose mean-flow values were "
            "rounded at different precisions. It does not alter either prospective "
            "source artifact and does not remove the separate CP-SAT candidate-family gap."
        ),
    }


def _audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = row["result"]
    job_count = int(result["instance"]["job_count"])
    ours = exact_completion_ledger(result["ours"]["schedule"], job_count=job_count)
    relations = {}
    ledgers = {}
    for policy, run in result["baselines"].items():
        ledger = exact_completion_ledger(run["schedule"], job_count=job_count)
        ledgers[policy] = ledger
        relations[policy] = exact_relation(ours, ledger)
    exact_nondominated = not any(
        relation == "baseline_dominates" for relation in relations.values()
    )
    old_nondominated = bool(result["ours_pareto_nondominated"])
    old_dominators = sorted(
        policy
        for policy, relation in result["baseline_relations"].items()
        if relation == "baseline_dominates"
    )
    corrected_same_actions = sorted(
        policy for policy in old_dominators if relations.get(policy) == "same_action"
    )
    return {
        "relative_path": str(row["relative_path"]),
        "old_ours_pareto_nondominated": old_nondominated,
        "exact_ledger_pareto_nondominated": exact_nondominated,
        "old_baseline_dominators": old_dominators,
        "corrected_same_action_policies": corrected_same_actions,
        "numeric_false_dominance": bool(
            not old_nondominated
            and exact_nondominated
            and old_dominators
            and set(old_dominators) == set(corrected_same_actions)
        ),
        "ours_exact_ledger": ours,
        "exact_relations": relations,
        "baseline_exact_ledgers": ledgers,
    }


def _validate_source_artifacts(
    family: Mapping[str, Any],
    hurink_compact: Mapping[str, Any],
    hurink_full: Mapping[str, Any],
) -> None:
    if family.get("schema_version") != "scheduleurm.fjsp_family_holdout.v1":
        raise FJSPNumericDispositionError("unexpected first-family schema")
    if hurink_compact.get("schema_version") != "scheduleurm.fjsp_hurink_holdout.v1.compact.v1":
        raise FJSPNumericDispositionError("unexpected Hurink compact schema")
    if hurink_full.get("schema_version") != "scheduleurm.fjsp_hurink_holdout.v1":
        raise FJSPNumericDispositionError("unexpected Hurink full schema")
    if len(family.get("rows") or ()) != 20 or len(hurink_full.get("rows") or ()) != 20:
        raise FJSPNumericDispositionError("expected two 20-instance artifacts")
    if hurink_compact.get("full_artifact_gzip_sha256") != _file_sha256(HURINK_FULL):
        raise FJSPNumericDispositionError("Hurink compact/full hash binding failed")


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# FJSP Numeric Disposition Gate",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Frozen source artifacts rewritten: `false`",
        "- Comparison coordinates: `(makespan_ticks, sum_completion_ticks)`",
        "",
        "| Holdout | Old nondominated | Exact-ledger nondominated | False dominance |",
        "|---|---:|---:|---:|",
    ]
    for label, group in report["groups"].items():
        count = int(group["instance_count"])
        lines.append(
            f"| `{label}` | {group['old_pareto_nondominated_count']}/{count} | "
            f"{group['exact_ledger_pareto_nondominated_count']}/{count} | "
            f"{group['numeric_false_dominance_count']} |"
        )
    lines.extend(["", "## Dispositions", ""])
    for row in report["numeric_false_dominance_rows"]:
        lines.append(
            f"- `{row['group']}:{row['relative_path']}`: old dominator(s) "
            f"`{','.join(row['old_baseline_dominators'])}` have the same exact "
            "schedule fingerprint and are reclassified as `same_action`."
        )
    lines.extend(["", str(report["claim_boundary"]), ""])
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any], *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise FJSPNumericDispositionError(f"invalid numeric value {value!r}") from exc
    if not result.is_finite():
        raise FJSPNumericDispositionError("nonfinite schedule value")
    return result


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    return "0" if text in {"-0", ""} else text


def _file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FJSPNumericDispositionError(f"expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=Path, default=FAMILY_ARTIFACT)
    parser.add_argument("--hurink-compact", type=Path, default=HURINK_COMPACT)
    parser.add_argument("--hurink-full", type=Path, default=HURINK_FULL)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_disposition(
        family_path=args.family,
        hurink_compact_path=args.hurink_compact,
        hurink_full_path=args.hurink_full,
    )
    write_outputs(report, json_path=args.output, markdown_path=args.markdown)
    print(json.dumps(report["gate"], indent=2, sort_keys=True))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
