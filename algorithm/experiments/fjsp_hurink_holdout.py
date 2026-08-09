"""Prospective Hurink-family FJSP confirmation after the v1 audit failure.

The v1 artifact is retained unchanged.  This runner uses untouched Hurink
subfamilies and separates source/policy protocol validity, empirical Pareto
performance, and time-bounded CP-SAT reference coverage.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_family_holdout import evaluate_instance
from algorithm.experiments.fjsp_instances import load_fjsp_instance


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "tests" / "data" / "fjsp_hurink_external_holdout"
DEFAULT_PREREGISTRATION = DEFAULT_ROOT / "preregistration.json"
DEFAULT_MANIFEST = DEFAULT_ROOT / "source_manifest.json"
DEFAULT_FULL = (
    REPO_ROOT / "md" / "experiment_artifacts" / "fjsp_hurink_holdout_20260809.json.gz"
)
DEFAULT_COMPACT = (
    REPO_ROOT / "md" / "experiment_artifacts" / "fjsp_hurink_holdout_20260809.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "fjsp_hurink_holdout_20260809.md"
MATCHED_REFERENCE_FLOOR_S = 0.05
SCHEMA_VERSION = "scheduleurm.fjsp_hurink_holdout.v1"


class FJSPHurinkHoldoutError(ValueError):
    """Raised when the prospective Hurink holdout contract fails closed."""


def validate_frozen_suite(
    *,
    suite_root: str | Path = DEFAULT_ROOT,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(suite_root)
    preregistration_file = Path(preregistration_path)
    manifest_file = Path(manifest_path)
    preregistration = _load_json(preregistration_file)
    manifest = _load_json(manifest_file)
    if preregistration.get("schema_version") != (
        "scheduleurm.fjsp_hurink_holdout.preregistration.v1"
    ):
        raise FJSPHurinkHoldoutError("unexpected preregistration schema")
    if manifest.get("schema_version") != "scheduleurm.fjsp_hurink_holdout.source.v1":
        raise FJSPHurinkHoldoutError("unexpected source manifest schema")
    prereg_hash = _file_sha256(preregistration_file)
    if manifest.get("preregistration_sha256") != prereg_hash:
        raise FJSPHurinkHoldoutError("source manifest does not bind preregistration")
    relative = preregistration_file.resolve().relative_to(REPO_ROOT.resolve())
    prereg_commit = _git_output(
        ["log", "-1", "--format=%H", "--", relative.as_posix()]
    ).decode().strip()
    if len(prereg_commit) != 40:
        raise FJSPHurinkHoldoutError("preregistration is not committed")
    if sha256(_git_output(["show", f"{prereg_commit}:{relative.as_posix()}"])).hexdigest() != prereg_hash:
        raise FJSPHurinkHoldoutError("committed preregistration differs from disk")
    for output in preregistration.get("outputs") or ():
        path = Path(str(output))
        if path.is_absolute() or ".." in path.parts:
            raise FJSPHurinkHoldoutError(f"unsafe output path: {output}")
        if subprocess.run(
            ["git", "cat-file", "-e", f"{prereg_commit}:{path.as_posix()}"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            raise FJSPHurinkHoldoutError(f"result predates preregistration: {output}")

    freeze = preregistration.get("implementation_freeze") or {}
    freeze_commit = str(freeze.get("repository_commit") or "")
    if len(freeze_commit) != 40:
        raise FJSPHurinkHoldoutError("invalid implementation freeze commit")
    verified_implementation = []
    for source, expected in sorted((freeze.get("implementation_sha256") or {}).items()):
        source_path = Path(str(source))
        current = _file_sha256(REPO_ROOT / source_path)
        committed = sha256(
            _git_output(["show", f"{freeze_commit}:{source_path.as_posix()}"])
        ).hexdigest()
        if current != expected or committed != expected:
            raise FJSPHurinkHoldoutError(f"implementation drifted: {source}")
        verified_implementation.append({"path": source, "sha256": expected})
    if not verified_implementation:
        raise FJSPHurinkHoldoutError("empty implementation freeze")

    selected = tuple(str(row) for row in preregistration["selection"]["members"])
    rows = manifest.get("instances")
    if not isinstance(rows, list) or tuple(
        str(row.get("relative_path")) for row in rows
    ) != selected:
        raise FJSPHurinkHoldoutError("manifest differs from registered selection")
    verified = []
    for row in rows:
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise FJSPHurinkHoldoutError(f"unsafe source path: {relative_path}")
        source = root / relative_path
        if _file_sha256(source) != str(row["sha256"]):
            raise FJSPHurinkHoldoutError(f"source hash mismatch: {relative_path}")
        instance = load_fjsp_instance(source)
        shape = {
            "job_count": instance.job_count,
            "machine_count": instance.machine_count,
            "operation_count": instance.operation_count,
        }
        if shape != {key: int(row[key]) for key in shape}:
            raise FJSPHurinkHoldoutError(f"source shape mismatch: {relative_path}")
        verified.append({**row, "source": source, "instance": instance})
    if len(verified) != 20:
        raise FJSPHurinkHoldoutError("Hurink holdout must contain 20 instances")
    return {
        "ready": True,
        "preregistration": preregistration,
        "preregistration_sha256": prereg_hash,
        "preregistration_commit": prereg_commit,
        "manifest_sha256": _file_sha256(manifest_file),
        "implementation": verified_implementation,
        "verified_instances": verified,
    }


def build_holdout_report(
    *,
    suite_root: str | Path = DEFAULT_ROOT,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    source = validate_frozen_suite(
        suite_root=suite_root,
        preregistration_path=preregistration_path,
        manifest_path=manifest_path,
    )
    protocol = source["preregistration"]["protocol"]
    if float(protocol["matched_reference_floor_s"]) != MATCHED_REFERENCE_FLOOR_S:
        raise FJSPHurinkHoldoutError("matched-reference floor drifted")
    rows = []
    for registered in source["verified_instances"]:
        rows.append(
            {
                "relative_path": registered["relative_path"],
                "subfamily": registered["subfamily"],
                "result": evaluate_instance(
                    registered["instance"],
                    lower_bound=float(registered["lower_bound"]),
                    upper_bound=float(registered["upper_bound"]),
                    run_cp_sat=bool(protocol["run_cp_sat_reference"]),
                    fixed_reference_budget_s=float(protocol["fixed_reference_budget_s"]),
                ),
            }
        )
    protocol_pass = bool(
        rows
        and all(
            row["result"]["ours"]["selection"]["exact_generated_family_argmax"]
            for row in rows
        )
        and all(
            all(
                baseline["feasibility"]["pass"] is True
                for baseline in row["result"]["baselines"].values()
            )
            for row in rows
        )
    )
    reference_ready_count = sum(row["result"]["reference_ready"] for row in rows)
    fixed_feasible_count = sum(
        row["result"]["exact_reference"]["fixed_budget"]["feasible_solution"]
        for row in rows
    )
    fixed_optimal_count = sum(
        row["result"]["exact_reference"]["fixed_budget"]["optimality_proved"]
        for row in rows
    )
    subfamily_counts: dict[str, int] = {}
    for row in rows:
        subfamily_counts[row["subfamily"]] = subfamily_counts.get(row["subfamily"], 0) + 1
    aggregate = {
        "instance_count": len(rows),
        "subfamily_counts": subfamily_counts,
        "ours_pareto_nondominated_count": sum(
            row["result"]["ours_pareto_nondominated"] for row in rows
        ),
        "ours_strict_every_baseline_count": sum(
            row["result"]["ours_strictly_dominates_every_baseline"] for row in rows
        ),
        "geomean_ours_makespan_over_public_upper_bound": _geomean(
            row["result"]["ours_makespan_over_public_upper_bound"] for row in rows
        ),
        "reference_ready_count": reference_ready_count,
        "fixed_budget_cp_sat_feasible_count": fixed_feasible_count,
        "fixed_budget_cp_sat_optimal_count": fixed_optimal_count,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "preregistration_sha256": source["preregistration_sha256"],
            "preregistration_commit": source["preregistration_commit"],
            "manifest_sha256": source["manifest_sha256"],
            "implementation": source["implementation"],
        },
        "protocol": protocol,
        "rows": rows,
        "aggregate": aggregate,
        "gate": {
            "pass": protocol_pass,
            "status": (
                "FJSP_HURINK_HOLDOUT_PASS"
                if protocol_pass
                else "FJSP_HURINK_HOLDOUT_FAIL"
            ),
            "source_and_policy_protocol_ready": protocol_pass,
            "complete_cp_sat_reference_coverage_ready": reference_ready_count == len(rows),
            "performance_nondominance_ready": bool(
                rows and aggregate["ours_pareto_nondominated_count"] == len(rows)
            ),
            "performance_superiority_claim_ready": bool(
                rows and aggregate["ours_strict_every_baseline_count"] == len(rows)
            ),
            "global_fjsp_optimality_claim_ready": False,
            "stochastic_stability_transfer_claim_ready": False,
        },
        "claim_boundary": {
            "protocol_does_not_depend_on_performance": True,
            "protocol_does_not_depend_on_cp_sat_returning_a_solution": True,
            "cp_sat_unknown_is_retained_as_missing_reference_coverage": True,
            "scope": "registered Hurink subfamilies and instances only",
        },
    }


def compact_certificate(report: Mapping[str, Any], full_hash: str) -> dict[str, Any]:
    compact = {
        "schema_version": f"{SCHEMA_VERSION}.compact.v1",
        "gate": report["gate"],
        "full_artifact_path": str(DEFAULT_FULL.relative_to(REPO_ROOT)),
        "full_artifact_gzip_sha256": full_hash,
        "source": report["source"],
        "protocol": report["protocol"],
        "aggregate": report["aggregate"],
        "rows": [
            {
                "relative_path": row["relative_path"],
                "subfamily": row["subfamily"],
                "instance": row["result"]["instance"]["name"],
                "ours_makespan_over_public_upper_bound": row["result"]["ours_makespan_over_public_upper_bound"],
                "ours_pareto_nondominated": row["result"]["ours_pareto_nondominated"],
                "ours_strictly_dominates_every_baseline": row["result"]["ours_strictly_dominates_every_baseline"],
                "reference_ready": row["result"]["reference_ready"],
                "fixed_cp_sat_status": row["result"]["exact_reference"]["fixed_budget"]["status"],
            }
            for row in report["rows"]
        ],
        "claim_boundary": report["claim_boundary"],
    }
    compact["compact_content_sha256_excluding_self"] = _digest(compact)
    return compact


def write_artifacts(
    report: Mapping[str, Any],
    *,
    full_path: str | Path = DEFAULT_FULL,
    compact_path: str | Path = DEFAULT_COMPACT,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    full = Path(full_path)
    compact = Path(compact_path)
    markdown = Path(markdown_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    full_hash = _file_sha256(full)
    certificate = compact_certificate(report, full_hash)
    compact.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(certificate), encoding="utf-8")
    return {
        "full": str(full),
        "full_sha256": full_hash,
        "compact": str(compact),
        "compact_sha256": _file_sha256(compact),
        "markdown": str(markdown),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    aggregate = compact["aggregate"]
    gate = compact["gate"]
    lines = [
        "# Prospective Hurink FJSP holdout",
        "",
        f"- Protocol status: `{gate['status']}`",
        f"- Instances: `{aggregate['instance_count']}`",
        f"- Subfamilies: `{json.dumps(aggregate['subfamily_counts'], sort_keys=True)}`",
        f"- Pareto-nondominated: `{aggregate['ours_pareto_nondominated_count']}/{aggregate['instance_count']}`",
        f"- Strict every-baseline dominance: `{aggregate['ours_strict_every_baseline_count']}/{aggregate['instance_count']}`",
        f"- Complete CP-SAT reference rows: `{aggregate['reference_ready_count']}/{aggregate['instance_count']}`",
        f"- Geometric mean makespan/public UB: `{aggregate['geomean_ours_makespan_over_public_upper_bound']:.6f}`",
        "",
        "| Subfamily | Instance | Makespan/UB | Nondominated | CP-SAT reference |",
        "|---|---|---:|---:|---:|",
    ]
    for row in compact["rows"]:
        lines.append(
            f"| {row['subfamily']} | {row['instance']} | "
            f"{row['ours_makespan_over_public_upper_bound']:.6f} | "
            f"{str(row['ours_pareto_nondominated']).lower()} | "
            f"{row['fixed_cp_sat_status']} |"
        )
    lines.extend(
        [
            "",
            "Source/policy protocol validity is independent of empirical dominance and",
            "time-bounded CP-SAT coverage. UNKNOWN solver rows remain missing references,",
            "not protocol failures or evidence of optimality.",
            "",
        ]
    )
    return "\n".join(lines)


def _geomean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows or any(value <= 0.0 or not math.isfinite(value) for value in rows):
        raise FJSPHurinkHoldoutError("geometric mean inputs must be positive")
    return math.exp(sum(math.log(value) for value in rows) / len(rows))


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(row)
            for key, row in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(row) for row in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FJSPHurinkHoldoutError("artifact contains non-finite value")
        return round(value, 9)
    return value


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FJSPHurinkHoldoutError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _git_output(args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_holdout_report(
        suite_root=args.suite_root,
        preregistration_path=args.preregistration,
        manifest_path=args.manifest,
    )
    outputs = write_artifacts(
        report,
        full_path=args.full,
        compact_path=args.compact,
        markdown_path=args.markdown,
    )
    print(json.dumps({"gate": report["gate"], "outputs": outputs}, indent=2))
    return 0 if report["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
