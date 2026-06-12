"""Reviewer-facing fabric-cover metric contract certificate.

This certificate wraps the measured finite-slice fabric calibration with the
explicit objects reviewers ask for: feature map, metric weights, projection
population, rho, L, Lrho, and exclusions.  It does not claim all future cluster
states are covered; it certifies the declared measured finite population and
reports the extra profiling required for broader claims.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .global_fabric_cover_calibration import build_global_fabric_cover_calibration


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


FINITE_FEATURE_KEYS = (
    "class_key",
    "regime_key",
    "post_count_bucket",
    "post_vram_bucket",
    "util_bucket",
    "resource_bucket",
    "task_kind",
)

NUMERIC_FEATURE_SCALES = {
    "post_vram_frac": 1.0,
    "used_vram_frac": 1.0,
    "util_pct": 100.0,
    "running_task_count": 8.0,
    "post_task_count": 8.0,
    "legacy_runtime_s": 3600.0,
}


def build_fabric_cover_contract_certificate() -> dict[str, Any]:
    calibration = build_global_fabric_cover_calibration()
    rows = []
    for row in calibration.get("rows") or []:
        exact = row.get("exact_cover") or {}
        representative = row.get("representative_cover") or {}
        curve = row.get("greedy_cover_curve") or []
        rows.append({
            "taskset": row.get("taskset"),
            "cover_population": "declared_measured_finite_action_slice",
            "full_action_count": row.get("full_action_count"),
            "profile_domains": row.get("profile_domains"),
            "projection_exact": {
                "projection": "identity_on_measured_action",
                "rho": exact.get("rho"),
                "L": row.get("L"),
                "Lrho": exact.get("Lrho"),
                "candidate_count": exact.get("candidate_count"),
                "slack_ready": float(exact.get("Lrho") or 0.0) == 0.0,
            },
            "projection_single_representative": {
                "projection": "nearest_to_calibrated_candidate_profile",
                "rho": representative.get("rho"),
                "L": row.get("L"),
                "Lrho": representative.get("Lrho"),
                "candidate_count": representative.get("candidate_count"),
                "worst_action_id": representative.get("worst_action_id"),
            },
            "best_nontrivial_cover": _best_nontrivial_cover(curve),
            "worst_perturbation": row.get("worst_perturbation"),
            "zero_metric_violation_count": row.get("zero_metric_violation_count"),
            "pair_count": row.get("perturbation_pair_count"),
            "exact_measured_slice_usable_for_theorem": row.get("exact_measured_slice_usable_for_theorem"),
        })
    return {
        "gate": "fabric_cover_metric_contract_certificate",
        "feature_map": {
            "finite_keys": list(FINITE_FEATURE_KEYS),
            "numeric_scales": dict(NUMERIC_FEATURE_SCALES),
            "metric": (
                "Hamming distance over finite keys plus clipped scaled l1 distance "
                "over numeric features, as implemented by algorithm.features.finite_feature_metric"
            ),
        },
        "projection_contract": {
            "exact_projection": "pi(a)=a on the declared measured finite action slice",
            "compressed_projection": "greedy k-center nearest-center projection on the same metric",
            "cover_population": (
                "declared q00/q01/q10/q11/hybrid measured action populations, not all future feasible states"
            ),
        },
        "calibration_artifact": "md/experiment_artifacts/global_fabric_cover_calibration.json",
        "rows": rows,
        "exact_measured_population_ready": all(
            bool(row.get("exact_measured_slice_usable_for_theorem")) for row in rows
        ),
        "general_future_population_ready": False,
        "pass": all(bool(row.get("exact_measured_slice_usable_for_theorem")) for row in rows),
        "exclusions": [
            "future unseen node/GPU regimes outside the measured finite action population",
            "unprofiled topology jumps where the finite metric has no sensitivity sample",
            "candidate generators claiming smaller covers without spending the reported Lrho",
            "statistical envelopes over future perturbation samples without a failure-probability table",
        ],
        "scope": (
            "Submission-facing metric contract for the measured finite fabric-cover claim. "
            "It closes exact measured-slice rho=0 and reports compressed-cover Lrho. "
            "It intentionally does not certify all future feasible cluster states."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    fmap = report.get("feature_map") or {}
    lines = [
        "# Fabric-Cover Metric Contract Certificate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `exact_measured_population_ready` | {str(bool(report.get('exact_measured_population_ready'))).lower()} |",
        f"| `general_future_population_ready` | {str(bool(report.get('general_future_population_ready'))).lower()} |",
        "",
        "## Feature Map",
        "",
        "| Component | Definition |",
        "|---|---|",
        f"| finite keys | `{', '.join(fmap.get('finite_keys') or [])}` |",
        f"| numeric scales | `{json.dumps(fmap.get('numeric_scales') or {}, sort_keys=True)}` |",
        f"| metric | {fmap.get('metric', '')} |",
        "",
        "## Taskset Contract Rows",
        "",
        "| Taskset | Actions | L | exact rho | exact Lrho | best compressed k | best compressed rho | best compressed Lrho | Exact usable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        exact = row.get("projection_exact") or {}
        best = row.get("best_nontrivial_cover") or {}
        lines.append(
            "| `{taskset}` | {actions} | {L:.9f} | {rho:.9f} | {Lrho:.9f} | {k} | {brho:.9f} | {bLrho:.9f} | {usable} |".format(
                taskset=row.get("taskset"),
                actions=int(row.get("full_action_count") or 0),
                L=float(exact.get("L") or 0.0),
                rho=float(exact.get("rho") or 0.0),
                Lrho=float(exact.get("Lrho") or 0.0),
                k=int(best.get("candidate_count") or 0),
                brho=float(best.get("rho") or 0.0),
                bLrho=float(best.get("Lrho") or 0.0),
                usable=str(bool(row.get("exact_measured_slice_usable_for_theorem"))).lower(),
            )
        )
    lines.extend([
        "",
        "## Exclusions",
        "",
    ])
    for item in report.get("exclusions") or []:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _best_nontrivial_cover(curve: list[Mapping[str, Any]]) -> dict[str, Any]:
    nonzero = [
        row for row in curve
        if int(row.get("candidate_count") or 0) > 1 and float(row.get("rho") or 0.0) > 0.0
    ]
    if not nonzero:
        return {}
    return min(nonzero, key=lambda row: (float(row.get("Lrho") or 0.0), int(row.get("candidate_count") or 0)))


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_fabric_cover_contract_certificate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.fabric_cover_contract_certificate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build reviewer-facing fabric-cover metric contract")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "fabric_cover_contract_certificate_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "fabric_cover_contract_certificate_20260612.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
