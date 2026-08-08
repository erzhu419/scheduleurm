"""Hash-locked official PSPLIB multi-mode suite benchmark sidecar.

The suite layer validates immutable source metadata before delegating each
official ``.mm`` instance to the strict parser and deterministic scheduling
policies in :mod:`mmrcpsp_instances` and :mod:`mmrcpsp_benchmark`.  Feasibility
is a gate.  Pareto and dominance fields are descriptive and never determine
whether the benchmark passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Mapping, Sequence

from algorithm.experiments.mmrcpsp_benchmark import (
    OURS_POLICY,
    POLICY_ORDER,
    build_mmrcpsp_benchmark,
)
from algorithm.experiments.mmrcpsp_instances import parse_psplib_mm


SCHEMA_VERSION = "scheduleurm.psplib.mm_suite_benchmark.v1"
MANIFEST_SCHEMA_VERSION = "scheduleurm.psplib.mm_source_manifest.v1"
OFFICIAL_J10_MM_INSTANCE_COUNT = 536
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SuiteIntegrityError(ValueError):
    """Raised when a suite cannot be tied to its declared immutable source."""


def build_psplib_mm_suite_benchmark(
    manifest_path: str | Path,
    *,
    instance_root: str | Path | None = None,
    policies: Sequence[str] = POLICY_ORDER,
) -> dict[str, Any]:
    """Validate and benchmark every instance registered by ``manifest_path``.

    The manifest is authoritative.  Missing, modified, duplicate, or
    unregistered ``.mm`` files fail closed before any policy result is emitted.
    """

    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    root = (
        Path(instance_root)
        if instance_root is not None
        else manifest_file.parent
    )
    verified = _verify_registered_instances(manifest, root=root)
    paths = tuple(row["path"] for row in verified)
    core = build_mmrcpsp_benchmark(paths, policies=policies)

    per_instance: dict[str, Any] = {}
    pareto_counts = {policy: 0 for policy in core["policy_order"]}
    makespan_best_counts = {policy: 0 for policy in core["policy_order"]}
    mean_flow_best_counts = {policy: 0 for policy in core["policy_order"]}
    makespan_ratios = {policy: [] for policy in core["policy_order"]}
    mean_flow_ratios = {policy: [] for policy in core["policy_order"]}
    ours_strictly_dominates_every_baseline = True

    for instance_name, row in core["instances"].items():
        policy_results = row["policy_results"]
        frontier = _pareto_frontier(policy_results, core["policy_order"])
        for policy in frontier:
            pareto_counts[policy] += 1

        feasible = {
            policy: result
            for policy, result in policy_results.items()
            if bool(result["feasible"])
        }
        if not feasible:
            best_makespan = None
            best_mean_flow = None
        else:
            best_makespan = min(
                float(result["metrics"]["makespan"])
                for result in feasible.values()
            )
            best_mean_flow = min(
                float(result["metrics"]["mean_flow_time"])
                for result in feasible.values()
            )
            for policy, result in feasible.items():
                makespan = float(result["metrics"]["makespan"])
                mean_flow = float(result["metrics"]["mean_flow_time"])
                if makespan == best_makespan:
                    makespan_best_counts[policy] += 1
                if mean_flow == best_mean_flow:
                    mean_flow_best_counts[policy] += 1
                makespan_ratios[policy].append(makespan / best_makespan)
                mean_flow_ratios[policy].append(mean_flow / best_mean_flow)

        dominance = _ours_dominance(policy_results, core["policy_order"])
        ours_strictly_dominates_every_baseline = (
            ours_strictly_dominates_every_baseline
            and dominance["ours_strictly_dominates_every_baseline"]
        )
        per_instance[instance_name] = {
            "pareto_policies": list(frontier),
            "best_makespan": best_makespan,
            "best_mean_flow_time": best_mean_flow,
            "ours_comparison": dominance,
        }

    suite_aggregate = {}
    for policy in core["policy_order"]:
        suite_aggregate[policy] = {
            **core["aggregate"][policy],
            "pareto_instance_count": pareto_counts[policy],
            "makespan_best_count_including_ties": makespan_best_counts[policy],
            "mean_flow_best_count_including_ties": mean_flow_best_counts[policy],
            "geometric_mean_makespan_ratio_to_instance_best": _geomean(
                makespan_ratios[policy]
            ),
            "geometric_mean_flow_ratio_to_instance_best": _geomean(
                mean_flow_ratios[policy]
            ),
        }

    gate_pass = bool(core["gate"]["pass"])
    all_registered = len(verified) == len(manifest["instances"])
    gate_pass = gate_pass and all_registered
    full_suite_coverage = (
        manifest["coverage"] == "complete_official_j10_multi_mode_archive"
        and len(manifest["instances"]) == OFFICIAL_J10_MM_INSTANCE_COUNT
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "official_psplib_j10_multi_mode_priority_rule_suite",
        "deterministic": True,
        "source_manifest": {
            "schema_version": manifest["schema_version"],
            "collection": manifest["collection"],
            "source_page": manifest["source_page"],
            "archive_url": manifest["archive_url"],
            "archive_sha256": manifest["archive_sha256"],
            "archive_size_bytes": manifest["archive_size_bytes"],
            "retrieved_utc": manifest["retrieved_utc"],
            "coverage": manifest["coverage"],
            "registered_instance_count": len(manifest["instances"]),
            "manifest_sha256": hashlib.sha256(
                manifest_file.read_bytes()
            ).hexdigest(),
        },
        "verified_instances": [
            {
                key: value
                for key, value in row.items()
                if key != "path"
            }
            for row in verified
        ],
        "policy_order": list(core["policy_order"]),
        "instances": core["instances"],
        "per_instance_pareto": per_instance,
        "suite_aggregate": suite_aggregate,
        "gate": {
            "status": (
                "OFFICIAL_PSPLIB_MM_SUITE_PASS"
                if gate_pass
                else "OFFICIAL_PSPLIB_MM_SUITE_FAIL"
            ),
            "pass": gate_pass,
            "source_integrity_verified": all_registered,
            "all_policy_schedules_feasible": core["gate"]
            ["all_policy_schedules_feasible"],
            "zero_resource_violation": core["gate"]["zero_resource_violation"],
            "zero_precedence_violation": core["gate"]
            ["zero_precedence_violation"],
            "dominance_not_required_for_pass": True,
        },
        "claim_boundary": {
            "official_source_scope": (
                "Only hash-registered files extracted from the declared official "
                "PSPLIB j10 multi-mode archive are evidence."
            ),
            "coverage_scope": manifest["coverage"],
            "algorithm_scope": core["claim_boundary"]["algorithm_scope"],
            "ours_strictly_dominates_every_baseline_on_every_instance": (
                ours_strictly_dominates_every_baseline
            ),
            "dominance_is_descriptive_not_a_gate": True,
            "full_psplib_j10_suite_claim_ready": full_suite_coverage,
            "exact_or_state_of_the_art_mmrcpsp_claim_ready": False,
            "strong_claim_ready": False,
            "strong_claim_blockers": [
                (
                    "The registered manifest coverage is not the complete official "
                    "j10 multi-mode archive."
                    if not full_suite_coverage
                    else "No exact optimality certificate is included."
                ),
                "The baselines are priority rules, not direct external solver binaries.",
                "No stochastic arrivals or measured industrial mode-switch costs are tested.",
            ],
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    """Render a deterministic audit report."""

    gate = report["gate"]
    source = report["source_manifest"]
    lines = [
        "# Official PSPLIB MMRCPSP Suite Benchmark",
        "",
        f"- Status: `{gate['status']}`",
        f"- Pass: `{str(bool(gate['pass'])).lower()}`",
        f"- Collection: `{source['collection']}`",
        f"- Coverage: `{source['coverage']}`",
        f"- Archive SHA-256: `{source['archive_sha256']}`",
        f"- Registered instances: `{source['registered_instance_count']}`",
        "",
        "## Per-Instance Pareto Sets",
        "",
        "| Instance | Pareto policies | Best makespan | Best mean flow | Ours dominates every baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in report["per_instance_pareto"].items():
        lines.append(
            f"| `{name}` | {', '.join(f'`{p}`' for p in row['pareto_policies'])} | "
            f"{_format(row['best_makespan'])} | "
            f"{_format(row['best_mean_flow_time'])} | "
            f"{str(bool(row['ours_comparison']['ours_strictly_dominates_every_baseline'])).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Suite Aggregates",
            "",
            "| Policy | Feasible | Mean makespan | Mean flow | Pareto count | GM makespan / best | GM flow / best |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    instance_count = len(report["per_instance_pareto"])
    for policy in report["policy_order"]:
        row = report["suite_aggregate"][policy]
        lines.append(
            f"| `{policy}` | {row['feasible_instance_count']}/{instance_count} | "
            f"{_format(row['mean_makespan'])} | "
            f"{_format(row['mean_flow_time'])} | "
            f"{row['pareto_instance_count']} | "
            f"{_format(row['geometric_mean_makespan_ratio_to_instance_best'])} | "
            f"{_format(row['geometric_mean_flow_ratio_to_instance_best'])} |"
        )

    boundary = report["claim_boundary"]
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- Official source scope: {boundary['official_source_scope']}",
            f"- Coverage scope: `{boundary['coverage_scope']}`",
            (
                "- Ours strictly dominates every baseline on every instance: "
                f"`{str(bool(boundary['ours_strictly_dominates_every_baseline_on_every_instance'])).lower()}`"
            ),
            "- Dominance is descriptive and is not a pass gate.",
            "- Exact or state-of-the-art MMRCPSP claim ready: `false`",
        ]
    )
    for blocker in boundary["strong_claim_blockers"]:
        lines.append(f"- Blocker: {blocker}")
    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Write deterministic JSON and Markdown artifacts atomically."""

    _atomic_write(
        Path(json_path),
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(Path(markdown_path), markdown_report(report))


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SuiteIntegrityError(f"cannot read source manifest: {path}") from exc
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SuiteIntegrityError(f"invalid JSON source manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise SuiteIntegrityError("source manifest must be a JSON object")
    required = {
        "schema_version",
        "collection",
        "source_page",
        "archive_url",
        "archive_sha256",
        "archive_size_bytes",
        "retrieved_utc",
        "coverage",
        "instances",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise SuiteIntegrityError(f"source manifest missing fields: {missing}")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise SuiteIntegrityError("unsupported source manifest schema")
    if manifest["collection"] != "PSPLIB j10 multi-mode":
        raise SuiteIntegrityError("manifest collection is not PSPLIB j10 multi-mode")
    if not str(manifest["source_page"]).startswith("https://"):
        raise SuiteIntegrityError("source_page must be HTTPS")
    if not str(manifest["archive_url"]).startswith("https://"):
        raise SuiteIntegrityError("archive_url must be HTTPS")
    _validate_sha256(manifest["archive_sha256"], label="archive_sha256")
    if not isinstance(manifest["archive_size_bytes"], int) or manifest[
        "archive_size_bytes"
    ] <= 0:
        raise SuiteIntegrityError("archive_size_bytes must be a positive integer")
    if not isinstance(manifest["instances"], list) or not manifest["instances"]:
        raise SuiteIntegrityError("manifest instances must be a nonempty list")
    return manifest


def _verify_registered_instances(
    manifest: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    registered: dict[str, Mapping[str, Any]] = {}
    for raw in manifest["instances"]:
        if not isinstance(raw, dict):
            raise SuiteIntegrityError("each manifest instance must be an object")
        required = {
            "file",
            "archive_member",
            "sha256",
            "size_bytes",
            "jobs_including_dummies",
            "horizon",
            "renewable_capacities",
            "nonrenewable_capacities",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SuiteIntegrityError(f"instance entry missing fields: {missing}")
        filename = raw["file"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise SuiteIntegrityError(f"unsafe or invalid instance filename: {filename!r}")
        if not filename.lower().endswith(".mm"):
            raise SuiteIntegrityError(f"registered instance is not .mm: {filename}")
        if filename in registered:
            raise SuiteIntegrityError(f"duplicate registered instance: {filename}")
        _validate_sha256(raw["sha256"], label=f"{filename} sha256")
        registered[filename] = raw

    if not root.is_dir():
        raise SuiteIntegrityError(f"instance root is not a directory: {root}")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".mm"
    }
    expected = set(registered)
    missing_files = sorted(expected - actual)
    extra_files = sorted(actual - expected)
    if missing_files or extra_files:
        raise SuiteIntegrityError(
            "suite file set differs from manifest; "
            f"missing={missing_files}, unregistered={extra_files}"
        )

    verified = []
    for filename in sorted(registered):
        metadata = registered[filename]
        path = root / filename
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != metadata["sha256"]:
            raise SuiteIntegrityError(
                f"{filename}: SHA-256 mismatch; expected {metadata['sha256']}, got {digest}"
            )
        if len(payload) != metadata["size_bytes"]:
            raise SuiteIntegrityError(
                f"{filename}: size mismatch; expected {metadata['size_bytes']}, got {len(payload)}"
            )
        instance = parse_psplib_mm(path)
        semantic = {
            "jobs_including_dummies": instance.declared_job_count,
            "horizon": instance.horizon,
            "renewable_capacities": list(instance.renewable_capacities),
            "nonrenewable_capacities": list(instance.nonrenewable_capacities),
        }
        for key, actual_value in semantic.items():
            if metadata[key] != actual_value:
                raise SuiteIntegrityError(
                    f"{filename}: semantic metadata mismatch for {key}; "
                    f"expected {metadata[key]!r}, got {actual_value!r}"
                )
        verified.append(
            {
                "file": filename,
                "archive_member": metadata["archive_member"],
                "sha256": digest,
                "size_bytes": len(payload),
                **semantic,
                "path": path,
            }
        )
    return verified


def _pareto_frontier(
    policy_results: Mapping[str, Mapping[str, Any]],
    policy_order: Sequence[str],
) -> tuple[str, ...]:
    feasible = [
        policy for policy in policy_order if bool(policy_results[policy]["feasible"])
    ]
    frontier = []
    for candidate in feasible:
        candidate_metrics = policy_results[candidate]["metrics"]
        candidate_point = (
            float(candidate_metrics["makespan"]),
            float(candidate_metrics["mean_flow_time"]),
        )
        dominated = False
        for other in feasible:
            if other == candidate:
                continue
            other_metrics = policy_results[other]["metrics"]
            other_point = (
                float(other_metrics["makespan"]),
                float(other_metrics["mean_flow_time"]),
            )
            if (
                other_point[0] <= candidate_point[0]
                and other_point[1] <= candidate_point[1]
                and other_point != candidate_point
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


def _ours_dominance(
    policy_results: Mapping[str, Mapping[str, Any]],
    policy_order: Sequence[str],
) -> dict[str, Any]:
    ours = policy_results[OURS_POLICY]
    comparisons = {}
    strictly_all = bool(ours["feasible"])
    for policy in policy_order:
        if policy == OURS_POLICY:
            continue
        baseline = policy_results[policy]
        if not ours["feasible"] or not baseline["feasible"]:
            relation = "not_comparable_infeasible"
            strictly_dominates = False
        else:
            ours_point = (
                float(ours["metrics"]["makespan"]),
                float(ours["metrics"]["mean_flow_time"]),
            )
            baseline_point = (
                float(baseline["metrics"]["makespan"]),
                float(baseline["metrics"]["mean_flow_time"]),
            )
            weak = ours_point[0] <= baseline_point[0] and ours_point[1] <= baseline_point[1]
            strictly_dominates = weak and ours_point != baseline_point
            if strictly_dominates:
                relation = "ours_strictly_dominates"
            elif ours_point == baseline_point:
                relation = "tie"
            elif weak:
                relation = "ours_weakly_dominates"
            else:
                relation = "tradeoff_or_baseline_dominates"
        comparisons[policy] = {
            "relation": relation,
            "ours_strictly_dominates": strictly_dominates,
        }
        strictly_all = strictly_all and strictly_dominates
    return {
        "by_baseline": comparisons,
        "ours_strictly_dominates_every_baseline": strictly_all,
        "descriptive_only": True,
    }


def _geomean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric-mean inputs must be positive and finite")
    return round(math.exp(fmean(math.log(value) for value in values)), 8)


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _validate_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SuiteIntegrityError(f"{label} must be a lowercase SHA-256 digest")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instance-root", type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_psplib_mm_suite_benchmark(
        args.manifest,
        instance_root=args.instance_root,
    )
    write_artifacts(
        report,
        json_path=args.json_out,
        markdown_path=args.markdown_out,
    )
    print(json.dumps(report["gate"], sort_keys=True))
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
