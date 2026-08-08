"""Hash-locked Brandimarte Mk01-Mk15 deterministic FJSP suite gate.

The suite runner delegates parsing and schedule construction to the existing
FJSP sidecar modules.  It deliberately stores compact per-instance summaries
instead of duplicating every operation-level trace in the suite artifact.

``BKS`` means the source manifest's best-known feasible upper bound.  Matching
that value is reported as ``matched_bks`` only; this heuristic gate never turns
an unmatched BKS, or an open lower/upper-bound interval, into an optimality
claim.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.fjsp_benchmark import (
    POLICY_LABELS,
    POLICY_ORDER,
    run_benchmark,
)
from algorithm.experiments.fjsp_instances import load_fjsp_instance


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_DIR = REPO_ROOT / "tests" / "data" / "fjsp_brandimarte_suite"
DEFAULT_MANIFEST = DEFAULT_SUITE_DIR / "source_manifest.json"
DEFAULT_JSON = DEFAULT_SUITE_DIR / "fjsp_brandimarte_suite_gate.json"
DEFAULT_MARKDOWN = DEFAULT_SUITE_DIR / "fjsp_brandimarte_suite_gate.md"
EXPECTED_SOURCE_COMMIT = "e5e62a01023ff827f658255b793fd1b86e5c1707"
EXPECTED_INSTANCE_IDS = tuple(f"mk{index:02d}" for index in range(1, 16))
EXPECTED_LICENSE = "CC-BY-SA-4.0"
OBJECTIVE_KEYS = ("makespan", "mean_flow_time")


class SuiteSourceError(ValueError):
    """Raised when the pinned suite source contract does not validate."""


def load_source_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteSourceError(f"cannot read source manifest: {source}") from exc
    if not isinstance(payload, dict):
        raise SuiteSourceError("source manifest root must be an object")
    return payload


def validate_source_manifest(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    root = Path(suite_dir).resolve()
    manifest_file = Path(manifest_path).resolve()
    manifest = load_source_manifest(manifest_file)
    source = _mapping(manifest.get("source"), "source")
    license_row = _mapping(manifest.get("license"), "license")
    rows = manifest.get("instances")
    if source.get("commit") != EXPECTED_SOURCE_COMMIT:
        raise SuiteSourceError("source commit does not match the pinned suite commit")
    if license_row.get("spdx") != EXPECTED_LICENSE:
        raise SuiteSourceError("suite license must remain CC-BY-SA-4.0")
    if not isinstance(rows, list):
        raise SuiteSourceError("manifest instances must be a list")
    ids = tuple(str(row.get("id") or "") for row in rows if isinstance(row, Mapping))
    if ids != EXPECTED_INSTANCE_IDS:
        raise SuiteSourceError("manifest must contain Mk01-Mk15 exactly once and in order")

    verified: list[dict[str, Any]] = []
    for expected_id, raw_row in zip(EXPECTED_INSTANCE_IDS, rows):
        row = _mapping(raw_row, f"instance {expected_id}")
        filename = str(row.get("file") or "")
        if filename != f"{expected_id}.txt":
            raise SuiteSourceError(f"unexpected filename for {expected_id}: {filename}")
        local_path = (root / filename).resolve()
        if local_path.parent != root:
            raise SuiteSourceError(f"instance path escapes suite root: {filename}")
        if not local_path.is_file():
            raise SuiteSourceError(f"missing suite instance: {filename}")
        raw_sha256 = sha256(local_path.read_bytes()).hexdigest()
        if raw_sha256 != str(row.get("sha256") or ""):
            raise SuiteSourceError(f"SHA-256 mismatch for {expected_id}")
        lower_bound = _positive_number(row.get("lower_bound"), f"{expected_id} lower bound")
        upper_bound = _positive_number(row.get("upper_bound"), f"{expected_id} upper bound")
        if lower_bound > upper_bound:
            raise SuiteSourceError(f"lower bound exceeds upper bound for {expected_id}")
        if bool(row.get("bounds_equal")) != _close(lower_bound, upper_bound):
            raise SuiteSourceError(f"bounds_equal is inconsistent for {expected_id}")

        instance = load_fjsp_instance(local_path)
        expected_size = f"{instance.job_count} x {instance.machine_count}"
        if str(row.get("size") or "") != expected_size:
            raise SuiteSourceError(f"declared size mismatch for {expected_id}")
        verified.append(
            {
                "id": expected_id,
                "path": local_path,
                "raw_sha256": raw_sha256,
                "canonical_sha256": instance.source_sha256,
                "instance": instance,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "bounds_equal": bool(row.get("bounds_equal")),
                "status": str(row.get("status") or "unknown"),
                "source_path": str(row.get("source_path") or ""),
            }
        )

    return {
        "pass": True,
        "manifest": manifest,
        "manifest_path": manifest_file,
        "manifest_sha256": sha256(manifest_file.read_bytes()).hexdigest(),
        "verified_instances": verified,
    }


def build_fjsp_suite_gate(
    suite_dir: str | Path = DEFAULT_SUITE_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    source_gate = validate_source_manifest(suite_dir, manifest_path)
    manifest = source_gate["manifest"]
    instance_rows: list[dict[str, Any]] = []
    for source_row in source_gate["verified_instances"]:
        result = run_benchmark(
            source_row["instance"],
            known_best_makespan=source_row["upper_bound"],
        )
        instance_rows.append(_instance_summary(source_row, result))

    aggregate = _aggregate(instance_rows)
    lower_bound_violations = [
        f"{row['instance_id']}:{policy}"
        for row in instance_rows
        for policy, result in row["policies"].items()
        if bool(result["below_source_lower_bound"])
    ]
    potential_bks_improvements = [
        f"{row['instance_id']}:{policy}"
        for row in instance_rows
        for policy, result in row["policies"].items()
        if bool(result["below_bks"]) and not bool(result["below_source_lower_bound"])
    ]
    all_feasible = all(
        bool(result["feasible"])
        for row in instance_rows
        for result in row["policies"].values()
    )
    pass_gate = bool(
        source_gate["pass"]
        and len(instance_rows) == len(EXPECTED_INSTANCE_IDS)
        and all_feasible
        and not lower_bound_violations
    )
    ours_aggregate_frontier = (
        "ours_robust_maxweight"
        in aggregate["aggregate_pareto_frontier_policies"]
    )
    ours_nondominated_count = sum(
        bool(row["ours_is_pareto_nondominated"]) for row in instance_rows
    )
    ours_bks_matches = aggregate["policies"]["ours_robust_maxweight"][
        "matched_bks_count"
    ]
    return {
        "schema_version": "scheduleurm.fjsp_brandimarte_suite_gate.v1",
        "deterministic": True,
        "suite": "Brandimarte Mk01-Mk15",
        "implementation": _implementation_manifest(),
        "source": {
            "manifest_sha256": source_gate["manifest_sha256"],
            "repository": manifest["source"]["repository"],
            "commit": manifest["source"]["commit"],
            "commit_url": manifest["source"]["commit_url"],
            "instance_root": manifest["source"]["instance_root"],
            "bounds_path": manifest["source"]["bounds_path"],
            "bounds_sha256": manifest["source"]["bounds_sha256"],
            "license": manifest["license"],
            "citation": manifest["citation"],
        },
        "semantics": {
            "problem": "standard_nonpreemptive_fjsp",
            "reassignment_extension_enabled": False,
            "policies": list(POLICY_ORDER),
            "bks": "source_manifest_upper_bound",
            "bks_gap": "(heuristic_makespan_minus_bks)/bks",
            "lower_bound_gap": "(heuristic_makespan_minus_lower_bound)/lower_bound",
            "pareto_objectives": list(OBJECTIVE_KEYS),
            "matched_bks_is_not_renamed_to_optimal": True,
            "heuristic_optimality_claim": False,
        },
        "instances": instance_rows,
        "aggregate": aggregate,
        "gate": {
            "pass": pass_gate,
            "source_manifest_valid": bool(source_gate["pass"]),
            "instance_count": len(instance_rows),
            "expected_instance_count": len(EXPECTED_INSTANCE_IDS),
            "policy_run_count": len(instance_rows) * len(POLICY_ORDER),
            "all_schedules_feasible": all_feasible,
            "source_lower_bound_violations": lower_bound_violations,
            "potential_bks_improvements_requiring_external_verification": (
                potential_bks_improvements
            ),
            "unsupported_optimality_claims": [],
            "gate_scope": "source_integrity_and_schedule_feasibility",
            "performance_superiority_claim_ready": False,
            "ours_aggregate_pareto_nondominated": ours_aggregate_frontier,
            "ours_instance_pareto_nondominated_count": ours_nondominated_count,
            "ours_instance_pareto_dominated_count": (
                len(instance_rows) - ours_nondominated_count
            ),
            "ours_matched_bks_count": ours_bks_matches,
        },
        "scope": (
            "Deterministic constructive-heuristic evidence on the hash-locked "
            "Brandimarte Mk01-Mk15 suite.  BKS matches are equality to the pinned "
            "source upper bound, not an independent optimality certificate."
        ),
    }


def artifact_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def markdown_report(report: Mapping[str, Any]) -> str:
    source = report["source"]
    gate = report["gate"]
    aggregate = report["aggregate"]
    lines = [
        "# Brandimarte Mk01-Mk15 FJSP Suite Gate",
        "",
        (
            "- Source-integrity and schedule-feasibility gate: "
            f"`{str(bool(gate['pass'])).lower()}`"
        ),
        "- Performance-superiority claim ready: `false`",
        (
            "- Ours Pareto-nondominated instances: "
            f"`{gate['ours_instance_pareto_nondominated_count']}/"
            f"{gate['instance_count']}`"
        ),
        f"- Ours BKS matches: `{gate['ours_matched_bks_count']}`",
        f"- Source commit: `{source['commit']}`",
        f"- Source manifest SHA-256: `{source['manifest_sha256']}`",
        f"- License: `{source['license']['spdx']}`",
        f"- Policy runs: `{gate['policy_run_count']}`",
        "",
        (
            "BKS denotes the pinned source upper bound. A zero BKS gap means only "
            "that the heuristic matched that value; this report makes no heuristic "
            "optimality claim."
        ),
        "",
        "## Aggregate",
        "",
        (
            "| Policy | Geo. makespan/BKS | Mean makespan/BKS | Worst makespan/BKS | "
            "Geo. flow/best | BKS matches | Makespan best/tie | Flow best/tie | "
            "Pareto instances |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in report["semantics"]["policies"]:
        row = aggregate["policies"][policy]
        lines.append(
            f"| {row['policy_label']} | {_fmt(row['geomean_makespan_over_bks'])} | "
            f"{_fmt(row['mean_makespan_over_bks'])} | "
            f"{_fmt(row['max_makespan_over_bks'])} | "
            f"{_fmt(row['geomean_mean_flow_to_instance_best'])} | "
            f"{row['matched_bks_count']} | {row['makespan_best_or_tie_count']} | "
            f"{row['mean_flow_best_or_tie_count']} | {row['pareto_frontier_count']} |"
        )

    lines.extend(
        [
            "",
            "## Ours Versus Each Baseline",
            "",
            (
                "| Baseline | Geo. makespan ours/base | Geo. flow ours/base | "
                "Ours Pareto-dominates | Baseline dominates ours | Tradeoff/equal |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline, row in aggregate["ours_vs_baselines"].items():
        lines.append(
            f"| {POLICY_LABELS[baseline]} | {_fmt(row['geomean_makespan_ratio'])} | "
            f"{_fmt(row['geomean_mean_flow_ratio'])} | "
            f"{row['ours_pareto_dominates_count']} | "
            f"{row['baseline_pareto_dominates_count']} | "
            f"{row['tradeoff_or_equal_count']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Instance Summary",
            "",
            (
                "| Instance | Size | LB | BKS | Status | Ours makespan | Ours BKS gap | "
                "Ours mean flow | Best makespan | Best mean flow | Pareto frontier |"
            ),
            "|---|---:|---:|---:|---|---:|---:|---:|---|---|---|",
        ]
    )
    for row in report["instances"]:
        ours = row["policies"]["ours_robust_maxweight"]
        lines.append(
            f"| `{row['instance_id']}` | {row['job_count']}x{row['machine_count']} | "
            f"{_fmt(row['lower_bound'])} | {_fmt(row['bks'])} | `{row['source_status']}` | "
            f"{_fmt(ours['makespan'])} | {_pct(ours['bks_gap_fraction'])} | "
            f"{_fmt(ours['mean_flow_time'])} | "
            f"{_best_text(row, 'makespan')} | {_best_text(row, 'mean_flow_time')} | "
            f"{', '.join(row['pareto_frontier_policies'])} |"
        )

    lines.extend(
        [
            "",
            "## Every Policy On Every Instance",
            "",
            (
                "| Instance | Policy | Makespan | BKS gap | LB gap | Mean flow | "
                "Matched BKS | Pareto | Feasible |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["instances"]:
        frontier = set(row["pareto_frontier_policies"])
        for policy in report["semantics"]["policies"]:
            result = row["policies"][policy]
            lines.append(
                f"| `{row['instance_id']}` | {result['policy_label']} | "
                f"{_fmt(result['makespan'])} | {_pct(result['bks_gap_fraction'])} | "
                f"{_pct(result['lower_bound_gap_fraction'])} | "
                f"{_fmt(result['mean_flow_time'])} | "
                f"{str(bool(result['matched_bks'])).lower()} | "
                f"{str(policy in frontier).lower()} | "
                f"{str(bool(result['feasible'])).lower()} |"
            )

    lines.extend(["", "## Scope", "", str(report["scope"]), ""])
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(artifact_json(report), encoding="utf-8")
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def _instance_summary(
    source_row: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    lower_bound = float(source_row["lower_bound"])
    bks = float(source_row["upper_bound"])
    policy_rows: dict[str, dict[str, Any]] = {}
    for policy in POLICY_ORDER:
        result = benchmark["policies"][policy]
        metrics = result["metrics"]
        makespan = float(metrics["makespan"])
        policy_rows[policy] = {
            "policy_label": str(result["policy_label"]),
            "makespan": _number(makespan),
            "mean_flow_time": _number(metrics["mean_flow_time"]),
            "reconfiguration_count": int(metrics["reconfiguration_count"]),
            "reconfiguration_time": _number(metrics["reconfiguration_time"]),
            "makespan_over_bks": _number(makespan / bks),
            "bks_gap_absolute": _number(makespan - bks),
            "bks_gap_fraction": _number((makespan - bks) / bks),
            "lower_bound_gap_absolute": _number(makespan - lower_bound),
            "lower_bound_gap_fraction": _number(
                (makespan - lower_bound) / lower_bound
            ),
            "matched_bks": _close(makespan, bks),
            "below_bks": makespan < bks - 1e-9,
            "below_source_lower_bound": makespan < lower_bound - 1e-9,
            "reported_as_optimal": False,
            "feasible": bool(result["feasibility"]["pass"]),
        }

    frontier = _pareto_frontier(policy_rows)
    best_by_metric = {
        metric: _best_policies(policy_rows, metric) for metric in OBJECTIVE_KEYS
    }
    ours_dominated_by = [
        policy
        for policy in POLICY_ORDER
        if policy != "ours_robust_maxweight"
        and _dominates(policy_rows[policy], policy_rows["ours_robust_maxweight"])
    ]
    instance = source_row["instance"]
    return {
        "instance_id": str(source_row["id"]),
        "source_path": str(source_row["source_path"]),
        "source_raw_sha256": str(source_row["raw_sha256"]),
        "source_canonical_sha256": str(source_row["canonical_sha256"]),
        "job_count": int(instance.job_count),
        "machine_count": int(instance.machine_count),
        "operation_count": int(instance.operation_count),
        "source_status": str(source_row["status"]),
        "lower_bound": _number(lower_bound),
        "bks": _number(bks),
        "bounds_equal": bool(source_row["bounds_equal"]),
        "policies": policy_rows,
        "best_policies_by_metric": best_by_metric,
        "pareto_frontier_policies": frontier,
        "ours_pareto_dominated_by": ours_dominated_by,
        "ours_is_pareto_nondominated": "ours_robust_maxweight" in frontier,
        "matched_bks_is_not_optimality_claim": True,
    }


def _aggregate(instance_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    policy_aggregate: dict[str, dict[str, Any]] = {}
    for policy in POLICY_ORDER:
        rows = [instance["policies"][policy] for instance in instance_rows]
        makespan_ratios = [float(row["makespan_over_bks"]) for row in rows]
        normalized_flows = []
        for instance in instance_rows:
            best_flow = min(
                float(result["mean_flow_time"])
                for result in instance["policies"].values()
            )
            normalized_flows.append(
                float(instance["policies"][policy]["mean_flow_time"]) / best_flow
            )
        policy_aggregate[policy] = {
            "policy_label": POLICY_LABELS[policy],
            "instance_count": len(rows),
            "geomean_makespan_over_bks": _number(_geomean(makespan_ratios)),
            "mean_makespan_over_bks": _number(statistics.fmean(makespan_ratios)),
            "median_makespan_over_bks": _number(statistics.median(makespan_ratios)),
            "max_makespan_over_bks": _number(max(makespan_ratios)),
            "mean_bks_gap_fraction": _number(
                statistics.fmean(float(row["bks_gap_fraction"]) for row in rows)
            ),
            "geomean_mean_flow_to_instance_best": _number(_geomean(normalized_flows)),
            "matched_bks_count": sum(bool(row["matched_bks"]) for row in rows),
            "below_bks_count": sum(bool(row["below_bks"]) for row in rows),
            "feasible_count": sum(bool(row["feasible"]) for row in rows),
            "makespan_best_or_tie_count": sum(
                policy in instance["best_policies_by_metric"]["makespan"]
                for instance in instance_rows
            ),
            "mean_flow_best_or_tie_count": sum(
                policy in instance["best_policies_by_metric"]["mean_flow_time"]
                for instance in instance_rows
            ),
            "pareto_frontier_count": sum(
                policy in instance["pareto_frontier_policies"]
                for instance in instance_rows
            ),
            "reported_as_optimal_count": 0,
        }

    aggregate_frontier = _pareto_frontier(
        {
            policy: {
                "makespan": row["geomean_makespan_over_bks"],
                "mean_flow_time": row["geomean_mean_flow_to_instance_best"],
            }
            for policy, row in policy_aggregate.items()
        }
    )
    return {
        "instance_count": len(instance_rows),
        "policy_run_count": len(instance_rows) * len(POLICY_ORDER),
        "policies": policy_aggregate,
        "aggregate_pareto_frontier_policies": aggregate_frontier,
        "ours_vs_baselines": {
            baseline: _ours_vs_baseline(instance_rows, baseline)
            for baseline in POLICY_ORDER
            if baseline != "ours_robust_maxweight"
        },
    }


def _ours_vs_baseline(
    instance_rows: Sequence[Mapping[str, Any]],
    baseline: str,
) -> dict[str, Any]:
    makespan_ratios: list[float] = []
    flow_ratios: list[float] = []
    ours_dominates = 0
    baseline_dominates = 0
    tradeoff_or_equal = 0
    makespan_wins = makespan_ties = makespan_losses = 0
    flow_wins = flow_ties = flow_losses = 0
    for instance in instance_rows:
        ours = instance["policies"]["ours_robust_maxweight"]
        other = instance["policies"][baseline]
        ours_makespan = float(ours["makespan"])
        other_makespan = float(other["makespan"])
        ours_flow = float(ours["mean_flow_time"])
        other_flow = float(other["mean_flow_time"])
        makespan_ratios.append(ours_makespan / other_makespan)
        flow_ratios.append(ours_flow / other_flow)
        makespan_wins += ours_makespan < other_makespan - 1e-9
        makespan_ties += _close(ours_makespan, other_makespan)
        makespan_losses += ours_makespan > other_makespan + 1e-9
        flow_wins += ours_flow < other_flow - 1e-9
        flow_ties += _close(ours_flow, other_flow)
        flow_losses += ours_flow > other_flow + 1e-9
        if _dominates(ours, other):
            ours_dominates += 1
        elif _dominates(other, ours):
            baseline_dominates += 1
        else:
            tradeoff_or_equal += 1
    return {
        "geomean_makespan_ratio": _number(_geomean(makespan_ratios)),
        "geomean_mean_flow_ratio": _number(_geomean(flow_ratios)),
        "makespan_win_count": makespan_wins,
        "makespan_tie_count": makespan_ties,
        "makespan_loss_count": makespan_losses,
        "mean_flow_win_count": flow_wins,
        "mean_flow_tie_count": flow_ties,
        "mean_flow_loss_count": flow_losses,
        "ours_pareto_dominates_count": ours_dominates,
        "baseline_pareto_dominates_count": baseline_dominates,
        "tradeoff_or_equal_count": tradeoff_or_equal,
    }


def _pareto_frontier(policy_rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        policy
        for policy in POLICY_ORDER
        if policy in policy_rows
        and not any(
            other != policy and _dominates(policy_rows[other], policy_rows[policy])
            for other in policy_rows
        )
    ]


def _best_policies(
    policy_rows: Mapping[str, Mapping[str, Any]],
    metric: str,
) -> list[str]:
    best = min(float(row[metric]) for row in policy_rows.values())
    return [
        policy
        for policy in POLICY_ORDER
        if policy in policy_rows and _close(float(policy_rows[policy][metric]), best)
    ]


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    pairs = [(float(left[key]), float(right[key])) for key in OBJECTIVE_KEYS]
    return all(lhs <= rhs + 1e-9 for lhs, rhs in pairs) and any(
        lhs < rhs - 1e-9 for lhs, rhs in pairs
    )


def _implementation_manifest() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("fjsp_benchmark.py"),
        Path(__file__).resolve().with_name("fjsp_instances.py"),
    )
    files: list[dict[str, str]] = []
    combined = sha256()
    for path in paths:
        data = path.read_bytes()
        relative = path.relative_to(REPO_ROOT).as_posix()
        files.append({"path": relative, "sha256": sha256(data).hexdigest()})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(data)
        combined.update(b"\0")
    return {"files": files, "combined_sha256": combined.hexdigest()}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuiteSourceError(f"{label} must be an object")
    return value


def _positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SuiteSourceError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise SuiteSourceError(f"{label} must be finite and positive")
    return number


def _geomean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows or any(value <= 0.0 or not math.isfinite(value) for value in rows):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in rows))


def _number(value: Any) -> int | float:
    rounded = round(float(value), 9)
    return int(rounded) if rounded.is_integer() else rounded


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9 * max(1.0, abs(float(right)))


def _fmt(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _pct(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%"


def _best_text(row: Mapping[str, Any], metric: str) -> str:
    policies = row["best_policies_by_metric"][metric]
    value = row["policies"][policies[0]][metric]
    return f"{_fmt(value)} ({', '.join(policies)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the hash-locked Brandimarte Mk01-Mk15 FJSP suite gate."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_fjsp_suite_gate(args.suite_dir, args.manifest)
    json_path, markdown_path = write_artifacts(
        report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
