"""Execute the committed MMRCPSP renewal-stream protocol and write artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.mmrcpsp_renewal_stream_benchmark import (
    OURS_POLICY,
    POLICY_ORDER,
    default_scenarios,
    run_mmrcpsp_renewal_stream_benchmark,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "tests" / "data" / "mmrcpsp_family_external_holdout"
SOURCE_PREREGISTRATION = SOURCE_ROOT / "preregistration.json"
SOURCE_PREREGISTRATION_SHA256 = "f77acefc01f20ac362e2fede94137a8d482878a0b62afb4862c2c68edcc2e245"
SOURCE_MANIFEST = SOURCE_ROOT / "source_manifest.json"
SOURCE_MANIFEST_SHA256 = "f7cdc7065b084269df7a6e9e7cf8344db395b15b1778d3992a297f4b077f9c8c"
PREREGISTRATION = REPO_ROOT / "tests" / "data" / "mmrcpsp_renewal_stream_v1" / "preregistration.json"
PREREGISTRATION_SHA256 = "8d2fa266bbaac3fefcdf2c67e26cccc71be47839586d6e3333a64277088fe191"
IMPLEMENTATION = REPO_ROOT / "algorithm" / "experiments" / "mmrcpsp_renewal_stream_benchmark.py"
IMPLEMENTATION_SHA256 = "15818c10a5384ab2eb18e509395dcaba992cce12cce0c084010a4c07b9f5a5b3"
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "mmrcpsp_renewal_stream_v1_20260810.json.gz"
)
DEFAULT_COMPACT = DEFAULT_FULL.with_suffix("").with_suffix(".json")
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "mmrcpsp_renewal_stream_v1_20260810.md"


class MMRCPSPRenewalArtifactError(ValueError):
    """Raised when the frozen stream protocol or artifact contract fails."""


def build_artifact() -> dict[str, Any]:
    protocol = _load_verified_json(PREREGISTRATION, PREREGISTRATION_SHA256)
    source_preregistration = _load_verified_json(
        SOURCE_PREREGISTRATION, SOURCE_PREREGISTRATION_SHA256
    )
    source_manifest = _load_verified_json(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256)
    members = tuple(str(row) for row in protocol["selection"]["members"])
    if len(members) != 25 or len(members) != len(set(members)):
        raise MMRCPSPRenewalArtifactError("registered member family is not 25 unique rows")
    if set(members) != set(source_preregistration["selection"]["members"]):
        raise MMRCPSPRenewalArtifactError("stream members differ from frozen source holdout")
    manifest_rows = {
        str(row["relative_path"]): row for row in source_manifest["instances"]
    }
    if set(members) != set(manifest_rows):
        raise MMRCPSPRenewalArtifactError("source manifest membership mismatch")
    instance_paths = []
    instance_commits: dict[str, str] = {}
    for member in members:
        relative = Path(member)
        if relative.is_absolute() or ".." in relative.parts:
            raise MMRCPSPRenewalArtifactError(f"unsafe member path: {member}")
        source = SOURCE_ROOT / relative
        expected = str(manifest_rows[member]["sha256"])
        instance_commits[member] = _committed_file_contract(source, expected)
        instance_paths.append(source)

    horizon = int(protocol["protocol"]["horizon"])
    seeds = tuple(int(seed) for seed in protocol["protocol"]["seeds"])
    policies = tuple(str(policy) for policy in protocol["protocol"]["policies"])
    scenarios = default_scenarios(horizon=horizon)
    if policies != POLICY_ORDER:
        raise MMRCPSPRenewalArtifactError("registered policy order changed")
    if tuple(row.scenario_id for row in scenarios) != tuple(
        str(row) for row in protocol["protocol"]["scenarios"]
    ):
        raise MMRCPSPRenewalArtifactError("registered scenario family changed")

    wrapper = Path(__file__).resolve()
    wrapper_hash = _file_sha256(wrapper)
    execution_contract = {
        "preregistration_commit": _committed_file_contract(
            PREREGISTRATION, PREREGISTRATION_SHA256
        ),
        "source_preregistration_commit": _committed_file_contract(
            SOURCE_PREREGISTRATION, SOURCE_PREREGISTRATION_SHA256
        ),
        "source_manifest_commit": _committed_file_contract(
            SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256
        ),
        "implementation_commit": _committed_file_contract(
            IMPLEMENTATION, IMPLEMENTATION_SHA256
        ),
        "wrapper_commit": _committed_file_contract(wrapper, wrapper_hash),
        "wrapper_sha256": wrapper_hash,
        "instance_commits": instance_commits,
        "stream_results_observed_before_protocol_commit": False,
        "static_instance_results_known_before_stream_protocol": True,
    }
    report = run_mmrcpsp_renewal_stream_benchmark(
        instance_paths,
        scenarios=scenarios,
        policies=policies,
        seeds=seeds,
    )
    artifact = {
        "schema_version": "scheduleurm.mmrcpsp_renewal_stream.artifact.v1",
        "protocol": protocol,
        "source_contract": {
            "source_preregistration_sha256": SOURCE_PREREGISTRATION_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_hashes_verified": True,
            "registered_member_count": len(members),
        },
        "execution_contract": execution_contract,
        "report": report,
    }
    artifact["artifact_sha256_excluding_self"] = _digest(artifact)
    return artifact


def compact_certificate(
    artifact: Mapping[str, Any], full_gzip_sha256: str
) -> dict[str, Any]:
    report = artifact["report"]
    rows = []
    for run in report["runs"]:
        performance = run["performance"]
        stability = run["stability_diagnostics"]
        rows.append(
            {
                "run_id": run["run_id"],
                "scenario_id": run["scenario"]["scenario_id"],
                "scenario_family": run["scenario"]["family"],
                "load_factor": run["scenario"]["load_factor"],
                "seed": run["seed"],
                "policy": run["policy"],
                "completed_projects": performance["completed_projects"],
                "final_backlog": performance["final_backlog"],
                "maximum_backlog": performance["maximum_backlog"],
                "time_average_queue": performance["time_average_queue"],
                "project_throughput_per_time": performance[
                    "project_throughput_per_time"
                ],
                "nominal_capacity_slack": stability["nominal_capacity_slack"],
                "mean_observed_high_backlog_drift_rate": stability[
                    "mean_observed_high_backlog_drift_rate"
                ],
                "empirical_negative_high_backlog_drift": stability[
                    "empirical_negative_high_backlog_drift"
                ],
                "run_gate_pass": run["gate"]["pass"],
                "exact_duration_normalized_oracle": run["gate"][
                    "exact_duration_normalized_oracle"
                ],
            }
        )
    comparisons = _performance_dispositions(rows)
    compact = {
        "schema_version": "scheduleurm.mmrcpsp_renewal_stream.compact.v1",
        "gate": report["gate"],
        "aggregate": report["aggregate"],
        "library_contract": report["library_contract"],
        "arrival_contract": report["arrival_contract"],
        "execution_contract": artifact["execution_contract"],
        "source_contract": artifact["source_contract"],
        "performance_dispositions": comparisons,
        "rows": rows,
        "claim_boundary": report["claim_boundary"],
        "full_artifact_gzip_sha256": full_gzip_sha256,
    }
    compact["artifact_sha256_excluding_self"] = _digest(compact)
    return compact


def write_outputs(
    artifact: Mapping[str, Any],
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
            raise MMRCPSPRenewalArtifactError(f"refusing to overwrite result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    full_hash = _file_sha256(full)
    compact = compact_certificate(artifact, full_hash)
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
        "markdown_sha256": _file_sha256(markdown),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    aggregate = compact["aggregate"]
    disposition = compact["performance_dispositions"]
    lines = [
        "# MMRCPSP renewal-stream experiment v1",
        "",
        f"- Status: `{compact['gate']['status']}`",
        f"- Registered classes: `{compact['source_contract']['registered_member_count']}`",
        f"- Runs: `{aggregate['protocol_pass_count']}/{aggregate['run_count']}` protocol PASS",
        f"- Ours exact registered-family oracle runs: `{aggregate['ours_exact_oracle_run_count']}`",
        f"- Ours Pareto-nondominated scenario/seed rows: `{disposition['ours_nondominated_count']}/{disposition['scenario_seed_count']}`",
        f"- Ours strictly improves every baseline: `{disposition['ours_strict_every_baseline_count']}/{disposition['scenario_seed_count']}`",
        "- Costs are final backlog and time-average queue; lower is better.",
        "- Static instance outcomes were known before this stream protocol; arrival tapes and stream outcomes were not.",
        "- Finite-sample drift remains diagnostic and is not a positive-recurrence proof.",
        "",
        "| Scenario | Seed | Ours final backlog | Ours average queue | Disposition |",
        "|---|---:|---:|---:|---|",
    ]
    ours_rows = {
        (row["scenario_id"], row["seed"]): row
        for row in compact["rows"]
        if row["policy"] == OURS_POLICY
    }
    for row in disposition["rows"]:
        ours = ours_rows[(row["scenario_id"], row["seed"])]
        lines.append(
            f"| {row['scenario_id']} | {row['seed']} | "
            f"{ours['final_backlog']} | {ours['time_average_queue']} | "
            f"{row['ours_disposition']} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(compact["claim_boundary"]), ""])
    return "\n".join(lines)


def _performance_dispositions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario_id"]), int(row["seed"])), []).append(row)
    dispositions = []
    for (scenario_id, seed), group in sorted(grouped.items()):
        ours = next(row for row in group if row["policy"] == OURS_POLICY)
        baselines = [row for row in group if row["policy"] != OURS_POLICY]
        dominators = [
            str(row["policy"])
            for row in baselines
            if _cost_dominates(row, ours)
        ]
        ours_dominates = [
            str(row["policy"])
            for row in baselines
            if _cost_dominates(ours, row)
        ]
        if dominators:
            disposition = "baseline_dominates"
        elif len(ours_dominates) == len(baselines):
            disposition = "ours_strict_every_baseline"
        else:
            disposition = "ours_nondominated_tradeoff_or_tie"
        dispositions.append(
            {
                "scenario_id": scenario_id,
                "seed": seed,
                "ours_disposition": disposition,
                "baseline_dominators": sorted(dominators),
                "baselines_strictly_dominated_by_ours": sorted(ours_dominates),
            }
        )
    counts = Counter(row["ours_disposition"] for row in dispositions)
    return {
        "scenario_seed_count": len(dispositions),
        "ours_nondominated_count": len(dispositions)
        - counts["baseline_dominates"],
        "ours_strict_every_baseline_count": counts[
            "ours_strict_every_baseline"
        ],
        "baseline_dominates_count": counts["baseline_dominates"],
        "performance_is_not_protocol_gate": True,
        "rows": dispositions,
    }


def _cost_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_cost = (float(left["final_backlog"]), float(left["time_average_queue"]))
    right_cost = (float(right["final_backlog"]), float(right["time_average_queue"]))
    return all(a <= b + 1e-9 for a, b in zip(left_cost, right_cost)) and any(
        a < b - 1e-9 for a, b in zip(left_cost, right_cost)
    )


def _load_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise MMRCPSPRenewalArtifactError(f"disk hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _committed_file_contract(path: Path, expected_sha256: str) -> str:
    source = path.resolve()
    relative = source.relative_to(REPO_ROOT)
    if _file_sha256(source) != expected_sha256:
        raise MMRCPSPRenewalArtifactError(f"disk hash mismatch: {relative}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise MMRCPSPRenewalArtifactError(f"file is not committed: {relative}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if sha256(committed).hexdigest() != expected_sha256:
        raise MMRCPSPRenewalArtifactError(f"committed hash mismatch: {relative}")
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
    artifact = build_artifact()
    outputs = write_outputs(
        artifact,
        full_path=args.full,
        compact_path=args.compact,
        markdown_path=args.markdown,
    )
    gate = artifact["report"]["gate"]
    print(json.dumps({"gate": gate, "outputs": outputs}, indent=2))
    return 0 if gate["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
