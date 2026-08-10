"""Refresh and bind the registered port trajectory ledger.

The trajectory family and all benchmark instances predate this refresh, so the
result is a reproducible registered replay, not a prospective holdout.  The
writer uses a deterministic gzip header and records the exact implementation
and input hashes used by the replay.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from algorithm.experiments.port_public_benchmark_adapter import (
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
)
from algorithm.experiments.port_trajectory_upgrade import (
    build_port_trajectory_upgrade,
    compact_certificate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_FULL_GZIP = (
    ARTIFACT_ROOT / "port_trajectory_registered_refresh_v1_20260810.json.gz"
)
DEFAULT_COMPACT = (
    ARTIFACT_ROOT / "port_trajectory_registered_refresh_v1_20260810.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_trajectory_registered_refresh_v1_20260810.md"
IMPLEMENTATION_PATHS = (
    Path("algorithm/experiments/port_trajectory_upgrade.py"),
    Path("algorithm/experiments/port_public_benchmark_adapter.py"),
    Path("algorithm/experiments/port_scheduling_benchmark.py"),
    Path("algorithm/experiments/port_scheduling_instances.py"),
)


class PortTrajectoryRefreshError(ValueError):
    """Raised when the registered replay or its provenance is invalid."""


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _without_observed_runtime(value: Any) -> Any:
    """Remove wall-clock telemetry while preserving every scheduling result."""
    if isinstance(value, Mapping):
        return {
            str(key): _without_observed_runtime(row)
            for key, row in sorted(value.items(), key=lambda item: str(item[0]))
            if "runtime_seconds_observed" not in str(key)
        }
    if isinstance(value, (list, tuple)):
        return [_without_observed_runtime(row) for row in value]
    return value


def _validate_full_report(report: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "scheduleurm.port_trajectory_upgrade.v2",
        "status": "PORT_TRAJECTORY_UPGRADE_PASS",
        "pass": True,
        "trajectory_family_size": 31,
        "evaluated_instance_count": 4,
        "evaluated_trajectory_instance_pairs": 124,
        "all_candidate_results_feasible": True,
        "trajectory_oracle_gap_alpha0": 0.0,
        "trajectory_oracle_gap_alpha1": 0.0,
        "per_instance_cherry_pick": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise PortTrajectoryRefreshError("registered trajectory protocol drifted")
    penalty = report.get("bounded_penalty_certificate") or {}
    if penalty.get("ready") is not True or penalty.get("queue_independent") is not True:
        raise PortTrajectoryRefreshError("bounded penalty certificate is open")
    synthetic = report.get("synthetic_four_resource") or {}
    if synthetic.get("all_three_migration_analogues_in_candidate_family") is not True:
        raise PortTrajectoryRefreshError("synthetic migration action family is incomplete")
    if synthetic.get("ours_pareto_nondominated_on_every_instance") is not True:
        raise PortTrajectoryRefreshError("registered synthetic Pareto result drifted")
    public = report.get("public_bacasp_s") or {}
    if public.get("source_core_feasibility_ready") is not True:
        raise PortTrajectoryRefreshError("public BACASP-S source-core gate is open")


def build_refresh_artifact(
    fixture: str | Path = DEFAULT_FIXTURE,
    manifest: str | Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the full deterministic ledger and its provenance-bound summary."""
    full = build_port_trajectory_upgrade(fixture, manifest)
    _validate_full_report(full)
    compact = compact_certificate(full)
    source_hashes = {
        path.as_posix(): _file_sha256(REPO_ROOT / path)
        for path in IMPLEMENTATION_PATHS
    }
    source_hashes[str(Path(fixture).resolve())] = _file_sha256(fixture)
    source_hashes[str(Path(manifest).resolve())] = _file_sha256(manifest)
    instance_rows = compact["synthetic_four_resource"]["instances"]
    ours_nondominated = sum(
        "trajectory_robust_global" in row["pareto"]["nondominated_policies"]
        for row in instance_rows
    )
    summary: dict[str, Any] = {
        "schema_version": "scheduleurm.port_trajectory_registered_refresh.compact.v1",
        "gate": {
            "pass": True,
            "status": "PORT_TRAJECTORY_REGISTERED_REFRESH_PASS",
            "registered_replay_ready": True,
            "prospective_holdout_claim_ready": False,
            "physical_port_claim_ready": False,
            "global_port_optimality_claim_ready": False,
        },
        "protocol": {
            "chronology": "registered instances and trajectory family predate this refresh",
            "replay_type": "retrospective deterministic registered replay",
            "implementation_and_input_sha256": source_hashes,
            "gzip_header": "mtime=0 and no FNAME field",
            "full_ledger_normalization": (
                "exclude only nondeterministic keys containing runtime_seconds_observed"
            ),
        },
        "ledger": compact,
        "result_summary": {
            "trajectory_family_size": full["trajectory_family_size"],
            "evaluated_context_count": full["evaluated_instance_count"],
            "trajectory_context_pair_count": full[
                "evaluated_trajectory_instance_pairs"
            ],
            "exact_generated_family_oracle_gap_alpha0": full[
                "trajectory_oracle_gap_alpha0"
            ],
            "all_candidate_results_feasible": full[
                "all_candidate_results_feasible"
            ],
            "synthetic_instance_count": len(instance_rows),
            "ours_synthetic_pareto_nondominated_count": int(ours_nondominated),
            "ours_synthetic_pareto_nondominated_total": len(instance_rows),
            "migration_analogues_in_candidate_family": compact[
                "synthetic_four_resource"
            ]["migration_semantics_preserved"],
            "public_source_core_ours_pareto_nondominated": compact[
                "public_bacasp_s"
            ]["ours_source_core_pareto_nondominated"],
        },
        "claim_boundary": {
            "supports": [
                "current-code deterministic reproduction of the registered 31-candidate trajectory family",
                "one locked global plan over one public and three synthetic port contexts",
                "exact generated-family selection and feasibility for all 124 trajectory-context pairs",
                "registered four-resource re-berthing, crane-reassignment, and yard-rehandle semantics",
            ],
            "does_not_support": [
                "a prospective port holdout",
                "measured operation or safety at a physical terminal",
                "unrestricted port optimality or universal policy superiority",
                "stochastic port stability from deterministic replay alone",
            ],
        },
    }
    summary["artifact_sha256_excluding_self"] = sha256(
        _canonical_bytes(summary)
    ).hexdigest()
    return full, summary


def _write_deterministic_gzip(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as zipped:
            zipped.write(raw)


def markdown_report(summary: Mapping[str, Any]) -> str:
    result = summary["result_summary"]
    ledger = summary["ledger"]
    lines = [
        "# Port trajectory registered refresh v1",
        "",
        f"- Status: `{summary['gate']['status']}`",
        "- Chronology: registered retrospective replay; this is not a prospective holdout.",
        f"- Full semantic ledger hash: `{ledger['full_artifact_hash']}`",
        f"- Candidate/context evaluations: `{result['trajectory_context_pair_count']}`.",
        f"- Generated-family oracle gap: `{result['exact_generated_family_oracle_gap_alpha0']}`.",
        "",
        "| Scope | Registered evidence | Result |",
        "|---|---:|---|",
        (
            "| Public BACASP-S source core | 1 context | ours Pareto nondominated: "
            f"`{str(result['public_source_core_ours_pareto_nondominated']).lower()}` |"
        ),
        (
            f"| Synthetic berth/quay/yard/gate | {result['synthetic_instance_count']} contexts | "
            f"ours Pareto nondominated {result['ours_synthetic_pareto_nondominated_count']}/"
            f"{result['ours_synthetic_pareto_nondominated_total']} |"
        ),
        (
            "| Reconfiguration family | 3 analogues | `"
            + "`, `".join(result["migration_analogues_in_candidate_family"])
            + "` |"
        ),
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Supports: {row}." for row in summary["claim_boundary"]["supports"])
    lines.extend(
        f"- Does not support: {row}."
        for row in summary["claim_boundary"]["does_not_support"]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(
    *,
    full_output: str | Path = DEFAULT_FULL_GZIP,
    compact_output: str | Path = DEFAULT_COMPACT,
    markdown_output: str | Path = DEFAULT_MARKDOWN,
    force: bool = False,
) -> dict[str, Any]:
    outputs = [Path(full_output), Path(compact_output), Path(markdown_output)]
    if not force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise PortTrajectoryRefreshError(f"refusing to overwrite artifacts: {existing}")
    full, summary = build_refresh_artifact()
    normalized_full = _without_observed_runtime(full)
    full_raw = (
        json.dumps(
            normalized_full,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _write_deterministic_gzip(outputs[0], full_raw)
    summary["full_gzip_sha256"] = _file_sha256(outputs[0])
    summary["full_decompressed_sha256"] = sha256(full_raw).hexdigest()
    summary["artifact_sha256_excluding_self"] = sha256(
        _canonical_bytes(
            {
                key: value
                for key, value in summary.items()
                if key != "artifact_sha256_excluding_self"
            }
        )
    ).hexdigest()
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text(markdown_report(summary), encoding="ascii")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-output", type=Path, default=DEFAULT_FULL_GZIP)
    parser.add_argument("--compact-output", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = write_artifacts(
        full_output=args.full_output,
        compact_output=args.compact_output,
        markdown_output=args.markdown_output,
        force=args.force,
    )
    print(json.dumps(summary["gate"], sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PortTrajectoryRefreshError",
    "build_refresh_artifact",
    "markdown_report",
    "write_artifacts",
]
