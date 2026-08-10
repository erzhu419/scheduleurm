"""Execute and serialize the committed MMRCPSP class-balance holdout."""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

from algorithm.experiments.mmrcpsp_class_balance_holdout import (
    PARETO_COST_COORDINATES,
    compact_class_balance_report,
    run_class_balance_holdout,
)
from algorithm.experiments.mmrcpsp_renewal_stream_benchmark import (
    POLICY_ORDER,
    default_scenarios,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    REPO_ROOT
    / "tests"
    / "data"
    / "mmrcpsp_class_balance_holdout_v1"
    / "preregistration.json"
)
PREREGISTRATION_SHA256 = "3bd68aa23cea5124aedd770f735e7a50c88b0a32f398f380864829b82344ac12"
IMPLEMENTATION = (
    REPO_ROOT / "algorithm" / "experiments" / "mmrcpsp_class_balance_holdout.py"
)
IMPLEMENTATION_SHA256 = "b795668723a2cdecf77f5781f9b20da29d17e77ae364802c624232d85d6fa983"
SOURCE_ARTIFACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "mmrcpsp_renewal_stream_v1_20260810.json.gz"
)
SOURCE_ARTIFACT_SHA256 = "385b2fda1de5f8e65e18e422f43415e6ac9a1946051617db1191c23947dbad6b"
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "mmrcpsp_class_balance_holdout_v1_20260810.json.gz"
)
DEFAULT_COMPACT = DEFAULT_FULL.with_suffix("").with_suffix(".json")
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "mmrcpsp_class_balance_holdout_v1_20260810.md"


class MMRCPSPClassBalanceArtifactError(ValueError):
    """Raised when the frozen class-balance artifact contract fails."""


def build_artifact() -> dict[str, Any]:
    protocol = _load_json(PREREGISTRATION, PREREGISTRATION_SHA256)
    source = _load_gzip_json(SOURCE_ARTIFACT, SOURCE_ARTIFACT_SHA256)
    stream = protocol["stream_protocol"]
    horizon = int(stream["horizon"])
    seeds = tuple(int(seed) for seed in stream["seeds"])
    policies = tuple(str(policy) for policy in stream["policies"])
    scenarios = default_scenarios(horizon=horizon)
    if policies != POLICY_ORDER:
        raise MMRCPSPClassBalanceArtifactError("registered policy order changed")
    if tuple(row.scenario_id for row in scenarios) != tuple(stream["scenarios"]):
        raise MMRCPSPClassBalanceArtifactError("registered scenario family changed")
    if tuple(protocol["metric_protocol"]["coordinates"]) != PARETO_COST_COORDINATES:
        raise MMRCPSPClassBalanceArtifactError("registered metric family changed")
    wrapper = Path(__file__).resolve()
    wrapper_hash = _file_sha256(wrapper)
    report = run_class_balance_holdout(
        source,
        scenarios=scenarios,
        seeds=seeds,
        policies=policies,
    )
    artifact = {
        "schema_version": "scheduleurm.mmrcpsp_class_balance_holdout.artifact.v1",
        "protocol": protocol,
        "execution_contract": {
            "preregistration_commit": _committed_file_contract(
                PREREGISTRATION, PREREGISTRATION_SHA256
            ),
            "implementation_commit": _committed_file_contract(
                IMPLEMENTATION, IMPLEMENTATION_SHA256
            ),
            "source_artifact_commit": _committed_file_contract(
                SOURCE_ARTIFACT, SOURCE_ARTIFACT_SHA256
            ),
            "wrapper_commit": _committed_file_contract(wrapper, wrapper_hash),
            "wrapper_sha256": wrapper_hash,
            "arrival_tapes_observed_before_protocol_commit": False,
            "v1_total_cost_result_known_before_metric_registration": True,
        },
        "source_artifact_sha256": SOURCE_ARTIFACT_SHA256,
        "report": report,
    }
    artifact["artifact_sha256_excluding_self"] = _digest(artifact)
    return artifact


def compact_certificate(
    artifact: Mapping[str, Any], full_gzip_sha256: str
) -> dict[str, Any]:
    compact = compact_class_balance_report(artifact["report"])
    compact["execution_contract"] = artifact["execution_contract"]
    compact["source_artifact_sha256"] = artifact["source_artifact_sha256"]
    compact["full_artifact_gzip_sha256"] = full_gzip_sha256
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
            raise MMRCPSPClassBalanceArtifactError(
                f"refusing to overwrite result: {path}"
            )
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
    dispositions = compact["performance_dispositions"]
    ours_rows = {
        (row["scenario_id"], row["seed"]): row
        for row in compact["rows"]
        if row["policy"] == "duration_normalized_robust_maxweight"
    }
    lines = [
        "# MMRCPSP class-balance arrival-tape holdout v1",
        "",
        f"- Status: `{compact['gate']['status']}`",
        f"- Runs: `{aggregate['protocol_pass_count']}/{aggregate['run_count']}` protocol PASS",
        f"- Ours exact registered-family oracle runs: `{aggregate['ours_exact_oracle_run_count']}`",
        f"- Ours Pareto-nondominated scenario/seeds: `{aggregate['ours_nondominated_count']}/{aggregate['scenario_seed_count']}`",
        f"- Ours strictly improves every baseline: `{aggregate['ours_strict_every_baseline_count']}/{aggregate['scenario_seed_count']}`",
        f"- Baseline dominates ours: `{aggregate['baseline_dominates_count']}/{aggregate['scenario_seed_count']}`",
        "- Frozen costs: total average queue, worst-class average queue, and worst-class final backlog.",
        "- The v1 total-only 0/21 result is retained; this is a new-tape metric holdout over the same known action library.",
        "- Performance does not determine protocol validity or imply positive recurrence.",
        "",
        "| Scenario | Seed | Total average queue | Worst-class average queue | Worst-class final backlog | Disposition |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for disposition in dispositions["rows"]:
        key = (disposition["scenario_id"], disposition["seed"])
        ours = ours_rows[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {ours['total_time_average_queue']} | "
            f"{ours['max_class_time_average_queue']} | "
            f"{ours['max_class_final_backlog']} | "
            f"{disposition['ours_disposition']} |"
        )
    lines.extend(["", "## Claim Boundary", "", str(compact["claim_boundary"]), ""])
    return "\n".join(lines)


def _load_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise MMRCPSPClassBalanceArtifactError(f"disk hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gzip_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise MMRCPSPClassBalanceArtifactError(f"disk hash mismatch: {path}")
    with gzip.open(path, "rt", encoding="ascii") as stream:
        return json.load(stream)


def _committed_file_contract(path: Path, expected_sha256: str) -> str:
    source = path.resolve()
    relative = source.relative_to(REPO_ROOT)
    if _file_sha256(source) != expected_sha256:
        raise MMRCPSPClassBalanceArtifactError(f"disk hash mismatch: {relative}")
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative.as_posix()],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise MMRCPSPClassBalanceArtifactError(f"file is not committed: {relative}")
    committed = subprocess.run(
        ["git", "show", f"{commit}:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if sha256(committed).hexdigest() != expected_sha256:
        raise MMRCPSPClassBalanceArtifactError(
            f"committed hash mismatch: {relative}"
        )
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
