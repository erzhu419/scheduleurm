"""Prospective R85/R86 confirmation for the factor-conditioned port policy."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_factor_policy_development import cell_key
from algorithm.experiments.port_public_external_holdout import (
    _aggregate,
    _digest,
    _evaluate_registered_instance,
    verify_suite_manifest,
)
from algorithm.experiments.port_scheduling_benchmark import BASELINE_POLICIES
from algorithm.experiments.port_trajectory_upgrade import (
    SOURCE_COST_METRICS,
    TRAJECTORY_POLICY,
    TrajectoryPlan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "tests" / "data" / "port_bacasp_s_factor_holdout_r85_r86"
DEFAULT_MANIFEST = SUITE_ROOT / "source_manifest.json"
DEFAULT_PREREGISTRATION = SUITE_ROOT / "preregistration.json"
DEFAULT_DEVELOPMENT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_factor_policy_development_r87_r90_20260809.json"
)
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_factor_policy_holdout_r85_r86_20260809.json.gz"
)
DEFAULT_COMPACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_factor_policy_holdout_r85_r86_20260809.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_factor_policy_holdout_r85_r86_20260809.md"
EXPECTED_REPLICATIONS = (85, 86)
SCHEMA_VERSION = "scheduleurm.port_factor_policy_holdout.v1"


class PortFactorHoldoutError(ValueError):
    """Raised when the prospective port confirmation contract fails closed."""


def verify_preregistration(
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    development_path: str | Path = DEFAULT_DEVELOPMENT,
) -> dict[str, Any]:
    preregistration_file = Path(preregistration_path)
    preregistration = _load_json(preregistration_file)
    if preregistration.get("schema_version") != (
        "scheduleurm.port_factor_policy_holdout.preregistration.v1"
    ):
        raise PortFactorHoldoutError("unexpected preregistration schema")
    if _file_sha256(manifest_path) != preregistration["suite_manifest_sha256"]:
        raise PortFactorHoldoutError("confirmation source manifest drifted")
    if _file_sha256(development_path) != preregistration["development_compact_sha256"]:
        raise PortFactorHoldoutError("frozen factor policy drifted")
    development = _load_json(development_path)
    if development.get("compact_content_sha256_excluding_self") != preregistration[
        "development_content_sha256"
    ]:
        raise PortFactorHoldoutError("development content digest drifted")
    mapping = development.get("factor_policy") or {}
    if len(mapping) != 27:
        raise PortFactorHoldoutError("factor policy must cover 27 cells")
    if _digest(mapping) != preregistration["factor_policy_sha256"]:
        raise PortFactorHoldoutError("factor-policy mapping drifted")

    relative = preregistration_file.resolve().relative_to(REPO_ROOT.resolve())
    prereg_commit = _git_output(
        ["log", "-1", "--format=%H", "--", relative.as_posix()]
    ).decode().strip()
    if len(prereg_commit) != 40:
        raise PortFactorHoldoutError("preregistration is not committed")
    if sha256(_git_output(["show", f"{prereg_commit}:{relative.as_posix()}"])).hexdigest() != _file_sha256(preregistration_file):
        raise PortFactorHoldoutError("committed preregistration differs from disk")
    for output in preregistration.get("outputs") or ():
        output_path = Path(str(output))
        if output_path.is_absolute() or ".." in output_path.parts:
            raise PortFactorHoldoutError(f"unsafe output path: {output}")
        if subprocess.run(
            ["git", "cat-file", "-e", f"{prereg_commit}:{output_path.as_posix()}"],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            raise PortFactorHoldoutError(f"result predates preregistration: {output}")

    freeze = preregistration.get("implementation_freeze") or {}
    freeze_commit = str(freeze.get("repository_commit") or "")
    if len(freeze_commit) != 40:
        raise PortFactorHoldoutError("invalid implementation freeze commit")
    verified_implementation = []
    for source, expected in sorted((freeze.get("implementation_sha256") or {}).items()):
        source_path = Path(str(source))
        current = _file_sha256(REPO_ROOT / source_path)
        committed = sha256(
            _git_output(["show", f"{freeze_commit}:{source_path.as_posix()}"])
        ).hexdigest()
        if current != expected or committed != expected:
            raise PortFactorHoldoutError(f"implementation drifted: {source}")
        verified_implementation.append({"path": source, "sha256": expected})
    if not verified_implementation:
        raise PortFactorHoldoutError("empty implementation freeze")
    return {
        "ready": True,
        "preregistration_commit": prereg_commit,
        "preregistration_sha256": _file_sha256(preregistration_file),
        "factor_policy": mapping,
        "development_sha256": _file_sha256(development_path),
        "implementation": verified_implementation,
    }


def build_holdout_report(
    *,
    suite_manifest_path: str | Path = DEFAULT_MANIFEST,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    development_path: str | Path = DEFAULT_DEVELOPMENT,
) -> dict[str, Any]:
    provenance = verify_suite_manifest(
        suite_manifest_path,
        expected_replications=EXPECTED_REPLICATIONS,
    )
    freeze = verify_preregistration(
        preregistration_path, suite_manifest_path, development_path
    )
    manifest = _load_json(suite_manifest_path)
    plans_by_id = {
        str(row["selected_plan_id"]): TrajectoryPlan(
            tuple(_plan_policies(development_path, str(row["selected_plan_id"])))
        )
        for row in freeze["factor_policy"].values()
    }
    rows = []
    failures = []
    for registered in manifest["instances"]:
        key = cell_key(
            int(registered["q_max_factor"]),
            str(registered["speed_setup_factor"]),
            str(registered["deadline_factor"]),
        )
        try:
            selected_id = str(freeze["factor_policy"][key]["selected_plan_id"])
            policy_plans = {
                TRAJECTORY_POLICY: plans_by_id[selected_id],
                **{
                    baseline: TrajectoryPlan((baseline,))
                    for baseline in BASELINE_POLICIES
                },
            }
            row = _evaluate_registered_instance(registered, policy_plans)
            row["factor_cell"] = key
            row["selected_plan_id"] = selected_id
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "upstream_path": registered.get("upstream_path"),
                    "factor_cell": key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    aggregate = _aggregate(rows, registered_replications=EXPECTED_REPLICATIONS)
    protocol_pass = bool(
        provenance["ready"]
        and freeze["ready"]
        and len(rows) == 54
        and not failures
        and all(row["selected_feasible"] for row in rows)
    )
    strict_count = aggregate["selected_strictly_dominates_every_baseline_count"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": {
            "pass": protocol_pass,
            "status": (
                "PORT_FACTOR_POLICY_HOLDOUT_PASS"
                if protocol_pass
                else "PORT_FACTOR_POLICY_HOLDOUT_FAIL"
            ),
            "registered_instance_count": 54,
            "completed_instance_count": len(rows),
            "failure_count": len(failures),
            "performance_nondominance_ready": bool(
                protocol_pass
                and aggregate["selected_pareto_nondominated_count"] == 54
            ),
            "performance_superiority_claim_ready": bool(
                protocol_pass and strict_count == 54
            ),
            "unrestricted_port_optimality_claim_ready": False,
            "stochastic_port_stability_claim_ready": False,
        },
        "provenance": provenance,
        "freeze": {
            key: value for key, value in freeze.items() if key != "factor_policy"
        },
        "factor_policy_sha256": _digest(freeze["factor_policy"]),
        "selected_plan_count": len(plans_by_id),
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
        "candidate_generation_on_holdout": False,
        "holdout_feedback_used_for_selection": False,
        "rows": rows,
        "failures": failures,
        "aggregate": aggregate,
        "selected_plan_histogram": dict(
            sorted(Counter(row["selected_plan_id"] for row in rows).items())
        ),
        "claim_boundary": {
            "scope": "BACASP-S fixed-slot source-core R85/R86 only",
            "mid_service_migration": False,
            "physical_terminal_deployment": False,
            "unrestricted_port_optimality": False,
            "stochastic_stability_transfer": False,
        },
    }
    report["scientific_content_sha256"] = _digest(report)
    return report


def _plan_policies(development_path: str | Path, plan_id: str) -> tuple[str, ...]:
    development = _load_json(development_path)
    for row in development["candidate_family"]["plans"]:
        if str(row["plan_id"]) == plan_id:
            return tuple(str(value) for value in row["policies"])
    raise PortFactorHoldoutError(f"selected plan absent from frozen family: {plan_id}")


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
    compact_payload = {
        "schema_version": f"{SCHEMA_VERSION}.compact.v1",
        "gate": report["gate"],
        "full_artifact_path": str(full.relative_to(REPO_ROOT)),
        "full_artifact_gzip_sha256": full_hash,
        "full_scientific_content_sha256": report["scientific_content_sha256"],
        "freeze": report["freeze"],
        "factor_policy_sha256": report["factor_policy_sha256"],
        "selected_plan_count": report["selected_plan_count"],
        "baseline_policies": report["baseline_policies"],
        "evaluation_metrics": report["evaluation_metrics"],
        "aggregate": report["aggregate"],
        "selected_plan_histogram": report["selected_plan_histogram"],
        "failures": report["failures"],
        "claim_boundary": report["claim_boundary"],
    }
    compact_payload["compact_content_sha256_excluding_self"] = _digest(compact_payload)
    compact.write_text(
        json.dumps(compact_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(markdown_report(compact_payload), encoding="utf-8")
    return {
        "full": str(full),
        "full_sha256": full_hash,
        "compact": str(compact),
        "compact_sha256": _file_sha256(compact),
        "markdown": str(markdown),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    gate = compact["gate"]
    aggregate = compact["aggregate"]
    lines = [
        "# BACASP-S factor-policy R85/R86 confirmation",
        "",
        f"- Status: `{gate['status']}`",
        f"- Completed: `{gate['completed_instance_count']}/{gate['registered_instance_count']}`",
        f"- Pareto-nondominated: `{aggregate['selected_pareto_nondominated_count']}/{gate['registered_instance_count']}`",
        f"- Strict every-baseline dominance: `{aggregate['selected_strictly_dominates_every_baseline_count']}/{gate['registered_instance_count']}`",
        f"- Distinct selected plans: `{compact['selected_plan_count']}`",
        "",
        "| Baseline | Metric | Geomean ours/baseline | 95% factor-cluster interval |",
        "|---|---|---:|---:|",
    ]
    for baseline, summary in aggregate["paired_policy_summaries"].items():
        for metric, values in summary["metrics"].items():
            lo, hi = values["cluster_bootstrap_95pct"]
            lines.append(
                f"| `{baseline}` | `{metric}` | "
                f"{values['geometric_mean_ratio_ours_over_baseline']:.6f} | "
                f"[{lo:.6f}, {hi:.6f}] |"
            )
    lines.extend(
        [
            "",
            "The confirmation is limited to the registered BACASP-S fixed-slot source",
            "core. It is not an unrestricted berth-allocation optimum, physical-terminal",
            "deployment, migration experiment, or stochastic-stability certificate.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PortFactorHoldoutError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _git_output(args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_holdout_report(
        suite_manifest_path=args.suite_manifest,
        preregistration_path=args.preregistration,
        development_path=args.development,
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
