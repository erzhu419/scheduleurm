"""Develop a factor-conditioned BACASP-S policy on non-confirmatory replications.

The finite trajectory family was registered before this module.  For each
physical Q x speed/setup x deadline cell, this development-only routine chooses
the plan minimizing worst normalized regret across R87--R90 and all three
registered source costs.  R85/R86 are never read here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.port_public_benchmark_adapter import (
    adapt_bacasp_s_to_port_instance,
    parse_bacasp_s_instance,
)
from algorithm.experiments.port_scheduling_benchmark import RECONFIGURATION_ACTIONS
from algorithm.experiments.port_trajectory_upgrade import (
    SOURCE_COST_METRICS,
    TrajectoryPlan,
    _run_public_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ARTIFACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_trajectory_upgrade_v2_full_20260809.json.gz"
)
CANDIDATE_ARTIFACT_SHA256 = (
    "46df76bd1d59cea5dc67fbc7d2e77d1ac55c0b8fb6e969e0f08c8549916ee69a"
)
DEVELOPMENT_MANIFESTS = (
    REPO_ROOT
    / "tests"
    / "data"
    / "port_bacasp_s_external_holdout_r87_r88"
    / "source_manifest.json",
    REPO_ROOT
    / "tests"
    / "data"
    / "port_bacasp_s_external_holdout_r89_r90"
    / "source_manifest.json",
)
DEVELOPMENT_MANIFEST_SHA256 = (
    "87089340e79a336cbcda85ba4638f40f33c92263c726f2769d9208fd2300e28f",
    "aaafa502a378dd853135e7f3ed4632edb85db909c44738da370ad764c41db97f",
)
EXPECTED_REPLICATIONS = (87, 88, 89, 90)
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_factor_policy_development_r87_r90_20260809.json.gz"
)
DEFAULT_COMPACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_factor_policy_development_r87_r90_20260809.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_factor_policy_development_r87_r90_20260809.md"
SCHEMA_VERSION = "scheduleurm.port_factor_policy_development.v1"


class PortFactorPolicyError(ValueError):
    """Raised when development inputs or finite-policy audits fail closed."""


def cell_key(q_max: int, speed_setup: str, deadline: str) -> str:
    return f"Q{int(q_max)}|S{str(speed_setup)}|D{str(deadline)}"


def load_candidate_plans(
    path: str | Path = CANDIDATE_ARTIFACT,
) -> tuple[TrajectoryPlan, ...]:
    source = Path(path)
    if _file_sha256(source) != CANDIDATE_ARTIFACT_SHA256:
        raise PortFactorPolicyError("registered trajectory-family artifact drifted")
    with gzip.open(source, "rt", encoding="ascii") as stream:
        artifact = json.load(stream)
    family = artifact.get("trajectory_family")
    if not isinstance(family, dict) or len(family) != 31:
        raise PortFactorPolicyError("registered trajectory family must contain 31 plans")
    plans = []
    for plan_id, row in sorted(family.items()):
        plan = TrajectoryPlan(tuple(row["plan"]["policies"]))
        if plan.plan_id != plan_id:
            raise PortFactorPolicyError(f"trajectory plan id mismatch: {plan_id}")
        plans.append(plan)
    if len({plan.plan_id for plan in plans}) != len(plans):
        raise PortFactorPolicyError("trajectory plan identifiers are not unique")
    return tuple(plans)


def load_development_rows(
    manifests: Sequence[str | Path] = DEVELOPMENT_MANIFESTS,
) -> list[dict[str, Any]]:
    if len(manifests) != len(DEVELOPMENT_MANIFEST_SHA256):
        raise PortFactorPolicyError("development manifest count drifted")
    rows = []
    seen_hashes: set[str] = set()
    for manifest_path, expected_hash in zip(
        manifests, DEVELOPMENT_MANIFEST_SHA256, strict=True
    ):
        path = Path(manifest_path)
        if _file_sha256(path) != expected_hash:
            raise PortFactorPolicyError(f"development manifest drifted: {path}")
        manifest = _load_json(path)
        for registered in manifest.get("instances") or ():
            replication = int(registered["replication"])
            if replication not in EXPECTED_REPLICATIONS:
                raise PortFactorPolicyError(
                    f"unregistered development replication: R{replication}"
                )
            source = REPO_ROOT / str(registered["local_path"])
            digest = _file_sha256(source)
            if digest != str(registered["sha256"]):
                raise PortFactorPolicyError(f"development source drifted: {source}")
            if digest in seen_hashes:
                raise PortFactorPolicyError("duplicate development instance hash")
            seen_hashes.add(digest)
            rows.append(dict(registered))
    if len(rows) != 108:
        raise PortFactorPolicyError("development suite must contain 108 instances")
    cells: dict[str, set[int]] = {}
    for row in rows:
        key = cell_key(
            int(row["q_max_factor"]),
            str(row["speed_setup_factor"]),
            str(row["deadline_factor"]),
        )
        cells.setdefault(key, set()).add(int(row["replication"]))
    if len(cells) != 27 or any(values != set(EXPECTED_REPLICATIONS) for values in cells.values()):
        raise PortFactorPolicyError("development factor-cell replication coverage is incomplete")
    return sorted(
        rows,
        key=lambda row: (
            int(row["q_max_factor"]),
            str(row["speed_setup_factor"]),
            str(row["deadline_factor"]),
            int(row["replication"]),
        ),
    )


def evaluate_registered_row(
    registered: Mapping[str, Any],
    plans: Sequence[TrajectoryPlan],
) -> dict[str, Any]:
    source_path = REPO_ROOT / str(registered["local_path"])
    source = parse_bacasp_s_instance(
        source_path, expected_sha256=str(registered["sha256"])
    )
    adapted = adapt_bacasp_s_to_port_instance(source)
    policy_metrics = {}
    policy_result_hashes = {}
    for plan in plans:
        result = _run_public_plan(adapted, plan)
        feasible = bool(
            result.get("pass") is True
            and result["public_source_feasibility_audit"].get(
                "resource_feasibility_ready"
            )
            is True
            and all(
                int(result["action_counts"].get(action_type, 0)) == 0
                for action_type in RECONFIGURATION_ACTIONS
            )
        )
        if not feasible:
            raise PortFactorPolicyError(
                f"{source_path.name}: infeasible development plan {plan.plan_id}"
            )
        metrics = {
            metric: float(result["public_source_core_metrics"][metric])
            for metric in SOURCE_COST_METRICS
        }
        if any(value <= 0.0 or not math.isfinite(value) for value in metrics.values()):
            raise PortFactorPolicyError(
                f"{source_path.name}: nonpositive development metric"
            )
        policy_metrics[plan.plan_id] = metrics
        policy_result_hashes[plan.plan_id] = result["trajectory_result_hash"]
    return {
        "upstream_path": registered["upstream_path"],
        "source_sha256": registered["sha256"],
        "replication": int(registered["replication"]),
        "q_max_factor": int(registered["q_max_factor"]),
        "speed_setup_factor": str(registered["speed_setup_factor"]),
        "deadline_factor": str(registered["deadline_factor"]),
        "cell_key": cell_key(
            int(registered["q_max_factor"]),
            str(registered["speed_setup_factor"]),
            str(registered["deadline_factor"]),
        ),
        "policy_metrics": policy_metrics,
        "policy_result_hashes": policy_result_hashes,
    }


def minimax_regret_selection(
    rows: Sequence[Mapping[str, Any]],
    plan_ids: Sequence[str],
) -> dict[str, Any]:
    if not rows or not plan_ids:
        raise PortFactorPolicyError("minimax selection requires rows and plans")
    plan_ids = tuple(sorted(str(value) for value in plan_ids))
    regret_values: dict[str, list[float]] = {plan_id: [] for plan_id in plan_ids}
    for row in rows:
        metrics = row["policy_metrics"]
        if set(metrics) != set(plan_ids):
            raise PortFactorPolicyError("development row has a different plan family")
        best = {
            metric: min(float(metrics[plan_id][metric]) for plan_id in plan_ids)
            for metric in SOURCE_COST_METRICS
        }
        for plan_id in plan_ids:
            for metric in SOURCE_COST_METRICS:
                regret_values[plan_id].append(
                    float(metrics[plan_id][metric]) / best[metric] - 1.0
                )
    diagnostics = {}
    for plan_id in plan_ids:
        values = regret_values[plan_id]
        diagnostics[plan_id] = {
            "maximum_normalized_regret": max(values),
            "mean_normalized_regret": statistics.fmean(values),
            "observation_count": len(values),
        }
    selected = min(
        plan_ids,
        key=lambda plan_id: (
            diagnostics[plan_id]["maximum_normalized_regret"],
            diagnostics[plan_id]["mean_normalized_regret"],
            plan_id,
        ),
    )
    return {
        "selected_plan_id": selected,
        "selection_objective": "lexicographic_minimum_(maximum_regret,mean_regret,plan_id)",
        "selected_diagnostics": diagnostics[selected],
        "candidate_diagnostics": diagnostics,
        "exact_finite_family_argmin": True,
    }


def build_development_report(*, workers: int = 1) -> dict[str, Any]:
    worker_count = int(workers)
    if worker_count <= 0:
        raise PortFactorPolicyError("workers must be positive")
    plans = load_candidate_plans()
    registered_rows = load_development_rows()
    plan_payload = tuple(plan.policies for plan in plans)
    if worker_count == 1:
        evaluated = [
            evaluate_registered_row(row, plans) for row in registered_rows
        ]
    else:
        tasks = [(row, plan_payload) for row in registered_rows]
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            evaluated = list(pool.map(_evaluate_task, tasks))
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in evaluated:
        by_cell.setdefault(str(row["cell_key"]), []).append(row)
    mappings = {}
    for key, cell_rows in sorted(by_cell.items()):
        if {int(row["replication"]) for row in cell_rows} != set(EXPECTED_REPLICATIONS):
            raise PortFactorPolicyError(f"cell lacks all development replications: {key}")
        mappings[key] = minimax_regret_selection(
            cell_rows, [plan.plan_id for plan in plans]
        )
    selected_ids = {row["selected_plan_id"] for row in mappings.values()}
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": {
            "pass": len(evaluated) == 108 and len(mappings) == 27,
            "status": "PORT_FACTOR_POLICY_DEVELOPMENT_PASS",
            "performance_superiority_claim_ready": False,
        },
        "source": {
            "candidate_artifact": str(CANDIDATE_ARTIFACT.relative_to(REPO_ROOT)),
            "candidate_artifact_sha256": CANDIDATE_ARTIFACT_SHA256,
            "development_manifests": [
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": digest,
                }
                for path, digest in zip(
                    DEVELOPMENT_MANIFESTS,
                    DEVELOPMENT_MANIFEST_SHA256,
                    strict=True,
                )
            ],
            "development_replications": list(EXPECTED_REPLICATIONS),
        },
        "candidate_family": {
            "count": len(plans),
            "plans": [plan.snapshot() for plan in plans],
        },
        "selection_protocol": {
            "conditioning_features": [
                "global available quay-crane count Q",
                "crane speed/setup factor",
                "deadline factor",
            ],
            "metrics": list(SOURCE_COST_METRICS),
            "objective": "minimax normalized regret with mean-regret and plan-id ties",
            "per_instance_tuning": False,
            "per_factor_cell_mapping": True,
            "confirmation_replications_observed": False,
        },
        "factor_policy": mappings,
        "selected_plan_count": len(selected_ids),
        "selected_plan_ids": sorted(selected_ids),
        "rows": evaluated,
        "claim_boundary": {
            "development_only": True,
            "r85_r86_used": False,
            "arbitrary_port_optimality": False,
            "mid_service_migration": False,
            "stochastic_stability_transfer": False,
        },
    }
    report["scientific_content_sha256"] = _digest(report)
    return report


def _evaluate_task(
    task: tuple[Mapping[str, Any], Sequence[Sequence[str]]],
) -> dict[str, Any]:
    registered, plan_payload = task
    plans = tuple(TrajectoryPlan(tuple(policies)) for policies in plan_payload)
    return evaluate_registered_row(registered, plans)


def compact_certificate(report: Mapping[str, Any], full_hash: str) -> dict[str, Any]:
    compact = {
        "schema_version": f"{SCHEMA_VERSION}.compact.v1",
        "gate": report["gate"],
        "full_artifact_path": str(DEFAULT_FULL.relative_to(REPO_ROOT)),
        "full_artifact_gzip_sha256": full_hash,
        "full_scientific_content_sha256": report["scientific_content_sha256"],
        "source": report["source"],
        "candidate_family": report["candidate_family"],
        "selection_protocol": report["selection_protocol"],
        "factor_policy": report["factor_policy"],
        "selected_plan_count": report["selected_plan_count"],
        "selected_plan_ids": report["selected_plan_ids"],
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
    lines = [
        "# BACASP-S factor-conditioned policy development",
        "",
        f"- Status: `{compact['gate']['status']}`",
        f"- Candidate plans: `{compact['candidate_family']['count']}`",
        f"- Factor cells: `{len(compact['factor_policy'])}`",
        f"- Distinct selected plans: `{compact['selected_plan_count']}`",
        "- Development replications: `R87/R88/R89/R90`",
        "- Confirmation replications used: `false`",
        "",
        "| Factor cell | Selected plan | Maximum regret | Mean regret |",
        "|---|---|---:|---:|",
    ]
    for key, row in compact["factor_policy"].items():
        diagnostic = row["selected_diagnostics"]
        lines.append(
            f"| `{key}` | `{row['selected_plan_id']}` | "
            f"{diagnostic['maximum_normalized_regret']:.6f} | "
            f"{diagnostic['mean_normalized_regret']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This artifact is development evidence only. It makes no performance claim",
            "on R85/R86 and no claim about unrestricted berth allocation, physical",
            "terminal deployment, or stochastic port stability.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PortFactorPolicyError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


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
            raise PortFactorPolicyError("artifact contains non-finite value")
        return round(value, 9)
    return value


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_development_report(workers=args.workers)
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
