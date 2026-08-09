"""Post-freeze BACASP-S LargeMB external-holdout evaluation.

The development trajectory search is completed before this module is allowed
to inspect holdout outcomes.  A preregistration binds the selected trajectory,
search configuration, implementation files, baselines, metrics, and the
a factor-complete source manifest and its registered replication pair.  The
holdout runner never generates, ranks, or tunes trajectory candidates.

The public instances cover berth/quay/specific-crane source semantics only.
They do not contain yard, gate, draft, or mid-service migration observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.port_public_benchmark_adapter import (
    DEFAULT_MANIFEST as DEVELOPMENT_SOURCE_MANIFEST,
    adapt_bacasp_s_to_port_instance,
    parse_bacasp_s_instance,
)
from algorithm.experiments.port_scheduling_benchmark import (
    BASELINE_POLICIES,
    RECONFIGURATION_ACTIONS,
)
from algorithm.experiments.port_trajectory_upgrade import (
    GLOBAL_TRAJECTORY_CONFIG,
    SOURCE_COST_METRICS,
    TRAJECTORY_POLICY,
    TrajectoryPlan,
    _pareto,
    _run_public_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = REPO_ROOT / "tests" / "data" / "port_bacasp_s_external_holdout_r89_r90"
DEFAULT_SUITE_MANIFEST = SUITE_ROOT / "source_manifest.json"
DEFAULT_PREREGISTRATION = SUITE_ROOT / "preregistration.json"
DEFAULT_FULL = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_bacasp_s_external_holdout_r89_r90_full_20260809.json.gz"
)
DEFAULT_COMPACT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_bacasp_s_external_holdout_r89_r90_20260809.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "port_bacasp_s_external_holdout_r89_r90_20260809.md"
SCHEMA_VERSION = "scheduleurm.port_bacasp_s_external_holdout.v1"
EXPECTED_REPOSITORY_COMMIT = "e8eda86ec0435105ca313afe85987e4f4ab2cbd7"
EXPECTED_REPLICATIONS = (89, 90)
EXPECTED_Q = (5, 10, 15)
EXPECTED_SPEED_SETUP = ("160-0.2", "200-0.15", "240-0.1")
EXPECTED_DEADLINE = ("1", "1.5", "2")
BOOTSTRAP_SEED = 20_260_809
BOOTSTRAP_REPLICATES = 5_000


class PortHoldoutContractError(ValueError):
    """Raised when provenance, freeze, or artifact contracts do not hold."""


def verify_suite_manifest(
    manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
    development_manifest_path: str | Path = DEVELOPMENT_SOURCE_MANIFEST,
    expected_replications: Sequence[int] = EXPECTED_REPLICATIONS,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != "scheduleurm.port_bacasp_s_suite_manifest.v1":
        raise PortHoldoutContractError("unexpected BACASP-S suite manifest schema")
    claimed_digest = str(manifest.get("manifest_content_sha256_excluding_self") or "")
    digest_payload = dict(manifest)
    digest_payload.pop("manifest_content_sha256_excluding_self", None)
    if _digest(digest_payload) != claimed_digest:
        raise PortHoldoutContractError("suite manifest content digest mismatch")
    if manifest.get("repository_commit") != EXPECTED_REPOSITORY_COMMIT:
        raise PortHoldoutContractError("suite is not pinned to the registered upstream commit")

    rows = list(manifest.get("instances") or ())
    if len(rows) != 54:
        raise PortHoldoutContractError("external holdout must contain exactly 54 instances")
    paths = [str(row.get("local_path") or "") for row in rows]
    upstream_paths = [str(row.get("upstream_path") or "") for row in rows]
    hashes = [str(row.get("sha256") or "") for row in rows]
    if len(set(paths)) != 54 or len(set(upstream_paths)) != 54 or len(set(hashes)) != 54:
        raise PortHoldoutContractError("holdout paths and hashes must be unique")

    expected_cells = {
        (q, speed, deadline)
        for q in EXPECTED_Q
        for speed in EXPECTED_SPEED_SETUP
        for deadline in EXPECTED_DEADLINE
    }
    cell_replicates: dict[tuple[int, str, str], set[int]] = defaultdict(set)
    verified_rows = []
    for row in rows:
        local_path = REPO_ROOT / str(row["local_path"])
        if not local_path.is_file():
            raise PortHoldoutContractError(f"missing holdout fixture: {local_path}")
        raw = local_path.read_bytes()
        checks = {
            "sha256": sha256(raw).hexdigest() == str(row["sha256"]),
            "byte_count": len(raw) == int(row["byte_count"]),
            "line_count": len(raw.splitlines()) == int(row["line_count"]),
            "raw_bytes_unmodified": row.get("raw_bytes_modified") is False,
            "split": row.get("split") == "external_holdout",
        }
        if not all(checks.values()):
            raise PortHoldoutContractError(
                f"fixture authenticity failed for {local_path.name}: {checks}"
            )
        cell = (
            int(row["q_max_factor"]),
            str(row["speed_setup_factor"]),
            str(row["deadline_factor"]),
        )
        cell_replicates[cell].add(int(row["replication"]))
        verified_rows.append({"local_path": str(local_path), "checks": checks})
    if set(cell_replicates) != expected_cells:
        raise PortHoldoutContractError("27-cell Q x S x D coverage is incomplete")
    registered_replications = tuple(sorted(int(value) for value in expected_replications))
    if len(registered_replications) != 2 or len(set(registered_replications)) != 2:
        raise PortHoldoutContractError("holdout must register exactly two replications")
    if any(
        tuple(sorted(values)) != registered_replications
        for values in cell_replicates.values()
    ):
        raise PortHoldoutContractError(
            "each factor cell must contain exactly the registered replication pair"
        )

    development = _load_json(development_manifest_path)
    development_fixture = development.get("fixture") or {}
    if str(development_fixture.get("sha256") or "") in set(hashes):
        raise PortHoldoutContractError("development fixture hash leaked into holdout")
    if str(development_fixture.get("upstream_path") or "") in set(upstream_paths):
        raise PortHoldoutContractError("development fixture path leaked into holdout")
    return {
        "ready": True,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_content_sha256_excluding_self": claimed_digest,
        "repository_commit": EXPECTED_REPOSITORY_COMMIT,
        "instance_count": len(rows),
        "factor_cell_count": len(cell_replicates),
        "replications_per_cell": 2,
        "registered_replications": list(registered_replications),
        "development_holdout_path_disjoint": True,
        "development_holdout_hash_disjoint": True,
        "verified_rows": verified_rows,
    }


def verify_preregistration(
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    suite_manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
) -> dict[str, Any]:
    path = Path(preregistration_path)
    prereg = _load_json(path)
    if prereg.get("schema_version") != "scheduleurm.port_bacasp_s_preregistration.v1":
        raise PortHoldoutContractError("unexpected port preregistration schema")
    claimed = str(prereg.get("preregistration_sha256_excluding_self") or "")
    payload = dict(prereg)
    payload.pop("preregistration_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise PortHoldoutContractError("preregistration digest mismatch")
    if prereg.get("holdout_outcomes_observed_before_freeze") is not False:
        raise PortHoldoutContractError("preregistration does not certify pre-outcome freeze")
    if _file_sha256(suite_manifest_path) != prereg.get("suite_manifest_file_sha256"):
        raise PortHoldoutContractError("suite manifest changed after preregistration")
    if prereg.get("frozen_search_config_sha256") != GLOBAL_TRAJECTORY_CONFIG.digest:
        raise PortHoldoutContractError("trajectory search configuration changed after freeze")
    if tuple(prereg.get("baseline_policies") or ()) != tuple(BASELINE_POLICIES):
        raise PortHoldoutContractError("baseline family changed after freeze")
    if tuple(prereg.get("evaluation_metrics") or ()) != tuple(SOURCE_COST_METRICS):
        raise PortHoldoutContractError("holdout metric family changed after freeze")
    plan = TrajectoryPlan(tuple(prereg["frozen_plan"]["policies"]))
    if plan.plan_id != prereg["frozen_plan"]["plan_id"]:
        raise PortHoldoutContractError("frozen plan id does not match its policies")
    for relative_path, expected_hash in sorted(
        (prereg.get("implementation_file_sha256") or {}).items()
    ):
        source = REPO_ROOT / relative_path
        if not source.is_file() or _file_sha256(source) != str(expected_hash):
            raise PortHoldoutContractError(
                f"implementation changed after preregistration: {relative_path}"
            )
    development_compact = REPO_ROOT / str(prereg["development_compact_artifact"])
    if _file_sha256(development_compact) != prereg["development_compact_file_sha256"]:
        raise PortHoldoutContractError("development certificate changed after freeze")
    development = _load_json(development_compact)
    if development.get("selected_global_plan_id") != plan.plan_id:
        raise PortHoldoutContractError("development-selected plan differs from frozen plan")
    if development.get("global_search_config_sha256") != GLOBAL_TRAJECTORY_CONFIG.digest:
        raise PortHoldoutContractError("development search configuration differs from freeze")
    return {
        "ready": True,
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "content_sha256_excluding_self": claimed,
        "frozen_plan": plan.snapshot(),
        "implementation_file_count": len(prereg["implementation_file_sha256"]),
        "holdout_feedback_used_for_selection": False,
    }


def build_holdout_report(
    *,
    suite_manifest_path: str | Path = DEFAULT_SUITE_MANIFEST,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    prereg = _load_json(preregistration_path)
    registered_replications = tuple(
        sorted(
            int(value)
            for value in prereg["split_rule"]["external_holdout_replications"]
        )
    )
    provenance = verify_suite_manifest(
        suite_manifest_path,
        expected_replications=registered_replications,
    )
    freeze = verify_preregistration(preregistration_path, suite_manifest_path)
    manifest = _load_json(suite_manifest_path)
    frozen_plan = TrajectoryPlan(tuple(prereg["frozen_plan"]["policies"]))
    policy_plans = {
        TRAJECTORY_POLICY: frozen_plan,
        **{policy: TrajectoryPlan((policy,)) for policy in BASELINE_POLICIES},
    }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for registered in manifest["instances"]:
        try:
            rows.append(_evaluate_registered_instance(registered, policy_plans))
        except Exception as exc:  # every registered row remains in the denominator
            failures.append(
                {
                    "upstream_path": registered.get("upstream_path"),
                    "replication": registered.get("replication"),
                    "q_max_factor": registered.get("q_max_factor"),
                    "speed_setup_factor": registered.get("speed_setup_factor"),
                    "deadline_factor": registered.get("deadline_factor"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    aggregate = _aggregate(rows, registered_replications=registered_replications)
    protocol_pass = bool(
        provenance["ready"]
        and freeze["ready"]
        and len(rows) + len(failures) == 54
        and not failures
        and all(row["selected_feasible"] for row in rows)
    )
    performance_ready = bool(
        protocol_pass and aggregate["selected_pareto_nondominated_count"] == 54
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate": {
            "pass": protocol_pass,
            "status": (
                "PORT_PUBLIC_EXTERNAL_HOLDOUT_PASS"
                if protocol_pass
                else "PORT_PUBLIC_EXTERNAL_HOLDOUT_FAIL"
            ),
            "registered_instance_count": 54,
            "completed_instance_count": len(rows),
            "failure_count": len(failures),
            "performance_nondominance_ready": performance_ready,
            "performance_superiority_claim_ready": False,
            "unrestricted_port_optimality_claim_ready": False,
        },
        "provenance": provenance,
        "freeze": freeze,
        "selected_policy": TRAJECTORY_POLICY,
        "frozen_plan": frozen_plan.snapshot(),
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
        "candidate_generation_on_holdout": False,
        "holdout_feedback_used_for_selection": False,
        "rows": rows,
        "failures": failures,
        "aggregate": aggregate,
        "theorem_mapping": {
            "low_level_action_score": "Q^T physical_lower_service minus action penalty",
            "low_level_oracle_audit": (
                "each decision records exact feasible-family oracle gap; the frozen periodic "
                "trajectory may intentionally choose a non-MaxWeight atomic policy"
            ),
            "outer_policy_role": (
                "the frozen trajectory was selected on development terminal virtual-service "
                "coordinates; holdout completion costs are evaluation outcomes only"
            ),
            "stochastic_boundary": (
                "this deterministic holdout does not instantiate a port arrival/service process, "
                "positive recurrence, or a port-specific slack certificate"
            ),
        },
        "claim_boundary": {
            "supports": [
                "post-freeze evaluation on 54 hash-bound disjoint BACASP-S LargeMB instances",
                "complete 27-cell Q x speed/setup x deadline coverage with "
                + "/".join(f"R{value}" for value in registered_replications)
                + " replication",
                "same-input comparison with four frozen simulator policies",
                "source-core feasibility and factor-stratified paired outcome summaries",
            ],
            "does_not_support": [
                "continuous-quay MILP reproduction or unrestricted port optimality",
                "comparison with the BACASP-S authors' exact or genetic algorithms",
                "public evidence for yard, gate, draft, or mid-service migration",
                "physical-port deployment or empirically calibrated reconfiguration costs",
                "automatic transfer of the server stochastic-stability certificate",
            ],
        },
    }
    report["scientific_content_sha256"] = _digest(report)
    return report


def _evaluate_registered_instance(
    registered: Mapping[str, Any],
    policy_plans: Mapping[str, TrajectoryPlan],
) -> dict[str, Any]:
    source_path = REPO_ROOT / str(registered["local_path"])
    source = parse_bacasp_s_instance(
        source_path,
        expected_sha256=str(registered["sha256"]),
    )
    adapted = adapt_bacasp_s_to_port_instance(source)
    results = {
        policy: _run_public_plan(adapted, plan)
        for policy, plan in policy_plans.items()
    }
    metrics = {
        policy: dict(result["public_source_core_metrics"])
        for policy, result in results.items()
    }
    pareto = _pareto(metrics, SOURCE_COST_METRICS)
    selected = results[TRAJECTORY_POLICY]
    selected_feasible = bool(
        selected.get("pass") is True
        and selected["public_source_feasibility_audit"].get(
            "resource_feasibility_ready"
        )
        is True
        and all(
            int(selected["action_counts"].get(action_type, 0)) == 0
            for action_type in RECONFIGURATION_ACTIONS
        )
    )
    oracle_gaps = [
        float(audit.get("oracle_gap_alpha0") or 0.0)
        for audit in selected.get("decision_audits") or ()
    ]
    theorem_ready = [
        bool(audit.get("theorem_ready"))
        for audit in selected.get("decision_audits") or ()
    ]
    baseline_relations = {
        baseline: _relation(
            metrics[TRAJECTORY_POLICY], metrics[baseline], SOURCE_COST_METRICS
        )
        for baseline in BASELINE_POLICIES
    }
    return {
        "upstream_path": registered["upstream_path"],
        "source_sha256": registered["sha256"],
        "replication": int(registered["replication"]),
        "q_max_factor": int(registered["q_max_factor"]),
        "speed_setup_factor": str(registered["speed_setup_factor"]),
        "deadline_factor": str(registered["deadline_factor"]),
        "vessel_count": source.vessel_count,
        "quay_length": source.quay_length,
        "crane_count": source.crane_count,
        "selected_feasible": selected_feasible,
        "selected_result_hash": selected["trajectory_result_hash"],
        "selected_source_core_metrics": metrics[TRAJECTORY_POLICY],
        "policy_source_core_metrics": metrics,
        "pareto": pareto,
        "selected_pareto_nondominated": (
            TRAJECTORY_POLICY in pareto["nondominated_policies"]
        ),
        "baseline_relations": baseline_relations,
        "selected_low_level_oracle_gap_max": max(oracle_gaps, default=0.0),
        "selected_low_level_exact_slot_count": sum(theorem_ready),
        "selected_low_level_slot_count": len(theorem_ready),
        "selected_mid_service_reconfiguration_count": sum(
            int(selected["action_counts"].get(action_type, 0))
            for action_type in RECONFIGURATION_ACTIONS
        ),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    registered_replications: Sequence[int] = EXPECTED_REPLICATIONS,
) -> dict[str, Any]:
    if not rows:
        return {
            "selected_pareto_nondominated_count": 0,
            "selected_strictly_dominates_every_baseline_count": 0,
            "paired_policy_summaries": {},
            "factor_cell_count": 0,
        }
    summaries = {}
    for baseline in BASELINE_POLICIES:
        metric_rows = {}
        for metric in SOURCE_COST_METRICS:
            ratios = [
                float(row["selected_source_core_metrics"][metric])
                / float(row["policy_source_core_metrics"][baseline][metric])
                for row in rows
            ]
            interval = _factor_cluster_bootstrap(rows, baseline, metric)
            metric_rows[metric] = {
                "geometric_mean_ratio_ours_over_baseline": _geomean(ratios),
                "cluster_bootstrap_95pct": interval,
            }
        relations = Counter(row["baseline_relations"][baseline] for row in rows)
        summaries[baseline] = {
            "metrics": metric_rows,
            "relation_counts": dict(sorted(relations.items())),
        }
    cells = {
        (
            int(row["q_max_factor"]),
            str(row["speed_setup_factor"]),
            str(row["deadline_factor"]),
        )
        for row in rows
    }
    return {
        "selected_pareto_nondominated_count": sum(
            bool(row["selected_pareto_nondominated"]) for row in rows
        ),
        "selected_strictly_dominates_every_baseline_count": sum(
            all(value == "ours_dominates" for value in row["baseline_relations"].values())
            for row in rows
        ),
        "factor_cell_count": len(cells),
        "paired_policy_summaries": summaries,
        "bootstrap": {
            "method": (
                "factor-cell cluster bootstrap; both registered replication rows "
                "retained per sampled cell"
            ),
            "registered_replications": [
                int(value) for value in registered_replications
            ],
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
    }


def _factor_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], baseline: str, metric: str
) -> list[float]:
    groups: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        cell = (
            int(row["q_max_factor"]),
            str(row["speed_setup_factor"]),
            str(row["deadline_factor"]),
        )
        groups[cell].append(
            math.log(
                float(row["selected_source_core_metrics"][metric])
                / float(row["policy_source_core_metrics"][baseline][metric])
            )
        )
    ordered = sorted(groups)
    rng = random.Random(
        f"{BOOTSTRAP_SEED}|{baseline}|{metric}|{len(ordered)}"
    )
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [groups[rng.choice(ordered)] for _ in ordered]
        values = [value for group in selected for value in group]
        draws.append(math.exp(statistics.fmean(values)))
    draws.sort()
    return [
        round(draws[int(0.025 * (len(draws) - 1))], 9),
        round(draws[int(0.975 * (len(draws) - 1))], 9),
    ]


def _relation(
    ours: Mapping[str, Any], baseline: Mapping[str, Any], metrics: Sequence[str]
) -> str:
    ours_no_worse = all(float(ours[key]) <= float(baseline[key]) + 1e-8 for key in metrics)
    ours_strict = any(float(ours[key]) < float(baseline[key]) - 1e-8 for key in metrics)
    base_no_worse = all(float(baseline[key]) <= float(ours[key]) + 1e-8 for key in metrics)
    base_strict = any(float(baseline[key]) < float(ours[key]) - 1e-8 for key in metrics)
    if ours_no_worse and ours_strict:
        return "ours_dominates"
    if base_no_worse and base_strict:
        return "baseline_dominates"
    if not ours_strict and not base_strict:
        return "tie"
    return "tradeoff"


def write_artifacts(
    report: Mapping[str, Any],
    *,
    full_path: str | Path = DEFAULT_FULL,
    compact_path: str | Path = DEFAULT_COMPACT,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> dict[str, str]:
    full_path = Path(full_path)
    compact_path = Path(compact_path)
    markdown_path = Path(markdown_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(full_payload)
    full_file_hash = _file_sha256(full_path)
    compact = compact_certificate(report, full_path, full_file_hash)
    compact_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path.write_text(
        json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(compact), encoding="utf-8")
    return {
        "full": str(full_path),
        "full_file_sha256": full_file_hash,
        "compact": str(compact_path),
        "compact_file_sha256": _file_sha256(compact_path),
        "markdown": str(markdown_path),
    }


def compact_certificate(
    report: Mapping[str, Any], full_path: Path, full_file_hash: str
) -> dict[str, Any]:
    rows = [
        {
            key: row[key]
            for key in (
                "upstream_path",
                "source_sha256",
                "replication",
                "q_max_factor",
                "speed_setup_factor",
                "deadline_factor",
                "selected_feasible",
                "selected_source_core_metrics",
                "policy_source_core_metrics",
                "selected_pareto_nondominated",
                "baseline_relations",
                "selected_low_level_oracle_gap_max",
            )
        }
        for row in report["rows"]
    ]
    compact = {
        "schema_version": f"{SCHEMA_VERSION}.compact.v1",
        "gate": report["gate"],
        "full_artifact_path": str(full_path.relative_to(REPO_ROOT)),
        "full_artifact_gzip_sha256": full_file_hash,
        "full_scientific_content_sha256": report["scientific_content_sha256"],
        "provenance": {
            key: report["provenance"][key]
            for key in (
                "manifest_file_sha256",
                "manifest_content_sha256_excluding_self",
                "repository_commit",
                "instance_count",
                "factor_cell_count",
                "replications_per_cell",
                "registered_replications",
                "development_holdout_path_disjoint",
                "development_holdout_hash_disjoint",
            )
        },
        "freeze": report["freeze"],
        "frozen_plan": report["frozen_plan"],
        "baseline_policies": report["baseline_policies"],
        "evaluation_metrics": report["evaluation_metrics"],
        "candidate_generation_on_holdout": False,
        "holdout_feedback_used_for_selection": False,
        "aggregate": report["aggregate"],
        "rows": rows,
        "claim_boundary": report["claim_boundary"],
    }
    compact["compact_content_sha256_excluding_self"] = _digest(compact)
    return compact


def verify_artifact_pair(compact_path: str | Path = DEFAULT_COMPACT) -> dict[str, Any]:
    compact_path = Path(compact_path)
    compact = _load_json(compact_path)
    claimed = compact.get("compact_content_sha256_excluding_self")
    payload = dict(compact)
    payload.pop("compact_content_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise PortHoldoutContractError("compact certificate digest mismatch")
    full_path = REPO_ROOT / str(compact["full_artifact_path"])
    if _file_sha256(full_path) != compact["full_artifact_gzip_sha256"]:
        raise PortHoldoutContractError("full compressed ledger hash mismatch")
    with gzip.open(full_path, "rt", encoding="ascii") as stream:
        full = json.load(stream)
    if full.get("scientific_content_sha256") != compact["full_scientific_content_sha256"]:
        raise PortHoldoutContractError("full scientific-content hash mismatch")
    full_payload = dict(full)
    full_claimed = full_payload.pop("scientific_content_sha256", None)
    if _digest(full_payload) != full_claimed:
        raise PortHoldoutContractError("full ledger scientific-content digest mismatch")
    return {
        "ready": True,
        "compact_file_sha256": _file_sha256(compact_path),
        "full_file_sha256": _file_sha256(full_path),
        "registered_instance_count": full["gate"]["registered_instance_count"],
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    gate = compact["gate"]
    aggregate = compact["aggregate"]
    replications = "/".join(
        f"R{int(value)}"
        for value in compact["provenance"]["registered_replications"]
    )
    lines = [
        f"# BACASP-S {replications} external holdout",
        "",
        f"- Status: `{gate['status']}`",
        f"- Instances: `{gate['completed_instance_count']}/{gate['registered_instance_count']}`.",
        f"- Pareto-nondominated: `{aggregate['selected_pareto_nondominated_count']}/{gate['registered_instance_count']}`.",
        f"- Strictly dominates every registered baseline: `{aggregate['selected_strictly_dominates_every_baseline_count']}/{gate['registered_instance_count']}`.",
        f"- Frozen plan: `{compact['frozen_plan']['plan_id']}`.",
        f"- Full gzip SHA256: `{compact['full_artifact_gzip_sha256']}`.",
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
            "This is a fixed-slot, source-core, post-freeze public holdout. It is not",
            "a continuous-quay optimum, author-algorithm comparison, physical-port",
            "deployment, or port stochastic-stability certificate.",
            "",
        ]
    )
    return "\n".join(lines)


def _geomean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise PortHoldoutContractError("geometric mean requires finite positive ratios")
    return round(math.exp(statistics.fmean(math.log(value) for value in values)), 9)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PortHoldoutContractError(f"expected JSON object: {path}")
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
            raise PortHoldoutContractError("artifact contains non-finite value")
        return round(value, 9)
    return value


def _digest(value: Any) -> str:
    return sha256(
        json.dumps(
            _canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_SUITE_MANIFEST)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.verify_only:
        print(json.dumps(verify_artifact_pair(args.compact), indent=2, sort_keys=True))
        return 0
    report = build_holdout_report(
        suite_manifest_path=args.suite_manifest,
        preregistration_path=args.preregistration,
    )
    outputs = write_artifacts(
        report,
        full_path=args.full,
        compact_path=args.compact,
        markdown_path=args.markdown,
    )
    print(json.dumps({"gate": report["gate"], "outputs": outputs}, indent=2, sort_keys=True))
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_holdout_report",
    "compact_certificate",
    "verify_artifact_pair",
    "verify_preregistration",
    "verify_suite_manifest",
    "write_artifacts",
]
