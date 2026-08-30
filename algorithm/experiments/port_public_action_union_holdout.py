"""Prospective BACASP-S holdout for the finite port trajectory-action union."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from algorithm.experiments.port_public_action_union import (
    OUTER_SCORE_CONFIG,
    POLICY,
    build_port_public_action_union,
    frozen_action_family,
)
from algorithm.experiments.port_public_benchmark_adapter import (
    adapt_bacasp_s_to_port_instance,
    parse_bacasp_s_instance,
)
from algorithm.experiments.port_public_external_holdout import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    REPO_ROOT,
    SOURCE_COST_METRICS,
    _aggregate,
    _digest,
    _file_sha256,
    _relation,
    verify_suite_manifest,
)
from algorithm.experiments.port_scheduling_benchmark import (
    BASELINE_POLICIES,
    RECONFIGURATION_ACTIONS,
)


SCHEMA_VERSION = "scheduleurm.port_public_action_union_holdout.v1"
EXPECTED_REPLICATIONS = (81, 82)
DEFAULT_SUITE_ROOT = REPO_ROOT / "tests/data/port_bacasp_s_action_union_holdout_r81_r82"
DEFAULT_MANIFEST = DEFAULT_SUITE_ROOT / "source_manifest.json"
DEFAULT_PREREGISTRATION = DEFAULT_SUITE_ROOT / "preregistration.json"
DEFAULT_FULL = REPO_ROOT / "md/experiment_artifacts/port_action_union_r81_r82_20260830_full.json.gz"
DEFAULT_COMPACT = REPO_ROOT / "md/experiment_artifacts/port_action_union_r81_r82_20260830.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md/port_action_union_r81_r82_20260830.md"


class PortActionUnionHoldoutError(ValueError):
    """Raised when the prospective action-union contract does not hold."""


def verify_preregistration(
    preregistration_path: str | Path,
    suite_manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(preregistration_path)
    prereg = _load_json(path)
    if prereg.get("schema_version") != "scheduleurm.port_action_union_preregistration.v1":
        raise PortActionUnionHoldoutError("unexpected action-union preregistration schema")
    claimed = str(prereg.get("preregistration_sha256_excluding_self") or "")
    payload = dict(prereg)
    payload.pop("preregistration_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise PortActionUnionHoldoutError("preregistration digest mismatch")
    if prereg.get("holdout_outcomes_observed_before_freeze") is not False:
        raise PortActionUnionHoldoutError("pre-outcome freeze is not certified")
    if _file_sha256(suite_manifest_path) != prereg.get("suite_manifest_file_sha256"):
        raise PortActionUnionHoldoutError("suite manifest changed after freeze")
    if tuple(prereg["split_rule"]["external_holdout_replications"]) != EXPECTED_REPLICATIONS:
        raise PortActionUnionHoldoutError("unexpected registered replications")

    expected_family = [plan.snapshot() for plan in frozen_action_family()]
    if prereg.get("frozen_action_family") != expected_family:
        raise PortActionUnionHoldoutError("registered action family changed after freeze")
    if prereg.get("outer_score_config") != OUTER_SCORE_CONFIG.snapshot():
        raise PortActionUnionHoldoutError("outer score configuration changed after freeze")
    if tuple(prereg.get("baseline_policies") or ()) != tuple(BASELINE_POLICIES):
        raise PortActionUnionHoldoutError("baseline family changed after freeze")
    if tuple(prereg.get("evaluation_metrics") or ()) != tuple(SOURCE_COST_METRICS):
        raise PortActionUnionHoldoutError("evaluation metrics changed after freeze")
    for relative, expected_hash in sorted(
        (prereg.get("implementation_file_sha256") or {}).items()
    ):
        source = REPO_ROOT / relative
        if not source.is_file() or _file_sha256(source) != expected_hash:
            raise PortActionUnionHoldoutError(
                f"implementation changed after preregistration: {relative}"
            )
    return {
        "ready": True,
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "content_sha256_excluding_self": claimed,
        "action_family_size": len(expected_family),
        "holdout_feedback_used_for_selection": False,
    }


def build_holdout_report(
    *,
    suite_manifest_path: str | Path = DEFAULT_MANIFEST,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
) -> dict[str, Any]:
    provenance = verify_suite_manifest(
        suite_manifest_path,
        expected_replications=EXPECTED_REPLICATIONS,
    )
    freeze = verify_preregistration(preregistration_path, suite_manifest_path)
    manifest = _load_json(suite_manifest_path)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for registered in manifest["instances"]:
        try:
            rows.append(_evaluate_registered_instance(registered))
        except Exception as exc:
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

    aggregate = _aggregate(rows, registered_replications=EXPECTED_REPLICATIONS)
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
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": {
            "pass": protocol_pass,
            "status": (
                "PORT_ACTION_UNION_HOLDOUT_PASS"
                if protocol_pass
                else "PORT_ACTION_UNION_HOLDOUT_FAIL"
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
        "selected_policy": POLICY,
        "frozen_action_family": [plan.snapshot() for plan in frozen_action_family()],
        "baseline_policies": list(BASELINE_POLICIES),
        "evaluation_metrics": list(SOURCE_COST_METRICS),
        "candidate_generation_on_holdout": False,
        "candidate_scoring_is_part_of_frozen_policy": True,
        "holdout_feedback_used_for_policy_revision": False,
        "rows": rows,
        "failures": failures,
        "aggregate": aggregate,
        "selected_plan_counts": dict(
            sorted(Counter(row["selected_plan_id"] for row in rows).items())
        ),
        "theorem_mapping": {
            "candidate_action": "complete registered port trajectory",
            "finite_family_exact_argmax": True,
            "outer_score": "candidate-independent terminal virtual service",
            "outer_penalty_bound": 0.0,
            "low_level_score": "statewise Q^T lower_service minus bounded penalty",
        },
        "claim_boundary": {
            "supports": [
                "post-freeze action-union evaluation on 54 disjoint BACASP-S instances",
                "exact finite-family trajectory selection under the registered simulator",
                "same-input paired comparison with every registered single-policy baseline",
            ],
            "does_not_support": [
                "unrestricted BACASP-S optimality",
                "comparison with BACASP-S author algorithms",
                "physical-port deployment or calibrated migration cost",
                "automatic port stochastic-recurrence transfer",
            ],
        },
    }
    report["scientific_content_sha256"] = _digest(report)
    return report


def _evaluate_registered_instance(registered: Mapping[str, Any]) -> dict[str, Any]:
    source = parse_bacasp_s_instance(
        REPO_ROOT / str(registered["local_path"]),
        expected_sha256=str(registered["sha256"]),
    )
    adapted = adapt_bacasp_s_to_port_instance(source)
    union = build_port_public_action_union(adapted)
    selected = union["selected_result"]
    candidate_metrics = {
        str(row["plan"]["plan_id"]): dict(row["source_core_metrics"])
        for row in union["candidate_family"]
    }
    baseline_metrics = {}
    for baseline in BASELINE_POLICIES:
        plan_id = f"plan|{baseline}"
        if plan_id not in candidate_metrics:
            raise PortActionUnionHoldoutError(f"missing baseline action: {baseline}")
        baseline_metrics[baseline] = candidate_metrics[plan_id]
    ours = dict(union["selected_source_core_metrics"])
    relations = {
        baseline: _relation(ours, baseline_metrics[baseline], SOURCE_COST_METRICS)
        for baseline in BASELINE_POLICIES
    }
    audits = list(selected.get("decision_audits") or ())
    oracle_gaps = [float(row.get("oracle_gap_alpha0") or 0.0) for row in audits]
    no_mid_service = all(
        int(selected["action_counts"].get(action_type, 0)) == 0
        for action_type in RECONFIGURATION_ACTIONS
    )
    selected_feasible = bool(
        union["gate"]["pass"]
        and selected.get("pass") is True
        and selected["public_source_feasibility_audit"].get(
            "resource_feasibility_ready"
        )
        is True
        and no_mid_service
    )
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
        "selected_plan_id": union["selection"]["selected_plan_id"],
        "selected_source_core_metrics": ours,
        "policy_source_core_metrics": baseline_metrics,
        "selected_pareto_nondominated": bool(
            union["selection"]["selected_pareto_nondominated"]
        ),
        "baseline_relations": relations,
        "selected_low_level_oracle_gap_max": max(oracle_gaps, default=0.0),
        "selected_low_level_exact_slot_count": sum(
            bool(row.get("theorem_ready")) for row in audits
        ),
        "selected_low_level_slot_count": len(audits),
        "candidate_count": int(union["selection"]["candidate_count"]),
        "action_union_report": union,
    }


def compact_certificate(
    report: Mapping[str, Any], full_path: Path, full_file_hash: str
) -> dict[str, Any]:
    compact: dict[str, Any] = {
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
        "frozen_action_family": report["frozen_action_family"],
        "baseline_policies": report["baseline_policies"],
        "evaluation_metrics": report["evaluation_metrics"],
        "candidate_generation_on_holdout": False,
        "holdout_feedback_used_for_policy_revision": False,
        "aggregate": report["aggregate"],
        "selected_plan_counts": report["selected_plan_counts"],
        "rows": [
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
                    "selected_plan_id",
                    "selected_source_core_metrics",
                    "policy_source_core_metrics",
                    "selected_pareto_nondominated",
                    "baseline_relations",
                    "selected_low_level_oracle_gap_max",
                    "candidate_count",
                )
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
    full = _repo_path(full_path)
    compact = _repo_path(compact_path)
    markdown = _repo_path(markdown_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")
    with full.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(payload)
    full_hash = _file_sha256(full)
    certificate = compact_certificate(report, full, full_hash)
    compact.parent.mkdir(parents=True, exist_ok=True)
    compact.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(markdown_report(certificate), encoding="utf-8")
    return {
        "full": str(full),
        "full_file_sha256": full_hash,
        "compact": str(compact),
        "compact_file_sha256": _file_sha256(compact),
        "markdown": str(markdown),
    }


def verify_artifact_pair(compact_path: str | Path = DEFAULT_COMPACT) -> dict[str, Any]:
    compact_path = Path(compact_path)
    compact = _load_json(compact_path)
    claimed = compact.get("compact_content_sha256_excluding_self")
    payload = dict(compact)
    payload.pop("compact_content_sha256_excluding_self", None)
    if _digest(payload) != claimed:
        raise PortActionUnionHoldoutError("compact digest mismatch")
    full_path = REPO_ROOT / str(compact["full_artifact_path"])
    if _file_sha256(full_path) != compact["full_artifact_gzip_sha256"]:
        raise PortActionUnionHoldoutError("full gzip hash mismatch")
    with gzip.open(full_path, "rt", encoding="ascii") as stream:
        full = json.load(stream)
    full_payload = dict(full)
    full_claim = full_payload.pop("scientific_content_sha256", None)
    if _digest(full_payload) != full_claim:
        raise PortActionUnionHoldoutError("full scientific digest mismatch")
    if full_claim != compact["full_scientific_content_sha256"]:
        raise PortActionUnionHoldoutError("compact/full scientific digest mismatch")
    return {
        "ready": True,
        "registered_instance_count": full["gate"]["registered_instance_count"],
        "compact_file_sha256": _file_sha256(compact_path),
        "full_file_sha256": _file_sha256(full_path),
    }


def markdown_report(compact: Mapping[str, Any]) -> str:
    gate = compact["gate"]
    aggregate = compact["aggregate"]
    lines = [
        "# BACASP-S R81/R82 finite action-union holdout",
        "",
        f"- Status: `{gate['status']}`",
        f"- Instances: `{gate['completed_instance_count']}/{gate['registered_instance_count']}`.",
        f"- Pareto-nondominated: `{aggregate['selected_pareto_nondominated_count']}/{gate['registered_instance_count']}`.",
        f"- Strictly dominates every baseline: `{aggregate['selected_strictly_dominates_every_baseline_count']}/{gate['registered_instance_count']}`.",
        f"- Candidate plans: `{len(compact['frozen_action_family'])}`.",
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
            "The action union is exact only over the registered deterministic candidate",
            "family. It is not an unrestricted BACASP-S optimum, author-algorithm",
            "comparison, physical-port deployment, or stochastic-stability certificate.",
            "",
        ]
    )
    return "\n".join(lines)


def _repo_path(path: str | Path) -> Path:
    output = Path(path)
    return output if output.is_absolute() else REPO_ROOT / output


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PortActionUnionHoldoutError(f"expected JSON object: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
