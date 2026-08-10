"""Aggregate v4 OR-domain evidence while preserving every negative result."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_PRIOR = ARTIFACT_ROOT / "or_generalization_certificate_20260809.json"
DEFAULT_FJSP = ARTIFACT_ROOT / "fjsp_exact_ledger_confirmation_v2_20260810.json"
DEFAULT_FJSP_REPLICATION = ARTIFACT_ROOT / "fjsp_exact_ledger_replication_audit_20260810.json"
DEFAULT_MMRCPSP_STREAM = ARTIFACT_ROOT / "mmrcpsp_renewal_stream_v1_20260810.json"
DEFAULT_PORT_STATEWISE = ARTIFACT_ROOT / "port_statewise_trajectory_v3_20260810.json"
DEFAULT_JSON = ARTIFACT_ROOT / "or_generalization_certificate_v4_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_v4_20260810.md"


class GeneralizationCertificateV4Error(ValueError):
    """Raised when a component artifact or claim boundary drifts."""


def build_certificate(
    *,
    prior_path: str | Path = DEFAULT_PRIOR,
    fjsp_path: str | Path = DEFAULT_FJSP,
    fjsp_replication_path: str | Path = DEFAULT_FJSP_REPLICATION,
    mmrcpsp_stream_path: str | Path = DEFAULT_MMRCPSP_STREAM,
    port_statewise_path: str | Path = DEFAULT_PORT_STATEWISE,
) -> dict[str, Any]:
    prior, prior_hash = _load(prior_path)
    fjsp, fjsp_hash = _load(fjsp_path)
    fjsp_replication, fjsp_replication_hash = _load(fjsp_replication_path)
    mmrcpsp, mmrcpsp_hash = _load(mmrcpsp_stream_path)
    port, port_hash = _load(port_statewise_path)
    _validate_prior(prior)
    _validate_fjsp(fjsp)
    _validate_fjsp_replication(fjsp_replication)
    _validate_mmrcpsp_stream(mmrcpsp)
    _validate_port_statewise(port)

    fjsp_relations: dict[str, int] = {}
    for row in fjsp["rows"]:
        relation = str(row["cp_sat_exact_relation"])
        fjsp_relations[relation] = fjsp_relations.get(relation, 0) + 1
    mm_disposition = mmrcpsp["performance_dispositions"]
    port_report = port["report"]
    port_selection = port_report["certificates"]["synthetic_migration"][
        "selection"
    ]
    report = {
        "schema_version": "scheduleurm.or_generalization_certificate.v4",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_V4_PASS_MIXED_EVIDENCE",
            "domain_count": 3,
            "prior_evidence_retained": True,
            "all_new_protocol_gates_pass": True,
            "all_negative_results_retained": True,
            "performance_superiority_claim_ready": False,
            "arbitrary_domain_optimality_claim_ready": False,
            "industrial_port_deployment_claim_ready": False,
            "positive_recurrence_from_finite_replay_claim_ready": False,
        },
        "prior_v3": {
            "artifact_sha256": prior_hash,
            "status": prior["gate"]["status"],
            "retained_without_reinterpretation": True,
        },
        "fjsp_exact_ledger_confirmation_v2": {
            "artifact_sha256": fjsp_hash,
            "status": fjsp["gate"]["status"],
            "instance_count": fjsp["candidate_family_aggregate"]["instance_count"],
            "baseline_union_nondominated_count": fjsp[
                "candidate_family_aggregate"
            ]["exact_nondominated_count"],
            "cp_sat_feasible_reference_count": fjsp[
                "cp_sat_reference_aggregate"
            ]["feasible_reference_count"],
            "cp_sat_makespan_optimal_status_count": fjsp[
                "cp_sat_reference_aggregate"
            ]["optimality_proved_count"],
            "cp_sat_exact_relations": fjsp_relations,
            "fixed_budget_cp_sat_dominates_count": fjsp_relations.get(
                "comparator_dominates", 0
            ),
            "global_fjsp_optimality_claim_ready": fjsp["gate"][
                "global_fjsp_optimality_claim_ready"
            ],
        },
        "fjsp_replication_audit": {
            "artifact_sha256": fjsp_replication_hash,
            "status": fjsp_replication["gate"]["status"],
            "candidate_exact_ledger_match_count": fjsp_replication[
                "candidate_exact_ledger_match_count"
            ],
            "cp_sat_qualitative_disposition_match_count": fjsp_replication[
                "cp_sat_qualitative_disposition_match_count"
            ],
            "cp_sat_numeric_gap_match_count": fjsp_replication[
                "cp_sat_numeric_gap_match_count"
            ],
            "strict_full_artifact_reproducibility_ready": fjsp_replication[
                "strict_full_artifact_reproducibility_ready"
            ],
        },
        "mmrcpsp_renewal_stream": {
            "artifact_sha256": mmrcpsp_hash,
            "status": mmrcpsp["gate"]["status"],
            "registered_class_count": mmrcpsp["source_contract"][
                "registered_member_count"
            ],
            "run_count": mmrcpsp["aggregate"]["run_count"],
            "protocol_pass_count": mmrcpsp["aggregate"]["protocol_pass_count"],
            "ours_exact_oracle_run_count": mmrcpsp["aggregate"][
                "ours_exact_oracle_run_count"
            ],
            "ours_nondominated_scenario_seed_count": mm_disposition[
                "ours_nondominated_count"
            ],
            "ours_strict_every_baseline_scenario_seed_count": mm_disposition[
                "ours_strict_every_baseline_count"
            ],
            "baseline_dominates_scenario_seed_count": mm_disposition[
                "baseline_dominates_count"
            ],
            "scenario_seed_count": mm_disposition["scenario_seed_count"],
            "finite_sample_drift_is_diagnostic": mmrcpsp["claim_boundary"][
                "finite_sample_drift_is_diagnostic_not_positive_recurrence_proof"
            ],
            "stochastic_stability_claim_ready": mmrcpsp["claim_boundary"][
                "stochastic_stability_claim_ready"
            ],
        },
        "port_statewise_trajectory_v3": {
            "artifact_sha256": port_hash,
            "status": port_report["status"],
            "source_core_ready": port_report["certificates"][
                "bacasp_source_core"
            ]["ready"],
            "synthetic_migration_ready": port_report["certificates"][
                "synthetic_migration"
            ]["ready"],
            "duration_normalized_oracle_gap_alpha0": port_selection[
                "oracle_gap_alpha0"
            ],
            "finite_family_penalty_bound_P0": port_selection[
                "finite_family_penalty_bound_P0"
            ],
            "score_semantic_pruning_ready": port_report[
                "score_semantic_pruning_ready"
            ],
            "terminal_pareto_is_diagnostic_only": True,
            "industrial_deployment_claim_ready": port_report[
                "industrial_deployment_claim_ready"
            ],
            "global_port_optimality_claim_ready": port_report[
                "global_port_optimality_claim_ready"
            ],
        },
        "common_theorem_interface": {
            "registered_action": (
                "a complete finite configuration trajectory with a duration, "
                "cumulative lower-service vector, and bounded penalty"
            ),
            "duration_normalized_score": (
                "(Q^T cumulative_lower_service - bounded_penalty) / (tau/H)"
            ),
            "statewise_feasibility": (
                "resource, precedence, capacity, and persistent-transition "
                "constraints are checked before oracle selection"
            ),
            "finite_family_boundary": (
                "exact oracle statements concern the registered feasible family, "
                "not an unrestricted domain-global optimizer"
            ),
        },
        "claim_boundary": {
            "supports": [
                "a common duration-normalized finite-trajectory interface across FJSP, MMRCPSP, and port scheduling",
                "pathwise queue recurrence, workload conservation, and drift-identity audits for the registered MMRCPSP stream",
                "persistent-resource and generalized transition-cost semantics in the synthetic port statewise model",
                "exact integer-ledger FJSP candidate-family comparisons with an independent fixed-budget CP-SAT reference",
            ],
            "does_not_support": [
                "global FJSP or MMRCPSP optimality",
                "dominance over unrestricted exact or state-of-the-art domain solvers",
                "bitwise repeatability of wall-clock-limited CP-SAT incumbents",
                "industrial terminal deployment or physical port safety certification",
                "positive recurrence inferred from finite replay",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def _validate_prior(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.or_generalization_certificate.v3":
        raise GeneralizationCertificateV4Error("unexpected prior certificate schema")
    if (report.get("gate") or {}).get("pass") is not True:
        raise GeneralizationCertificateV4Error("prior certificate gate is open")


def _validate_fjsp(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.fjsp_exact_ledger_confirmation.v2.compact.v1":
        raise GeneralizationCertificateV4Error("unexpected FJSP confirmation schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True or gate.get("global_fjsp_optimality_claim_ready") is not False:
        raise GeneralizationCertificateV4Error("FJSP confirmation boundary drifted")
    candidate = report.get("candidate_family_aggregate") or {}
    reference = report.get("cp_sat_reference_aggregate") or {}
    if candidate.get("instance_count") != 20 or candidate.get("exact_nondominated_count") != 20:
        raise GeneralizationCertificateV4Error("FJSP candidate-family result drifted")
    if reference.get("feasible_reference_count") != 20:
        raise GeneralizationCertificateV4Error("FJSP CP-SAT reference coverage drifted")
    if sum(row.get("cp_sat_exact_relation") == "comparator_dominates" for row in report["rows"]) != 14:
        raise GeneralizationCertificateV4Error("FJSP CP-SAT counterexample count drifted")


def _validate_fjsp_replication(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.fjsp_exact_ledger.replication_audit.v1":
        raise GeneralizationCertificateV4Error("unexpected FJSP replication schema")
    if (report.get("gate") or {}).get("pass") is not True:
        raise GeneralizationCertificateV4Error("FJSP replication audit failed")
    if report.get("candidate_exact_ledger_match_count") != 20:
        raise GeneralizationCertificateV4Error("FJSP candidate ledgers did not reproduce")
    if report.get("cp_sat_qualitative_disposition_match_count") != 20:
        raise GeneralizationCertificateV4Error("FJSP CP-SAT disposition drifted")
    if report.get("cp_sat_numeric_gap_match_count") != 17:
        raise GeneralizationCertificateV4Error("FJSP CP-SAT numeric audit drifted")
    if report.get("strict_full_artifact_reproducibility_ready") is not False:
        raise GeneralizationCertificateV4Error("FJSP replication boundary was erased")


def _validate_mmrcpsp_stream(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.mmrcpsp_renewal_stream.compact.v1":
        raise GeneralizationCertificateV4Error("unexpected MMRCPSP stream schema")
    if (report.get("gate") or {}).get("pass") is not True:
        raise GeneralizationCertificateV4Error("MMRCPSP stream protocol failed")
    if (report.get("source_contract") or {}).get("registered_member_count") != 25:
        raise GeneralizationCertificateV4Error("MMRCPSP class count drifted")
    aggregate = report.get("aggregate") or {}
    if aggregate.get("run_count") != 105 or aggregate.get("protocol_pass_count") != 105:
        raise GeneralizationCertificateV4Error("MMRCPSP run matrix is incomplete")
    if aggregate.get("ours_exact_oracle_run_count") != 21:
        raise GeneralizationCertificateV4Error("MMRCPSP exact-oracle audit drifted")
    boundary = report.get("claim_boundary") or {}
    if boundary.get("stochastic_stability_claim_ready") is not False:
        raise GeneralizationCertificateV4Error("MMRCPSP finite replay overclaims stability")


def _validate_port_statewise(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != "scheduleurm.port.statewise_trajectory.v3.artifact.v1":
        raise GeneralizationCertificateV4Error("unexpected port statewise schema")
    report = artifact.get("report") or {}
    if report.get("pass") is not True:
        raise GeneralizationCertificateV4Error("port statewise protocol failed")
    if report.get("score_semantic_pruning_ready") is not True:
        raise GeneralizationCertificateV4Error("port score-semantic audit failed")
    if report.get("industrial_deployment_claim_ready") is not False:
        raise GeneralizationCertificateV4Error("port artifact overclaims deployment")
    if report.get("global_port_optimality_claim_ready") is not False:
        raise GeneralizationCertificateV4Error("port artifact overclaims optimality")
    certificates = report.get("certificates") or {}
    if (certificates.get("bacasp_source_core") or {}).get("ready") is not True:
        raise GeneralizationCertificateV4Error("port source-core certificate failed")
    if (certificates.get("synthetic_migration") or {}).get("ready") is not True:
        raise GeneralizationCertificateV4Error("port synthetic migration certificate failed")


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp_exact_ledger_confirmation_v2"]
    replication = report["fjsp_replication_audit"]
    mm = report["mmrcpsp_renewal_stream"]
    port = report["port_statewise_trajectory_v3"]
    lines = [
        "# OR Generalization Certificate v4",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Protocol validity, finite-family oracle exactness, and performance superiority remain separate claims.",
        "",
        "| Domain | New evidence | Protocol | Performance disposition | Boundary |",
        "|---|---|---|---|---|",
        (
            f"| FJSP | 20 unused Hurink-family instances | `{fjsp['status']}` | "
            f"candidate family nondominated 20/20; CP-SAT dominates {fjsp['fixed_budget_cp_sat_dominates_count']}/20 | no global optimum |"
        ),
        (
            f"| MMRCPSP | 25-class renewal stream, 105 runs | `{mm['status']}` | "
            f"ours nondominated {mm['ours_nondominated_scenario_seed_count']}/{mm['scenario_seed_count']} scenario/seeds | finite drift only |"
        ),
        (
            f"| Port | statewise synthetic transition model | `{port['status']}` | "
            f"oracle gap {port['duration_normalized_oracle_gap_alpha0']}; P0={port['finite_family_penalty_bound_P0']} | no industrial deployment |"
        ),
        "",
        "## Reproducibility and retained counterexamples",
        "",
        f"- FJSP frozen candidate ledgers reproduce {replication['candidate_exact_ledger_match_count']}/20.",
        f"- CP-SAT qualitative relations reproduce {replication['cp_sat_qualitative_disposition_match_count']}/20, but exact numeric gaps reproduce only {replication['cp_sat_numeric_gap_match_count']}/20 because the comparator is wall-clock limited.",
        "- The MMRCPSP stream does not convert empirical tail drift into a recurrence theorem.",
        "- The BACASP source-core and synthetic migration certificates remain non-substitutable.",
        "",
        "## Claim Boundary",
        "",
    ]
    lines.extend(f"- Supports: {claim}" for claim in report["claim_boundary"]["supports"])
    lines.extend(
        f"- Does not support: {claim}"
        for claim in report["claim_boundary"]["does_not_support"]
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output = Path(json_path)
    markdown = Path(markdown_path)
    for path in (output, markdown):
        if path.exists():
            raise GeneralizationCertificateV4Error(f"refusing to overwrite result: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(report), encoding="utf-8")
    return output, markdown


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise GeneralizationCertificateV4Error(f"expected JSON object: {source}")
    return payload, sha256(raw).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--fjsp", type=Path, default=DEFAULT_FJSP)
    parser.add_argument("--fjsp-replication", type=Path, default=DEFAULT_FJSP_REPLICATION)
    parser.add_argument("--mmrcpsp-stream", type=Path, default=DEFAULT_MMRCPSP_STREAM)
    parser.add_argument("--port-statewise", type=Path, default=DEFAULT_PORT_STATEWISE)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_certificate(
        prior_path=args.prior,
        fjsp_path=args.fjsp,
        fjsp_replication_path=args.fjsp_replication,
        mmrcpsp_stream_path=args.mmrcpsp_stream,
        port_statewise_path=args.port_statewise,
    )
    output, markdown = write_outputs(
        report, json_path=args.json, markdown_path=args.markdown
    )
    print(
        json.dumps(
            {"gate": report["gate"], "json": str(output), "markdown": str(markdown)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
