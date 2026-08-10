"""Aggregate reproducible OR-domain evidence without erasing counterexamples."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_PRIOR = ARTIFACT_ROOT / "or_generalization_certificate_v4_20260810.json"
DEFAULT_BALANCE = (
    ARTIFACT_ROOT / "mmrcpsp_class_balance_holdout_v1_20260810.json"
)
DEFAULT_REPRODUCIBILITY = (
    ARTIFACT_ROOT / "mmrcpsp_reproducibility_audit_20260810.json"
)
DEFAULT_PORT = ARTIFACT_ROOT / "port_statewise_trajectory_v3_20260810.json"
DEFAULT_JSON = ARTIFACT_ROOT / "or_generalization_certificate_v5_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_v5_20260810.md"


class GeneralizationCertificateV5Error(ValueError):
    """Raised when a component result or its claim boundary drifts."""


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise GeneralizationCertificateV5Error(f"expected JSON object: {source}")
    return payload, sha256(raw).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_prior(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.or_generalization_certificate.v4":
        raise GeneralizationCertificateV5Error("unexpected v4 certificate schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True:
        raise GeneralizationCertificateV5Error("v4 protocol gate is open")
    if gate.get("all_negative_results_retained") is not True:
        raise GeneralizationCertificateV5Error("v4 negative evidence was erased")
    if gate.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateV5Error("v4 overclaims superiority")
    fjsp = report.get("fjsp_exact_ledger_confirmation_v2") or {}
    if fjsp.get("fixed_budget_cp_sat_dominates_count") != 14:
        raise GeneralizationCertificateV5Error("FJSP counterexample count drifted")
    mm = report.get("mmrcpsp_renewal_stream") or {}
    if mm.get("ours_nondominated_scenario_seed_count") != 0:
        raise GeneralizationCertificateV5Error("MMRCPSP total-cost result drifted")
    if mm.get("baseline_dominates_scenario_seed_count") != 21:
        raise GeneralizationCertificateV5Error("MMRCPSP total-cost counterexamples drifted")


def _validate_balance(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.mmrcpsp_class_balance_holdout.compact.v1":
        raise GeneralizationCertificateV5Error("unexpected class-balance schema")
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    boundary = report.get("claim_boundary") or {}
    if gate.get("pass") is not True:
        raise GeneralizationCertificateV5Error("class-balance protocol gate is open")
    expected = {
        "run_count": 175,
        "protocol_pass_count": 175,
        "ours_exact_oracle_run_count": 35,
        "scenario_seed_count": 35,
        "ours_nondominated_count": 35,
        "ours_strict_every_baseline_count": 0,
        "baseline_dominates_count": 0,
    }
    if any(aggregate.get(key) != value for key, value in expected.items()):
        raise GeneralizationCertificateV5Error("class-balance result drifted")
    if boundary.get("v1_total_cost_result_retained") is not True:
        raise GeneralizationCertificateV5Error("v1 total-cost result was erased")
    if boundary.get("metric_family_registered_after_v1_total_cost_result") is not True:
        raise GeneralizationCertificateV5Error("metric chronology was erased")
    if boundary.get("arrival_tapes_observed_before_metric_freeze") is not False:
        raise GeneralizationCertificateV5Error("arrival-tape holdout boundary drifted")
    if boundary.get("new_structural_instance_holdout") is not False:
        raise GeneralizationCertificateV5Error("known-library boundary was erased")


def _validate_reproducibility(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.mmrcpsp.reproducibility_audit.v1":
        raise GeneralizationCertificateV5Error("unexpected reproducibility schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True:
        raise GeneralizationCertificateV5Error("semantic reproduction failed")
    if gate.get("semantic_reproducibility_ready_count") != 2:
        raise GeneralizationCertificateV5Error("reproduction pair coverage drifted")
    if gate.get("bitwise_path_independent_gzip_claim_ready") is not False:
        raise GeneralizationCertificateV5Error("gzip boundary was erased")


def _validate_port(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.port.statewise_trajectory.v3.artifact.v1":
        raise GeneralizationCertificateV5Error("unexpected port artifact schema")
    body = report.get("report") or {}
    certificates = body.get("certificates") or {}
    source = certificates.get("bacasp_source_core") or {}
    synthetic = certificates.get("synthetic_migration") or {}
    if body.get("pass") is not True or source.get("ready") is not True:
        raise GeneralizationCertificateV5Error("port source-core gate is open")
    if synthetic.get("ready") is not True:
        raise GeneralizationCertificateV5Error("port synthetic migration gate is open")
    source_pareto = source.get("source_core_pareto") or {}
    robust_dominators = (source_pareto.get("dominated_by") or {}).get(
        "robust_maxweight"
    )
    if robust_dominators != ["reconfiguration_greedy", "spt_static"]:
        raise GeneralizationCertificateV5Error("port source-core counterexample drifted")
    if body.get("industrial_deployment_claim_ready") is not False:
        raise GeneralizationCertificateV5Error("port artifact overclaims deployment")


def build_certificate(
    *,
    prior_path: str | Path = DEFAULT_PRIOR,
    balance_path: str | Path = DEFAULT_BALANCE,
    reproducibility_path: str | Path = DEFAULT_REPRODUCIBILITY,
    port_path: str | Path = DEFAULT_PORT,
) -> dict[str, Any]:
    prior, prior_hash = _load(prior_path)
    balance, balance_hash = _load(balance_path)
    reproducibility, reproducibility_hash = _load(reproducibility_path)
    port, port_hash = _load(port_path)
    _validate_prior(prior)
    _validate_balance(balance)
    _validate_reproducibility(reproducibility)
    _validate_port(port)

    v1_mm = prior["mmrcpsp_renewal_stream"]
    balance_aggregate = balance["aggregate"]
    port_body = port["report"]
    port_source = port_body["certificates"]["bacasp_source_core"]
    port_synthetic = port_body["certificates"]["synthetic_migration"]
    port_selection = port_synthetic["selection"]
    report: dict[str, Any] = {
        "schema_version": "scheduleurm.or_generalization_certificate.v5",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_V5_PASS_MIXED_REPRODUCIBLE_EVIDENCE",
            "domain_count": 3,
            "all_protocol_gates_pass": True,
            "all_negative_results_retained": True,
            "semantic_reproducibility_audited": True,
            "performance_superiority_claim_ready": False,
            "arbitrary_domain_optimality_claim_ready": False,
            "positive_recurrence_from_finite_replay_claim_ready": False,
            "industrial_port_deployment_claim_ready": False,
        },
        "prior_v4": {
            "artifact_sha256": prior_hash,
            "status": prior["gate"]["status"],
            "retained_without_reinterpretation": True,
        },
        "fjsp": prior["fjsp_exact_ledger_confirmation_v2"],
        "fjsp_reproducibility": prior["fjsp_replication_audit"],
        "port": {
            "artifact_sha256": port_hash,
            "status": port_body["status"],
            "source_core": {
                "ready": port_source["ready"],
                "completed_source_vessels": port_source["source_core_metrics"][
                    "robust_maxweight"
                ]["completed_source_vessels"],
                "pareto_metrics": port_source["source_core_pareto"]["metrics"],
                "nondominated_policies": port_source["source_core_pareto"][
                    "nondominated_policies"
                ],
                "robust_maxweight_dominated_by": port_source[
                    "source_core_pareto"
                ]["dominated_by"]["robust_maxweight"],
                "mid_service_migration_evidence": port_source[
                    "mid_service_migration_evidence"
                ],
                "scope_boundary": port_source["scope_boundary"],
            },
            "synthetic_migration": {
                "ready": port_synthetic["ready"],
                "physical_terminal_observations_used": port_synthetic[
                    "physical_terminal_observations_used"
                ],
                "selected_candidate_id": port_selection["selected_candidate"][
                    "candidate_id"
                ],
                "duration_normalized_oracle_gap_alpha0": port_selection[
                    "oracle_gap_alpha0"
                ],
                "finite_family_penalty_bound_P0": port_selection[
                    "finite_family_penalty_bound_P0"
                ],
                "persistent_transition_resources_checked": port_selection[
                    "persistent_transition_resources_checked"
                ],
                "scope_boundary": port_synthetic["scope_boundary"],
            },
            "industrial_deployment_claim_ready": port_body[
                "industrial_deployment_claim_ready"
            ],
            "global_port_optimality_claim_ready": port_body[
                "global_port_optimality_claim_ready"
            ],
        },
        "mmrcpsp": {
            "v1_total_queue_result": {
                "status": v1_mm["status"],
                "scenario_seed_count": v1_mm["scenario_seed_count"],
                "ours_nondominated_count": v1_mm[
                    "ours_nondominated_scenario_seed_count"
                ],
                "baseline_dominates_count": v1_mm[
                    "baseline_dominates_scenario_seed_count"
                ],
                "pareto_coordinates": [
                    "final_backlog",
                    "time_average_queue",
                ],
                "retained_negative_result": True,
            },
            "prospective_class_balance_arrival_tape_holdout": {
                "artifact_sha256": balance_hash,
                "status": balance["gate"]["status"],
                "run_count": balance_aggregate["run_count"],
                "scenario_seed_count": balance_aggregate["scenario_seed_count"],
                "ours_exact_oracle_run_count": balance_aggregate[
                    "ours_exact_oracle_run_count"
                ],
                "ours_nondominated_count": balance_aggregate[
                    "ours_nondominated_count"
                ],
                "ours_strict_every_baseline_count": balance_aggregate[
                    "ours_strict_every_baseline_count"
                ],
                "baseline_dominates_count": balance_aggregate[
                    "baseline_dominates_count"
                ],
                "pareto_coordinates": balance["pareto_cost_coordinates"],
                "known_instance_and_action_library": balance["claim_boundary"][
                    "known_instance_and_action_library"
                ],
                "new_arrival_tapes": True,
                "new_structural_instance_holdout": balance["claim_boundary"][
                    "new_structural_instance_holdout"
                ],
                "metric_family_registered_after_v1_total_cost_result": balance[
                    "claim_boundary"
                ]["metric_family_registered_after_v1_total_cost_result"],
                "performance_superiority_claim_ready": False,
            },
            "reproducibility_audit": {
                "artifact_sha256": reproducibility_hash,
                "status": reproducibility["gate"]["status"],
                "pair_count": reproducibility["gate"]["pair_count"],
                "semantic_reproducibility_ready_count": reproducibility["gate"][
                    "semantic_reproducibility_ready_count"
                ],
                "raw_gzip_container_match_count": reproducibility["gate"][
                    "raw_gzip_container_match_count"
                ],
                "path_independent_gzip_claim_ready": reproducibility["gate"][
                    "bitwise_path_independent_gzip_claim_ready"
                ],
            },
        },
        "theorem_to_experiment_interpretation": {
            "throughput_stability_statement": (
                "The renewal-frame MaxWeight theorem controls drift under its "
                "slack assumptions; it is not a theorem of total-delay or "
                "makespan optimality."
            ),
            "v1_total_delay_result": (
                "The shortest registered trajectory baseline dominates the "
                "one-project-at-a-time abstraction on aggregate delay costs."
            ),
            "class_balance_result": (
                "On prospectively frozen arrival tapes, the normalized robust "
                "policy is nondominated on total queue plus worst-class queue "
                "costs, but never strictly dominates every baseline."
            ),
            "logical_conclusion": (
                "The two results identify an objective tradeoff and do not "
                "contradict the stability theorem or imply global delay "
                "superiority."
            ),
        },
        "claim_boundary": {
            "supports": [
                "a duration-normalized finite-trajectory scheduling interface across FJSP, MMRCPSP, and port models",
                "exact registered-family oracle audits in all 35 MMRCPSP class-balance scenario/seed cells",
                "a prospective arrival-tape MMRCPSP class-balance Pareto result with 35/35 nondominated cells",
                "semantic reproduction of both deterministic MMRCPSP result matrices",
                "statewise port transition feasibility with a measured finite-family oracle gap and penalty bound",
            ],
            "does_not_support": [
                "global FJSP, MMRCPSP, or port optimality",
                "universal delay, makespan, or performance superiority",
                "erasing the 21/21 MMRCPSP aggregate-delay counterexamples",
                "calling the known MMRCPSP instance/action library a new structural holdout",
                "positive recurrence inferred from finite replay",
                "industrial terminal deployment or physical port safety certification",
                "path-independent bitwise gzip identity",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]
    mm = report["mmrcpsp"]
    v1 = mm["v1_total_queue_result"]
    balance = mm["prospective_class_balance_arrival_tape_holdout"]
    port = report["port"]
    lines = [
        "# OR Generalization Certificate v5",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- This certificate treats protocol validity, registered-family oracle exactness, reproducibility, and performance as separate claims.",
        "",
        "| Domain | Evidence | Result | Retained boundary |",
        "|---|---|---|---|",
        (
            f"| FJSP | 20 frozen Hurink-family ledgers | candidate family nondominated 20/20; fixed-budget CP-SAT dominates {fjsp['fixed_budget_cp_sat_dominates_count']}/20 | no global FJSP optimum |"
        ),
        (
            f"| MMRCPSP total queue | {v1['scenario_seed_count']} scenario/seed cells | baseline dominates {v1['baseline_dominates_count']}/{v1['scenario_seed_count']} | retained negative result |"
        ),
        (
            f"| MMRCPSP class balance | {balance['scenario_seed_count']} prospectively frozen arrival tapes | ours nondominated {balance['ours_nondominated_count']}/{balance['scenario_seed_count']}; strict-all {balance['ours_strict_every_baseline_count']}/{balance['scenario_seed_count']} | known instance/action library |"
        ),
        (
            f"| Port | BACASP-S source-core plus separate synthetic migration | source robust MaxWeight dominated by {', '.join(port['source_core']['robust_maxweight_dominated_by'])}; synthetic oracle gap {port['synthetic_migration']['duration_normalized_oracle_gap_alpha0']}, P0={port['synthetic_migration']['finite_family_penalty_bound_P0']} | no industrial deployment |"
        ),
        "",
        "## Mathematical interpretation",
        "",
        f"- {report['theorem_to_experiment_interpretation']['throughput_stability_statement']}",
        f"- {report['theorem_to_experiment_interpretation']['v1_total_delay_result']}",
        f"- {report['theorem_to_experiment_interpretation']['class_balance_result']}",
        f"- {report['theorem_to_experiment_interpretation']['logical_conclusion']}",
        "",
        "## Reproducibility",
        "",
        f"- MMRCPSP semantic reruns: {mm['reproducibility_audit']['semantic_reproducibility_ready_count']}/{mm['reproducibility_audit']['pair_count']} exact.",
        "- Raw gzip container hashes differ because the FNAME header stores the output basename; decompressed results do not differ.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Supports: {claim}." for claim in report["claim_boundary"]["supports"])
    lines.extend(
        f"- Does not support: {claim}."
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
            raise GeneralizationCertificateV5Error(
                f"refusing to overwrite result: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown.write_text(markdown_report(report), encoding="utf-8")
    return output, markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--balance", type=Path, default=DEFAULT_BALANCE)
    parser.add_argument(
        "--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY
    )
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_certificate(
        prior_path=args.prior,
        balance_path=args.balance,
        reproducibility_path=args.reproducibility,
        port_path=args.port,
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
