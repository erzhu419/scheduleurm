"""Aggregate prospective OR-domain evidence without erasing counterexamples."""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DATA_ROOT = REPO_ROOT / "tests" / "data"

DEFAULT_FJSP_KACEM = (
    DATA_ROOT / "fjsp_kacem_external_holdout" / "fjsp_kacem_external_holdout_gate.json"
)
DEFAULT_FJSP_FAMILY = (
    DATA_ROOT
    / "fjsp_family_external_holdout"
    / "fjsp_family_external_holdout_gate.json"
)
DEFAULT_FJSP_HURINK = ARTIFACT_ROOT / "fjsp_hurink_holdout_20260809.json"
DEFAULT_FJSP_NUMERIC = ARTIFACT_ROOT / "fjsp_numeric_disposition_gate_20260810.json"
DEFAULT_MM_V1 = (
    DATA_ROOT
    / "mmrcpsp_psplib_external_holdout"
    / "mmrcpsp_psplib_external_holdout_gate.json"
)
DEFAULT_MM_REPAIR = ARTIFACT_ROOT / "mmrcpsp_holdout1_repair_regression_20260809.json"
DEFAULT_MM_V2 = (
    DATA_ROOT
    / "mmrcpsp_psplib_external_holdout_v2"
    / "mmrcpsp_psplib_external_holdout_v2_gate.json"
)
DEFAULT_MM_FAMILY = (
    DATA_ROOT
    / "mmrcpsp_family_external_holdout"
    / "mmrcpsp_family_external_holdout_gate.json"
)
DEFAULT_PORT_TRAJECTORY = ARTIFACT_ROOT / "port_trajectory_upgrade_compact_20260809.json"
DEFAULT_PORT_FAILED = ARTIFACT_ROOT / "port_bacasp_s_external_holdout_r89_r90_20260809.json"
DEFAULT_PORT_CONFIRMATION = ARTIFACT_ROOT / "port_bacasp_s_external_holdout_r87_r88_20260809.json"
DEFAULT_PORT_FACTOR = ARTIFACT_ROOT / "port_factor_policy_holdout_r85_r86_20260809.json.gz"
DEFAULT_JSON = ARTIFACT_ROOT / "or_generalization_certificate_20260809.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_20260809.md"


class GeneralizationCertificateError(ValueError):
    """Raised when a component artifact violates its declared boundary."""


def build_certificate(
    *,
    fjsp_kacem_path: str | Path = DEFAULT_FJSP_KACEM,
    fjsp_family_path: str | Path = DEFAULT_FJSP_FAMILY,
    fjsp_hurink_path: str | Path = DEFAULT_FJSP_HURINK,
    fjsp_numeric_path: str | Path = DEFAULT_FJSP_NUMERIC,
    mm_v1_path: str | Path = DEFAULT_MM_V1,
    mm_repair_path: str | Path = DEFAULT_MM_REPAIR,
    mm_v2_path: str | Path = DEFAULT_MM_V2,
    mm_family_path: str | Path = DEFAULT_MM_FAMILY,
    port_trajectory_path: str | Path = DEFAULT_PORT_TRAJECTORY,
    port_failed_path: str | Path = DEFAULT_PORT_FAILED,
    port_confirmation_path: str | Path = DEFAULT_PORT_CONFIRMATION,
    port_factor_path: str | Path = DEFAULT_PORT_FACTOR,
) -> dict[str, Any]:
    loaded = {
        "fjsp_kacem": _load(fjsp_kacem_path),
        "fjsp_family": _load(fjsp_family_path),
        "fjsp_hurink": _load(fjsp_hurink_path),
        "fjsp_numeric": _load(fjsp_numeric_path),
        "mm_v1": _load(mm_v1_path),
        "mm_repair": _load(mm_repair_path),
        "mm_v2": _load(mm_v2_path),
        "mm_family": _load(mm_family_path),
        "port_trajectory": _load(port_trajectory_path),
        "port_failed": _load(port_failed_path),
        "port_confirmation": _load(port_confirmation_path),
        "port_factor": _load(port_factor_path),
    }
    payload = {key: value[0] for key, value in loaded.items()}
    hashes = {key: value[1] for key, value in loaded.items()}

    _validate_fjsp_kacem(payload["fjsp_kacem"])
    _validate_fjsp_family(payload["fjsp_family"])
    _validate_fjsp_hurink(payload["fjsp_hurink"])
    _validate_fjsp_numeric(
        payload["fjsp_numeric"],
        family_sha256=hashes["fjsp_family"],
        hurink_sha256=hashes["fjsp_hurink"],
    )
    _validate_mm_v1(payload["mm_v1"])
    _validate_mm_repair(payload["mm_repair"], hashes["mm_v1"])
    _validate_mm_v2(payload["mm_v2"])
    _validate_mm_family(payload["mm_family"])
    _validate_port_trajectory(payload["port_trajectory"])
    _validate_port_failed(payload["port_failed"])
    _validate_port_confirmation(payload["port_confirmation"])
    _validate_port_factor(payload["port_factor"])

    fjsp_family = payload["fjsp_family"]
    fjsp_hurink = payload["fjsp_hurink"]
    fjsp_numeric = payload["fjsp_numeric"]
    mm_family = payload["mm_family"]
    port_failed = payload["port_failed"]
    port_confirmation = payload["port_confirmation"]
    port_factor = payload["port_factor"]

    report = {
        "schema_version": "scheduleurm.or_generalization_certificate.v3",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_CERTIFICATE_PASS_MIXED_EVIDENCE",
            "domain_count": 3,
            "all_registered_outcomes_retained": True,
            "all_component_protocols_auditable": True,
            "all_selected_policies_pareto_nondominated": False,
            "performance_superiority_claim_ready": False,
            "arbitrary_domain_optimality_claim_ready": False,
            "stochastic_stability_transfers_without_domain_model": False,
        },
        "fjsp": {
            "kacem_small_holdout": {
                "artifact_sha256": hashes["fjsp_kacem"],
                "instance_count": payload["fjsp_kacem"]["gate"]["instance_count"],
                "pareto_nondominated_count": payload["fjsp_kacem"]["gate"]
                ["ours_instance_pareto_nondominated_count"],
            },
            "first_family_holdout": {
                "artifact_sha256": hashes["fjsp_family"],
                "status": fjsp_family["gate"]["status"],
                "protocol_pass": fjsp_family["gate"]["pass"],
                "instance_count": 20,
                "pareto_nondominated_count": fjsp_numeric["groups"]
                ["first_family_holdout"]
                ["exact_ledger_pareto_nondominated_count"],
                "source_reported_pareto_nondominated_count": fjsp_family
                ["aggregate"]["ours_pareto_nondominated_count"],
                "numeric_false_dominance_count": fjsp_numeric["groups"]
                ["first_family_holdout"]["numeric_false_dominance_count"],
                "fixed_cp_sat_dominates_generated_family_count": fjsp_numeric
                ["groups"]["first_family_holdout"]
                ["fixed_cp_sat_dominates_generated_family_count"],
                "strict_every_baseline_count": fjsp_family["aggregate"]
                ["ours_strict_every_baseline_count"],
            },
            "hurink_confirmation": {
                "artifact_sha256": hashes["fjsp_hurink"],
                "status": fjsp_hurink["gate"]["status"],
                "protocol_pass": fjsp_hurink["gate"]["pass"],
                "complete_reference_coverage": fjsp_hurink["gate"]
                ["complete_cp_sat_reference_coverage_ready"],
                "instance_count": 20,
                "pareto_nondominated_count": fjsp_numeric["groups"]
                ["hurink_confirmation"]
                ["exact_ledger_pareto_nondominated_count"],
                "source_reported_pareto_nondominated_count": fjsp_hurink
                ["aggregate"]["ours_pareto_nondominated_count"],
                "numeric_false_dominance_count": fjsp_numeric["groups"]
                ["hurink_confirmation"]["numeric_false_dominance_count"],
                "fixed_cp_sat_dominates_generated_family_count": fjsp_numeric
                ["groups"]["hurink_confirmation"]
                ["fixed_cp_sat_dominates_generated_family_count"],
                "strict_every_baseline_count": fjsp_hurink["aggregate"]
                ["ours_strict_every_baseline_count"],
            },
            "numeric_disposition": {
                "artifact_sha256": hashes["fjsp_numeric"],
                "status": fjsp_numeric["gate"]["status"],
                "source_protocol_artifacts_immutable": fjsp_numeric["gate"]
                ["source_protocol_artifacts_immutable"],
                "baseline_union_pareto_nondominance_ready": fjsp_numeric["gate"]
                ["baseline_union_pareto_nondominance_ready"],
                "fixed_cp_sat_candidate_family_gap_open": fjsp_numeric["gate"]
                ["fixed_cp_sat_candidate_family_gap_open"],
            },
            "performance_superiority_claim_ready": False,
            "global_optimality_claim_ready": False,
        },
        "mmrcpsp": {
            "v1_failure_artifact_sha256": hashes["mm_v1"],
            "v1_failure_retained": payload["mm_v1"]["gate"]["pass"] is False,
            "opened_row_repair": {
                "artifact_sha256": hashes["mm_repair"],
                "pass_count": payload["mm_repair"]["pass_count"],
                "instance_count": payload["mm_repair"]["instance_count"],
                "prospective_claim_ready": payload["mm_repair"]
                ["prospective_external_holdout_claim_ready"],
            },
            "disjoint_v2": {
                "artifact_sha256": hashes["mm_v2"],
                "instance_count": payload["mm_v2"]["gate"]
                ["registered_instance_count"],
                "pareto_nondominated_count": payload["mm_v2"]["gate"]
                ["ours_instance_pareto_nondominated_count"],
            },
            "multi_family_confirmation": {
                "artifact_sha256": hashes["mm_family"],
                "status": mm_family["gate"]["status"],
                "protocol_pass": mm_family["gate"]["pass"],
                **_performance_counts(mm_family),
                "public_optimum_attained_count": mm_family["aggregate"]
                ["ours_attains_public_optimum_count"],
            },
            "performance_superiority_claim_ready": False,
            "global_optimality_claim_ready": False,
        },
        "port": {
            "generated_trajectory_family": {
                "artifact_sha256": hashes["port_trajectory"],
                "family_size": payload["port_trajectory"]["trajectory_family_size"],
                "oracle_gap_alpha0": payload["port_trajectory"]
                ["trajectory_oracle_gap_alpha0"],
                "performance_superiority_claim_ready": payload["port_trajectory"]
                ["performance_superiority_claim_ready"],
            },
            "pre_adapter_failure": {
                "artifact_sha256": hashes["port_failed"],
                "status": port_failed["gate"]["status"],
                "completed_instance_count": port_failed["gate"]
                ["completed_instance_count"],
                "registered_instance_count": port_failed["gate"]
                ["registered_instance_count"],
            },
            "r87_r88_confirmation": {
                "artifact_sha256": hashes["port_confirmation"],
                "status": port_confirmation["gate"]["status"],
                "protocol_pass": port_confirmation["gate"]["pass"],
                **_performance_counts(port_confirmation),
            },
            "r85_r86_factor_policy_confirmation": {
                "artifact_sha256": hashes["port_factor"],
                "status": port_factor["gate"]["status"],
                "protocol_pass": port_factor["gate"]["pass"],
                "candidate_generation_on_holdout": port_factor
                ["candidate_generation_on_holdout"],
                "holdout_feedback_used_for_selection": port_factor
                ["holdout_feedback_used_for_selection"],
                **_performance_counts(port_factor),
            },
            "performance_superiority_claim_ready": False,
            "unrestricted_optimality_claim_ready": False,
        },
        "common_theorem_interface": {
            "candidate_action": (
                "a finite configuration trajectory containing placement, mode, "
                "or registered reconfiguration decisions"
            ),
            "score": "Q^T lower_service(trajectory) - bounded_penalty(trajectory)",
            "finite_family_oracle": (
                "the selector is audited against its explicitly generated finite "
                "trajectory family; this is not a domain-global optimum"
            ),
            "duration_boundary": (
                "variable-duration trajectories require the renewal-frame drift "
                "model before any stochastic recurrence transfer"
            ),
        },
        "claim_boundary": {
            "supports": [
                "hash-bound external-instance protocol portability across FJSP, MMRCPSP, and BACASP-S",
                "prospective multi-family MMRCPSP evidence with retained earlier failure",
                "exact-ledger FJSP baseline-union nondominance with immutable source artifacts",
                "fixed-budget CP-SAT and port counterexamples that delimit the finite selector",
                "finite trajectory-family oracle and bounded-penalty accounting",
            ],
            "does_not_support": [
                "global FJSP, MMRCPSP, berth-allocation, or quay-crane optimality",
                "dominance over arbitrary exact or state-of-the-art domain solvers",
                "physical terminal deployment or measured industrial migration cost",
                "automatic transfer of server stochastic stability without a domain arrival and service model",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def _performance_counts(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate") or {}
    gate = report.get("gate") or {}
    instance_count = int(
        aggregate.get("instance_count")
        or gate.get("registered_instance_count")
        or gate.get("instance_count")
        or 0
    )
    return {
        "instance_count": instance_count,
        "pareto_nondominated_count": int(
            aggregate.get("ours_pareto_nondominated_count")
            or aggregate.get("selected_pareto_nondominated_count")
            or 0
        ),
        "strict_every_baseline_count": int(
            aggregate.get("ours_strict_every_baseline_count")
            or aggregate.get("selected_strictly_dominates_every_baseline_count")
            or 0
        ),
    }


def _validate_fjsp_kacem(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    if report.get("schema_version") != "scheduleurm.fjsp_external_holdout.v1":
        raise GeneralizationCertificateError("unexpected Kacem FJSP schema")
    if gate.get("pass") is not True or gate.get("instance_count") != 4:
        raise GeneralizationCertificateError("Kacem FJSP holdout is not closed")
    if gate.get("ours_instance_pareto_nondominated_count") != 4:
        raise GeneralizationCertificateError("Kacem FJSP result drifted")
    if gate.get("global_fjsp_optimality_claim") is not False:
        raise GeneralizationCertificateError("Kacem FJSP boundary drifted")


def _validate_fjsp_family(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if report.get("schema_version") != "scheduleurm.fjsp_family_holdout.v1":
        raise GeneralizationCertificateError("unexpected FJSP family schema")
    if gate.get("pass") is not False or gate.get("status") != "FJSP_FAMILY_HOLDOUT_FAIL":
        raise GeneralizationCertificateError("FJSP family failure was overwritten")
    if aggregate.get("instance_count") != 20 or aggregate.get("ours_pareto_nondominated_count") != 19:
        raise GeneralizationCertificateError("FJSP family counterexample drifted")
    _require_false_boundaries(gate, "FJSP family")


def _validate_fjsp_hurink(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if report.get("schema_version") != "scheduleurm.fjsp_hurink_holdout.v1.compact.v1":
        raise GeneralizationCertificateError("unexpected Hurink FJSP schema")
    if gate.get("pass") is not True or gate.get("source_and_policy_protocol_ready") is not True:
        raise GeneralizationCertificateError("Hurink FJSP protocol is not closed")
    if gate.get("complete_cp_sat_reference_coverage_ready") is not True:
        raise GeneralizationCertificateError("Hurink reference coverage drifted")
    if aggregate.get("instance_count") != 20 or aggregate.get("ours_pareto_nondominated_count") != 19:
        raise GeneralizationCertificateError("Hurink FJSP counterexample drifted")
    _require_false_boundaries(gate, "Hurink FJSP")


def _validate_fjsp_numeric(
    report: Mapping[str, Any], *, family_sha256: str, hurink_sha256: str
) -> None:
    gate = report.get("gate") or {}
    groups = report.get("groups") or {}
    source = report.get("source_sha256") or {}
    if report.get("schema_version") != "scheduleurm.fjsp_numeric_disposition.v2":
        raise GeneralizationCertificateError("unexpected FJSP numeric disposition schema")
    if gate.get("pass") is not True:
        raise GeneralizationCertificateError("FJSP numeric disposition is not closed")
    if gate.get("source_protocol_artifacts_immutable") is not True:
        raise GeneralizationCertificateError("FJSP numeric disposition rewrote source artifacts")
    if source.get("family") != family_sha256 or source.get("hurink_compact") != hurink_sha256:
        raise GeneralizationCertificateError("FJSP numeric disposition source binding drifted")
    expected = {
        "first_family_holdout": (20, 1, 7),
        "hurink_confirmation": (20, 1, 15),
    }
    for label, (nondominated, false_dominance, cp_sat_gap) in expected.items():
        row = groups.get(label) or {}
        if row.get("exact_ledger_pareto_nondominated_count") != nondominated:
            raise GeneralizationCertificateError(f"{label} exact-ledger result drifted")
        if row.get("numeric_false_dominance_count") != false_dominance:
            raise GeneralizationCertificateError(f"{label} numeric disposition drifted")
        if row.get("fixed_cp_sat_dominates_generated_family_count") != cp_sat_gap:
            raise GeneralizationCertificateError(f"{label} CP-SAT gap drifted")
    if gate.get("candidate_family_vs_fixed_reference_dominance_claim_ready") is not False:
        raise GeneralizationCertificateError("FJSP candidate-family boundary drifted")


def _validate_mm_v1(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    if report.get("schema_version") != "scheduleurm.mmrcpsp_external_holdout.v1":
        raise GeneralizationCertificateError("unexpected MMRCPSP v1 schema")
    if gate.get("pass") is not False or report.get("first_failure") is None:
        raise GeneralizationCertificateError("MMRCPSP v1 failure was overwritten")


def _validate_mm_repair(report: Mapping[str, Any], v1_hash: str) -> None:
    if report.get("schema_version") != "scheduleurm.mmrcpsp_holdout1_repair_regression.v1":
        raise GeneralizationCertificateError("unexpected MMRCPSP repair schema")
    if report.get("original_prospective_failure_artifact_sha256") != v1_hash:
        raise GeneralizationCertificateError("MMRCPSP repair does not bind v1 failure")
    if report.get("pass_count") != 56 or report.get("failure_count") != 0:
        raise GeneralizationCertificateError("MMRCPSP repair regression is incomplete")
    if report.get("prospective_external_holdout_claim_ready") is not False:
        raise GeneralizationCertificateError("opened-row repair overclaims prospectivity")


def _validate_mm_v2(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    if report.get("schema_version") != "scheduleurm.mmrcpsp_external_holdout.v2":
        raise GeneralizationCertificateError("unexpected MMRCPSP v2 schema")
    if gate.get("pass") is not True or gate.get("registered_instance_count") != 56:
        raise GeneralizationCertificateError("MMRCPSP v2 holdout is not closed")
    if gate.get("ours_instance_pareto_nondominated_count") != 56:
        raise GeneralizationCertificateError("MMRCPSP v2 result drifted")
    if gate.get("global_mmrcpsp_optimality_claim") is not False:
        raise GeneralizationCertificateError("MMRCPSP v2 boundary drifted")


def _validate_mm_family(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if report.get("schema_version") != "scheduleurm.mmrcpsp_family_holdout.v1":
        raise GeneralizationCertificateError("unexpected MMRCPSP family schema")
    if gate.get("pass") is not True or gate.get("protocol_and_feasibility_ready") is not True:
        raise GeneralizationCertificateError("MMRCPSP family protocol is not closed")
    if aggregate.get("instance_count") != 25 or aggregate.get("ours_pareto_nondominated_count") != 25:
        raise GeneralizationCertificateError("MMRCPSP family result drifted")
    _require_false_boundaries(gate, "MMRCPSP family")


def _validate_port_trajectory(report: Mapping[str, Any]) -> None:
    if not str(report.get("schema_version", "")).endswith(".compact.v1"):
        raise GeneralizationCertificateError("unexpected port trajectory schema")
    if report.get("pass") is not True or report.get("registered_instance_pareto_ready") is not True:
        raise GeneralizationCertificateError("port trajectory gate is not closed")
    if report.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateError("port trajectory overclaims performance")
    if report.get("trajectory_oracle_gap_alpha0") != 0.0:
        raise GeneralizationCertificateError("port finite-family oracle gap is nonzero")


def _validate_port_failed(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    if report.get("schema_version") != "scheduleurm.port_bacasp_s_external_holdout.v1.compact.v1":
        raise GeneralizationCertificateError("unexpected failed BACASP-S schema")
    if gate.get("pass") is not False or gate.get("completed_instance_count") != 36:
        raise GeneralizationCertificateError("BACASP-S pre-adapter failure was overwritten")


def _validate_port_confirmation(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if report.get("schema_version") != "scheduleurm.port_bacasp_s_external_holdout.v1.compact.v1":
        raise GeneralizationCertificateError("unexpected BACASP-S confirmation schema")
    if gate.get("pass") is not True or gate.get("completed_instance_count") != 54:
        raise GeneralizationCertificateError("BACASP-S R87/R88 protocol is not closed")
    if aggregate.get("selected_pareto_nondominated_count") != 12:
        raise GeneralizationCertificateError("BACASP-S R87/R88 counterexamples drifted")
    if gate.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateError("BACASP-S R87/R88 overclaims performance")


def _validate_port_factor(report: Mapping[str, Any]) -> None:
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if report.get("schema_version") != "scheduleurm.port_factor_policy_holdout.v1":
        raise GeneralizationCertificateError("unexpected BACASP-S factor-policy schema")
    if gate.get("pass") is not True or gate.get("completed_instance_count") != 54:
        raise GeneralizationCertificateError("BACASP-S factor-policy protocol is not closed")
    if report.get("candidate_generation_on_holdout") is not False:
        raise GeneralizationCertificateError("BACASP-S factor policy used holdout generation")
    if report.get("holdout_feedback_used_for_selection") is not False:
        raise GeneralizationCertificateError("BACASP-S factor policy used holdout feedback")
    if aggregate.get("factor_cell_count") != 27:
        raise GeneralizationCertificateError("BACASP-S factor-cell coverage drifted")
    if gate.get("unrestricted_port_optimality_claim_ready") is not False:
        raise GeneralizationCertificateError("BACASP-S factor policy overclaims optimality")


def _require_false_boundaries(gate: Mapping[str, Any], label: str) -> None:
    for key in (
        "performance_superiority_claim_ready",
        "stochastic_stability_transfer_claim_ready",
    ):
        if gate.get(key) is not False:
            raise GeneralizationCertificateError(f"{label} boundary drifted: {key}")


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]
    mm = report["mmrcpsp"]
    port = report["port"]
    rows = [
        ("FJSP", "first public-family holdout", fjsp["first_family_holdout"]),
        ("FJSP", "Hurink confirmation", fjsp["hurink_confirmation"]),
        ("MMRCPSP", "PSPLIB multi-family confirmation", mm["multi_family_confirmation"]),
        ("Port", "BACASP-S R87/R88", port["r87_r88_confirmation"]),
        ("Port", "factor-policy R85/R86", port["r85_r86_factor_policy_confirmation"]),
    ]
    lines = [
        "# OR Generalization Certificate",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Protocol completeness and performance superiority are separate gates.",
        "- Numeric dispositions and substantive counterexamples are kept separate.",
        "",
        "| Domain | Prospective protocol | Status | Pareto-nondominated | Strict every-baseline | Fixed CP-SAT dominates |",
        "|---|---|---|---:|---:|---:|",
    ]
    for domain, protocol, row in rows:
        cp_sat_count = row.get("fixed_cp_sat_dominates_generated_family_count")
        cp_sat_text = (
            f"{cp_sat_count}/{row['instance_count']}"
            if cp_sat_count is not None
            else "n/a"
        )
        lines.append(
            f"| {domain} | {protocol} | `{row['status']}` | "
            f"{row['pareto_nondominated_count']}/{row['instance_count']} | "
            f"{row['strict_every_baseline_count']}/{row['instance_count']} | "
            f"{cp_sat_text} |"
        )
    lines.extend(
        [
            "",
            "## Retained negative evidence",
            "",
            "- The immutable FJSP source artifacts each reported 19/20 under mixed floating-point precision; the exact-ledger disposition reclassifies one same-action row in each artifact and yields 20/20 baseline-union nondominance.",
            "- Fixed-budget CP-SAT trajectories still dominate the generated family on 7/20 first-family and 15/20 Hurink rows; this is a substantive candidate-family gap, not a numerical artifact.",
            "- The R89/R90 BACASP-S adapter-failure artifact is retained; R87/R88 contains dominated cases.",
            "- The original MMRCPSP failure remains bound to its nonprospective repair regression.",
            "",
            "## Claim boundary",
            "",
        ]
    )
    lines.extend(f"- Supports: {row}" for row in report["claim_boundary"]["supports"])
    lines.extend(
        f"- Does not support: {row}"
        for row in report["claim_boundary"]["does_not_support"]
    )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    report: Mapping[str, Any],
    json_path: str | Path = DEFAULT_JSON,
    markdown_path: str | Path = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    output_json = Path(json_path)
    output_markdown = Path(markdown_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(markdown_report(report), encoding="utf-8")
    return output_json, output_markdown


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    decoded = gzip.decompress(raw) if source.suffix == ".gz" else raw
    payload = json.loads(decoded)
    if not isinstance(payload, dict):
        raise GeneralizationCertificateError(f"expected JSON object: {source}")
    return payload, sha256(raw).hexdigest()


def _digest(report: Mapping[str, Any]) -> str:
    payload = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fjsp-kacem", type=Path, default=DEFAULT_FJSP_KACEM)
    parser.add_argument("--fjsp-family", type=Path, default=DEFAULT_FJSP_FAMILY)
    parser.add_argument("--fjsp-hurink", type=Path, default=DEFAULT_FJSP_HURINK)
    parser.add_argument("--fjsp-numeric", type=Path, default=DEFAULT_FJSP_NUMERIC)
    parser.add_argument("--mm-v1", type=Path, default=DEFAULT_MM_V1)
    parser.add_argument("--mm-repair", type=Path, default=DEFAULT_MM_REPAIR)
    parser.add_argument("--mm-v2", type=Path, default=DEFAULT_MM_V2)
    parser.add_argument("--mm-family", type=Path, default=DEFAULT_MM_FAMILY)
    parser.add_argument("--port-trajectory", type=Path, default=DEFAULT_PORT_TRAJECTORY)
    parser.add_argument("--port-failed", type=Path, default=DEFAULT_PORT_FAILED)
    parser.add_argument("--port-confirmation", type=Path, default=DEFAULT_PORT_CONFIRMATION)
    parser.add_argument("--port-factor", type=Path, default=DEFAULT_PORT_FACTOR)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_certificate(
        fjsp_kacem_path=args.fjsp_kacem,
        fjsp_family_path=args.fjsp_family,
        fjsp_hurink_path=args.fjsp_hurink,
        fjsp_numeric_path=args.fjsp_numeric,
        mm_v1_path=args.mm_v1,
        mm_repair_path=args.mm_repair,
        mm_v2_path=args.mm_v2,
        mm_family_path=args.mm_family,
        port_trajectory_path=args.port_trajectory,
        port_failed_path=args.port_failed,
        port_confirmation_path=args.port_confirmation,
        port_factor_path=args.port_factor,
    )
    json_path, markdown_path = write_artifacts(report, args.json, args.markdown)
    print(
        json.dumps(
            {"gate": report["gate"], "json": str(json_path), "markdown": str(markdown_path)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
