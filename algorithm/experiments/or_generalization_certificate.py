"""Aggregate the port, FJSP, and MMRCPSP generalization evidence."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FJSP = (
    REPO_ROOT
    / "tests"
    / "data"
    / "fjsp_kacem_external_holdout"
    / "fjsp_kacem_external_holdout_gate.json"
)
DEFAULT_MM_V1 = (
    REPO_ROOT
    / "tests"
    / "data"
    / "mmrcpsp_psplib_external_holdout"
    / "mmrcpsp_psplib_external_holdout_gate.json"
)
DEFAULT_MM_REPAIR = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "mmrcpsp_holdout1_repair_regression_20260809.json"
)
DEFAULT_MM_V2 = (
    REPO_ROOT
    / "tests"
    / "data"
    / "mmrcpsp_psplib_external_holdout_v2"
    / "mmrcpsp_psplib_external_holdout_v2_gate.json"
)
DEFAULT_PORT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "port_trajectory_upgrade_compact_20260809.json"
)
DEFAULT_JSON = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "or_generalization_certificate_20260809.json"
)
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_20260809.md"


class GeneralizationCertificateError(ValueError):
    """Raised when a component artifact violates its declared claim boundary."""


def build_certificate(
    *,
    fjsp_path: str | Path = DEFAULT_FJSP,
    mm_v1_path: str | Path = DEFAULT_MM_V1,
    mm_repair_path: str | Path = DEFAULT_MM_REPAIR,
    mm_v2_path: str | Path = DEFAULT_MM_V2,
    port_path: str | Path = DEFAULT_PORT,
) -> dict[str, Any]:
    fjsp, fjsp_hash = _load(fjsp_path)
    mm_v1, mm_v1_hash = _load(mm_v1_path)
    mm_repair, mm_repair_hash = _load(mm_repair_path)
    mm_v2, mm_v2_hash = _load(mm_v2_path)
    port, port_hash = _load(port_path)
    _validate_fjsp(fjsp)
    _validate_mm_v1(mm_v1)
    _validate_mm_repair(mm_repair, mm_v1_hash)
    _validate_mm_v2(mm_v2)
    _validate_port(port)

    report = {
        "schema_version": "scheduleurm.or_generalization_certificate.v1",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_CERTIFICATE_PASS",
            "component_count": 3,
            "all_component_protocol_gates_auditable": True,
            "all_successful_selected_policies_pareto_nondominated": True,
            "arbitrary_domain_optimality_claim_ready": False,
            "stochastic_stability_transfers_without_domain_model": False,
        },
        "fjsp": {
            "artifact_sha256": fjsp_hash,
            "suite": fjsp["suite"],
            "protocol": "post_freeze_external_holdout",
            "instance_count": fjsp["gate"]["instance_count"],
            "ours_pareto_nondominated_count": fjsp["gate"]
            ["ours_instance_pareto_nondominated_count"],
            "performance_superiority_claim_ready": fjsp["gate"]
            ["performance_superiority_claim_ready"],
            "aggregate": fjsp["aggregate"],
            "global_optimality_claim": fjsp["gate"]["global_fjsp_optimality_claim"],
        },
        "mmrcpsp": {
            "v1_failure_artifact_sha256": mm_v1_hash,
            "v1_prospective_gate_pass": mm_v1["gate"]["pass"],
            "v1_first_failure": mm_v1["first_failure"],
            "repair_regression_artifact_sha256": mm_repair_hash,
            "repair_regression": {
                "instance_count": mm_repair["instance_count"],
                "pass_count": mm_repair["pass_count"],
                "selected_nondominated_count": mm_repair[
                    "selected_nondominated_count"
                ],
                "prospective_external_holdout_claim_ready": mm_repair[
                    "prospective_external_holdout_claim_ready"
                ],
                "repair_implementation_commit": mm_repair[
                    "repair_implementation_commit"
                ],
            },
            "v2_artifact_sha256": mm_v2_hash,
            "v2_protocol": "disjoint_post_repair_external_holdout",
            "v2_instance_count": mm_v2["gate"]["registered_instance_count"],
            "v2_ours_pareto_nondominated_count": mm_v2["gate"]
            ["ours_instance_pareto_nondominated_count"],
            "v2_performance_superiority_claim_ready": mm_v2["gate"]
            ["performance_superiority_claim_ready"],
            "v2_aggregate": mm_v2["aggregate"],
            "global_optimality_claim": mm_v2["gate"]
            ["global_mmrcpsp_optimality_claim"],
        },
        "port": {
            "artifact_sha256": port_hash,
            "full_event_ledger_hash": port["full_artifact_hash"],
            "selected_global_plan_id": port["selected_global_plan_id"],
            "trajectory_family_size": port["trajectory_family_size"],
            "generated_family_oracle_gap": port["trajectory_oracle_gap_alpha0"],
            "public_source_core_pareto": port["public_bacasp_s"]
            ["source_core_pareto"],
            "synthetic_instance_pareto": [
                {
                    "instance": row["instance"],
                    "pareto": row["pareto"],
                }
                for row in port["synthetic_four_resource"]["instances"]
            ],
            "selected_pareto_nondominated_on_every_registered_instance": bool(
                port["public_bacasp_s"]["ours_source_core_pareto_nondominated"]
                and port["synthetic_four_resource"]
                ["ours_pareto_nondominated_on_every_instance"]
            ),
            "performance_superiority_claim_ready": port[
                "performance_superiority_claim_ready"
            ],
            "bounded_penalty_certificate": port["bounded_penalty_certificate"],
        },
        "common_theorem_interface": {
            "candidate_action": (
                "a complete finite configuration trajectory, including placement, "
                "mode, or registered reconfiguration decisions"
            ),
            "score": "Q^T lower_service(trajectory) - bounded_penalty(trajectory)",
            "finite_family_oracle": (
                "each domain selects an exact maximizer over its explicitly generated "
                "finite trajectory family"
            ),
            "duration_boundary": (
                "trajectory durations require the separate variable-duration frame "
                "drift theorem before any stochastic recurrence claim"
            ),
        },
        "claim_boundary": {
            "supports": [
                "finite trajectory-action portability across compute, port, FJSP, and MMRCPSP models",
                "post-freeze external holdout evidence for the registered FJSP and repaired MMRCPSP suites",
                "hash-bound public-source and synthetic four-resource port evidence",
                "exact generated-family oracle and bounded-penalty certificates",
            ],
            "does_not_support": [
                "global FJSP, MMRCPSP, or continuous-port optimality",
                "dominance over arbitrary exact or state-of-the-art domain solvers",
                "physical port deployment or measured industrial reconfiguration cost",
                "automatic transfer of the server stochastic-stability certificate without a domain arrival/service model",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def _validate_fjsp(report: Mapping[str, Any]) -> None:
    gate = report.get("gate", {})
    if report.get("schema_version") != "scheduleurm.fjsp_external_holdout.v1":
        raise GeneralizationCertificateError("unexpected FJSP schema")
    if not gate.get("pass") or gate.get("instance_count") != 4:
        raise GeneralizationCertificateError("FJSP external holdout is not closed")
    if gate.get("ours_instance_pareto_nondominated_count") != 4:
        raise GeneralizationCertificateError("FJSP selected policy is dominated")
    if gate.get("global_fjsp_optimality_claim") is not False:
        raise GeneralizationCertificateError("FJSP claim boundary drift")


def _validate_mm_v1(report: Mapping[str, Any]) -> None:
    gate = report.get("gate", {})
    if report.get("schema_version") != "scheduleurm.mmrcpsp_external_holdout.v1":
        raise GeneralizationCertificateError("unexpected MMRCPSP v1 schema")
    if gate.get("pass") is not False or report.get("first_failure") is None:
        raise GeneralizationCertificateError("MMRCPSP v1 failure was overwritten")


def _validate_mm_repair(report: Mapping[str, Any], v1_hash: str) -> None:
    if report.get("schema_version") != (
        "scheduleurm.mmrcpsp_holdout1_repair_regression.v1"
    ):
        raise GeneralizationCertificateError("unexpected MMRCPSP repair schema")
    if report.get("original_prospective_failure_artifact_sha256") != v1_hash:
        raise GeneralizationCertificateError("MMRCPSP repair does not bind v1 failure")
    if report.get("pass_count") != 56 or report.get("failure_count") != 0:
        raise GeneralizationCertificateError("MMRCPSP repair regression is incomplete")
    if report.get("prospective_external_holdout_claim_ready") is not False:
        raise GeneralizationCertificateError("repair regression overclaims prospectivity")


def _validate_mm_v2(report: Mapping[str, Any]) -> None:
    gate = report.get("gate", {})
    if report.get("schema_version") != "scheduleurm.mmrcpsp_external_holdout.v2":
        raise GeneralizationCertificateError("unexpected MMRCPSP v2 schema")
    if not gate.get("pass") or gate.get("registered_instance_count") != 56:
        raise GeneralizationCertificateError("MMRCPSP v2 holdout is not closed")
    if gate.get("ours_instance_pareto_nondominated_count") != 56:
        raise GeneralizationCertificateError("MMRCPSP v2 selected policy is dominated")
    if gate.get("global_mmrcpsp_optimality_claim") is not False:
        raise GeneralizationCertificateError("MMRCPSP v2 claim boundary drift")


def _validate_port(report: Mapping[str, Any]) -> None:
    if not str(report.get("schema_version", "")).endswith(".compact.v1"):
        raise GeneralizationCertificateError("unexpected port compact schema")
    if report.get("pass") is not True or report.get(
        "registered_instance_pareto_ready"
    ) is not True:
        raise GeneralizationCertificateError("port trajectory gate is not closed")
    if report.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateError("port performance claim boundary drift")
    if report.get("trajectory_oracle_gap_alpha0") != 0.0:
        raise GeneralizationCertificateError("port generated-family oracle gap is nonzero")


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]
    mm = report["mmrcpsp"]
    port = report["port"]
    lines = [
        "# OR Generalization Certificate",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Common action: complete finite configuration trajectory.",
        "- Common selector: exact generated-family maximizer of lower service minus bounded penalty.",
        "",
        "| Domain | Protocol | Instances | Ours Pareto-nondominated | Strong global claim |",
        "|---|---|---:|---:|---:|",
        f"| FJSP | post-freeze Kacem holdout | {fjsp['instance_count']} | {fjsp['ours_pareto_nondominated_count']}/{fjsp['instance_count']} | false |",
        f"| MMRCPSP | disjoint post-repair PSPLIB holdout | {mm['v2_instance_count']} | {mm['v2_ours_pareto_nondominated_count']}/{mm['v2_instance_count']} | false |",
        f"| Port | public BACASP-S plus registered synthetic four-resource instances | 4 | 4/4 | false |",
        "",
        "## MMRCPSP prospective repair protocol",
        "",
        f"- v1 holdout: `FAIL`, first counterexample `{mm['v1_first_failure']['instance']}`.",
        f"- Opened-row repair regression: `{mm['repair_regression']['pass_count']}/{mm['repair_regression']['instance_count']}`; not prospective.",
        f"- Disjoint v2 holdout: `{mm['v2_ours_pareto_nondominated_count']}/{mm['v2_instance_count']}` Pareto-nondominated.",
        "",
        "## Claim boundary",
        "",
    ]
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
    payload = json.loads(raw)
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
    parser.add_argument("--fjsp", type=Path, default=DEFAULT_FJSP)
    parser.add_argument("--mm-v1", type=Path, default=DEFAULT_MM_V1)
    parser.add_argument("--mm-repair", type=Path, default=DEFAULT_MM_REPAIR)
    parser.add_argument("--mm-v2", type=Path, default=DEFAULT_MM_V2)
    parser.add_argument("--port", type=Path, default=DEFAULT_PORT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_certificate(
        fjsp_path=args.fjsp,
        mm_v1_path=args.mm_v1,
        mm_repair_path=args.mm_repair,
        mm_v2_path=args.mm_v2,
        port_path=args.port,
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
