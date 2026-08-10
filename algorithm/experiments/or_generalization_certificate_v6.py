"""Aggregate the strongest reproducible cross-domain OR evidence to date."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_V5 = ARTIFACT_ROOT / "or_generalization_certificate_v5_20260810.json"
DEFAULT_PORT_REFRESH = (
    ARTIFACT_ROOT / "port_trajectory_registered_refresh_v1_20260810.json"
)
DEFAULT_PORT_FRAME = ARTIFACT_ROOT / "port_frame_bridge_certificate_v1_20260810.json"
DEFAULT_MM_STRUCTURAL = (
    REPO_ROOT
    / "tests"
    / "data"
    / "mmrcpsp_family_external_holdout"
    / "mmrcpsp_family_external_holdout_gate.json"
)
DEFAULT_JSON = ARTIFACT_ROOT / "or_generalization_certificate_v6_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_v6_20260810.md"


class GeneralizationCertificateV6Error(ValueError):
    """Raised when a prerequisite artifact or retained boundary drifts."""


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise GeneralizationCertificateV6Error(f"expected JSON object: {source}")
    return payload, sha256(raw).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_v5(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.or_generalization_certificate.v5":
        raise GeneralizationCertificateV6Error("unexpected v5 certificate schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True or gate.get("all_negative_results_retained") is not True:
        raise GeneralizationCertificateV6Error("v5 protocol or negative-evidence gate is open")
    if gate.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateV6Error("v5 performance boundary drifted")


def _validate_port_refresh(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != (
        "scheduleurm.port_trajectory_registered_refresh.compact.v1"
    ):
        raise GeneralizationCertificateV6Error("unexpected port refresh schema")
    gate = report.get("gate") or {}
    result = report.get("result_summary") or {}
    if gate.get("pass") is not True or gate.get("registered_replay_ready") is not True:
        raise GeneralizationCertificateV6Error("port registered replay gate is open")
    expected = {
        "trajectory_family_size": 31,
        "evaluated_context_count": 4,
        "trajectory_context_pair_count": 124,
        "exact_generated_family_oracle_gap_alpha0": 0.0,
        "all_candidate_results_feasible": True,
        "ours_synthetic_pareto_nondominated_count": 3,
        "ours_synthetic_pareto_nondominated_total": 3,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise GeneralizationCertificateV6Error("port refreshed result drifted")
    if gate.get("prospective_holdout_claim_ready") is not False:
        raise GeneralizationCertificateV6Error("port chronology boundary drifted")


def _validate_port_frame(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.port_frame_bridge_certificate.v1":
        raise GeneralizationCertificateV6Error("unexpected port frame schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True:
        raise GeneralizationCertificateV6Error("port deterministic frame bridge is open")
    if gate.get("deterministic_virtual_frame_interface_ready") is not True:
        raise GeneralizationCertificateV6Error("port virtual frame interface is open")
    for boundary in (
        "physical_service_frame_interface_ready",
        "stochastic_port_arrival_service_model_ready",
        "port_slack_certificate_ready",
        "local_return_certificate_ready",
        "physical_port_positive_recurrence_claim_ready",
    ):
        if gate.get(boundary) is not False:
            raise GeneralizationCertificateV6Error(f"port boundary drifted: {boundary}")


def _validate_mm_structural(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.mmrcpsp_family_holdout.v1":
        raise GeneralizationCertificateV6Error("unexpected MMRCPSP structural schema")
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if gate.get("pass") is not True or gate.get("protocol_and_feasibility_ready") is not True:
        raise GeneralizationCertificateV6Error("MMRCPSP structural holdout gate is open")
    expected = {
        "instance_count": 25,
        "ours_pareto_nondominated_count": 25,
        "ours_strict_every_baseline_count": 11,
        "ours_attains_public_optimum_count": 7,
    }
    if any(aggregate.get(key) != value for key, value in expected.items()):
        raise GeneralizationCertificateV6Error("MMRCPSP structural result drifted")
    if gate.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateV6Error("MMRCPSP superiority boundary drifted")


def build_certificate(
    *,
    v5_path: str | Path = DEFAULT_V5,
    port_refresh_path: str | Path = DEFAULT_PORT_REFRESH,
    port_frame_path: str | Path = DEFAULT_PORT_FRAME,
    mm_structural_path: str | Path = DEFAULT_MM_STRUCTURAL,
) -> dict[str, Any]:
    v5, v5_hash = _load(v5_path)
    port_refresh, port_refresh_hash = _load(port_refresh_path)
    port_frame, port_frame_hash = _load(port_frame_path)
    mm_structural, mm_structural_hash = _load(mm_structural_path)
    _validate_v5(v5)
    _validate_port_refresh(port_refresh)
    _validate_port_frame(port_frame)
    _validate_mm_structural(mm_structural)

    mm_aggregate = mm_structural["aggregate"]
    port_result = port_refresh["result_summary"]
    report: dict[str, Any] = {
        "schema_version": "scheduleurm.or_generalization_certificate.v6",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_V6_PASS_REGISTERED_REPRODUCIBLE_MIXED_EVIDENCE",
            "domain_count": 3,
            "all_protocol_gates_pass": True,
            "all_negative_results_retained": True,
            "structural_and_arrival_holdouts_separated": True,
            "deterministic_virtual_frame_bridge_ready": True,
            "performance_superiority_claim_ready": False,
            "physical_port_stability_claim_ready": False,
            "arbitrary_domain_optimality_claim_ready": False,
        },
        "inputs": {
            "v5_sha256": v5_hash,
            "port_refresh_sha256": port_refresh_hash,
            "port_frame_sha256": port_frame_hash,
            "mmrcpsp_structural_sha256": mm_structural_hash,
        },
        "fjsp": v5["fjsp"],
        "fjsp_reproducibility": v5["fjsp_reproducibility"],
        "mmrcpsp": {
            "prospective_structural_family_holdout": {
                "status": mm_structural["gate"]["status"],
                "instance_count": mm_aggregate["instance_count"],
                "families": mm_aggregate["family_counts"],
                "ours_pareto_nondominated_count": mm_aggregate[
                    "ours_pareto_nondominated_count"
                ],
                "ours_strict_every_baseline_count": mm_aggregate[
                    "ours_strict_every_baseline_count"
                ],
                "ours_attains_public_optimum_count": mm_aggregate[
                    "ours_attains_public_optimum_count"
                ],
                "geomean_ours_makespan_over_public_optimum": mm_aggregate[
                    "geomean_ours_makespan_over_public_optimum"
                ],
                "performance_superiority_claim_ready": False,
            },
            "renewal_stream_total_queue": v5["mmrcpsp"]["v1_total_queue_result"],
            "prospective_class_balance_arrival_tapes": v5["mmrcpsp"][
                "prospective_class_balance_arrival_tape_holdout"
            ],
            "reproducibility_audit": v5["mmrcpsp"]["reproducibility_audit"],
        },
        "port": {
            "registered_trajectory_refresh": {
                "status": port_refresh["gate"]["status"],
                **port_result,
                "prospective_holdout_claim_ready": False,
            },
            "deterministic_frame_bridge": {
                "status": port_frame["gate"]["status"],
                "frame_rows": port_frame["frame_rows"],
                "finite_family": port_frame["finite_family"],
                "lean_source_sha256": port_frame["lean"]["source"]["sha256"],
                "lean_compile_exit_code": port_frame["lean"]["build"][
                    "compile_exit_code"
                ],
                "deterministic_virtual_frame_interface_ready": True,
                "physical_service_frame_interface_ready": False,
                "physical_port_positive_recurrence_claim_ready": False,
            },
            "source_core_and_statewise_migration": v5["port"],
        },
        "theorem_to_experiment_interpretation": {
            "common_interface": (
                "FJSP, MMRCPSP, and registered port actions are finite variable-duration trajectory families "
                "with exact or audited finite-family selection and bounded penalties."
            ),
            "stability_boundary": (
                "The Lean renewal-frame theorem is conditional on physical cumulative service, stochastic "
                "arrivals, positive slack, and local return; deterministic benchmark validity does not supply them."
            ),
            "performance_boundary": (
                "Pareto nondominance and exact family argmax are reported separately from global optimum or "
                "universal performance superiority."
            ),
        },
        "claim_boundary": {
            "supports": [
                "20-instance FJSP confirmation with independent exact-reference ledgers",
                "25-instance prospective multi-family MMRCPSP structural holdout",
                "35-cell prospective MMRCPSP arrival-tape class-balance holdout",
                "current-code reproducible 31-by-4 registered port trajectory ledger",
                "compiled deterministic virtual-frame bridge for registered port trajectories",
            ],
            "does_not_support": [
                "global FJSP, MMRCPSP, or port optimality",
                "universal delay, makespan, or performance superiority",
                "erasing MMRCPSP aggregate-delay or port source-core counterexamples",
                "calling the registered port replay a prospective holdout",
                "equating port terminal virtual-cost coordinates with physical service",
                "physical-port positive recurrence or industrial deployment",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]
    mm = report["mmrcpsp"]
    structural = mm["prospective_structural_family_holdout"]
    aggregate = mm["renewal_stream_total_queue"]
    balance = mm["prospective_class_balance_arrival_tapes"]
    port = report["port"]["registered_trajectory_refresh"]
    lines = [
        "# OR Generalization Certificate v6",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Structural holdout, arrival-tape holdout, registered replay, theorem interface, and performance are separate claims.",
        "",
        "| Domain/evidence | Scope | Result | Retained boundary |",
        "|---|---:|---|---|",
        (
            f"| FJSP exact-ledger confirmation | 20 instances | candidate family nondominated 20/20; fixed-budget CP-SAT dominates {fjsp['fixed_budget_cp_sat_dominates_count']}/20 | no global FJSP optimum |"
        ),
        (
            f"| MMRCPSP structural family holdout | {structural['instance_count']} instances | nondominated {structural['ours_pareto_nondominated_count']}/{structural['instance_count']}; strict-all {structural['ours_strict_every_baseline_count']}/{structural['instance_count']}; optimum {structural['ours_attains_public_optimum_count']}/{structural['instance_count']} | no universal superiority |"
        ),
        (
            f"| MMRCPSP aggregate queue | {aggregate['scenario_seed_count']} cells | baseline dominates {aggregate['baseline_dominates_count']}/{aggregate['scenario_seed_count']} | negative result retained |"
        ),
        (
            f"| MMRCPSP class balance | {balance['scenario_seed_count']} new arrival tapes | nondominated {balance['ours_nondominated_count']}/{balance['scenario_seed_count']}; strict-all {balance['ours_strict_every_baseline_count']}/{balance['scenario_seed_count']} | known structural library |"
        ),
        (
            f"| Port registered trajectories | {port['trajectory_context_pair_count']} pairs | feasible all; oracle gap {port['exact_generated_family_oracle_gap_alpha0']}; synthetic Pareto {port['ours_synthetic_pareto_nondominated_count']}/{port['ours_synthetic_pareto_nondominated_total']} | retrospective replay, not physical deployment |"
        ),
        "",
        "## Port frame boundary",
        "",
        "The registered port trajectory family has a compiled deterministic virtual-frame bridge: all 31 trajectories fit each context's positive candidate-independent horizon, and exact finite-family ordering is preserved after division by the common horizon. The terminal coordinates remain virtual cost service. No stochastic vessel-arrival slack, local-return, or physical-port positive-recurrence claim is made.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- Supports: {row}." for row in report["claim_boundary"]["supports"])
    lines.extend(
        f"- Does not support: {row}."
        for row in report["claim_boundary"]["does_not_support"]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(
    *,
    json_output: str | Path = DEFAULT_JSON,
    markdown_output: str | Path = DEFAULT_MARKDOWN,
    force: bool = False,
) -> dict[str, Any]:
    outputs = [Path(json_output), Path(markdown_output)]
    if not force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise GeneralizationCertificateV6Error(
                f"refusing to overwrite artifacts: {existing}"
            )
    report = build_certificate()
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(markdown_report(report), encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = write_artifacts(
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        force=args.force,
    )
    print(json.dumps(report["gate"], sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "GeneralizationCertificateV6Error",
    "build_certificate",
    "markdown_report",
    "write_artifacts",
]
