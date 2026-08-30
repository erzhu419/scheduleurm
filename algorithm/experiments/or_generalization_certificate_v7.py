"""Aggregate prospective cross-domain action-union evidence without erasing failures."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_V6 = ARTIFACT_ROOT / "or_generalization_certificate_v6_20260810.json"
DEFAULT_FJSP_UNION = ARTIFACT_ROOT / "fjsp_solver_action_union_holdout_v1.json"
DEFAULT_PORT_NEGATIVE = (
    ARTIFACT_ROOT / "port_bacasp_s_external_holdout_r83_r84_20260830.json"
)
DEFAULT_PORT_UNION = ARTIFACT_ROOT / "port_action_union_r81_r82_20260830.json"
DEFAULT_JSON = ARTIFACT_ROOT / "or_generalization_certificate_v7_20260830.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "or_generalization_certificate_v7_20260830.md"
DEFAULT_TEX = REPO_ROOT / "paper" / "generated" / "or_generalization_v7_results.tex"


class GeneralizationCertificateV7Error(ValueError):
    """Raised when an input protocol or retained claim boundary drifts."""


def _load(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise GeneralizationCertificateV7Error(f"expected JSON object: {source}")
    return payload, sha256(raw).hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_v6(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "scheduleurm.or_generalization_certificate.v6":
        raise GeneralizationCertificateV7Error("unexpected v6 certificate schema")
    gate = report.get("gate") or {}
    if gate.get("pass") is not True or gate.get("all_negative_results_retained") is not True:
        raise GeneralizationCertificateV7Error("v6 protocol or negative-evidence gate is open")
    if gate.get("performance_superiority_claim_ready") is not False:
        raise GeneralizationCertificateV7Error("v6 performance boundary drifted")


def _validate_fjsp_union(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != (
        "scheduleurm.fjsp_solver_action_union_holdout.v1.compact.v1"
    ):
        raise GeneralizationCertificateV7Error("unexpected FJSP action-union schema")
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    expected_gate = {
        "pass": True,
        "completed_instance_count": 20,
        "failure_count": 0,
        "complete_solver_action_coverage_ready": True,
        "performance_nondominance_ready": True,
        "global_fjsp_optimality_claim_ready": False,
    }
    if any(gate.get(key) != value for key, value in expected_gate.items()):
        raise GeneralizationCertificateV7Error("FJSP action-union gate drifted")
    expected_aggregate = {
        "solver_feasible_count": 20,
        "exact_union_argmax_count": 20,
        "selected_nondominated_count": 20,
        "fixed_policy_dominator_count": 0,
    }
    if any(aggregate.get(key) != value for key, value in expected_aggregate.items()):
        raise GeneralizationCertificateV7Error("FJSP action-union result drifted")
    if (aggregate.get("solver_relation_counts") or {}).get("same_action") != 20:
        raise GeneralizationCertificateV7Error("FJSP solver relation drifted")
    rows = report.get("rows") or []
    if len(rows) != 20 or any(int(row.get("effective_union_action_count") or 0) <= 0 for row in rows):
        raise GeneralizationCertificateV7Error("FJSP effective action ledger is incomplete")


def _validate_port_negative(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != (
        "scheduleurm.port_bacasp_s_external_holdout.v1.compact.v1"
    ):
        raise GeneralizationCertificateV7Error("unexpected port negative-holdout schema")
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if not (
        gate.get("pass") is True
        and gate.get("completed_instance_count") == 54
        and gate.get("failure_count") == 0
        and gate.get("performance_nondominance_ready") is False
        and gate.get("performance_superiority_claim_ready") is False
        and aggregate.get("selected_pareto_nondominated_count") == 27
        and aggregate.get("selected_strictly_dominates_every_baseline_count") == 12
    ):
        raise GeneralizationCertificateV7Error("port negative holdout was not retained")
    for baseline in ("spt_static", "reconfiguration_greedy"):
        relations = (
            (aggregate.get("paired_policy_summaries") or {})
            .get(baseline, {})
            .get("relation_counts", {})
        )
        if relations != {
            "baseline_dominates": 27,
            "ours_dominates": 12,
            "tradeoff": 15,
        }:
            raise GeneralizationCertificateV7Error(
                f"port negative relation drifted: {baseline}"
            )


def _validate_port_union(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != (
        "scheduleurm.port_public_action_union_holdout.v1.compact.v1"
    ):
        raise GeneralizationCertificateV7Error("unexpected port action-union schema")
    gate = report.get("gate") or {}
    aggregate = report.get("aggregate") or {}
    if not (
        gate.get("pass") is True
        and gate.get("completed_instance_count") == 54
        and gate.get("failure_count") == 0
        and gate.get("performance_superiority_claim_ready") is False
        and gate.get("unrestricted_port_optimality_claim_ready") is False
        and report.get("candidate_generation_on_holdout") is False
        and report.get("holdout_feedback_used_for_policy_revision") is False
        and aggregate.get("factor_cell_count") == 27
    ):
        raise GeneralizationCertificateV7Error("port action-union protocol drifted")


def build_certificate(
    *,
    v6_path: str | Path = DEFAULT_V6,
    fjsp_union_path: str | Path = DEFAULT_FJSP_UNION,
    port_negative_path: str | Path = DEFAULT_PORT_NEGATIVE,
    port_union_path: str | Path = DEFAULT_PORT_UNION,
) -> dict[str, Any]:
    v6, v6_hash = _load(v6_path)
    fjsp_union, fjsp_hash = _load(fjsp_union_path)
    port_negative, port_negative_hash = _load(port_negative_path)
    port_union, port_union_hash = _load(port_union_path)
    _validate_v6(v6)
    _validate_fjsp_union(fjsp_union)
    _validate_port_negative(port_negative)
    _validate_port_union(port_union)

    fjsp_aggregate = fjsp_union["aggregate"]
    fjsp_union_action_counts = [
        int(row["effective_union_action_count"]) for row in fjsp_union["rows"]
    ]
    port_negative_aggregate = port_negative["aggregate"]
    port_union_aggregate = port_union["aggregate"]
    report: dict[str, Any] = {
        "schema_version": "scheduleurm.or_generalization_certificate.v7",
        "gate": {
            "pass": True,
            "status": "OR_GENERALIZATION_V7_PASS_PROSPECTIVE_ACTION_UNIONS",
            "domain_count": 3,
            "all_protocol_gates_pass": True,
            "all_negative_results_retained": True,
            "fjsp_solver_action_union_holdout_ready": True,
            "port_action_union_holdout_ready": True,
            "performance_superiority_claim_ready": False,
            "physical_port_stability_claim_ready": False,
            "arbitrary_domain_optimality_claim_ready": False,
        },
        "inputs": {
            "v6_sha256": v6_hash,
            "fjsp_solver_action_union_sha256": fjsp_hash,
            "port_r83_r84_negative_sha256": port_negative_hash,
            "port_r81_r82_action_union_sha256": port_union_hash,
        },
        "fjsp": {
            "prospective_solver_action_union_holdout": {
                "status": fjsp_union["gate"]["status"],
                "instance_count": fjsp_union["gate"]["completed_instance_count"],
                "trajectory_generator_count": 7,
                "effective_union_action_count_min": min(fjsp_union_action_counts),
                "effective_union_action_count_max": max(fjsp_union_action_counts),
                "solver_feasible_count": fjsp_aggregate["solver_feasible_count"],
                "exact_union_argmax_count": fjsp_aggregate["exact_union_argmax_count"],
                "selected_nondominated_count": fjsp_aggregate[
                    "selected_nondominated_count"
                ],
                "base_family_relation_counts": fjsp_aggregate[
                    "base_family_relation_counts"
                ],
                "solver_relation_counts": fjsp_aggregate["solver_relation_counts"],
                "geomean_selected_over_base": fjsp_aggregate[
                    "geomean_selected_over_base"
                ],
                "geomean_selected_over_solver": fjsp_aggregate[
                    "geomean_selected_over_solver"
                ],
                "global_fjsp_optimality_claim_ready": False,
            },
            "previous_fixed_family_evidence": v6["fjsp"],
        },
        "mmrcpsp": v6["mmrcpsp"],
        "port": {
            "prospective_statewise_external_holdout": {
                "status": port_negative["gate"]["status"],
                "instance_count": port_negative["gate"]["completed_instance_count"],
                "selected_pareto_nondominated_count": port_negative_aggregate[
                    "selected_pareto_nondominated_count"
                ],
                "selected_strictly_dominates_every_baseline_count": (
                    port_negative_aggregate[
                        "selected_strictly_dominates_every_baseline_count"
                    ]
                ),
                "paired_policy_summaries": port_negative_aggregate[
                    "paired_policy_summaries"
                ],
                "retained_negative_result": True,
                "performance_nondominance_ready": False,
            },
            "prospective_finite_action_union_holdout": {
                "status": port_union["gate"]["status"],
                "instance_count": port_union["gate"]["completed_instance_count"],
                "candidate_count": len(port_union["frozen_action_family"]),
                "selected_pareto_nondominated_count": port_union_aggregate[
                    "selected_pareto_nondominated_count"
                ],
                "selected_strictly_dominates_every_baseline_count": (
                    port_union_aggregate[
                        "selected_strictly_dominates_every_baseline_count"
                    ]
                ),
                "paired_policy_summaries": port_union_aggregate[
                    "paired_policy_summaries"
                ],
                "selected_plan_counts": port_union["selected_plan_counts"],
                "performance_nondominance_ready": port_union["gate"][
                    "performance_nondominance_ready"
                ],
                "performance_superiority_claim_ready": False,
            },
            "deterministic_frame_and_previous_registered_evidence": v6["port"],
        },
        "theorem_to_experiment_interpretation": {
            "common_interface": (
                "Each domain exposes a finite family of feasible variable-duration "
                "trajectory actions and audits exact selection over that registered family."
            ),
            "action_union_interpretation": (
                "An external solver or policy trajectory is admitted as another finite "
                "candidate action; matching it is action-space closure, not solver superiority."
            ),
            "stability_boundary": (
                "Deterministic trajectory feasibility does not supply stochastic arrivals, "
                "physical service, positive slack, or local return."
            ),
        },
        "claim_boundary": {
            "supports": [
                "prospective 20-instance FJSP finite action-union evaluation",
                "prospective 54-instance port statewise-policy counterexample",
                "prospective 54-instance port finite action-union evaluation",
                "prospective MMRCPSP structural and arrival-tape holdouts",
                "finite-family selector portability across three scheduling domains",
            ],
            "does_not_support": [
                "global FJSP, MMRCPSP, or port optimality",
                "beating CP-SAT after its trajectory is admitted to the candidate family",
                "erasing the port statewise-policy or MMRCPSP aggregate-delay counterexamples",
                "universal delay, makespan, or source-objective superiority",
                "physical-port positive recurrence or industrial deployment",
            ],
        },
    }
    report["artifact_sha256_excluding_self"] = _digest(report)
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]["prospective_solver_action_union_holdout"]
    mm = report["mmrcpsp"]
    port_negative = report["port"]["prospective_statewise_external_holdout"]
    port_union = report["port"]["prospective_finite_action_union_holdout"]
    aggregate = mm["renewal_stream_total_queue"]
    lines = [
        "# OR Generalization Certificate v7",
        "",
        f"- Status: `{report['gate']['status']}`",
        "- Protocol validity, action-space closure, and universal superiority remain separate claims.",
        "",
        "| Domain/evidence | Scope | Result | Retained boundary |",
        "|---|---:|---|---|",
        (
            f"| FJSP solver-action union | {fjsp['instance_count']} prospective instances | "
            f"exact union {fjsp['exact_union_argmax_count']}/{fjsp['instance_count']}; "
            f"same solver trajectory {fjsp['solver_relation_counts']['same_action']}/"
            f"{fjsp['instance_count']} | action-space closure, not CP-SAT superiority |"
        ),
        (
            f"| MMRCPSP aggregate queue | {aggregate['scenario_seed_count']} cells | "
            f"baseline dominates {aggregate['baseline_dominates_count']}/"
            f"{aggregate['scenario_seed_count']} | negative result retained |"
        ),
        (
            f"| Port statewise external holdout | {port_negative['instance_count']} "
            f"prospective instances | nondominated "
            f"{port_negative['selected_pareto_nondominated_count']}/"
            f"{port_negative['instance_count']} | negative result retained |"
        ),
        (
            f"| Port finite action union | {port_union['instance_count']} prospective "
            f"instances | nondominated {port_union['selected_pareto_nondominated_count']}/"
            f"{port_union['instance_count']}; strict-all "
            f"{port_union['selected_strictly_dominates_every_baseline_count']}/"
            f"{port_union['instance_count']} | registered family only |"
        ),
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


def latex_macros(report: Mapping[str, Any]) -> str:
    fjsp = report["fjsp"]["prospective_solver_action_union_holdout"]
    mm = report["mmrcpsp"]["renewal_stream_total_queue"]
    port_negative = report["port"]["prospective_statewise_external_holdout"]
    port_union = report["port"]["prospective_finite_action_union_holdout"]
    base_relations = fjsp["base_family_relation_counts"]
    values = {
        "FjspUnionInstanceCount": fjsp["instance_count"],
        "FjspUnionExactArgmaxCount": fjsp["exact_union_argmax_count"],
        "FjspUnionSameSolverCount": fjsp["solver_relation_counts"]["same_action"],
        "FjspUnionImprovesBaseCount": base_relations["ours_dominates"],
        "FjspUnionSameBaseCount": base_relations["same_action"],
        "FjspUnionTradeoffBaseCount": base_relations["tradeoff"],
        "FjspUnionMakespanRatio": (
            f"{fjsp['geomean_selected_over_base']['makespan']:.4f}"
        ),
        "FjspUnionFlowRatio": (
            f"{fjsp['geomean_selected_over_base']['sum_completion']:.4f}"
        ),
        "MmrcpspAggregateBaselineDominatesCount": mm["baseline_dominates_count"],
        "MmrcpspAggregateCellCount": mm["scenario_seed_count"],
        "PortStatewiseInstanceCount": port_negative["instance_count"],
        "PortStatewiseNondominatedCount": (
            port_negative["selected_pareto_nondominated_count"]
        ),
        "PortStatewiseStrictAllCount": (
            port_negative["selected_strictly_dominates_every_baseline_count"]
        ),
        "PortUnionInstanceCount": port_union["instance_count"],
        "PortUnionCandidateCount": port_union["candidate_count"],
        "PortUnionNondominatedCount": port_union[
            "selected_pareto_nondominated_count"
        ],
        "PortUnionStrictAllCount": port_union[
            "selected_strictly_dominates_every_baseline_count"
        ],
    }
    lines = ["% Generated by or_generalization_certificate_v7.py; do not edit."]
    lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()
    )
    return "\n".join(lines) + "\n"


def write_artifacts(
    *,
    json_output: str | Path = DEFAULT_JSON,
    markdown_output: str | Path = DEFAULT_MARKDOWN,
    tex_output: str | Path = DEFAULT_TEX,
    force: bool = False,
) -> dict[str, Any]:
    outputs = [Path(json_output), Path(markdown_output), Path(tex_output)]
    if not force:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise GeneralizationCertificateV7Error(
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
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    outputs[2].write_text(latex_macros(report), encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = write_artifacts(
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        tex_output=args.tex_output,
        force=args.force,
    )
    print(json.dumps(report["gate"], sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "GeneralizationCertificateV7Error",
    "build_certificate",
    "latex_macros",
    "markdown_report",
    "write_artifacts",
]
