"""Build a reviewer-facing dashboard for Scheduleurm OR claim gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


GATES: tuple[dict[str, Any], ...] = (
    {
        "gate": "stability_theorem",
        "artifact": "../proof/ScheduleurmUpload.lean",
        "raw": "../proof/Scheduleurm/build.log",
        "scoped_claim": "Lean-backed robust candidate MaxWeight theorem spine and theorem-name crosswalk.",
        "scoped_key": None,
        "strong_claim": "Automatic production-wide stability without verifying model assumptions.",
        "strong_key": None,
        "blocker": "Operational claims still require the matching stochastic/load certificate.",
        "next_threshold": "Keep theorem names, build log, and no-sorry audit synchronized with the final manuscript.",
    },
    {
        "gate": "declared_finite_domain_positive_cover_gate",
        "artifact": "md/experiment_artifacts/declared_finite_domain_positive_cover_gate_20260612.json",
        "raw": "md/declared_finite_domain_positive_cover_gate_20260612.md",
        "scoped_claim": "Declared finite service-cache domain is classified and has positive lower-service rows or measured boundaries.",
        "scoped_key": "declared_finite_positive_cover_ready",
        "strong_claim": "Arbitrary all-state positive-service fabric cover.",
        "strong_key": "positive_service_all_state_cover_ready",
        "blocker": "Unmeasured future states must be probed before entering the positive theorem population.",
        "next_threshold": "Service-cache v2 with timestamps/sample windows plus perturbation profiling for any broader state universe.",
    },
    {
        "gate": "all_state_conservative_cover_gate",
        "artifact": "md/experiment_artifacts/all_state_conservative_cover_gate_20260612.json",
        "raw": "md/all_state_conservative_cover_gate_20260612.md",
        "scoped_claim": "All scheduler-visible states are safely partitioned into measured-admitted or zero-service probe/defer states.",
        "scoped_key": "all_state_safety_cover_ready",
        "strong_claim": "Positive-service stability for unknown future arrivals.",
        "strong_key": "positive_service_all_state_cover_ready",
        "blocker": "Unknown states have zero theorem service until measured.",
        "next_threshold": "Measure and admit future buckets or prove a finite all-state universe.",
    },
    {
        "gate": "future_production_admission_contract",
        "artifact": "md/experiment_artifacts/future_production_admission_contract_20260612.json",
        "raw": "md/future_production_admission_contract_20260612.md",
        "scoped_claim": "Strict telemetry contract routes measured production jobs to theorem trace and unknown jobs to probe.",
        "scoped_key": "future_production_automatic_theorem_closure_ready",
        "strong_claim": "Every future production job is theorem-grade without probe.",
        "strong_key": "future_jobs_all_theorem_grade_without_probe",
        "blocker": "Future unmeasured jobs must remain outside the positive theorem stream.",
        "next_threshold": "Keep strict admission on by default in trace-only production telemetry.",
    },
    {
        "gate": "gavel_service_unit_equivalence_certificate",
        "artifact": "md/experiment_artifacts/gavel_service_unit_equivalence_certificate_20260612.json",
        "raw": "md/gavel_service_unit_equivalence_certificate_20260612.md",
        "scoped_claim": "Bounded same-trace Gavel adapter compatibility and native simulator metric extraction for q01/q11.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Measured service-unit equivalence and direct full-stack same-workload superiority.",
        "strong_key": "gavel_service_unit_equivalence_ready",
        "blocker": "No calibration yet proves a Gavel simulator step equals a Scheduleurm measured lower-service unit.",
        "next_threshold": "Run q01/q11 service-unit calibration with holdout relative-error certificate.",
    },
    {
        "gate": "gavel_service_unit_calibration_gate",
        "artifact": "md/experiment_artifacts/gavel_service_unit_calibration_gate_20260612.json",
        "raw": "md/gavel_service_unit_calibration_gate_20260612.md",
        "scoped_claim": "q01/q11 profile-aware same-workload native Gavel simulator calibration against paired Scheduleurm holdout windows.",
        "scoped_key": "profile_aware_model_calibration_ready",
        "strong_claim": "Scalar service-unit equivalence between Gavel simulator time and Scheduleurm measured service units.",
        "strong_key": "gavel_service_unit_equivalence_ready",
        "blocker": "The profile-aware same-workload model passes, but the single scalar service-unit map fails the q01/q11 p95 relative-error threshold.",
        "next_threshold": "Promote only if a single service-unit scale reaches holdout p95 relative error <= 5% on the paired q01/q11 windows.",
    },
    {
        "gate": "gavel_resident_delay_jct_holdout_gate",
        "artifact": "md/experiment_artifacts/gavel_resident_delay_jct_holdout_20260613.json",
        "raw": "md/gavel_resident_delay_jct_holdout_20260613.md",
        "scoped_claim": "Measured corner-case resident-delay/JCT holdout shows immediate co-location beats defer under a resident-alone challenge rate.",
        "scoped_key": "gavel_style_resident_delay_jct_holdout_ready",
        "strong_claim": "Direct full-stack Gavel/Pollux/Sia/IADeep/Salus superiority.",
        "strong_key": "strong_claim_ready",
        "blocker": "Holdout is a measured-service Scheduleurm slice, not an external full-stack scheduler run.",
        "next_threshold": "Promote only after direct same-workload external full-stack execution with service-unit/JCT equivalence.",
    },
    {
        "gate": "sota_fullstack_superiority_gate",
        "artifact": "md/experiment_artifacts/sota_fullstack_superiority_gate_20260613.json",
        "raw": "md/sota_fullstack_superiority_gate_20260613.md",
        "scoped_claim": "Hard blocker certificate for direct full-stack SOTA claims plus policy-semantics fallback readiness.",
        "scoped_key": "hard_blocker_certificate_ready",
        "strong_claim": "Directly beats Gavel/Pollux/Sia/IADeep/Salus binaries or full stacks.",
        "strong_key": "direct_fullstack_sota_superiority_ready",
        "blocker": "Missing service-unit equivalence, Docker/Kubernetes/Go/native runtime, or scope match.",
        "next_threshold": "Only promote an adapter row after direct same-workload full-stack validation.",
    },
    {
        "gate": "sota_candidate_union_gate",
        "artifact": "md/experiment_artifacts/sota_candidate_union_gate_20260613.json",
        "raw": "md/sota_candidate_union_gate_20260613.md",
        "scoped_claim": "Scheduleurm can evaluate a candidate set that explicitly contains SOTA-style policy-family actions; the fixed Pareto-slack online union policy closes both SOTA-style measured-cache envelopes within tolerance.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The measured-cache Pareto-slack replay result directly dominates every external full-stack system on every metric.",
        "strong_key": "strong_claim_ready",
        "blocker": "The fixed Pareto-slack union closes the policy-semantics measured-cache envelopes, but this is still not an external binary/full-stack run.",
        "next_threshold": "Use Pareto-slack policy-semantics dominance for the theory-facing claim; promote direct-system language only after full-stack same-workload execution.",
    },
    {
        "gate": "sota_algorithm_upgrade_gate",
        "artifact": "md/experiment_artifacts/sota_algorithm_upgrade_gate_20260613.json",
        "raw": "md/sota_algorithm_upgrade_gate_20260613.md",
        "scoped_claim": "Opt-in SOTA-facing algorithm upgrade closes adaptive scalarized union, state-dependent marginal cache, bounded lookahead dispatch, reusable ETA/LCB, and expanded SOTA action families.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The production scheduler default directly beats every external full-stack SOTA system.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is replay/certificate evidence; launched production traces and direct external full-stack execution remain separate.",
        "next_threshold": "Enable the opt-in policy on controlled launches and collect global-action theorem traces before production-default language.",
    },
    {
        "gate": "sota_native_execution_attempts",
        "artifact": "md/experiment_artifacts/sota_native_execution_attempts_20260613.json",
        "raw": "md/sota_native_execution_attempts_20260613.md",
        "scoped_claim": "Native execution ledger records Gavel native simulator, Pollux policy-layer optimizer tests, Sia official-artifact probe, and Decima simulator entrypoint execution on this host.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Direct full-stack SOTA superiority over Gavel/Pollux/Sia/IADeep/Salus.",
        "strong_key": "direct_fullstack_sota_superiority_ready",
        "blocker": "No adapter row has direct full-stack same-workload execution; Sia still lacks its official solver/simulator environment or AdaptDL/Kubernetes path, Pollux still lacks Kubernetes/AdaptDLJob stack, and Gavel lacks service-unit equivalence/live cluster execution.",
        "next_threshold": "Run at least one external system end-to-end on identical workload, cluster resources, and JCT/service-unit accounting before using direct superiority language.",
    },
    {
        "gate": "online_ablation_summary_ci",
        "artifact": "md/experiment_artifacts/online_ablation_summary_ci_20260612.json",
        "raw": "md/online_ablation_summary_ci_20260612.md",
        "scoped_claim": "Distributional online replay and ablation summary over existing measured-cache scenarios.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Direct external binary execution or stochastic service generalization.",
        "strong_key": "strong_claim_ready",
        "blocker": "Rows are replay-policy semantics on measured cache, not live external system runs.",
        "next_threshold": "Keep distributional replay separate from live external execution and stochastic LCB lower-service certification.",
    },
    {
        "gate": "corner_case_lower_service_gate",
        "artifact": "md/experiment_artifacts/corner_case_lower_service_gate_20260613.json",
        "raw": "md/corner_case_lower_service_gate_20260613.md",
        "scoped_claim": "Controlled node007/node001 corner-case rows are admitted into a standalone service cache with positive row-level lower-service slack.",
        "scoped_key": "corner_case_lower_service_capacity_ready",
        "strong_claim": "Production-wide or all-state stability for every future corner case.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is a controlled finite measured slice; future states still require service admission.",
        "next_threshold": "Merge only explicitly admitted corner-case rows into theorem-facing replay populations; keep default replay cache unchanged.",
    },
    {
        "gate": "selected_profile_holdout_lcb_gate",
        "artifact": "md/experiment_artifacts/selected_profile_holdout_lcb_gate_20260612.json",
        "raw": "md/selected_profile_holdout_lcb_gate_20260612.md",
        "scoped_claim": "Selected-profile aggregate-window stochastic LCB lower-service capacity certificate for the high-backlog theorem profiles.",
        "scoped_key": "selected_profile_stochastic_lcb_ready",
        "strong_claim": "Positive absolute or diagonal mean-service eta for the same selected profiles.",
        "strong_key": "diagonal_normalized_eta_ready",
        "blocker": "The LCB lower-service capacity certificate passes, but absolute and diagonal mean-service eta remain negative because heterogeneous workload units have different scales.",
        "next_threshold": "Use the LCB lower-service certificate for the main theorem; promote mean-service eta only after the absolute or diagonal eta gate becomes positive.",
    },
    {
        "gate": "controlled_launched_completion_gate",
        "artifact": "md/experiment_artifacts/controlled_production_completion_gate_20260612.json",
        "raw": "md/controlled_production_completion_gate_20260612.md",
        "scoped_claim": "32-task controlled launched completion with theorem-grade oracle trace and canary recorder contract.",
        "scoped_key": "controlled_32_task_completion_ready",
        "strong_claim": "Organic production-wide launched completion.",
        "strong_key": "large_scale_organic_launched_completion_ready",
        "blocker": "Controlled 32-task completion is closed; organic production evidence still requires natural arrivals.",
        "next_threshold": "Accumulate >=64 organic launches, >=50 completions, >=3 workload domains, >=2 nodes, and zero unadmitted launched rows.",
    },
    {
        "gate": "organic_production_canary_recorder_gate",
        "artifact": "md/experiment_artifacts/organic_production_canary_recorder_gate_20260612.json",
        "raw": "md/organic_production_canary_recorder_gate_20260612.md",
        "scoped_claim": "Strict organic production canary recorder and theorem admission/trace contract are ready.",
        "scoped_key": "organic_production_canary_recorder_ready",
        "strong_claim": "Large organic production launched-completion trace.",
        "strong_key": "large_scale_organic_launched_completion_ready",
        "blocker": "Organic thresholds need enough natural launches, completions, domains, nodes, and zero unadmitted launched rows.",
        "next_threshold": "Accumulate >=64 organic launches, >=50 completions, >=3 domains, and >=2 nodes.",
    },
    {
        "gate": "production_launch_completion_gate",
        "artifact": "md/experiment_artifacts/production_launch_completion_gate_20260612.json",
        "raw": "md/production_launch_completion_gate_20260612.md",
        "scoped_claim": "Large-scale active-production progress and non-invasive theorem shadow trace.",
        "scoped_key": "large_scale_active_progress_ready",
        "strong_claim": "Large-scale launched production completion trace.",
        "strong_key": "large_scale_launched_completion_ready",
        "blocker": "Rolling snapshots may lack a theorem shadow subset; launched completion also requires queued production plus low GPU utilization.",
        "next_threshold": "Recover a nonempty theorem shadow subset and launch only when the live gate reports launch_safe=true.",
    },
    {
        "gate": "global_theorem_dispatcher_prototype_gate",
        "artifact": "md/experiment_artifacts/global_theorem_dispatcher_prototype_gate_20260612.json",
        "raw": "md/global_theorem_dispatcher_prototype_gate_20260612.md",
        "scoped_claim": "Pure bounded global robust-MaxWeight action selector has an exact oracle-gap certificate over its enumerated family.",
        "scoped_key": "global_action_dispatch_ready",
        "strong_claim": "Live scheduler default is a deployed global batch MaxWeight dispatcher.",
        "strong_key": "live_scheduler_default_global_dispatcher_ready",
        "blocker": "Prototype is wired only as an opt-in scheduler soft-hint A/B hook; it is not the live default and does not by itself certify launched global-action traces.",
        "next_threshold": "Run controlled launches with global_theorem_maxweight_v1 and collect launched global-action theorem traces.",
    },
    {
        "gate": "or_algorithm_upgrade_gate",
        "artifact": "md/experiment_artifacts/or_algorithm_upgrade_gate_20260613.json",
        "raw": "md/or_algorithm_upgrade_gate_20260613.md",
        "scoped_claim": "Optional algorithm-layer upgrade validates global batch candidate construction, state-dependent marginal service rows, online LCB/ETA, and backlog-aware guarded replay without regressions.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "The production scheduler default is a launched global batch MaxWeight dispatcher or the system directly beats external full stacks.",
        "strong_key": "strong_claim_ready",
        "blocker": "The gate is replay/certificate evidence for an opt-in algorithm surface; production launched global-action traces and external full-stack execution remain separate.",
        "next_threshold": "Enable the optional hook for controlled live launches and collect global-action theorem traces before promoting production-default language.",
    },
    {
        "gate": "reviewer_environment_manifest_gate",
        "artifact": "md/experiment_artifacts/reviewer_environment_manifest_gate_20260612.json",
        "raw": "md/reviewer_environment_manifest_gate_20260612.md",
        "scoped_claim": "Current-host reviewer environment and one-command reproduction contract are present.",
        "scoped_key": "scoped_claim_ready",
        "strong_claim": "Clean Docker/Nix container proof.",
        "strong_key": "clean_container_ready",
        "blocker": "No Dockerfile.reviewer or Nix flake is supplied.",
        "next_threshold": "Run the full reproduction script inside a clean container or Nix environment.",
    },
)


def build_gate_status_dashboard() -> dict[str, Any]:
    rows = [_dashboard_row(spec) for spec in GATES]
    scoped_ready_count = sum(1 for row in rows if bool(row.get("scoped_claim_ready")))
    strong_ready_count = sum(1 for row in rows if bool(row.get("strong_claim_ready")))
    return {
        "gate": "gate_status_dashboard",
        "status": "CLAIM_LADDER_DASHBOARD_READY",
        "gate_pass": True,
        "scoped_claim_ready": True,
        "strong_claim_ready": False,
        "pass_meaning": (
            "reviewer-facing claim ladder; a scoped pass row must not be read as "
            "the adjacent strong claim unless strong_claim_ready is true"
        ),
        "row_count": len(rows),
        "scoped_ready_count": scoped_ready_count,
        "strong_ready_count": strong_ready_count,
        "scoped_pending_count": len(rows) - scoped_ready_count,
        "rows": rows,
        "pass": True,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Gate Status Dashboard",
        "",
        "This dashboard separates scoped claims from adjacent strong claims. A scoped pass is not a universal claim.",
        "",
        "| Gate | Scoped claim | Scoped ready | Strong claim | Strong ready | Status | Blocker | Next threshold | Artifact | Raw evidence |",
        "|---|---|---:|---|---:|---|---|---|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{gate}` | {scoped} | {scoped_ready} | {strong} | {strong_ready} | `{status}` | {blocker} | {next_threshold} | `{artifact}` | `{raw}` |".format(
                gate=row.get("gate"),
                scoped=_cell(row.get("scoped_claim")),
                scoped_ready=str(bool(row.get("scoped_claim_ready"))).lower(),
                strong=_cell(row.get("strong_claim")),
                strong_ready=str(bool(row.get("strong_claim_ready"))).lower(),
                status=row.get("status") or "",
                blocker=_cell(row.get("blocker")),
                next_threshold=_cell(row.get("next_threshold")),
                artifact=row.get("artifact_path"),
                raw=row.get("raw_evidence_path"),
            )
        )
    return "\n".join(lines) + "\n"


def _dashboard_row(spec: Mapping[str, Any]) -> dict[str, Any]:
    artifact = str(spec.get("artifact") or "")
    data = _load_artifact(artifact)
    scoped_key = spec.get("scoped_key")
    strong_key = spec.get("strong_key")
    scoped_ready = bool(data.get(scoped_key)) if scoped_key else Path(REPO_ROOT / artifact).exists()
    strong_ready = bool(data.get(strong_key)) if strong_key else False
    status = str(data.get("status") or ("SCOPED_PASS" if scoped_ready else "SCOPED_PENDING"))
    return {
        "gate": spec.get("gate"),
        "scoped_claim": spec.get("scoped_claim"),
        "scoped_claim_ready": scoped_ready,
        "strong_claim": spec.get("strong_claim"),
        "strong_claim_ready": strong_ready,
        "status": status,
        "blocker": "" if strong_ready else spec.get("blocker"),
        "next_threshold": spec.get("next_threshold"),
        "artifact_path": artifact,
        "raw_evidence_path": spec.get("raw"),
        "artifact_exists": _artifact_path(artifact).exists(),
    }


def _load_artifact(path: str) -> dict[str, Any]:
    p = _artifact_path(path)
    if not p.exists() or p.suffix != ".json":
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _artifact_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_gate_status_dashboard()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.gate_status_dashboard")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build reviewer-facing gate status dashboard")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "gate_status_dashboard_20260612.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "gate_status_dashboard.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
