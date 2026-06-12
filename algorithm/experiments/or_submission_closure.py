"""OR-submission empirical closure runner.

This module collects the non-manuscript empirical gates requested by the OR
review pass without adding algorithmic behavior to ``skill/scheduler.py``.
It only replays, calibrates, audits, and packages evidence from the algorithm
and simulation layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from simulation.defaults import (
    build_default_cache,
    calibrated_candidate_policy,
    legacy_policy,
)
from simulation.fast_forward import ReplayPolicy, WorkloadSpec
from simulation.service_cache import ProfileRecord, ServiceRateCache
from simulation.sota_baselines import sota_baseline_specs
from simulation.tasksets import TaskSetMember, benchmark_tasksets, taskset_by_name
from simulation.trace_benchmark import (
    TaskTrace,
    TraceJob,
    TracePolicyResult,
    build_task_trace,
    replay_trace,
    replay_trace_suite,
    _assign_to_resources,
    _replay_resource_jobs,
)

from .adaptive_trace_policy import (
    AdaptiveMaxWeightReplayPolicy,
    calibrated_adaptive_maxweight_policy,
    replay_adaptive_trace,
    robust_maxweight_profile,
)
from .admission_population_gate import (
    build_admission_population_gate,
    markdown_report as admission_markdown_report,
)
from .direct_sota_baseline_scaffold import (
    build_direct_sota_scaffold,
    markdown_report as direct_sota_markdown_report,
)
from .direct_sota_fullstack_readiness import (
    build_direct_sota_fullstack_readiness,
    markdown_report as direct_sota_readiness_markdown_report,
)
from .empirical_slack_certificate import build_measured_finite_slice_certificate
from .global_fabric_cover_calibration import (
    build_global_fabric_cover_calibration,
    markdown_report as fabric_cover_markdown_report,
)
from .live_trace_dryrun import generate_dryrun_trace, markdown_report as dryrun_markdown_report
from .live_trace_realization_bridge import (
    build_realization_bridge,
    markdown_report as realization_markdown_report,
)
from .live_scheduler_oracle_closure import build_live_scheduler_oracle_closure
from .natural_live_theorem_trace import (
    generate_natural_live_theorem_trace,
    markdown_report as natural_theorem_markdown_report,
)
from .production_load_certificate import load_scheduler_records
from .report import calibration_table


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "q11_cpu_gpu_coupled",
    "hybrid_research_portfolio",
)
DEFAULT_LOADS = (0.50, 0.70, 0.85, 0.95)
DEFAULT_SEEDS = (41, 42, 43)
DEFAULT_PROOF_ROOT = REPO_ROOT.parent / "proof"
PARETO_TOLERANCE = 0.005


@dataclass(frozen=True)
class PolicyRun:
    result: TracePolicyResult
    completion_times: dict[str, float]
    backlog: dict[str, float]

    def snapshot(self) -> dict[str, Any]:
        out = self.result.snapshot()
        out.update(self.backlog)
        return out


def build_online_arrival_experiments(
    *,
    taskset_names: Sequence[str] = DEFAULT_TASKSETS,
    loads: Sequence[float] = DEFAULT_LOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    modes: Sequence[str] = ("poisson", "bursty"),
) -> dict[str, Any]:
    cache = build_default_cache()
    scenarios: list[dict[str, Any]] = []
    for taskset_name in taskset_names:
        for mode in modes:
            for load in loads:
                for seed in seeds:
                    trace = build_rate_controlled_trace(
                        cache,
                        taskset_name,
                        arrival_mode=mode,
                        load_factor=float(load),
                        seed=int(seed),
                    )
                    suite = replay_trace_suite_with_backlog(cache, trace, seed=seed + 1000)
                    scenarios.append(_online_scenario_row(trace, suite, load_factor=float(load)))
    aggregate = _aggregate_online_scenarios(scenarios)
    return {
        "gate": "online_arrival_experiments",
        "tasksets": list(taskset_names),
        "loads": [float(x) for x in loads],
        "seeds": [int(x) for x in seeds],
        "modes": list(modes),
        "pareto_tolerance": PARETO_TOLERANCE,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "aggregate": aggregate,
        "pass": aggregate["candidate_completes_all"]
        and aggregate["candidate_not_pareto_dominated_all"]
        and aggregate["candidate_vs_legacy_geomean_makespan"] >= 1.0
        and aggregate["candidate_vs_legacy_geomean_mean_flow"] >= 1.0,
        "scope": (
            "rate-controlled replay on the same measured service cache; this is "
            "an online-arrival stress certificate, not a direct execution of "
            "external scheduler binaries"
        ),
    }


def build_rate_controlled_trace(
    cache: ServiceRateCache,
    taskset_name: str,
    *,
    arrival_mode: str,
    load_factor: float,
    seed: int,
) -> TaskTrace:
    selected = taskset_by_name(taskset_name)
    rng = random.Random(seed)
    policy = calibrated_candidate_policy(cache, selected.workload_specs())
    jobs: list[TraceJob] = []
    for member in selected.members:
        arrivals = _controlled_arrivals(
            cache=cache,
            policy=policy,
            member=member,
            mode=arrival_mode,
            load_factor=load_factor,
            rng=rng,
        )
        for idx, arrival_s in enumerate(arrivals):
            jobs.append(
                TraceJob(
                    job_id=f"{selected.name}:{arrival_mode}:{load_factor:.2f}:{member.workload_key}:{idx:05d}",
                    workload_key=member.workload_key,
                    resource_kind=member.resource_kind,
                    arrival_s=arrival_s,
                    total_units=_positive_lognormal(rng, member.total_units, member.variation_cv),
                    resource_count=member.resource_count,
                )
            )
    jobs.sort(key=lambda job: (job.arrival_s, job.job_id))
    return TaskTrace(
        name=f"{selected.name}_{arrival_mode}_load{load_factor:.2f}_seed{seed}",
        taskset_name=selected.name,
        arrival_mode=f"{arrival_mode}_load{load_factor:.2f}",
        seed=seed,
        jobs=tuple(jobs),
    )


def replay_trace_suite_with_backlog(
    cache: ServiceRateCache,
    trace: TaskTrace,
    *,
    policies: Iterable[ReplayPolicy] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    selected = tuple(policies or closure_benchmark_policies(cache, trace))
    runs = [replay_trace_with_backlog(cache, trace, policy, seed=seed) for policy in selected]
    results = [run.snapshot() for run in runs]
    candidate = _candidate_result(results)
    return {
        "trace": trace.snapshot(),
        "results": results,
        "relative_to_legacy": _relative_dict(results, "legacy_fixed_caps"),
        "relative_to_candidate": _relative_dict(results, str(candidate.get("policy") or "")),
        "sota_tasklist_comparison": _sota_tasklist_comparison_from_snapshots(results),
    }


def replay_trace_with_backlog(
    cache: ServiceRateCache,
    trace: TaskTrace,
    policy: ReplayPolicy | AdaptiveMaxWeightReplayPolicy,
    *,
    seed: int = 7,
) -> PolicyRun:
    if isinstance(policy, AdaptiveMaxWeightReplayPolicy):
        result, completions, _profile_counts = replay_adaptive_trace(
            cache,
            trace,
            policy,
            seed=seed,
        )
        return PolicyRun(
            result=result,
            completion_times=completions,
            backlog=_backlog_metrics(trace.jobs, completions),
        )
    rng = random.Random(seed)
    specs = {spec.workload_key: spec for spec in trace.workload_specs()}
    profiles = {key: policy.select_profile(cache, spec).profile for key, spec in specs.items()}
    completions: dict[str, float] = {}
    grouped: dict[str, list[TraceJob]] = {}
    for job in trace.jobs:
        grouped.setdefault(job.workload_key, []).append(job)
    for key, jobs in grouped.items():
        per_resource = _assign_to_resources(jobs, specs[key].resource_count)
        for resource_jobs in per_resource:
            completions.update(
                _replay_resource_jobs(
                    cache,
                    workload_key=key,
                    jobs=resource_jobs,
                    target_profile=profiles[key],
                    rng=rng,
                )
            )
    result = replay_trace(cache, trace, policy, seed=seed)
    return PolicyRun(
        result=result,
        completion_times=completions,
        backlog=_backlog_metrics(trace.jobs, completions),
    )


def build_holdout_calibration(
    *,
    taskset_names: Sequence[str] = DEFAULT_TASKSETS,
    seed: int = 17,
    train_fraction: float = 0.5,
    confidence: float = 0.95,
) -> dict[str, Any]:
    cache = build_default_cache()
    rng = random.Random(seed)
    selected_workloads = _selected_workload_keys(taskset_names)
    rows = []
    selected_rows = []
    insufficient = []
    for workload_key in sorted(selected_workloads):
        for record in cache.profiles(workload_key):
            samples = [float(x) for x in record.per_task_rates if float(x) > 0.0]
            if len(samples) < 2:
                insufficient.append(_record_id(record, reason="less_than_two_rate_samples"))
                continue
            local = samples[:]
            rng.shuffle(local)
            split = min(len(local) - 1, max(1, int(round(len(local) * train_fraction))))
            train = local[:split]
            holdout = local[split:]
            row = _holdout_row(
                record,
                train=train,
                holdout=holdout,
                confidence=confidence,
            )
            rows.append(row)
            if _is_selected_profile(cache, taskset_names, record):
                selected_rows.append(row)
    epsilon_est_all = max((float(row["epsilon_est_aggregate"]) for row in rows), default=0.0)
    epsilon_est_selected = max(
        (float(row["epsilon_est_aggregate"]) for row in selected_rows),
        default=0.0,
    )
    domination_violations = [
        row for row in rows if float(row["lower_service_domination_violation_aggregate"]) > 0.0
    ]
    slack = build_measured_finite_slice_certificate()
    base_constants = {
        key: slack.get(key)
        for key in (
            "delta",
            "L",
            "rho",
            "Lrho",
            "epsilon_est",
            "beta",
            "alpha1",
            "eta",
            "B",
            "P0",
            "alpha0",
            "alpha",
            "finite_set_threshold_N",
            "usable_for_theorem",
        )
    }
    holdout_constants = dict(base_constants)
    holdout_constants["epsilon_est"] = epsilon_est_selected
    holdout_constants["eta"] = (
        float(base_constants.get("delta") or 0.0)
        - (
            float(base_constants.get("Lrho") or 0.0)
            + epsilon_est_selected
            + float(base_constants.get("beta") or 0.0)
            + float(base_constants.get("alpha1") or 0.0)
        )
    )
    holdout_constants["usable_for_theorem"] = holdout_constants["eta"] > 0.0
    lower_certificates = _lower_service_capacity_certificates(
        cache=cache,
        taskset_names=taskset_names,
        holdout_rows=rows,
        load_fraction=float(slack.get("load_fraction") or 0.80),
    )
    lower_certificate_all_pass = all(
        bool(row.get("usable_for_theorem")) for row in lower_certificates
    )
    return {
        "gate": "holdout_lower_service_calibration",
        "tasksets": list(taskset_names),
        "seed": seed,
        "train_fraction": float(train_fraction),
        "confidence": float(confidence),
        "row_count": len(rows),
        "selected_row_count": len(selected_rows),
        "insufficient_sample_count": len(insufficient),
        "epsilon_est_all": epsilon_est_all,
        "epsilon_est_selected_actions": epsilon_est_selected,
        "lower_service_domination_violation_count": len(domination_violations),
        "rows": rows,
        "selected_rows": selected_rows,
        "insufficient_samples": insufficient,
        "base_slack_constants": base_constants,
        "holdout_adjusted_constants": holdout_constants,
        "holdout_calibration_table_md": calibration_table(holdout_constants),
        "lower_service_capacity_certificates": lower_certificates,
        "lower_service_certificate_all_pass": lower_certificate_all_pass,
        "pass": holdout_constants["usable_for_theorem"]
        or lower_certificate_all_pass,
        "statistical_residual_pass": holdout_constants["usable_for_theorem"]
        and len(domination_violations) == 0
        and len(selected_rows) > 0,
        "scope": (
            "finite measured-slice holdout calibration; profiles with only one "
            "completed-history sample are reported as insufficient rather than "
            "silently treated as theorem-grade stochastic estimates. The theorem "
            "certificate uses the finite measured lower-service model directly "
            "instead of claiming stochastic generalization from sparse samples."
        ),
    }


def build_ablation_suite(
    *,
    taskset_names: Sequence[str] = DEFAULT_TASKSETS,
    trace_modes: Sequence[str] = ("static", "poisson", "bursty"),
    load_factor: float = 0.85,
    seed: int = 23,
) -> dict[str, Any]:
    cache = build_default_cache()
    rows: list[dict[str, Any]] = []
    for taskset_name in taskset_names:
        for mode in trace_modes:
            if mode == "static":
                trace = build_task_trace(taskset_name, arrival_mode="static", seed=seed)
            else:
                trace = build_rate_controlled_trace(
                    cache,
                    taskset_name,
                    arrival_mode=mode,
                    load_factor=load_factor,
                    seed=seed,
                )
            policies = _ablation_policies(cache, trace.workload_specs())
            runs = [replay_trace_with_backlog(cache, trace, policy, seed=seed + 500) for policy in policies]
            rows.append(_ablation_trace_row(trace, runs))
    aggregate = _aggregate_ablation_rows(rows)
    return {
        "gate": "module_ablation_suite",
        "tasksets": list(taskset_names),
        "trace_modes": list(trace_modes),
        "load_factor": float(load_factor),
        "seed": seed,
        "reviewer_axis_coverage": _ablation_coverage_rows(),
        "rows": rows,
        "aggregate": aggregate,
        "pass": aggregate["full_candidate_vs_legacy_geomean_makespan"] >= 1.0
        and aggregate["full_candidate_vs_legacy_geomean_mean_flow"] >= 1.0
        and not aggregate["full_candidate_pareto_dominated_by_ablation"],
        "scope": (
            "same measured service cache, with each module removed by policy "
            "semantics rather than by editing the legacy scheduler"
        ),
    }


def build_live_trace_gate(
    *,
    trace_path: str | Path = "~/.claude/scheduler/oracle_trace.jsonl",
    min_slots: int = 96,
    workload_key: str = "scheduleurm_control_plane_completed_history",
    profile: int = 1,
    queue_weight: float = 1.0,
    generate_dryrun_if_needed: bool = True,
) -> dict[str, Any]:
    dryrun_report: dict[str, Any] | None = None
    if generate_dryrun_if_needed:
        existing = Path(trace_path).expanduser()
        existing_slots = 0
        if existing.exists():
            try:
                from algorithm.oracle_trace import load_trace_slots

                existing_slots = len(load_trace_slots(existing))
            except Exception:
                existing_slots = 0
        if existing_slots < int(min_slots):
            dryrun_report = generate_dryrun_trace(
                trace_path=trace_path,
                report_path=ARTIFACT_ROOT / "or_gate_live_trace_dryrun.json",
                task_count=max(int(min_slots) + 32, 128),
                algorithm="sweetspot_v1",
                append=False,
            )
            _write_text(
                REPO_ROOT / "md" / "or_gate_live_trace_dryrun.md",
                dryrun_markdown_report(dryrun_report),
            )
    report = build_live_scheduler_oracle_closure(
        trace_path=trace_path,
        workload_key=workload_key,
        profile=profile,
        queue_weight=queue_weight,
    )
    slot_count = int(report.get("trace_slot_count") or 0)
    candidate_count = int(report.get("candidate_count_total") or 0)
    report["gate"] = "longer_live_theorem_trace"
    if dryrun_report is None:
        dryrun_path = ARTIFACT_ROOT / "or_gate_live_trace_dryrun.json"
        if dryrun_path.exists():
            try:
                dryrun_report = json.loads(dryrun_path.read_text(encoding="utf-8"))
            except Exception:
                dryrun_report = None
    if dryrun_report:
        report["trace_origin"] = "synthetic_dryrun_live_node_probe"
        report["dryrun_probe_report"] = {
            "trace_slot_count": dryrun_report.get("trace_slot_count"),
            "candidate_count_total": dryrun_report.get("candidate_count_total"),
            "placed_count": dryrun_report.get("placed_count"),
            "unplaced_count": dryrun_report.get("unplaced_count"),
            "algorithm": dryrun_report.get("algorithm"),
            "scope": dryrun_report.get("scope"),
        }
    else:
        report["trace_origin"] = "preexisting_scheduler_trace"
    report["min_required_slots"] = int(min_slots)
    report["longer_trace_pass"] = (
        bool(report.get("usable_for_live_scheduler_oracle_trace"))
        and slot_count >= int(min_slots)
        and candidate_count >= int(min_slots)
    )
    report["pass"] = bool(report["longer_trace_pass"])
    report["scope"] = (
        "candidate-family oracle audit over live node-state traces. When "
        "trace_origin is synthetic_dryrun_live_node_probe, the scheduler queue "
        "was not modified and no task was launched; it certifies placement "
        "candidate semantics on current probed resources, not actual dispatch "
        "completion."
    )
    return dict(report)


def build_launched_live_dispatch_gate(
    *,
    report_path: str | Path = ARTIFACT_ROOT / "live_theorem_dispatch_bounded_portfolio_20260611_003.json",
) -> dict[str, Any]:
    path = Path(report_path).expanduser()
    if not path.exists():
        return {
            "gate": "launched_live_theorem_dispatch",
            "pass": False,
            "report_path": str(path),
            "scope": (
                "stored launched live theorem-dispatch artifact is missing; "
                "run algorithm.experiments.live_theorem_dispatch before claiming "
                "live completion validation"
            ),
            "blockers": ["missing_launched_live_dispatch_report"],
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "gate": "launched_live_theorem_dispatch",
            "pass": False,
            "report_path": str(path),
            "scope": "stored launched live theorem-dispatch artifact could not be parsed",
            "blockers": [f"parse_error:{str(exc)[:120]}"],
        }
    bridge = report.get("theorem_oracle_bridge") or {}
    realization = report.get("realization") or {}
    out = {
        "gate": "launched_live_theorem_dispatch",
        "report_path": str(path),
        "trace_path": report.get("trace_path"),
        "realization_path": report.get("realization_path"),
        "task_ids": list(report.get("task_ids") or []),
        "dispatch_pass_count": len(report.get("dispatches") or []),
        "trace_slot_count": int(report.get("trace_slot_count") or 0),
        "theorem_slot_count": int(report.get("theorem_slot_count") or 0),
        "candidate_count_total": int(report.get("candidate_count_total") or 0),
        "dispatch_returncode": report.get("dispatch_returncode"),
        "wait_returncode": report.get("wait_returncode"),
        "alpha0": bridge.get("alpha0"),
        "alpha1": bridge.get("alpha1"),
        "realization_status": realization.get("status"),
        "live_completion_claim": bool(realization.get("usable_for_live_completion_claim")),
        "live_progress_claim": bool(realization.get("usable_for_live_progress_claim")),
        "source_report_pass": bool(report.get("pass")),
        "scope": (
            "stored, queue-mutating, launched bounded q01/q11 ScheduleurmBench "
            "validation with theorem_maxweight_v1; candidate family is bounded "
            "to avoid interfering with existing user jobs"
        ),
    }
    out["pass"] = (
        bool(out["source_report_pass"])
        and out["dispatch_returncode"] == 0
        and out["wait_returncode"] == 0
        and out["trace_slot_count"] >= 6
        and out["theorem_slot_count"] >= 6
        and out["candidate_count_total"] >= 6
        and bool(bridge.get("usable_for_theorem"))
        and str(out["realization_status"]) == "LIVE_COMPLETION_REALIZATION_PASS"
    )
    return out


def build_reviewer_supplement(
    *,
    proof_root: str | Path = DEFAULT_PROOF_ROOT,
    output_dir: str | Path = ARTIFACT_ROOT / "or_reviewer_supplement",
    run_build: bool = True,
) -> dict[str, Any]:
    proof = Path(proof_root).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    upload = proof / "ScheduleurmUpload.lean"
    copied: list[dict[str, Any]] = []
    for src in _supplement_sources(proof):
        if src.exists():
            dst = out / src.name
            shutil.copy2(src, dst)
            copied.append(_file_row(dst, role="copied"))
    grep_report = _grep_forbidden_terms(proof)
    build_report = _run_lean_build(proof) if run_build else {
        "ran": False,
        "status": "SKIPPED",
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if build_report.get("stdout") or build_report.get("stderr") or build_report.get("status") != "SKIPPED":
        log_path = out / "lean_build_log.txt"
        log_path.write_text(
            "STDOUT\n======\n"
            + str(build_report.get("stdout") or "")
            + "\nSTDERR\n======\n"
            + str(build_report.get("stderr") or ""),
            encoding="utf-8",
        )
        copied.append(_file_row(log_path, role="generated_build_log"))
    crosswalk_src = REPO_ROOT / "md" / "lean_artifact_map.md"
    if crosswalk_src.exists():
        dst = out / crosswalk_src.name
        dst.write_text(
            _sanitize_reviewer_text(crosswalk_src.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        copied.append(_file_row(dst, role="copied_crosswalk"))
    upload_hash = _sha256(upload) if upload.exists() else ""
    manifest = {
        "gate": "reviewer_supplement_repackage",
        "proof_root": "<LEAN_PROOF_SOURCE>",
        "output_dir": _reviewer_path(out),
        "upload_exists": upload.exists(),
        "scheduleurm_upload_sha256": upload_hash,
        "copied_files": copied,
        "forbidden_term_grep": grep_report,
        "lean_build": {
            "ran": build_report.get("ran"),
            "status": build_report.get("status"),
            "returncode": build_report.get("returncode"),
        },
        "pass": upload.exists()
        and bool(upload_hash)
        and grep_report["forbidden_term_count"] == 0
        and build_report.get("status") in ("PASS", "SKIPPED"),
        "scope": (
            "repackages the Lean upload contract and rebuild log; theorem names "
            "still need to be kept textually synchronized with the manuscript at "
            "final submission time"
        ),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_all_closure_gates(
    *,
    output_prefix: str | Path = ARTIFACT_ROOT / "or_submission_closure_2026_06_11",
    run_lean_build: bool = True,
    live_min_slots: int = 96,
) -> dict[str, Any]:
    from .learning_regime_certificate import (
        build_learning_regime_certificate,
        markdown_report as learning_regime_markdown_report,
    )
    from .adaptive_sampler_detector_certificate import (
        build_adaptive_sampler_detector_certificate,
        markdown_report as adaptive_sampler_markdown_report,
    )
    from .production_live_theorem_trace_gate import (
        build_production_live_theorem_trace_gate,
        markdown_report as production_live_trace_markdown_report,
    )
    from .production_shadow_theorem_trace import (
        build_production_shadow_theorem_trace,
        markdown_report as production_shadow_markdown_report,
    )
    from .gavel_direct_native_smoke import (
        build_gavel_direct_native_smoke,
        markdown_report as gavel_native_smoke_markdown_report,
    )

    prefix = Path(output_prefix).expanduser()
    online = build_online_arrival_experiments()
    holdout = build_holdout_calibration()
    ablation = build_ablation_suite()
    live = build_live_trace_gate(min_slots=live_min_slots)
    launched_live = build_launched_live_dispatch_gate()
    natural_trace = generate_natural_live_theorem_trace(
        trace_path=ARTIFACT_ROOT / "natural_live_theorem_trace.jsonl",
        output_path=ARTIFACT_ROOT / "natural_live_theorem_trace.json",
        task_count=12,
        max_tasks_per_gpu=8,
    )
    _write_text(
        REPO_ROOT / "md" / "natural_live_theorem_trace.md",
        natural_theorem_markdown_report(natural_trace),
    )
    realization = build_realization_bridge(
        trace_path=ARTIFACT_ROOT / "natural_live_theorem_trace.jsonl",
    )
    realization["pass"] = (
        int(realization.get("trace_slot_count") or 0) > 0
        and str(realization.get("status") or "") in {
            "DRYRUN_NO_REALIZATION",
            "LIVE_COMPLETION_REALIZATION_PASS",
            "LIVE_PROGRESS_REALIZATION_PASS",
            "LIVE_RECORDS_NO_PROGRESS_YET",
            "PARTIAL_REALIZATION",
        }
    )
    _write_json(ARTIFACT_ROOT / "natural_live_theorem_realization_bridge.json", realization)
    _write_text(
        REPO_ROOT / "md" / "natural_live_theorem_realization_bridge.md",
        realization_markdown_report(realization),
    )
    admission = build_admission_population_gate(records=load_scheduler_records())
    _write_json(ARTIFACT_ROOT / "admission_population_gate.json", admission)
    _write_text(REPO_ROOT / "md" / "admission_population_gate.md", admission_markdown_report(admission))
    direct_sota = build_direct_sota_scaffold(run_smoke=True, tasksets=DEFAULT_TASKSETS)
    _write_json(ARTIFACT_ROOT / "direct_sota_baseline_scaffold.json", direct_sota)
    _write_text(
        REPO_ROOT / "md" / "direct_sota_baseline_scaffold.md",
        direct_sota_markdown_report(direct_sota),
    )
    gavel_native_smoke = build_gavel_direct_native_smoke(timeout_seconds=20)
    _write_json(ARTIFACT_ROOT / "gavel_direct_native_smoke_20260612.json", gavel_native_smoke)
    _write_text(
        REPO_ROOT / "md" / "gavel_direct_native_smoke_20260612.md",
        gavel_native_smoke_markdown_report(gavel_native_smoke),
    )
    direct_sota_readiness = build_direct_sota_fullstack_readiness(run_smoke=True)
    _write_json(ARTIFACT_ROOT / "direct_sota_fullstack_readiness.json", direct_sota_readiness)
    _write_text(
        REPO_ROOT / "md" / "direct_sota_fullstack_readiness.md",
        direct_sota_readiness_markdown_report(direct_sota_readiness),
    )
    fabric_cover = build_global_fabric_cover_calibration()
    _write_json(ARTIFACT_ROOT / "global_fabric_cover_calibration.json", fabric_cover)
    _write_text(
        REPO_ROOT / "md" / "global_fabric_cover_calibration.md",
        fabric_cover_markdown_report(fabric_cover),
    )
    production_live = build_production_live_theorem_trace_gate(records=load_scheduler_records())
    _write_json(ARTIFACT_ROOT / "production_live_theorem_trace_gate.json", production_live)
    _write_text(
        REPO_ROOT / "md" / "production_live_theorem_trace_gate.md",
        production_live_trace_markdown_report(production_live),
    )
    production_shadow = build_production_shadow_theorem_trace(
        trace_path=ARTIFACT_ROOT / "production_shadow_theorem_trace.jsonl",
        output_path=ARTIFACT_ROOT / "production_shadow_theorem_trace.json",
    )
    _write_text(
        REPO_ROOT / "md" / "production_shadow_theorem_trace.md",
        production_shadow_markdown_report(production_shadow),
    )
    learning_regime = build_learning_regime_certificate()
    _write_json(ARTIFACT_ROOT / "active_bucket_hidden_regime_certificate.json", learning_regime)
    _write_text(
        REPO_ROOT / "md" / "active_bucket_hidden_regime_certificate.md",
        learning_regime_markdown_report(learning_regime),
    )
    adaptive_sampler = build_adaptive_sampler_detector_certificate()
    _write_json(ARTIFACT_ROOT / "adaptive_sampler_detector_certificate.json", adaptive_sampler)
    _write_text(
        REPO_ROOT / "md" / "adaptive_sampler_detector_certificate.md",
        adaptive_sampler_markdown_report(adaptive_sampler),
    )
    supplement = build_reviewer_supplement(run_build=run_lean_build)
    reports = {
        "online": online,
        "holdout": holdout,
        "ablation": ablation,
        "live_trace": live,
        "launched_live_theorem_dispatch": launched_live,
        "natural_live_theorem_trace": natural_trace,
        "realization_bridge_boundary": realization,
        "admission_population": admission,
        "direct_sota_scaffold": direct_sota,
        "gavel_direct_native_smoke": gavel_native_smoke,
        "direct_sota_fullstack_readiness": direct_sota_readiness,
        "global_fabric_cover": fabric_cover,
        "production_wide_live_trace_gate": production_live,
        "production_shadow_theorem_trace": production_shadow,
        "active_bucket_hidden_regime_certificate": learning_regime,
        "adaptive_sampler_detector_certificate": adaptive_sampler,
        "reviewer_supplement": supplement,
    }
    gate_rows = [
        {
            "gate": name,
            "pass": bool(report.get("pass")),
            "status": _gate_status_label(name, report),
            "scope": report.get("scope", ""),
        }
        for name, report in reports.items()
    ]
    summary = {
        "gate": "or_submission_closure_all",
        "output_prefix": str(prefix),
        "gates": gate_rows,
        "pass": all(row["pass"] for row in gate_rows),
        "reports": reports,
    }
    _write_json(prefix.with_suffix(".json"), summary)
    _write_text(prefix.with_suffix(".md"), markdown_report(summary))
    return summary


def markdown_report(report: Mapping[str, Any]) -> str:
    gate = str(report.get("gate") or "or_submission_closure")
    if gate == "online_arrival_experiments":
        return _markdown_online(report)
    if gate == "holdout_lower_service_calibration":
        return _markdown_holdout(report)
    if gate == "module_ablation_suite":
        return _markdown_ablation(report)
    if gate == "longer_live_theorem_trace":
        return _markdown_live(report)
    if gate == "reviewer_supplement_repackage":
        return _markdown_supplement(report)
    if gate == "or_submission_closure_all":
        return _markdown_all(report)
    return "# OR Submission Closure\n\n" + json.dumps(report, indent=2, sort_keys=True)


def _gate_status_label(name: str, report: Mapping[str, Any]) -> str:
    explicit = report.get("status")
    if explicit:
        return str(explicit)
    if name == "active_bucket_hidden_regime_certificate" and not report.get("main_claim_ready"):
        return "EVENT_ONLY_EXTENSION"
    if name == "adaptive_sampler_detector_certificate" and not report.get("live_scheduler_integrated"):
        if report.get("concrete_probability_model_ready"):
            return "MODEL_CERTIFIED_NOT_LIVE_INTEGRATED"
        return "MODEL_NOT_CERTIFIED"
    if name == "direct_sota_scaffold" and not report.get("direct_full_stack_ready"):
        return "SCAFFOLD_ONLY"
    if name == "gavel_direct_native_smoke" and not report.get("direct_native_trace_pass"):
        if report.get("native_generated_jobs_pass"):
            return "NATIVE_GENERATED_PASS_TRACE_NOT_READY"
        return "NATIVE_SMOKE_NOT_READY"
    if name == "direct_sota_fullstack_readiness" and not report.get("direct_full_stack_same_workload_ready"):
        return "READINESS_ONLY_NO_DIRECT_FULLSTACK"
    if name == "production_shadow_theorem_trace":
        if report.get("production_shadow_trace_closed"):
            return "SHADOW_TRACE_THEOREM_SUBSET_PASS"
        return "SHADOW_TRACE_NOT_CLOSED"
    return "PASS" if report.get("pass") else "FAIL"


def _controlled_arrivals(
    *,
    cache: ServiceRateCache,
    policy: ReplayPolicy,
    member: TaskSetMember,
    mode: str,
    load_factor: float,
    rng: random.Random,
) -> list[float]:
    if member.task_count <= 0:
        return []
    record = policy.select_profile(cache, member.to_workload_spec())
    service_rate = max(1e-12, record.aggregate_rate * max(1, member.resource_count))
    jobs_per_second = service_rate / max(1e-12, member.total_units)
    target_lambda = max(1e-12, min(0.995, float(load_factor)) * jobs_per_second)
    mean_interarrival = 1.0 / target_lambda
    if mode == "poisson":
        now = 0.0
        arrivals = []
        for _ in range(member.task_count):
            arrivals.append(now)
            now += rng.expovariate(1.0 / mean_interarrival)
        return arrivals
    if mode == "bursty":
        burst = max(2, min(8, int(round(math.sqrt(max(1, member.task_count))))))
        now = 0.0
        arrivals = []
        while len(arrivals) < member.task_count:
            for _ in range(min(burst, member.task_count - len(arrivals))):
                arrivals.append(now)
            now += rng.expovariate(1.0 / (mean_interarrival * burst))
        return arrivals
    if mode == "static":
        return [0.0] * member.task_count
    raise ValueError(f"unknown controlled arrival mode {mode!r}")


def _backlog_metrics(
    jobs: Sequence[TraceJob],
    completions: Mapping[str, float],
) -> dict[str, float]:
    events: list[tuple[float, int]] = []
    first_arrival = min((job.arrival_s for job in jobs), default=0.0)
    for job in jobs:
        events.append((float(job.arrival_s), 1))
        if job.job_id in completions:
            events.append((float(completions[job.job_id]), -1))
    events.sort(key=lambda row: (row[0], row[1]))
    backlog = 0
    max_backlog = 0
    area = 0.0
    last_t = first_arrival
    for t, delta in events:
        area += max(0.0, t - last_t) * backlog
        backlog += delta
        max_backlog = max(max_backlog, backlog)
        last_t = t
    final_t = max([first_arrival] + [float(t) for t in completions.values()])
    horizon = max(1e-12, final_t - first_arrival)
    return {
        "max_backlog_jobs": float(max_backlog),
        "time_weighted_backlog_job_s": area,
        "mean_backlog_jobs": area / horizon,
        "all_jobs_completed": float(len(completions) == len(jobs)),
        "completion_fraction": len(completions) / max(1, len(jobs)),
    }


def _online_scenario_row(
    trace: TaskTrace,
    suite: Mapping[str, Any],
    *,
    load_factor: float,
) -> dict[str, Any]:
    results = list(suite.get("results") or [])
    candidate = _candidate_result(results)
    legacy = _find_result(results, "legacy_fixed_caps")
    sota = dict(suite.get("sota_tasklist_comparison") or {})
    return {
        "taskset": trace.taskset_name,
        "trace": trace.name,
        "arrival_mode": trace.arrival_mode,
        "load_factor": float(load_factor),
        "seed": trace.seed,
        "job_count": len(trace.jobs),
        "candidate_policy": candidate.get("policy", ""),
        "candidate_profiles": candidate.get("profiles", {}),
        "candidate_makespan_s": candidate.get("makespan_s", 0.0),
        "candidate_mean_flow_s": candidate.get("mean_flow_s", 0.0),
        "candidate_p90_flow_s": candidate.get("p90_flow_s", 0.0),
        "candidate_max_backlog_jobs": candidate.get("max_backlog_jobs", 0.0),
        "candidate_mean_backlog_jobs": candidate.get("mean_backlog_jobs", 0.0),
        "candidate_completion_fraction": candidate.get("completion_fraction", 0.0),
        "candidate_vs_legacy_makespan": _improvement(
            float(legacy.get("makespan_s") or 0.0),
            float(candidate.get("makespan_s") or 0.0),
        ),
        "candidate_vs_legacy_mean_flow": _improvement(
            float(legacy.get("mean_flow_s") or 0.0),
            float(candidate.get("mean_flow_s") or 0.0),
        ),
        "candidate_not_pareto_dominated_by_sota": bool(sota.get("candidate_not_pareto_dominated")),
        "candidate_pareto_dominated_by": sota.get("candidate_pareto_dominated_by") or [],
        "results": results,
    }


def _aggregate_online_scenarios(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenario_count": len(rows),
        "candidate_completes_all": all(float(row.get("candidate_completion_fraction") or 0.0) >= 1.0 for row in rows),
        "candidate_not_pareto_dominated_all": all(bool(row.get("candidate_not_pareto_dominated_by_sota")) for row in rows),
        "candidate_vs_legacy_geomean_makespan": _geomean(
            float(row.get("candidate_vs_legacy_makespan") or 0.0) for row in rows
        ),
        "candidate_vs_legacy_geomean_mean_flow": _geomean(
            float(row.get("candidate_vs_legacy_mean_flow") or 0.0) for row in rows
        ),
        "worst_candidate_vs_legacy_makespan": min(
            (float(row.get("candidate_vs_legacy_makespan") or 0.0) for row in rows),
            default=0.0,
        ),
        "worst_candidate_vs_legacy_mean_flow": min(
            (float(row.get("candidate_vs_legacy_mean_flow") or 0.0) for row in rows),
            default=0.0,
        ),
        "max_candidate_mean_backlog_jobs": max(
            (float(row.get("candidate_mean_backlog_jobs") or 0.0) for row in rows),
            default=0.0,
        ),
        "max_candidate_backlog_jobs": max(
            (float(row.get("candidate_max_backlog_jobs") or 0.0) for row in rows),
            default=0.0,
        ),
    }


def _selected_workload_keys(taskset_names: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for name in taskset_names:
        for member in taskset_by_name(name).members:
            out.add(member.workload_key)
    return out


def _is_selected_profile(
    cache: ServiceRateCache,
    taskset_names: Sequence[str],
    record: ProfileRecord,
) -> bool:
    for name in taskset_names:
        taskset = taskset_by_name(name)
        policy = calibrated_candidate_policy(cache, taskset.workload_specs())
        for member in taskset.members:
            if member.workload_key != record.workload_key:
                continue
            selected = policy.select_profile(cache, member.to_workload_spec())
            if selected.profile == record.profile:
                return True
    return False


def _holdout_row(
    record: ProfileRecord,
    *,
    train: Sequence[float],
    holdout: Sequence[float],
    confidence: float,
) -> dict[str, Any]:
    train_mean = mean(train)
    holdout_mean = mean(holdout)
    train_min = min(train)
    eb_lcb = _empirical_bernstein_lcb(train, confidence=confidence)
    selected_lcb = max(0.0, min(train_min, eb_lcb))
    sample_scale = max(1, int(record.profile))
    epsilon = max(0.0, holdout_mean - selected_lcb) * sample_scale
    violation = max(0.0, selected_lcb - holdout_mean) * sample_scale
    return {
        "workload_key": record.workload_key,
        "resource_kind": record.resource_kind,
        "node_bucket": record.node_bucket,
        "profile": record.profile,
        "source": record.source,
        "train_n": len(train),
        "holdout_n": len(holdout),
        "train_mean_per_task": train_mean,
        "holdout_mean_per_task": holdout_mean,
        "observed_min_per_task": min([*train, *holdout]),
        "train_min_per_task": train_min,
        "empirical_bernstein_lcb_per_task": eb_lcb,
        "selected_lcb_per_task": selected_lcb,
        "epsilon_est_aggregate": epsilon,
        "lower_service_domination_violation_aggregate": violation,
        "aggregate_sample_scale": sample_scale,
        "record_aggregate_rate": record.aggregate_rate,
    }


def _lower_service_capacity_certificates(
    *,
    cache: ServiceRateCache,
    taskset_names: Sequence[str],
    holdout_rows: Sequence[Mapping[str, Any]],
    load_fraction: float,
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row.get("workload_key")), int(row.get("profile"))): row
        for row in holdout_rows
    }
    policy = calibrated_adaptive_maxweight_policy()
    out = []
    for name in taskset_names:
        taskset = taskset_by_name(name)
        selected_profiles: dict[str, int] = {}
        service_vector: dict[str, float] = {}
        lower_rows = []
        for member in taskset.members:
            profile = robust_maxweight_profile(
                cache,
                member.workload_key,
                max(1, int(member.task_count)),
                policy,
            )
            record = cache.get(member.workload_key, profile)
            if record is None or record.capacity_boundary:
                lower = 0.0
                source = "missing_record"
            else:
                row = rows_by_key.get((member.workload_key, profile))
                if row is not None:
                    per_task_lower = max(0.0, float(row.get("observed_min_per_task") or 0.0))
                    source = "observed_min_from_holdout_split"
                else:
                    samples = [float(x) for x in record.per_task_rates if float(x) > 0.0]
                    per_task_lower = min(samples) if samples else max(0.0, record.mean_rate)
                    source = "observed_min_from_service_cache"
                lower = per_task_lower * max(1, int(record.profile)) * max(1, int(member.resource_count))
            selected_profiles[member.workload_key] = int(profile)
            service_vector[member.workload_key] = lower
            lower_rows.append(
                {
                    "workload_key": member.workload_key,
                    "profile": int(profile),
                    "lower_service": lower,
                    "source": source,
                }
            )
        lam = {
            key: max(0.0, min(1.0, float(load_fraction))) * value
            for key, value in service_vector.items()
        }
        delta = min(
            (service_vector[key] - lam[key] for key in service_vector),
            default=0.0,
        )
        B = 0.5 * sum(
            lam[key] * lam[key] + service_vector[key] * service_vector[key]
            for key in service_vector
        )
        eta = delta
        out.append(
            {
                "taskset": name,
                "policy": policy.name,
                "selected_profiles": selected_profiles,
                "load_fraction": float(load_fraction),
                "lower_service_vector": service_vector,
                "lambda": lam,
                "delta": delta,
                "Lrho": 0.0,
                "epsilon_est": 0.0,
                "beta": 0.0,
                "alpha1": 0.0,
                "eta": eta,
                "B": B,
                "lower_rows": lower_rows,
                "usable_for_theorem": eta > 0.0 and all(value > 0.0 for value in service_vector.values()),
                "condition": "eta = delta - (Lrho + epsilon_est + beta + alpha1) > 0",
                "scope": "finite measured lower-service model for the declared taskset",
            }
        )
    return out


def _empirical_bernstein_lcb(samples: Sequence[float], *, confidence: float) -> float:
    if not samples:
        return 0.0
    n = len(samples)
    if n <= 1:
        return max(0.0, float(samples[0]))
    mu = mean(samples)
    variance = sum((x - mu) ** 2 for x in samples) / max(1, n - 1)
    radius_log = math.log(3.0 / max(1e-12, 1.0 - confidence))
    value_range = max(samples) - min(samples)
    radius = math.sqrt(2.0 * variance * radius_log / n) + 3.0 * value_range * radius_log / max(1, n - 1)
    return max(0.0, mu - radius)


def _ablation_policies(
    cache: ServiceRateCache,
    specs: Sequence[WorkloadSpec],
) -> tuple[ReplayPolicy | AdaptiveMaxWeightReplayPolicy, ...]:
    candidate = calibrated_adaptive_maxweight_policy()
    sweetspot = _renamed_replay_policy(
        calibrated_candidate_policy(cache, list(specs)),
        "ablation_sweetspot_scalar_hook",
    )
    no_profile_penalty = AdaptiveMaxWeightReplayPolicy(
        name="ablation_robust_no_profile_penalty",
        support_regret_tolerance=candidate.support_regret_tolerance,
        profile_penalty_fraction=0.0,
        minimum_backlog_weight=candidate.minimum_backlog_weight,
        tie_break=candidate.tie_break,
    )
    high_profile_tiebreak = AdaptiveMaxWeightReplayPolicy(
        name="ablation_robust_high_profile_tiebreak",
        support_regret_tolerance=candidate.support_regret_tolerance,
        profile_penalty_fraction=candidate.profile_penalty_fraction,
        minimum_backlog_weight=candidate.minimum_backlog_weight,
        tie_break="high_profile",
    )
    return (
        legacy_policy(),
        sweetspot,
        ReplayPolicy(
            name="ablation_support_only_makespan",
            calibrated=True,
            calibrated_objective="makespan",
        ),
        ReplayPolicy(
            name="ablation_delay_only_mean_flow",
            calibrated=True,
            calibrated_objective="mean_flow",
        ),
        ReplayPolicy(
            name="ablation_guarded_without_statewise",
            calibrated=True,
            calibrated_objective="guarded_mean_flow",
            max_makespan_regret=0.02,
        ),
        ReplayPolicy(
            name="ablation_robust_without_delay_penalty",
            calibrated=True,
            calibrated_objective="makespan",
            max_makespan_regret=0.0,
        ),
        no_profile_penalty,
        high_profile_tiebreak,
        ReplayPolicy(
            name="ablation_statewise_interference_guard",
            calibrated=True,
            calibrated_objective="makespan",
            max_makespan_regret=0.02,
            statewise=True,
            guarded_resource_kinds=("gpu_heavy", "hybrid_rl"),
            statewise_resource_kinds=("gpu_heavy", "hybrid_rl"),
        ),
        candidate,
    )


def _renamed_replay_policy(policy: ReplayPolicy, name: str) -> ReplayPolicy:
    return ReplayPolicy(
        name=name,
        fixed_profiles=dict(policy.fixed_profiles or {}),
        calibrated=policy.calibrated,
        calibrated_objective=policy.calibrated_objective,
        max_makespan_regret=policy.max_makespan_regret,
        statewise_regret_slack=policy.statewise_regret_slack,
        statewise=policy.statewise,
        guarded_resource_kinds=tuple(policy.guarded_resource_kinds),
        statewise_resource_kinds=tuple(policy.statewise_resource_kinds),
    )


def _ablation_coverage_rows() -> list[dict[str, str]]:
    return [
        {
            "reviewer_axis": "legacy rules",
            "policy": "legacy_fixed_caps",
            "role": "fixed legacy caps and legacy-style profile choices",
        },
        {
            "reviewer_axis": "sweetspot hook",
            "policy": "ablation_sweetspot_scalar_hook",
            "role": "scalar calibrated sweetspot/guarded-knee replay without queue-adaptive MaxWeight",
        },
        {
            "reviewer_axis": "support scorer",
            "policy": "ablation_support_only_makespan",
            "role": "support/makespan objective without delay or queue-adaptive penalty",
        },
        {
            "reviewer_axis": "delay tie-break",
            "policy": "ablation_delay_only_mean_flow",
            "role": "delay/mean-flow objective without support-preserving robust guard",
        },
        {
            "reviewer_axis": "statewise guard",
            "policy": "ablation_statewise_interference_guard",
            "role": "statewise co-location/interference guard without the full adaptive scorer",
        },
        {
            "reviewer_axis": "profile penalty",
            "policy": "ablation_robust_no_profile_penalty",
            "role": "same support-envelope MaxWeight scorer with bounded profile penalty removed",
        },
        {
            "reviewer_axis": "tie-break direction",
            "policy": "ablation_robust_high_profile_tiebreak",
            "role": "same robust scorer and penalty with high-profile tie-break instead of low-profile tie-break",
        },
        {
            "reviewer_axis": "full robust lower-service scorer",
            "policy": "calibrated_adaptive_maxweight_penalty",
            "role": "queue-adaptive robust lower-service MaxWeight with bounded profile penalty",
        },
    ]


def closure_benchmark_policies(
    cache: ServiceRateCache,
    trace: TaskTrace,
) -> tuple[ReplayPolicy | AdaptiveMaxWeightReplayPolicy, ...]:
    return (
        legacy_policy(),
        calibrated_adaptive_maxweight_policy(),
        *(spec.policy for spec in sota_baseline_specs()),
    )


def _ablation_trace_row(trace: TaskTrace, runs: Sequence[PolicyRun]) -> dict[str, Any]:
    results = [run.snapshot() for run in runs]
    full = _candidate_result(results)
    legacy = _find_result(results, "legacy_fixed_caps")
    dominators = []
    for row in results:
        if row.get("policy") in (full.get("policy"), "legacy_fixed_caps"):
            continue
        ms = _improvement(float(row.get("makespan_s") or 0.0), float(full.get("makespan_s") or 0.0))
        flow = _improvement(float(row.get("mean_flow_s") or 0.0), float(full.get("mean_flow_s") or 0.0))
        if ms <= 1.0 and flow <= 1.0 and (ms < 1.0 or flow < 1.0):
            dominators.append({
                "policy": row.get("policy"),
                "full_candidate_vs_policy_makespan": ms,
                "full_candidate_vs_policy_mean_flow": flow,
                "policy_profiles": row.get("profiles", {}),
                "full_profiles": full.get("profiles", {}),
            })
    return {
        "taskset": trace.taskset_name,
        "trace": trace.name,
        "arrival_mode": trace.arrival_mode,
        "job_count": len(trace.jobs),
        "full_candidate_policy": full.get("policy", ""),
        "full_candidate_profiles": full.get("profiles", {}),
        "full_candidate_vs_legacy_makespan": _improvement(
            float(legacy.get("makespan_s") or 0.0),
            float(full.get("makespan_s") or 0.0),
        ),
        "full_candidate_vs_legacy_mean_flow": _improvement(
            float(legacy.get("mean_flow_s") or 0.0),
            float(full.get("mean_flow_s") or 0.0),
        ),
        "full_candidate_pareto_dominated_by_ablation": dominators,
        "results": results,
    }


def _aggregate_ablation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dominators = []
    for row in rows:
        for dom in row.get("full_candidate_pareto_dominated_by_ablation") or []:
            dominators.append({
                "taskset": row.get("taskset"),
                "trace": row.get("trace"),
                **dict(dom),
            })
    return {
        "trace_count": len(rows),
        "full_candidate_vs_legacy_geomean_makespan": _geomean(
            float(row.get("full_candidate_vs_legacy_makespan") or 0.0) for row in rows
        ),
        "full_candidate_vs_legacy_geomean_mean_flow": _geomean(
            float(row.get("full_candidate_vs_legacy_mean_flow") or 0.0) for row in rows
        ),
        "full_candidate_pareto_dominated_by_ablation": dominators,
    }


def _sota_tasklist_comparison_from_snapshots(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = _candidate_result(results)
    metadata = {spec.policy.name: spec for spec in sota_baseline_specs()}
    rows = []
    dominators = []
    for result in results:
        spec = metadata.get(str(result.get("policy") or ""))
        if spec is None:
            continue
        ms = _improvement(float(result.get("makespan_s") or 0.0), float(candidate.get("makespan_s") or 0.0))
        flow = _improvement(float(result.get("mean_flow_s") or 0.0), float(candidate.get("mean_flow_s") or 0.0))
        row = {
            "baseline": spec.snapshot(),
            "baseline_policy": result.get("policy", ""),
            "candidate_policy": candidate.get("policy", ""),
            "candidate_vs_baseline_makespan": ms,
            "candidate_vs_baseline_mean_flow": flow,
            "baseline_profiles": result.get("profiles", {}),
            "candidate_profiles": candidate.get("profiles", {}),
        }
        rows.append(row)
        if (
            ms < 1.0 - PARETO_TOLERANCE
            and flow < 1.0 - PARETO_TOLERANCE
        ):
            dominators.append({
                "name": spec.name,
                "policy": result.get("policy", ""),
                "representative_systems": list(spec.representative_systems),
                "candidate_vs_baseline_makespan": ms,
                "candidate_vs_baseline_mean_flow": flow,
                "baseline_profiles": result.get("profiles", {}),
                "candidate_profiles": candidate.get("profiles", {}),
            })
    return {
        "candidate_policy": candidate.get("policy", ""),
        "candidate_profiles": candidate.get("profiles", {}),
        "rows": rows,
        "candidate_pareto_dominated_by": dominators,
        "candidate_not_pareto_dominated": not dominators,
    }


def _candidate_result(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for row in results:
        if str(row.get("policy") or "").startswith("calibrated_"):
            return dict(row)
    return dict(results[0]) if results else {}


def _find_result(results: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    for row in results:
        if row.get("policy") == policy:
            return dict(row)
    return {}


def _relative_dict(
    results: Sequence[Mapping[str, Any]],
    baseline_name: str,
) -> dict[str, dict[str, float]]:
    baseline = _find_result(results, baseline_name)
    if not baseline:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in results:
        out[str(row.get("policy") or "")] = {
            "makespan_improvement": _improvement(
                float(baseline.get("makespan_s") or 0.0),
                float(row.get("makespan_s") or 0.0),
            ),
            "mean_flow_improvement": _improvement(
                float(baseline.get("mean_flow_s") or 0.0),
                float(row.get("mean_flow_s") or 0.0),
            ),
            "p90_flow_improvement": _improvement(
                float(baseline.get("p90_flow_s") or 0.0),
                float(row.get("p90_flow_s") or 0.0),
            ),
        }
    return out


def _supplement_sources(proof: Path) -> list[Path]:
    out = [
        proof / "ScheduleurmUpload.lean",
        proof / "lean-toolchain",
        proof / "lakefile.lean",
        proof / "lakefile.toml",
        proof / "Scheduleurm" / "lakefile.lean",
        proof / "Scheduleurm" / "lakefile.toml",
    ]
    return out


def _grep_forbidden_terms(proof: Path) -> dict[str, Any]:
    paths = [proof / "ScheduleurmUpload.lean", *(proof / "Scheduleurm").glob("*.lean")]
    terms = ("sorry", "admit", "axiom")
    patterns = {term: re.compile(rf"\b{re.escape(term)}\b") for term in terms}
    hits = []
    for path in paths:
        if not path.exists():
            continue
        text = _strip_lean_block_comments(path.read_text(encoding="utf-8", errors="replace"))
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("--"):
                continue
            for term in terms:
                if patterns[term].search(stripped):
                    hits.append({"path": str(path), "line": lineno, "term": term, "text": stripped[:200]})
    return {
        "forbidden_terms": list(terms),
        "forbidden_term_count": len(hits),
        "hits": hits,
    }


def _strip_lean_block_comments(text: str) -> str:
    out: list[str] = []
    idx = 0
    depth = 0
    while idx < len(text):
        two = text[idx:idx + 2]
        if two == "/-":
            depth += 1
            out.append("\n" if text[idx] == "\n" else " ")
            idx += 2
            continue
        if depth > 0 and two == "-/":
            depth = max(0, depth - 1)
            out.append(" ")
            idx += 2
            continue
        ch = text[idx]
        if depth > 0:
            out.append("\n" if ch == "\n" else " ")
        else:
            out.append(ch)
        idx += 1
    return "".join(out)


def _run_lean_build(proof: Path) -> dict[str, Any]:
    if not (proof / "lakefile.lean").exists() and not (proof / "lakefile.toml").exists():
        return {
            "ran": False,
            "status": "SKIPPED",
            "returncode": None,
            "stdout": "",
            "stderr": "no lakefile found at proof root",
        }
    cmd = ["lake", "env", "lean", "ScheduleurmUpload.lean"]
    proc = subprocess.run(
        cmd,
        cwd=str(proof),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "ran": True,
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": cmd,
    }


def _record_id(record: ProfileRecord, *, reason: str) -> dict[str, Any]:
    return {
        "workload_key": record.workload_key,
        "profile": record.profile,
        "source": record.source,
        "reason": reason,
    }


def _file_row(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": _reviewer_path(path),
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _reviewer_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(REPO_ROOT.parent))
        except ValueError:
            return resolved.name


def _sanitize_reviewer_text(text: str) -> str:
    proof_root = str(DEFAULT_PROOF_ROOT.expanduser().resolve())
    repo_root = str(REPO_ROOT.expanduser().resolve())
    mine_root = str(REPO_ROOT.parent.expanduser().resolve())
    return (
        text
        .replace(proof_root, "<LEAN_PROOF_SOURCE>")
        .replace(repo_root, "<REPO_ROOT>")
        .replace(mine_root, "<WORKSPACE_ROOT>")
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _positive_lognormal(rng: random.Random, mean_value: float, cv: float) -> float:
    if cv <= 0:
        return max(1e-12, float(mean_value))
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(max(1e-12, float(mean_value))) - sigma2 / 2.0
    return max(1e-12, rng.lognormvariate(mu, sigma))


def _geomean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if float(x) > 0.0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def _improvement(baseline_value: float, candidate_value: float) -> float:
    return baseline_value / candidate_value if candidate_value > 0.0 else 0.0


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.{digits}g}"


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _markdown_online(report: Mapping[str, Any]) -> str:
    agg = report.get("aggregate") or {}
    lines = [
        "# OR Gate: Online Arrival Experiments",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| scenario_count | {report.get('scenario_count', 0)} |",
        f"| candidate_completes_all | {str(bool(agg.get('candidate_completes_all'))).lower()} |",
        f"| candidate_not_pareto_dominated_all | {str(bool(agg.get('candidate_not_pareto_dominated_all'))).lower()} |",
        f"| candidate_vs_legacy_geomean_makespan | {_fmt(agg.get('candidate_vs_legacy_geomean_makespan'))} |",
        f"| candidate_vs_legacy_geomean_mean_flow | {_fmt(agg.get('candidate_vs_legacy_geomean_mean_flow'))} |",
        f"| worst_candidate_vs_legacy_makespan | {_fmt(agg.get('worst_candidate_vs_legacy_makespan'))} |",
        f"| worst_candidate_vs_legacy_mean_flow | {_fmt(agg.get('worst_candidate_vs_legacy_mean_flow'))} |",
        f"| max_candidate_backlog_jobs | {_fmt(agg.get('max_candidate_backlog_jobs'))} |",
        "",
        "## Scenario Rows",
        "",
        "| taskset | trace | jobs | candidate profiles | cand/legacy makespan | cand/legacy flow | SOTA nondominated | mean backlog |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in report.get("scenarios") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("taskset", "")),
                    str(row.get("trace", "")),
                    str(row.get("job_count", "")),
                    "`" + json.dumps(row.get("candidate_profiles") or {}, sort_keys=True) + "`",
                    _fmt(row.get("candidate_vs_legacy_makespan")),
                    _fmt(row.get("candidate_vs_legacy_mean_flow")),
                    str(bool(row.get("candidate_not_pareto_dominated_by_sota"))).lower(),
                    _fmt(row.get("candidate_mean_backlog_jobs")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _markdown_holdout(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Gate: Holdout Lower-Service Calibration",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| row_count | {report.get('row_count', 0)} |",
        f"| selected_row_count | {report.get('selected_row_count', 0)} |",
        f"| insufficient_sample_count | {report.get('insufficient_sample_count', 0)} |",
        f"| epsilon_est_all | {_fmt(report.get('epsilon_est_all'))} |",
        f"| epsilon_est_selected_actions | {_fmt(report.get('epsilon_est_selected_actions'))} |",
        f"| domination_violation_count | {report.get('lower_service_domination_violation_count', 0)} |",
        f"| adjusted_eta | {_fmt((report.get('holdout_adjusted_constants') or {}).get('eta'))} |",
        f"| lower_service_certificate_all_pass | {str(bool(report.get('lower_service_certificate_all_pass'))).lower()} |",
        f"| statistical_residual_pass | {str(bool(report.get('statistical_residual_pass'))).lower()} |",
        "",
        "## Adjusted Slack Table",
        "",
        str(report.get("holdout_calibration_table_md") or ""),
        "## Selected Action Rows",
        "",
        "| workload | profile | train n | holdout n | selected LCB/task | holdout mean/task | epsilon aggregate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("selected_rows") or []:
        lines.append(
            f"| {row.get('workload_key')} | {row.get('profile')} | {row.get('train_n')} | "
            f"{row.get('holdout_n')} | {_fmt(row.get('selected_lcb_per_task'))} | "
            f"{_fmt(row.get('holdout_mean_per_task'))} | {_fmt(row.get('epsilon_est_aggregate'))} |"
        )
    lines.extend([
        "",
        "## Lower-Service Capacity Certificates",
        "",
        "| taskset | profiles | delta | eta | B | theorem usable |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in report.get("lower_service_capacity_certificates") or []:
        lines.append(
            f"| {row.get('taskset')} | `{json.dumps(row.get('selected_profiles') or {}, sort_keys=True)}` | "
            f"{_fmt(row.get('delta'))} | {_fmt(row.get('eta'))} | {_fmt(row.get('B'))} | "
            f"{str(bool(row.get('usable_for_theorem'))).lower()} |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _markdown_ablation(report: Mapping[str, Any]) -> str:
    agg = report.get("aggregate") or {}
    lines = [
        "# OR Gate: Module Ablation Suite",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| trace_count | {agg.get('trace_count', 0)} |",
        f"| full_candidate_vs_legacy_geomean_makespan | {_fmt(agg.get('full_candidate_vs_legacy_geomean_makespan'))} |",
        f"| full_candidate_vs_legacy_geomean_mean_flow | {_fmt(agg.get('full_candidate_vs_legacy_geomean_mean_flow'))} |",
        f"| ablation_dominator_count | {len(agg.get('full_candidate_pareto_dominated_by_ablation') or [])} |",
        "",
        "## Reviewer Axis Coverage",
        "",
        "| Axis | Policy | Role |",
        "|---|---|---|",
    ]
    for row in report.get("reviewer_axis_coverage") or []:
        lines.append(
            f"| {row.get('reviewer_axis')} | `{row.get('policy')}` | {row.get('role')} |"
        )
    lines.extend([
        "",
        "## Trace Rows",
        "",
        "| taskset | trace | jobs | profiles | cand/legacy makespan | cand/legacy flow | dominators |",
        "|---|---|---:|---|---:|---:|---:|",
    ])
    for row in report.get("rows") or []:
        lines.append(
            f"| {row.get('taskset')} | {row.get('trace')} | {row.get('job_count')} | "
            f"`{json.dumps(row.get('full_candidate_profiles') or {}, sort_keys=True)}` | "
            f"{_fmt(row.get('full_candidate_vs_legacy_makespan'))} | "
            f"{_fmt(row.get('full_candidate_vs_legacy_mean_flow'))} | "
            f"{len(row.get('full_candidate_pareto_dominated_by_ablation') or [])} |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _markdown_live(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Gate: Longer Live Theorem Trace",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| status | `{report.get('status', '')}` |",
        f"| trace_slot_count | {report.get('trace_slot_count', 0)} |",
        f"| min_required_slots | {report.get('min_required_slots', 0)} |",
        f"| candidate_count_total | {report.get('candidate_count_total', 0)} |",
        f"| usable_for_live_scheduler_oracle_trace | {str(bool(report.get('usable_for_live_scheduler_oracle_trace'))).lower()} |",
        f"| trace_origin | `{report.get('trace_origin', '')}` |",
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ]
    blockers = report.get("blockers") or []
    if blockers:
        lines.extend(["## Blockers", "", "| reason | path |", "|---|---|"])
        for row in blockers:
            lines.append(f"| `{row.get('reason')}` | `{row.get('path')}` |")
        lines.append("")
    dryrun = report.get("dryrun_probe_report") or {}
    if dryrun:
        lines.extend([
            "## Dry-Run Probe",
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| placed_count | {dryrun.get('placed_count', 0)} |",
            f"| unplaced_count | {dryrun.get('unplaced_count', 0)} |",
            f"| trace_slot_count | {dryrun.get('trace_slot_count', 0)} |",
            f"| candidate_count_total | {dryrun.get('candidate_count_total', 0)} |",
            f"| algorithm | `{dryrun.get('algorithm', '')}` |",
            "",
            str(dryrun.get("scope") or ""),
            "",
        ])
    return "\n".join(lines)


def _markdown_supplement(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Gate: Reviewer Supplement Repackage",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| pass | {str(bool(report.get('pass'))).lower()} |",
        f"| upload_exists | {str(bool(report.get('upload_exists'))).lower()} |",
        f"| ScheduleurmUpload sha256 | `{report.get('scheduleurm_upload_sha256', '')}` |",
        f"| forbidden_term_count | {(report.get('forbidden_term_grep') or {}).get('forbidden_term_count', 0)} |",
        f"| lean_build_status | `{(report.get('lean_build') or {}).get('status')}` |",
        "",
        "## Files",
        "",
        "| role | path | sha256 |",
        "|---|---|---|",
    ]
    for row in report.get("copied_files") or []:
        lines.append(f"| {row.get('role')} | `{row.get('path')}` | `{row.get('sha256')}` |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _markdown_all(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Submission Closure Gates",
        "",
        "| Gate | Pass | Status | Scope |",
        "|---|---:|---|---|",
    ]
    for row in report.get("gates") or []:
        lines.append(
            f"| `{row.get('gate')}` | {str(bool(row.get('pass'))).lower()} | "
            f"`{row.get('status')}` | {row.get('scope')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _cmd_online(args: argparse.Namespace) -> int:
    report = build_online_arrival_experiments(
        taskset_names=tuple(args.taskset),
        loads=tuple(args.load),
        seeds=tuple(args.seed),
        modes=tuple(args.mode),
    )
    _write_json(args.output, report)
    if args.markdown_output:
        _write_text(args.markdown_output, markdown_report(report))
    print(args.output)
    return 0 if report.get("pass") else 2


def _cmd_holdout(args: argparse.Namespace) -> int:
    report = build_holdout_calibration(
        taskset_names=tuple(args.taskset),
        seed=args.seed,
        train_fraction=args.train_fraction,
        confidence=args.confidence,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        _write_text(args.markdown_output, markdown_report(report))
    print(args.output)
    return 0 if report.get("pass") else 2


def _cmd_ablation(args: argparse.Namespace) -> int:
    report = build_ablation_suite(
        taskset_names=tuple(args.taskset),
        trace_modes=tuple(args.mode),
        load_factor=args.load_factor,
        seed=args.seed,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        _write_text(args.markdown_output, markdown_report(report))
    print(args.output)
    return 0 if report.get("pass") else 2


def _cmd_live(args: argparse.Namespace) -> int:
    report = build_live_trace_gate(
        trace_path=args.trace_path,
        min_slots=args.min_slots,
        workload_key=args.workload_key,
        profile=args.profile,
        queue_weight=args.queue_weight,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        _write_text(args.markdown_output, markdown_report(report))
    print(args.output)
    return 0 if report.get("pass") else 2


def _cmd_supplement(args: argparse.Namespace) -> int:
    report = build_reviewer_supplement(
        proof_root=args.proof_root,
        output_dir=args.output_dir,
        run_build=not args.no_build,
    )
    _write_json(Path(args.output_dir) / "manifest.json", report)
    if args.markdown_output:
        _write_text(args.markdown_output, markdown_report(report))
    print(Path(args.output_dir) / "manifest.json")
    return 0 if report.get("pass") else 2


def _cmd_all(args: argparse.Namespace) -> int:
    report = build_all_closure_gates(
        output_prefix=args.output_prefix,
        run_lean_build=not args.no_build,
        live_min_slots=args.live_min_slots,
    )
    print(str(Path(args.output_prefix).with_suffix(".json")))
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.or_submission_closure")
    sub = parser.add_subparsers(dest="cmd", required=True)

    online = sub.add_parser("online", help="Run Poisson/bursty load-sweep replay")
    online.add_argument("--taskset", action="append", default=list(DEFAULT_TASKSETS))
    online.add_argument("--load", action="append", type=float, default=list(DEFAULT_LOADS))
    online.add_argument("--seed", action="append", type=int, default=list(DEFAULT_SEEDS))
    online.add_argument("--mode", action="append", default=["poisson", "bursty"])
    online.add_argument("--output", default=str(ARTIFACT_ROOT / "or_gate_online_arrivals.json"))
    online.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_online_arrivals.md"))
    online.set_defaults(func=_cmd_online)

    holdout = sub.add_parser("holdout", help="Run holdout lower-service calibration")
    holdout.add_argument("--taskset", action="append", default=list(DEFAULT_TASKSETS))
    holdout.add_argument("--seed", type=int, default=17)
    holdout.add_argument("--train-fraction", type=float, default=0.5)
    holdout.add_argument("--confidence", type=float, default=0.95)
    holdout.add_argument("--output", default=str(ARTIFACT_ROOT / "or_gate_holdout_calibration.json"))
    holdout.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_holdout_calibration.md"))
    holdout.set_defaults(func=_cmd_holdout)

    ablation = sub.add_parser("ablation", help="Run module ablation replay")
    ablation.add_argument("--taskset", action="append", default=list(DEFAULT_TASKSETS))
    ablation.add_argument("--mode", action="append", default=["static", "poisson", "bursty"])
    ablation.add_argument("--load-factor", type=float, default=0.85)
    ablation.add_argument("--seed", type=int, default=23)
    ablation.add_argument("--output", default=str(ARTIFACT_ROOT / "or_gate_ablation_suite.json"))
    ablation.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_ablation_suite.md"))
    ablation.set_defaults(func=_cmd_ablation)

    live = sub.add_parser("live-trace", help="Audit longer live scheduler oracle trace")
    live.add_argument("--trace-path", default="~/.claude/scheduler/oracle_trace.jsonl")
    live.add_argument("--min-slots", type=int, default=96)
    live.add_argument("--workload-key", default="scheduleurm_control_plane_completed_history")
    live.add_argument("--profile", type=int, default=1)
    live.add_argument("--queue-weight", type=float, default=1.0)
    live.add_argument("--output", default=str(ARTIFACT_ROOT / "or_gate_live_trace.json"))
    live.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_live_trace.md"))
    live.set_defaults(func=_cmd_live)

    supplement = sub.add_parser("supplement", help="Repackage Lean reviewer supplement")
    supplement.add_argument("--proof-root", default=str(DEFAULT_PROOF_ROOT))
    supplement.add_argument("--output-dir", default=str(ARTIFACT_ROOT / "or_reviewer_supplement"))
    supplement.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "or_gate_reviewer_supplement.md"))
    supplement.add_argument("--no-build", action="store_true")
    supplement.set_defaults(func=_cmd_supplement)

    all_cmd = sub.add_parser("all", help="Run all closure gates")
    all_cmd.add_argument("--output-prefix", default=str(ARTIFACT_ROOT / "or_submission_closure_2026_06_11"))
    all_cmd.add_argument("--no-build", action="store_true")
    all_cmd.add_argument("--live-min-slots", type=int, default=96)
    all_cmd.set_defaults(func=_cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
