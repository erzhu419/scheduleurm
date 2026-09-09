"""Run controlled CPU probes for the full-factorial ETA design.

This runner is the CPU-side counterpart of
``full_factorial_eta_probe_runner``.  It bypasses the legacy scheduler path,
deploys the task-native progress benchmarks to a named node, and writes a
service-cache snapshot keyed by workload, node bucket, load state, and profile.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache

from .remote_workload_selected_profile_probe import (
    _run_remote,
    _run_remote_capture,
    _scheduler_node,
    _scheduler_module,
    _write_remote_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
REMOTE_ROOT = "/tmp/scheduleurm_progress_wrapper_pkg"
REMOTE_EXP_DIR = f"{REMOTE_ROOT}/algorithm/experiments"
DEFAULT_DESIGN = ARTIFACT_ROOT / "full_factorial_eta_design_rows_20260629.csv"
RATE_RE = re.compile(r"\brate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")
TOTAL_RATE_RE = re.compile(r"\btotal_rate=([0-9]+(?:\.[0-9]+)?)\s+step/s\b")


@dataclass(frozen=True)
class CpuProbeConfig:
    steps: int
    work_items: int
    timeout_s: int
    mode: str = "cpu"
    use_parallel: bool = True
    log_interval: int = 2


CPU_CONFIGS: dict[str, CpuProbeConfig] = {
    "light_control_local": CpuProbeConfig(steps=160, work_items=8000, timeout_s=600, mode="light", use_parallel=False),
    "cpu_heavy_local_bench": CpuProbeConfig(steps=160, work_items=90000, timeout_s=2400, mode="cpu", use_parallel=True),
    "freqduet_cpu_ablation_c17_32": CpuProbeConfig(steps=160, work_items=130000, timeout_s=3000, mode="cpu", use_parallel=True),
    "sumo_eval_cpu": CpuProbeConfig(steps=160, work_items=70000, timeout_s=2400, mode="cpu", use_parallel=True),
}


def build_full_factorial_cpu_eta_probe_runner(
    *,
    design_csv: Path = DEFAULT_DESIGN,
    allow_launch: bool = False,
    tier: str = "T0_core_curve",
    resource_state: str = "empty",
    hardware_groups: set[str] | None = None,
    workloads: set[str] | None = None,
    nodes: set[str] | None = None,
    max_rows: int = 1,
    max_profiles: int = 0,
    run_prefix: str = "full_factorial_cpu_eta_probe_20260630",
) -> dict[str, Any]:
    selected = _select_rows(
        design_csv,
        tier=tier,
        resource_state=resource_state,
        hardware_groups=hardware_groups,
        workloads=workloads,
        nodes=nodes,
        max_rows=max_rows,
    )
    cache = ServiceRateCache()
    rows = []
    for row in selected:
        manifest = _manifest(row, run_prefix=run_prefix, max_profiles=max_profiles)
        if allow_launch and manifest.get("launch_ready"):
            result = _launch(manifest)
            rows.append(result)
            _add_result_to_cache(cache, result)
        else:
            rows.append(manifest)
    launched = [row for row in rows if row.get("launched")]
    admitted = [row for row in rows if row.get("measurement_valid")]
    return {
        "gate": "full_factorial_cpu_eta_probe_runner",
        "allow_launch": bool(allow_launch),
        "run_prefix": run_prefix,
        "tier": tier,
        "resource_state": resource_state,
        "selected_count": len(selected),
        "launched_count": len(launched),
        "admitted_count": len(admitted),
        "pass": bool(rows) and (not allow_launch or bool(launched)),
        "status": "CPU_PROBE_RUNNER_LAUNCHED" if allow_launch else "CPU_PROBE_RUNNER_MANIFEST",
        "rows": rows,
        "admitted_rows": admitted,
        "service_cache_snapshot": cache.snapshot(),
        "scope": (
            "Controlled full-factorial CPU ETA probes.  Launches bypass the "
            "legacy scheduler dispatch path and accept only task-native "
            "progress/tqdm rate samples with a stable tail gate."
        ),
    }


def _select_rows(
    design_csv: Path,
    *,
    tier: str,
    resource_state: str,
    hardware_groups: set[str] | None,
    workloads: set[str] | None,
    nodes: set[str] | None,
    max_rows: int,
) -> list[dict[str, str]]:
    out = []
    for row in csv.DictReader(design_csv.open(encoding="utf-8")):
        if row.get("tier") != tier:
            continue
        if row.get("resource_state") != resource_state:
            continue
        if "pending" not in row.get("status", ""):
            continue
        if row.get("resource_kind") != "cpu":
            continue
        if hardware_groups and row.get("hardware_group") not in hardware_groups:
            continue
        if workloads and row.get("workload_key") not in workloads and row.get("family") not in workloads:
            continue
        if nodes and row.get("node") not in nodes:
            continue
        if not _parse_profiles(row.get("missing_profiles")):
            continue
        out.append(row)
        if max_rows > 0 and len(out) >= max_rows:
            break
    return out


def _manifest(row: Mapping[str, str], *, run_prefix: str, max_profiles: int) -> dict[str, Any]:
    workload_key = str(row.get("workload_key") or "")
    config = CPU_CONFIGS.get(workload_key)
    profiles = _parse_profiles(row.get("missing_profiles"))
    if max_profiles > 0:
        profiles = profiles[: int(max_profiles)]
    node = str(row.get("node") or "")
    resident_supported = _resident_state_supported(row)
    use_parallel = bool(config.use_parallel) if config is not None else False
    runner_profile_axis = (
        "allocation_workers" if use_parallel else "colocation_count"
    )
    design_profile_axis = str(row.get("profile_axis") or runner_profile_axis)
    profile_axis_match = design_profile_axis == runner_profile_axis
    launch_ready = bool(
        config is not None
        and profiles
        and resident_supported
        and profile_axis_match
    )
    run_id = _safe_id(f"{run_prefix}_{node}_{workload_key}_{row.get('workload_env')}_{row.get('resource_state')}")
    return {
        "row_id": row.get("row_id"),
        "run_id": run_id,
        "node": node,
        "node_bucket": row.get("node_bucket"),
        "hardware_group": row.get("hardware_group"),
        "workload_key": workload_key,
        "workload_env": row.get("workload_env"),
        "family": row.get("family"),
        "resource_state": row.get("resource_state"),
        "resident_mix": row.get("resident_mix"),
        "profiles": profiles,
        "profile_axis": runner_profile_axis,
        "design_profile_axis": design_profile_axis,
        "profile_axis_match": profile_axis_match,
        "allocation_worker_profiles": profiles if use_parallel else ([1] if profiles else []),
        "colocation_count_profiles": ([1] if profiles else []) if use_parallel else profiles,
        "launch_ready": launch_ready,
        "pending_reason": "" if launch_ready else _pending_reason(
            config=config,
            profiles=profiles,
            resident_supported=resident_supported,
            profile_axis_match=profile_axis_match,
        ),
        "launched": False,
        "measurement_valid": False,
        "eta_source_required": "tqdm/progress",
        "legacy_scheduler_limits_bypassed": True,
        "config": None if config is None else {
            "steps": config.steps,
            "work_items": config.work_items,
            "timeout_s": config.timeout_s,
            "mode": config.mode,
            "use_parallel": config.use_parallel,
            "log_interval": config.log_interval,
        },
    }


def _resident_state_supported(row: Mapping[str, str]) -> bool:
    state = str(row.get("resource_state") or "empty")
    # These rows express load directly through the manifest's explicit worker
    # allocation or co-location axis; no separate resident process is launched.
    return state in {"empty", "full_loaded"}


def _pending_reason(
    *,
    config: CpuProbeConfig | None,
    profiles: list[int],
    resident_supported: bool,
    profile_axis_match: bool,
) -> str:
    if config is None:
        return "no_cpu_probe_config"
    if not profiles:
        return "no_missing_profiles"
    if not resident_supported:
        return "resident_load_probe_not_implemented_for_state"
    if not profile_axis_match:
        return "profile_axis_mismatch_requires_native_colocation_runner"
    return "not_launch_ready"


def _cpu_profile_dimensions(profile: int, *, use_parallel: bool) -> tuple[int, int]:
    requested = max(1, int(profile))
    if use_parallel:
        return requested, 1
    return 1, requested


def _cpu_probe_steps(config: Mapping[str, Any], *, use_parallel: bool) -> int:
    default_steps = 80 if use_parallel else 160
    return max(1, int(config.get("steps") or default_steps))


def _launch(manifest: Mapping[str, Any]) -> dict[str, Any]:
    node = str(manifest["node"])
    run_id = str(manifest["run_id"])
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    report_dir = run_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _deploy_cpu_scripts(node, raw_dir)
    config = manifest.get("config") or {}
    profile_rows = []
    for profile in [int(x) for x in manifest.get("profiles") or []]:
        profile_rows.append(
            _measure_profile(
                node=node,
                run_id=run_id,
                raw_dir=raw_dir,
                report_dir=report_dir,
                profile=profile,
                workload_key=str(manifest.get("workload_key") or ""),
                config=config,
            )
        )
    valid = all(bool(row.get("measurement_valid")) for row in profile_rows)
    return {
        **dict(manifest),
        "launched": True,
        "measurement_valid": valid,
        "profile_rows": profile_rows,
        "reason": "" if valid else "one_or_more_cpu_profiles_missing_stable_progress_rate",
    }


def _measure_profile(
    *,
    node: str,
    run_id: str,
    raw_dir: Path,
    report_dir: Path,
    profile: int,
    workload_key: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    use_parallel = bool(config.get("use_parallel"))
    allocation_workers, colocation_count = _cpu_profile_dimensions(
        profile, use_parallel=use_parallel
    )
    steps = _cpu_probe_steps(config, use_parallel=use_parallel)
    phase = f"profile_{int(profile)}_per_resource"
    label = _safe_id(f"{run_id}_{phase}")
    if _windows_node(node):
        return _measure_profile_windows(
            node=node,
            run_id=run_id,
            raw_dir=raw_dir,
            report_dir=report_dir,
            profile=profile,
            workload_key=workload_key,
            config=config,
            label=label,
            phase=phase,
        )
    if use_parallel:
        cmd = (
            _remote_python_prefix()
            + f"\"$PY\" -u {REMOTE_EXP_DIR}/cpu_parallel_progress_compat_benchmark.py "
            f"--steps {steps} "
            f"--workers {allocation_workers} "
            f"--work-items {int(config.get('work_items') or 90000)} "
            f"--log-interval {int(config.get('log_interval') or 2)} "
            f"--label {shlex.quote(label)}"
        )
    else:
        cmd = _light_concurrent_cmd(
            profile=colocation_count, label=label, config=config
        )
    rc, out, err = _run_remote_capture(
        node,
        f"cd /tmp && {cmd}",
        raw_dir / f"{phase}_run",
        timeout_s=int(config.get("timeout_s") or 1800),
    )
    log_path = raw_dir / f"{phase}.log"
    log_path.write_text((out or "") + ("\nSTDERR:\n" + err if err else ""), encoding="utf-8")
    if use_parallel:
        rates = _progress_rates(out or "", prefer_total=True)
        stable = _stable_tail(rates)
        per_task_stable_rates = [float(stable.get("mean_rate") or 0.0)] if bool(stable.get("ready")) else []
        rc_valid = int(rc) == 0
        natural_completion_count = 1 if "CPU_PARALLEL_DONE" in (out or "") else 0
    else:
        sections = _split_cpu_log_sections(out or "")
        per_task_stable_rates = []
        rates = []
        for section in sections:
            section_rates = _progress_rates(section)
            rates.extend(section_rates)
            stable = _stable_tail(section_rates)
            if bool(stable.get("ready")):
                per_task_stable_rates.append(float(stable.get("mean_rate") or 0.0))
        rc_valid = int(rc) == 0 or (
            len(sections) == colocation_count
            and all("CPU_BENCH_DONE" in section for section in sections)
        )
        natural_completion_count = sum(
            1 for section in sections if "CPU_BENCH_DONE" in section
        )
        stable = {
            "ready": len(per_task_stable_rates) == colocation_count,
            "samples": len(rates),
            "mean_rate": sum(per_task_stable_rates),
            "cv": 0.0,
            "last_two_relative_delta": 0.0,
            "tail": per_task_stable_rates,
        }
    aggregate = float(sum(per_task_stable_rates))
    stable_progress_valid = bool(stable.get("ready")) and aggregate > 0.0
    natural_completion_observed = natural_completion_count == colocation_count
    rc_accepted = bool(rc_valid or stable_progress_valid)
    measurement_valid = rc_accepted and stable_progress_valid
    summary = {
        "phase": phase,
        "probe": "full_factorial_cpu_eta_probe_runner",
        "run_id": run_id,
        "workload_key": workload_key,
        "profile": int(profile),
        "allocation_workers": allocation_workers,
        "colocation_count": colocation_count,
        "steps": steps,
        "total_units": steps,
        "returncode": int(rc),
        "returncode_overridden_by_done_logs": bool(rc_valid and int(rc) != 0),
        "running_count": colocation_count,
        "returncode_valid_count": colocation_count if rc_valid else 0,
        "returncode_accepted_count": (
            colocation_count if rc_accepted else 0
        ),
        "returncode_acceptance_rule": "strict_zero_or_stable_required_rate",
        "require_stable_rate": True,
        "terminate_on_stable": False,
        "steady_state_rate_ready": stable_progress_valid,
        "natural_completion_observed": natural_completion_observed,
        "natural_completion_count": natural_completion_count,
        "require_completion_model": False,
        "completion_model_ready": False,
        "completion_model_count": 0,
        "completion_model_ready_count": 0,
        "all_completion_models_ready": False,
        "service_semantics": "steady_state_throughput",
        "rate_units": ["step"],
        "rates_unit_s": rates,
        "stable_rates_unit_s": per_task_stable_rates,
        "aggregate_stable_rate_unit_s": aggregate,
        "mean_stable_rate_unit_s": aggregate,
        "stable_rate_ready_count": len(per_task_stable_rates),
        "all_stable_rate_ready": bool(stable.get("ready")),
        "measurement_valid": measurement_valid,
        "placement_valid": rc_accepted,
        "status_counts": {"sampled": 1 if rates else 0, "missing_rate": 0 if rates else 1},
        "stable_tail": stable,
        "eta_source": "tqdm/progress",
        "capacity_boundary": False,
    }
    summary_path = report_dir / f"{phase}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{phase}_summary.md").write_text(_summary_markdown(summary, summary_path), encoding="utf-8")
    return {
        "profile": int(profile),
        "allocation_workers": allocation_workers,
        "colocation_count": colocation_count,
        "steps": steps,
        "summary_path": str(summary_path),
        "path_exists": True,
        "measurement_valid": measurement_valid,
        "stable_rate_ready_count": int(summary["stable_rate_ready_count"]),
        "aggregate_stable_rate_unit_s": aggregate,
        "steady_state_rate_ready": stable_progress_valid,
        "natural_completion_observed": natural_completion_observed,
        "completion_model_ready": False,
        "eta_source": "tqdm/progress",
        "returncode": int(rc),
    }


def _add_result_to_cache(cache: ServiceRateCache, result: Mapping[str, Any]) -> None:
    for profile_row in result.get("profile_rows") or []:
        if not profile_row.get("measurement_valid"):
            continue
        path = Path(str(profile_row.get("summary_path") or ""))
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        requested_profile = int(
            profile_row.get("profile") or summary.get("profile") or 0
        )
        allocation_workers, colocation_count = _summary_cpu_dimensions(
            summary,
            result=result,
            requested_profile=requested_profile,
        )
        total_units = _summary_total_units(summary, result=result)
        aggregate = float(summary.get("aggregate_stable_rate_unit_s") or 0.0)
        per_task_rates = tuple(float(x) for x in (summary.get("stable_rates_unit_s") or []) if float(x) > 0.0)
        if requested_profile <= 0 or aggregate <= 0.0 or total_units <= 0.0:
            continue
        cache.add(
            ProfileRecord(
                workload_key=str(result.get("workload_key")),
                command_fingerprint="live_tqdm_service_cache_v2_full_factorial_cpu_20260630",
                resource_kind=str(result.get("family") or "cpu"),
                node_bucket=str(result.get("node_bucket")),
                profile=colocation_count,
                unit="step",
                total_units=total_units,
                aggregate_rate=aggregate,
                per_task_rates=per_task_rates or (aggregate,),
                source=str(path),
                allocation_workers=allocation_workers,
                colocation_count=colocation_count,
                workload_env=str(result.get("workload_env")),
                resource_state=str(result.get("resource_state")),
                resident_mix=str(result.get("resident_mix") or ""),
                eta_source="tqdm/progress",
                stable_rate_ready=True,
                hardware_class=str(result.get("hardware_group")),
            ),
            force_replace=True,
        )


def _summary_cpu_dimensions(
    summary: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    requested_profile: int,
) -> tuple[int, int]:
    if (
        summary.get("allocation_workers") is not None
        or summary.get("colocation_count") is not None
    ):
        return (
            max(1, int(summary.get("allocation_workers") or 1)),
            max(
                1,
                int(summary.get("colocation_count") or requested_profile or 1),
            ),
        )
    config = result.get("config") or {}
    return _cpu_profile_dimensions(
        requested_profile,
        use_parallel=bool(config.get("use_parallel")),
    )


def _summary_total_units(
    summary: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
) -> float:
    config = result.get("config") or {}
    for value in (
        summary.get("steps"),
        summary.get("total_units"),
        config.get("steps"),
    ):
        try:
            units = float(value)
        except (TypeError, ValueError):
            continue
        if units > 0.0 and math.isfinite(units):
            return units
    return 0.0


def _deploy_cpu_scripts(node: str, raw_dir: Path) -> None:
    if _windows_node(node):
        _deploy_cpu_scripts_windows(node, raw_dir)
        return
    remote_mkdir = (
        f"mkdir -p {REMOTE_EXP_DIR} && touch {REMOTE_ROOT}/algorithm/__init__.py "
        f"{REMOTE_EXP_DIR}/__init__.py"
    )
    _run_remote(node, remote_mkdir, raw_dir / "deploy_cpu_mkdir", timeout_s=45)
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    for name in ("cpu_progress_benchmark.py", "cpu_parallel_progress_benchmark.py"):
        _write_remote_text(
            node,
            f"{REMOTE_EXP_DIR}/{name}",
            (local_dir / name).read_text(encoding="utf-8"),
            raw_dir / f"deploy_{name}",
        )
    for name in ("cpu_progress_compat_benchmark.py", "cpu_parallel_progress_compat_benchmark.py"):
        _write_remote_text(
            node,
            f"{REMOTE_EXP_DIR}/{name}",
            (local_dir / name).read_text(encoding="utf-8"),
            raw_dir / f"deploy_{name}",
        )


def _deploy_cpu_scripts_windows(node: str, raw_dir: Path) -> None:
    scheduler = _scheduler_module()
    mkdir = (
        "$dir = Join-Path $env:TEMP 'scheduleurm_progress_wrapper_pkg\\algorithm\\experiments'; "
        "New-Item -Force -ItemType Directory -Path $dir | Out-Null; "
        "Write-Output $dir"
    )
    _run_windows_ps_capture(scheduler, node, mkdir, raw_dir / "deploy_cpu_windows_mkdir", timeout_s=60)
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    for name in ("cpu_progress_compat_benchmark.py", "cpu_parallel_progress_compat_benchmark.py"):
        text = (local_dir / name).read_text(encoding="utf-8")
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        ps = (
            "$dir = Join-Path $env:TEMP 'scheduleurm_progress_wrapper_pkg\\algorithm\\experiments'; "
            "New-Item -Force -ItemType Directory -Path $dir | Out-Null; "
            f"$bytes = [Convert]::FromBase64String('{encoded}'); "
            "$text = [Text.Encoding]::UTF8.GetString($bytes); "
            f"$path = Join-Path $dir '{name}'; "
            "$enc = New-Object System.Text.UTF8Encoding $false; "
            "[IO.File]::WriteAllText($path, $text, $enc); "
            "Write-Output $path"
        )
        _run_windows_ps_capture(scheduler, node, ps, raw_dir / f"deploy_windows_{name}", timeout_s=60)


def _measure_profile_windows(
    *,
    node: str,
    run_id: str,
    raw_dir: Path,
    report_dir: Path,
    profile: int,
    workload_key: str,
    config: Mapping[str, Any],
    label: str,
    phase: str,
) -> dict[str, Any]:
    scheduler = _scheduler_module()
    py = _windows_python(node)
    use_parallel = bool(config.get("use_parallel"))
    allocation_workers, colocation_count = _cpu_profile_dimensions(
        profile, use_parallel=use_parallel
    )
    steps = _cpu_probe_steps(config, use_parallel=use_parallel)
    if use_parallel:
        ps = _windows_parallel_ps(
            py=py,
            profile=allocation_workers,
            label=label,
            config=config,
        )
    else:
        ps = _windows_light_ps(
            py=py,
            profile=colocation_count,
            label=label,
            config=config,
        )
    rc, out, err = _run_windows_ps_capture(
        scheduler,
        node,
        ps,
        raw_dir / f"{phase}_run",
        timeout_s=int(config.get("timeout_s") or 1800),
    )
    log_path = raw_dir / f"{phase}.log"
    log_path.write_text((out or "") + ("\nSTDERR:\n" + err if err else ""), encoding="utf-8")
    if use_parallel:
        rates = _progress_rates(out or "", prefer_total=True)
        stable = _stable_tail(rates)
        per_task_stable_rates = [float(stable.get("mean_rate") or 0.0)] if bool(stable.get("ready")) else []
        rc_valid = int(rc) == 0
        natural_completion_count = 1 if "CPU_PARALLEL_DONE" in (out or "") else 0
    else:
        sections = _split_cpu_log_sections(out or "")
        rates = []
        per_task_stable_rates = []
        for section in sections:
            section_rates = _progress_rates(section)
            rates.extend(section_rates)
            stable = _stable_tail(section_rates)
            if bool(stable.get("ready")):
                per_task_stable_rates.append(float(stable.get("mean_rate") or 0.0))
        rc_valid = int(rc) == 0 or (
            len(sections) == colocation_count
            and all("CPU_BENCH_DONE" in section for section in sections)
        )
        natural_completion_count = sum(
            1 for section in sections if "CPU_BENCH_DONE" in section
        )
        stable = {
            "ready": len(per_task_stable_rates) == colocation_count,
            "samples": len(rates),
            "mean_rate": sum(per_task_stable_rates),
            "cv": 0.0,
            "last_two_relative_delta": 0.0,
            "tail": per_task_stable_rates,
        }
    aggregate = float(sum(per_task_stable_rates))
    stable_progress_valid = bool(stable.get("ready")) and aggregate > 0.0
    natural_completion_observed = natural_completion_count == colocation_count
    rc_accepted = bool(rc_valid or stable_progress_valid)
    measurement_valid = rc_accepted and stable_progress_valid
    summary = {
        "phase": phase,
        "probe": "full_factorial_cpu_eta_probe_runner_windows",
        "run_id": run_id,
        "workload_key": workload_key,
        "profile": int(profile),
        "allocation_workers": allocation_workers,
        "colocation_count": colocation_count,
        "steps": steps,
        "total_units": steps,
        "returncode": int(rc),
        "returncode_overridden_by_done_logs": bool(rc_valid and int(rc) != 0),
        "running_count": colocation_count,
        "returncode_valid_count": colocation_count if rc_valid else 0,
        "returncode_accepted_count": (
            colocation_count if rc_accepted else 0
        ),
        "returncode_acceptance_rule": "strict_zero_or_stable_required_rate",
        "require_stable_rate": True,
        "terminate_on_stable": False,
        "steady_state_rate_ready": stable_progress_valid,
        "natural_completion_observed": natural_completion_observed,
        "natural_completion_count": natural_completion_count,
        "require_completion_model": False,
        "completion_model_ready": False,
        "completion_model_count": 0,
        "completion_model_ready_count": 0,
        "all_completion_models_ready": False,
        "service_semantics": "steady_state_throughput",
        "rate_units": ["step"],
        "rates_unit_s": rates,
        "stable_rates_unit_s": per_task_stable_rates,
        "aggregate_stable_rate_unit_s": aggregate,
        "mean_stable_rate_unit_s": aggregate,
        "stable_rate_ready_count": len(per_task_stable_rates),
        "all_stable_rate_ready": bool(stable.get("ready")),
        "measurement_valid": measurement_valid,
        "placement_valid": rc_accepted,
        "status_counts": {"sampled": 1 if rates else 0, "missing_rate": 0 if rates else 1},
        "stable_tail": stable,
        "eta_source": "tqdm/progress",
        "capacity_boundary": False,
    }
    summary_path = report_dir / f"{phase}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{phase}_summary.md").write_text(_summary_markdown(summary, summary_path), encoding="utf-8")
    return {
        "profile": int(profile),
        "allocation_workers": allocation_workers,
        "colocation_count": colocation_count,
        "steps": steps,
        "summary_path": str(summary_path),
        "path_exists": True,
        "measurement_valid": measurement_valid,
        "stable_rate_ready_count": int(summary["stable_rate_ready_count"]),
        "aggregate_stable_rate_unit_s": aggregate,
        "steady_state_rate_ready": stable_progress_valid,
        "natural_completion_observed": natural_completion_observed,
        "completion_model_ready": False,
        "eta_source": "tqdm/progress",
        "returncode": int(rc),
    }


def _windows_parallel_ps(*, py: str, profile: int, label: str, config: Mapping[str, Any]) -> str:
    steps = int(config.get("steps") or 80)
    work_items = int(config.get("work_items") or 90000)
    log_interval = int(config.get("log_interval") or 2)
    return (
        "$dir = Join-Path $env:TEMP 'scheduleurm_progress_wrapper_pkg\\algorithm\\experiments'; "
        "$script = Join-Path $dir 'cpu_parallel_progress_compat_benchmark.py'; "
        f"& {_ps_quote(py)} -u $script --steps {steps} --workers {int(profile)} "
        f"--work-items {work_items} --log-interval {log_interval} --label {_ps_quote(label)}"
    )


def _windows_light_ps(*, py: str, profile: int, label: str, config: Mapping[str, Any]) -> str:
    count = max(1, int(profile))
    steps = int(config.get("steps") or 160)
    work_items = int(config.get("work_items") or 8000)
    mode = str(config.get("mode") or "light")
    return (
        "$dir = Join-Path $env:TEMP 'scheduleurm_progress_wrapper_pkg\\algorithm\\experiments'; "
        "$script = Join-Path $dir 'cpu_progress_compat_benchmark.py'; "
        f"$out = Join-Path $env:TEMP 'scheduleurm_cpu_light_{_safe_id(label)}'; "
        "if (Test-Path $out) { Remove-Item -Recurse -Force $out }; "
        "New-Item -Force -ItemType Directory -Path $out | Out-Null; "
        "$procs = @(); "
        f"for ($i = 0; $i -lt {count}; $i++) {{ "
        "$log = Join-Path $out ($i.ToString() + '.log'); "
        f"$args = @('-u', $script, '--steps', '{steps}', '--mode', {_ps_quote(mode)}, '--work-items', '{work_items}', '--label', ({_ps_quote(label)} + '_' + $i)); "
        f"$procs += Start-Process -FilePath {_ps_quote(py)} -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError ($log + '.err') -NoNewWindow -PassThru; "
        "} "
        "$rc = 0; foreach ($p in $procs) { Wait-Process -Id $p.Id; if ($p.ExitCode -ne 0) { $rc = 1 } } "
        "Get-ChildItem -Path $out -Filter '*.log' | Sort-Object Name | ForEach-Object { "
        "Write-Output ('__SCHEDULEURM_CPU_LOG_BEGIN__ ' + $_.Name); "
        "Get-Content -LiteralPath $_.FullName; "
        "Write-Output ('__SCHEDULEURM_CPU_LOG_END__ ' + $_.Name); "
        "}; "
        "exit $rc"
    )


def _run_windows_ps_capture(scheduler: Any, node: str, ps: str, prefix: Path, *, timeout_s: int) -> tuple[int, str, str]:
    try:
        rc, out, err = scheduler._run_windows_ps(node, ps, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out = (exc.stdout or b"").decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = (exc.stderr or b"").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        err = (err + "\n" if err else "") + f"ScheduleurmProbeTimeout: windows PowerShell command timed out after {timeout_s}s"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".stdout").write_text(out or "", encoding="utf-8")
    prefix.with_suffix(".stderr").write_text(err or "", encoding="utf-8")
    prefix.with_suffix(".meta.json").write_text(
        json.dumps({"cmd": ["scheduler._run_windows_ps", node, ps], "returncode": int(rc)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return int(rc), out or "", err or ""


def _windows_python(node: str) -> str:
    scheduler = _scheduler_module()
    cfg = getattr(scheduler, "NODES", {}).get(node, {}) or {}
    return str(cfg.get("windows_python") or "python")


def _ps_quote(text: Any) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def _remote_python_prefix() -> str:
    return (
        'PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"; '
        'if [ -z "$PY" ]; then echo "ScheduleurmProbeError: no python available" >&2; exit 127; fi; '
    )


def _light_concurrent_cmd(*, profile: int, label: str, config: Mapping[str, Any]) -> str:
    count = max(1, int(profile))
    steps = int(config.get("steps") or 160)
    work_items = int(config.get("work_items") or 8000)
    mode = shlex.quote(str(config.get("mode") or "light"))
    safe_label = shlex.quote(label)
    run_dir = f"/tmp/scheduleurm_cpu_light_{_safe_id(label)}"
    return (
        f"set -e; OUT={shlex.quote(run_dir)}; rm -rf \"$OUT\"; mkdir -p \"$OUT\"; "
        "PIDS=\"\"; "
        f"for i in $(seq 0 {count - 1}); do "
        f"{_remote_python_prefix()} \"$PY\" -u {REMOTE_EXP_DIR}/cpu_progress_compat_benchmark.py "
        f"--steps {steps} --mode {mode} --work-items {work_items} "
        f"--label {safe_label}_$i > \"$OUT/$i.log\" 2>&1 & "
        "PIDS=\"$PIDS $!\"; "
        "done; "
        "RC=0; for p in $PIDS; do wait $p || RC=1; done; "
        "for f in \"$OUT\"/*.log; do "
        "b=$(basename \"$f\"); echo \"__SCHEDULEURM_CPU_LOG_BEGIN__ $b\"; "
        "cat \"$f\"; echo \"__SCHEDULEURM_CPU_LOG_END__ $b\"; "
        "done; exit $RC"
    )


def _split_cpu_log_sections(text: str) -> list[str]:
    sections = []
    current: list[str] | None = None
    for line in (text or "").splitlines():
        if line.startswith("__SCHEDULEURM_CPU_LOG_BEGIN__"):
            current = []
            continue
        if line.startswith("__SCHEDULEURM_CPU_LOG_END__"):
            if current is not None:
                sections.append("\n".join(current))
            current = None
            continue
        if current is not None:
            current.append(line)
    return sections or [text or ""]


def _stable_tail(rates: list[float], *, min_samples: int = 8, tail_n: int = 5) -> dict[str, Any]:
    clean = [float(x) for x in rates if float(x) > 0.0 and math.isfinite(float(x))]
    if len(clean) < min_samples:
        return {"ready": False, "samples": len(clean), "mean_rate": 0.0, "cv": float("inf"), "last_two_relative_delta": float("inf")}
    tail = clean[-min(tail_n, len(clean)):]
    mean_rate = fmean(tail)
    variance = fmean((x - mean_rate) ** 2 for x in tail) if len(tail) > 1 else 0.0
    cv = math.sqrt(max(0.0, variance)) / max(mean_rate, 1e-9)
    rel_delta = abs(tail[-1] - tail[-2]) / max(abs(tail[-2]), 1e-9) if len(tail) >= 2 else 0.0
    return {
        "ready": cv <= 0.10 and rel_delta <= 0.08,
        "samples": len(clean),
        "mean_rate": mean_rate,
        "cv": cv,
        "last_two_relative_delta": rel_delta,
        "tail": tail,
    }


def _progress_rates(text: str, *, prefer_total: bool = False) -> list[float]:
    """Return task-native progress-window rates, excluding final average rates."""
    out: list[float] = []
    for line in (text or "").splitlines():
        if "DONE" in line:
            continue
        if "PROGRESS" not in line and not line.startswith("Step "):
            continue
        match = TOTAL_RATE_RE.search(line) if prefer_total else RATE_RE.search(line)
        if match is None:
            match = RATE_RE.search(line)
        if not match:
            continue
        value = float(match.group(1))
        if value > 0.0 and math.isfinite(value):
            out.append(value)
    return out


def _parse_profiles(text: Any) -> list[int]:
    raw = str(text or "").replace(";", ",")
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return sorted(dict.fromkeys(out))


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_")[:180]


def _windows_node(node: str) -> bool:
    try:
        scheduler = _scheduler_module()
        cfg = getattr(scheduler, "NODES", {}).get(node, {}) or {}
        return str(cfg.get("os") or "").lower().startswith("windows")
    except Exception:
        return False


def _summary_markdown(summary: Mapping[str, Any], summary_path: Path) -> str:
    return "\n".join([
        "# Full-Factorial CPU ETA Profile",
        "",
        f"Summary: `{summary_path}`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `phase` | `{summary.get('phase')}` |",
        f"| `allocation_workers` | {int(summary.get('allocation_workers') or 1)} |",
        f"| `colocation_count` | {int(summary.get('colocation_count') or 1)} |",
        f"| `steps` | {int(summary.get('steps') or 0)} |",
        f"| `measurement_valid` | `{str(bool(summary.get('measurement_valid'))).lower()}` |",
        f"| `steady_state_rate_ready` | `{str(bool(summary.get('steady_state_rate_ready'))).lower()}` |",
        f"| `natural_completion_observed` | `{str(bool(summary.get('natural_completion_observed'))).lower()}` |",
        f"| `aggregate_stable_rate_unit_s` | {float(summary.get('aggregate_stable_rate_unit_s') or 0.0):.6g} |",
        f"| `samples` | {len(summary.get('rates_unit_s') or [])} |",
        f"| `stable_tail_cv` | {float((summary.get('stable_tail') or {}).get('cv') or 0.0):.6g} |",
        "",
    ])


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full-Factorial CPU ETA Probe Runner",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allow launch: `{str(bool(report.get('allow_launch'))).lower()}`",
        f"- Selected rows: `{report.get('selected_count')}`",
        f"- Launched rows: `{report.get('launched_count')}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        "",
        "| Row | Node | Workload | Env | State | Profiles | Valid | Rates |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        rates = [
            (
                f"w{int(profile_row.get('allocation_workers') or 1)}"
                f"/c{int(profile_row.get('colocation_count') or profile_row.get('profile') or 1)}"
                f"={float(profile_row.get('aggregate_stable_rate_unit_s') or 0.0):.6g}"
            )
            for profile_row in row.get("profile_rows") or []
        ]
        lines.append(
            f"| `{row.get('row_id')}` | `{row.get('node')}` | `{row.get('workload_key')}` | "
            f"`{row.get('workload_env')}` | `{row.get('resource_state')}` | `{row.get('profiles')}` | "
            f"{str(bool(row.get('measurement_valid'))).lower()} | `{', '.join(rates)}` |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--design-csv", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--tier", default="T0_core_curve")
    parser.add_argument("--resource-state", default="empty")
    parser.add_argument("--hardware-groups", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--max-profiles", type=int, default=0)
    parser.add_argument("--run-prefix", default="full_factorial_cpu_eta_probe_20260630")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "full_factorial_cpu_eta_probe_runner_20260630.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_full_factorial_cpu_eta_probe_20260630.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "full_factorial_cpu_eta_probe_runner_20260630.md")
    args = parser.parse_args()
    report = build_full_factorial_cpu_eta_probe_runner(
        design_csv=args.design_csv,
        allow_launch=bool(args.allow_launch),
        tier=str(args.tier),
        resource_state=str(args.resource_state),
        hardware_groups={x.strip() for x in args.hardware_groups.split(",") if x.strip()} or None,
        workloads={x.strip() for x in args.workloads.split(",") if x.strip()} or None,
        nodes={x.strip() for x in args.nodes.split(",") if x.strip()} or None,
        max_rows=int(args.max_rows),
        max_profiles=int(args.max_profiles),
        run_prefix=str(args.run_prefix),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.cache_output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_output.write_text(json.dumps(report["service_cache_snapshot"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
