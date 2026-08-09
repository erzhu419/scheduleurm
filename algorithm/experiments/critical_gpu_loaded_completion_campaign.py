"""Natural-completion GPU trajectory probes under a controlled resident task.

The older marginal-under-load runner stopped the add-one task as soon as a
stable service window appeared.  That is useful for service discovery but is
not a completion-time experiment.  This campaign keeps both the resident and
the target alive until natural exit, records each task's native progress and
completion model, and admits only the finite mixed-workload actions used by the
state-dependent Scheduleurm candidate family.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shlex
import time
from typing import Any, Iterable, Mapping

from algorithm.experiments.critical_gpu_completion_campaign import (
    NODE_SPECS,
    TEMPLATE_ROOT,
    WORKLOAD_SPECS,
    _node_environment_gate,
)
from algorithm.experiments.progress_units import parse_completion_model
from algorithm.experiments.remote_workload_selected_profile_probe import (
    RUN_ROOT,
    _deploy_progress_wrapper,
    _fetch_remote_file_bytes,
    _last_stable_rate,
    _render_workload_command,
    _run_remote_capture,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
CAMPAIGN_DATE = "20260809"
PROTOCOL = "critical_gpu_loaded_natural_completion_v2"
CAMPAIGN_PREFIX = "critical_gpu_loaded_v2"
TRAINING_WAVES = (1, 2, 3)
CALIBRATION_WAVES = tuple(range(4, 13))
HOLDOUT_WAVE = 13
ALL_WAVES = TRAINING_WAVES + CALIBRATION_WAVES + (HOLDOUT_WAVE,)
MEASUREMENT_CODE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "algorithm" / "experiments" / "critical_gpu_completion_campaign.py",
    REPO_ROOT / "algorithm" / "experiments" / "remote_workload_selected_profile_probe.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_wrapper.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_units.py",
    REPO_ROOT / "algorithm" / "experiments" / "gpu_progress_benchmark.py",
    REPO_ROOT / "algorithm" / "experiments" / "torch_cnn_progress_benchmark.py",
    REPO_ROOT / "algorithm" / "experiments" / "torch_llm_progress_benchmark.py",
    *(TEMPLATE_ROOT / spec.template for spec in WORKLOAD_SPECS),
)


@dataclass(frozen=True)
class LoadedTrajectorySpec:
    scenario_id: str
    resident_workload: str
    target_workload: str
    resource_state: str
    resident_mix: str
    resident_total_units: int
    target_total_units: int
    gpu: int = 0
    readiness_observations: int = 3


SCENARIOS = (
    LoadedTrajectorySpec(
        "cnn_after_llm",
        "gpu_llm_distilgpt2",
        "gpu_cnn_torch_resnet50",
        "mixed_colocation",
        "resident_llm_target_cnn",
        12000,
        300,
    ),
    LoadedTrajectorySpec(
        "llm_after_cnn",
        "gpu_cnn_torch_resnet50",
        "gpu_llm_distilgpt2",
        "mixed_colocation",
        "resident_cnn_target_llm",
        600,
        600,
    ),
    LoadedTrajectorySpec(
        "rl_after_cnn",
        "gpu_cnn_torch_resnet50",
        "hybrid_rl_resac_ant",
        "mixed_colocation",
        "resident_cnn_target_hybrid_rl",
        18000,
        40,
    ),
    LoadedTrajectorySpec(
        "cnn_after_rl",
        "hybrid_rl_resac_ant",
        "gpu_cnn_torch_resnet50",
        "mixed_colocation",
        "resident_hybrid_rl_target_cnn",
        80,
        600,
    ),
)


def build_loaded_completion_campaign(
    *,
    node: str,
    wave: int,
    scenario_ids: Iterable[str] | None = None,
    allow_launch: bool = False,
    keep_remote_output: bool = False,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unknown node {node!r}")
    if int(wave) not in ALL_WAVES:
        raise ValueError(f"wave must be one of {ALL_WAVES!r}")
    selected_ids = {str(value) for value in scenario_ids or ()}
    unknown = selected_ids - {row.scenario_id for row in SCENARIOS}
    if unknown:
        raise ValueError(f"unknown scenarios: {sorted(unknown)!r}")
    selected = tuple(
        row for row in SCENARIOS
        if not selected_ids or row.scenario_id in selected_ids
    )
    if not selected:
        raise ValueError("no loaded trajectory scenarios selected")

    initial_manifest = measurement_code_manifest()
    selection_suffix = ""
    if selected_ids:
        selection_digest = hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode("utf-8")
        ).hexdigest()[:12]
        selection_suffix = f"_sel{selection_digest}"
    campaign_id = (
        f"{CAMPAIGN_PREFIX}_{node}_r{int(wave):02d}_{CAMPAIGN_DATE}"
        f"{selection_suffix}_code{initial_manifest['sha256'][:12]}"
    )
    output_path = ARTIFACT_ROOT / f"{campaign_id}.json"
    environment = _node_environment_gate(NODE_SPECS[node])
    common = {
        "gate": "critical_gpu_loaded_completion_campaign",
        "schema_version": 1,
        "protocol": PROTOCOL,
        "campaign_id": campaign_id,
        "node": node,
        "hardware_class": NODE_SPECS[node].hardware_class,
        "node_bucket": NODE_SPECS[node].node_bucket,
        "wave": int(wave),
        "split_role": _split_role(int(wave)),
        "allow_launch": bool(allow_launch),
        "environment_gate": environment,
        "legacy_scheduler_limits_bypassed": True,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "full_resident_target_overlap_required": True,
        "terminate_on_stable": False,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "pre_registered_split": {
            "training": list(TRAINING_WAVES),
            "calibration": list(CALIBRATION_WAVES),
            "holdout": HOLDOUT_WAVE,
        },
        "measurement_code_manifest": initial_manifest,
        "ordinary_running_tasks_touched": False,
        "scenario_parallelism": 1,
        "scenario_parallelism_scope": "sequential_independent_action_measurement",
        "cross_node_parallelism_allowed": True,
        "gpu_assignment": {
            scenario.scenario_id: gpu
            for scenario, gpu in _scenario_gpu_assignments(
                selected,
                NODE_SPECS[node].gpus,
                wave=int(wave),
            )
        },
        "scenario_ids": [row.scenario_id for row in selected],
        "selection": {
            "scenario_ids": sorted(selected_ids),
            "filtered": bool(selected_ids),
        },
    }
    if not allow_launch:
        report = {
            **common,
            "status": "MANIFEST_ONLY",
            "pass": False,
            "rows": [_manifest_row(node, wave, row) for row in selected],
        }
        _write_report(report, output_path)
        return report
    if not bool(environment.get("ready")):
        report = {
            **common,
            "status": "WAIT_ENVIRONMENT",
            "pass": False,
            "rows": [],
        }
        _write_report(report, output_path)
        return report

    deploy_dir = RUN_ROOT / campaign_id / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    _deploy_progress_wrapper(
        node,
        deploy_dir,
        cwd=str(REPO_ROOT),
        output_root=NODE_SPECS[node].output_root,
    )
    rows = []
    for scenario, gpu in _scenario_gpu_assignments(
        selected,
        NODE_SPECS[node].gpus,
        wave=int(wave),
    ):
        rows.append(
            _run_loaded_trajectory(
                node=node,
                wave=int(wave),
                campaign_id=campaign_id,
                scenario=scenario,
                gpu=gpu,
                cleanup_remote_output=not keep_remote_output,
            )
        )
    report = {
        **common,
        "rows": rows,
        "ready_row_count": sum(bool(row.get("ready")) for row in rows),
        "capacity_boundary_count": sum(
            bool(row.get("capacity_boundary")) for row in rows
        ),
    }
    report["pass"] = bool(rows) and all(
        bool(row.get("ready")) or bool(row.get("capacity_boundary"))
        for row in rows
    )
    final_manifest = measurement_code_manifest()
    report["final_measurement_code_manifest"] = final_manifest
    report["measurement_code_unchanged"] = bool(
        initial_manifest["sha256"] == final_manifest["sha256"]
    )
    if not report["measurement_code_unchanged"]:
        report["pass"] = False
    report["status"] = "PASS" if report["pass"] else "INCOMPLETE"
    _write_report(report, output_path)
    return report


def measurement_code_manifest() -> dict[str, Any]:
    aggregate = hashlib.sha256()
    rows = []
    for path in sorted(MEASUREMENT_CODE_PATHS, key=lambda item: str(item)):
        payload = path.read_bytes()
        relative = str(path.relative_to(REPO_ROOT))
        digest = hashlib.sha256(payload).hexdigest()
        rows.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(payload)
        aggregate.update(b"\0")
    return {"sha256": aggregate.hexdigest(), "files": rows}


def _scenario_gpu_assignments(
    scenarios: Iterable[LoadedTrajectorySpec],
    gpus: Iterable[int],
    *,
    wave: int,
) -> tuple[tuple[LoadedTrajectorySpec, int], ...]:
    """Rotate sequential scenarios across physical GPUs between waves."""

    gpu_ids = tuple(int(gpu) for gpu in gpus)
    if not gpu_ids:
        raise ValueError("loaded trajectory campaign requires at least one GPU")
    rotation = (int(wave) - 1) % len(gpu_ids)
    return tuple(
        (scenario, gpu_ids[(rotation + index) % len(gpu_ids)])
        for index, scenario in enumerate(tuple(scenarios))
    )


def _run_loaded_trajectory(
    *,
    node: str,
    wave: int,
    campaign_id: str,
    scenario: LoadedTrajectorySpec,
    gpu: int,
    cleanup_remote_output: bool,
) -> dict[str, Any]:
    node_spec = NODE_SPECS[node]
    resident_spec = _workload(scenario.resident_workload)
    target_spec = _workload(scenario.target_workload)
    run_id = f"{campaign_id}_{scenario.scenario_id}"
    run_dir = RUN_ROOT / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    remote_root = f"{node_spec.output_root.rstrip('/')}/{run_id}"
    remote_logs = f"/tmp/scheduleurm_loaded_completion/{run_id}"
    resident_iters = int(scenario.resident_total_units)
    resident = _render(
        node=node,
        wave=wave,
        scenario=scenario,
        role="resident",
        spec=resident_spec,
        max_iters=resident_iters,
        index=0,
        output_root=remote_root,
    )
    target = _render(
        node=node,
        wave=wave,
        scenario=scenario,
        role="target",
        spec=target_spec,
        max_iters=int(scenario.target_total_units),
        index=1,
        output_root=remote_root,
    )
    resident_timeout = max(
        int(resident_spec.timeout_s) * 2,
        int(target_spec.timeout_s) * 2,
    )
    target_timeout = int(target_spec.timeout_s)
    readiness_timeout = 900 if "hybrid_rl" in scenario.resident_workload else 300
    script = _remote_script(
        gpu=int(gpu),
        remote_logs=remote_logs,
        resident_cmd=str(resident["cmd"]),
        target_cmd=str(target["cmd"]),
        resident_timeout_s=resident_timeout,
        target_timeout_s=target_timeout,
        readiness_timeout_s=readiness_timeout,
        readiness_observations=int(scenario.readiness_observations),
    )
    started = time.time()
    rc, out, err = _run_remote_capture(
        node,
        script,
        raw_dir / "loaded_trajectory",
        timeout_s=resident_timeout + target_timeout + readiness_timeout + 300,
    )
    resident_log = _fetch_log(node, f"{remote_logs}/resident.log", raw_dir / "resident")
    target_log = _fetch_log(node, f"{remote_logs}/target.log", raw_dir / "target")
    diagnostics = _fetch_log(
        node,
        f"{remote_logs}/diagnostics.log",
        raw_dir / "diagnostics",
    )
    gpu_monitor = _fetch_log(
        node,
        f"{remote_logs}/gpu_monitor.log",
        raw_dir / "gpu_monitor",
    )
    resident_model = parse_completion_model(resident_log) or {}
    target_model = parse_completion_model(target_log) or {}
    resident_stable = _last_stable_rate(resident_log)
    target_stable = _last_stable_rate(target_log)
    markers = _markers(out)
    overlap_service = _resident_overlap_service(
        resident_log,
        start_line=markers.get("TARGET_START_LINE"),
        end_line=markers.get("TARGET_END_LINE"),
        start_epoch_s=(
            float(markers["TARGET_START_NS"]) / 1_000_000_000.0
            if "TARGET_START_NS" in markers
            else None
        ),
        end_epoch_s=(
            float(markers["TARGET_END_NS"]) / 1_000_000_000.0
            if "TARGET_END_NS" in markers
            else None
        ),
    )
    resident_rc = int(markers.get("RESIDENT_RC", rc))
    target_rc = int(markers.get("TARGET_RC", rc))
    alive_start = bool(int(markers.get("RESIDENT_ALIVE_AT_TARGET_START", 0)))
    alive_end = bool(int(markers.get("RESIDENT_ALIVE_AT_TARGET_END", 0)))
    ready_count = int(markers.get("RESIDENT_READY_OBSERVATIONS", 0))
    target_ready = bool(target_model.get("completion_model_ready"))
    resident_ready = bool(resident_model.get("completion_model_ready"))
    ready = bool(
        int(rc) == 0
        and resident_rc == 0
        and target_rc == 0
        and target_ready
        and resident_ready
        and alive_start
        and alive_end
        and ready_count >= int(scenario.readiness_observations)
        and bool(overlap_service.get("ready"))
    )
    capacity_boundary = bool(
        not ready
        and (
            _oom_like(resident_log)
            or _oom_like(target_log)
            or resident_rc in {134, 137, 139}
            or target_rc in {134, 137, 139}
        )
    )
    row = {
        "scenario_id": scenario.scenario_id,
        "node": node,
        "node_bucket": node_spec.node_bucket,
        "hardware_class": node_spec.hardware_class,
        "gpu": int(gpu),
        "wave": int(wave),
        "split_role": _split_role(wave),
        "resource_state": scenario.resource_state,
        "resident_mix": scenario.resident_mix,
        "resident_workload_key": resident_spec.workload_key,
        "resident_workload_env": resident_spec.workload_env,
        "target_workload_key": target_spec.workload_key,
        "target_workload_env": target_spec.workload_env,
        "post_colocation_count": 2,
        "resident_total_units": resident_iters,
        "target_total_units": int(scenario.target_total_units),
        "resident_completion_model": resident_model,
        "target_completion_model": target_model,
        "resident_stable_rate": float(
            resident_stable.get("last_rate")
            or resident_stable.get("mean_rate")
            or 0.0
        ),
        "target_stable_rate": float(
            target_stable.get("last_rate")
            or target_stable.get("mean_rate")
            or 0.0
        ),
        "resident_overlap_service": overlap_service,
        "resident_alive_at_target_start": alive_start,
        "resident_alive_at_target_end": alive_end,
        "resident_ready_observations": ready_count,
        "resident_returncode": resident_rc,
        "target_returncode": target_rc,
        "remote_returncode": int(rc),
        "resident_process_group_id": markers.get("RESIDENT_PGID"),
        "target_process_group_id": markers.get("TARGET_PGID"),
        "target_start_ns": markers.get("TARGET_START_NS"),
        "target_end_ns": markers.get("TARGET_END_NS"),
        "elapsed_wall_s": time.time() - started,
        "ready": ready,
        "capacity_boundary": capacity_boundary,
        "reason": "" if ready else _reason(
            rc=rc,
            resident_rc=resident_rc,
            target_rc=target_rc,
            resident_ready=resident_ready,
            target_ready=target_ready,
            alive_start=alive_start,
            alive_end=alive_end,
            ready_count=ready_count,
            required_count=scenario.readiness_observations,
            overlap_ready=bool(overlap_service.get("ready")),
            stderr=err,
        ),
        "eta_source": "task_native_tqdm_natural_completion_loaded_trajectory",
        "legacy_scheduler_limits_bypassed": True,
        "ordinary_running_tasks_touched": False,
        "resident_log_path": str(raw_dir / "resident.log"),
        "target_log_path": str(raw_dir / "target.log"),
        "diagnostics_log_path": str(raw_dir / "diagnostics.log"),
        "diagnostics_sha256": hashlib.sha256(diagnostics.encode("utf-8")).hexdigest(),
        "gpu_monitor_log_path": str(raw_dir / "gpu_monitor.log"),
        "gpu_monitor_sha256": hashlib.sha256(gpu_monitor.encode("utf-8")).hexdigest(),
    }
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "summary.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if cleanup_remote_output:
        cleanup = f"rm -rf {shlex.quote(remote_logs)} {shlex.quote(remote_root)}"
        _run_remote_capture(
            node,
            cleanup,
            raw_dir / "cleanup",
            timeout_s=120,
        )
    return row


def _render(
    *,
    node: str,
    wave: int,
    scenario: LoadedTrajectorySpec,
    role: str,
    spec: Any,
    max_iters: int,
    index: int,
    output_root: str,
) -> Mapping[str, Any]:
    node_spec = NODE_SPECS[node]
    values = dict(spec.extra_values)
    values.update(
        {
            "torch_python": node_spec.torch_python,
            "bapr_python": node_spec.bapr_python,
            "workload_pythonpath": node_spec.workload_pythonpath,
            "hf_cache_dir": node_spec.hf_cache_dir,
            "checkpoint_interval": str(max(1, int(max_iters) // 2)),
        }
    )
    if spec.workload_key == "gpu_heavy_jax_matmul":
        values["torch_python"] = node_spec.bapr_python
    template = (TEMPLATE_ROOT / spec.template).read_text(encoding="utf-8").strip()
    return _render_workload_command(
        template,
        run_id=f"critical_gpu_loaded_{node}_{scenario.scenario_id}_r{wave:02d}",
        phase=role,
        count_per_gpu=2,
        index=index,
        gpu_idx=int(scenario.gpu),
        seed_base=920_000 + int(wave) * 100,
        max_iters=int(max_iters),
        output_root=output_root,
        node=node,
        terminate_on_stable=False,
        stable_windows=int(spec.stable_windows),
        min_rate_samples=int(spec.min_rate_samples),
        stable_cv=0.10,
        stable_rel_delta=0.08,
        stable_skip_samples=int(spec.stable_skip_samples),
        stable_cycle_units=int(spec.stable_cycle_units),
        extra_template_values=values,
    )


def _remote_script(
    *,
    gpu: int,
    remote_logs: str,
    resident_cmd: str,
    target_cmd: str,
    resident_timeout_s: int,
    target_timeout_s: int,
    readiness_timeout_s: int,
    readiness_observations: int,
) -> str:
    quoted_logs = shlex.quote(remote_logs)
    return "\n".join(
        (
            "set +e",
            f"REMOTE_LOGS={quoted_logs}",
            'rm -rf "$REMOTE_LOGS"',
            'mkdir -p "$REMOTE_LOGS"',
            'DIAG="$REMOTE_LOGS/diagnostics.log"',
            'GPU_MONITOR="$REMOTE_LOGS/gpu_monitor.log"',
            f"GPU={int(gpu)}",
            'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits > "$DIAG" 2>&1 || true',
            ': > "$GPU_MONITOR"',
            '(',
            '  while true; do',
            '    printf "__GPU_SAMPLE_NS__ %s\\n" "$(date +%s%N)"',
            '    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true',
            (
                '    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory '
                '--format=csv,noheader,nounits 2>/dev/null | '
                'while IFS=, read -r PROC_PID PROC_UUID PROC_MEM; do '
                'PROC_PID=$(printf "%s" "$PROC_PID" | tr -d " "); '
                'PROC_UUID=$(printf "%s" "$PROC_UUID" | tr -d " "); '
                'PROC_MEM=$(printf "%s" "$PROC_MEM" | tr -d " "); '
                'PROC_PGID=$(ps -o pgid= -p "$PROC_PID" 2>/dev/null | tr -d " "); '
                '[ -n "$PROC_PGID" ] || PROC_PGID=-1; '
                'printf "__GPU_PROC__ %s %s %s %s\\n" '
                '"$PROC_PID" "$PROC_PGID" "$PROC_UUID" "$PROC_MEM"; done'
            ),
            '    sleep 1',
            '  done',
            ') >> "$GPU_MONITOR" 2>&1 &',
            'GPU_MONITOR_PID=$!',
            'trap \'kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true; wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true\' EXIT INT TERM',
            (
                f"CUDA_VISIBLE_DEVICES={int(gpu)} setsid timeout --foreground "
                f"{int(resident_timeout_s)}s {resident_cmd} "
                '> "$REMOTE_LOGS/resident.log" 2>&1 &'
            ),
            "RESIDENT_PID=$!",
            "READY_OBS=0",
            f"READY_DEADLINE=$((SECONDS + {int(readiness_timeout_s)}))",
            'while [ "$SECONDS" -lt "$READY_DEADLINE" ]; do',
            '  if ! kill -0 "$RESIDENT_PID" >/dev/null 2>&1; then break; fi',
            '  READY_OBS=$(grep -c "ScheduleurmProgress" "$REMOTE_LOGS/resident.log" 2>/dev/null || true)',
            f'  if [ "$READY_OBS" -ge {int(readiness_observations)} ]; then break; fi',
            "  sleep 0.25",
            "done",
            "RESIDENT_ALIVE_START=0",
            'if kill -0 "$RESIDENT_PID" >/dev/null 2>&1; then RESIDENT_ALIVE_START=1; fi',
            'TARGET_START_NS=$(date +%s%N)',
            'TARGET_START_LINE=$(wc -l < "$REMOTE_LOGS/resident.log")',
            (
                f"CUDA_VISIBLE_DEVICES={int(gpu)} setsid timeout --foreground "
                f"{int(target_timeout_s)}s {target_cmd} "
                '> "$REMOTE_LOGS/target.log" 2>&1 &'
            ),
            "TARGET_PID=$!",
            'wait "$TARGET_PID"',
            "TARGET_RC=$?",
            'TARGET_END_NS=$(date +%s%N)',
            'TARGET_END_LINE=$(wc -l < "$REMOTE_LOGS/resident.log")',
            "RESIDENT_ALIVE_END=0",
            'if kill -0 "$RESIDENT_PID" >/dev/null 2>&1; then RESIDENT_ALIVE_END=1; fi',
            'wait "$RESIDENT_PID"',
            "RESIDENT_RC=$?",
            'kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true',
            'wait "$GPU_MONITOR_PID" >/dev/null 2>&1 || true',
            'trap - EXIT INT TERM',
            'nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$DIAG" 2>&1 || true',
            'echo "__RESIDENT_RC__ $RESIDENT_RC"',
            'echo "__TARGET_RC__ $TARGET_RC"',
            'echo "__RESIDENT_PGID__ $RESIDENT_PID"',
            'echo "__TARGET_PGID__ $TARGET_PID"',
            'echo "__RESIDENT_ALIVE_AT_TARGET_START__ $RESIDENT_ALIVE_START"',
            'echo "__RESIDENT_ALIVE_AT_TARGET_END__ $RESIDENT_ALIVE_END"',
            'echo "__RESIDENT_READY_OBSERVATIONS__ $READY_OBS"',
            'echo "__TARGET_START_NS__ $TARGET_START_NS"',
            'echo "__TARGET_END_NS__ $TARGET_END_NS"',
            'echo "__TARGET_START_LINE__ $TARGET_START_LINE"',
            'echo "__TARGET_END_LINE__ $TARGET_END_LINE"',
            "exit 0",
        )
    )


def _fetch_log(node: str, remote_path: str, prefix: Path) -> str:
    try:
        payload = _fetch_remote_file_bytes(
            node,
            remote_path,
            prefix,
            timeout_s=180,
        )
    except RuntimeError as exc:
        text = f"ScheduleurmLoadedFetchError: {exc!r}\n"
    else:
        text = payload.decode("utf-8", errors="replace")
    destination = prefix.with_suffix(".log")
    destination.write_text(text, encoding="utf-8")
    return text


def _markers(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, value in re.findall(r"__(\w+)__\s+(-?\d+)", str(text or "")):
        out[str(name)] = int(value)
    return out


def _resident_overlap_service(
    text: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    start_epoch_s: float | None = None,
    end_epoch_s: float | None = None,
) -> dict[str, Any]:
    lines = str(text or "").splitlines()
    explicit_bounds = bool(
        start_line is not None
        and end_line is not None
        and start_epoch_s is not None
        and end_epoch_s is not None
        and 0 <= int(start_line) <= int(end_line) <= len(lines)
    )
    if explicit_bounds:
        start_index = int(start_line)
        end_index = int(end_line)
        start_epoch = float(start_epoch_s)
        end_epoch = float(end_epoch_s)
        segment = lines[start_index:end_index]
        boundary_rule = "remote_line_count_and_nanosecond_bounds"
    else:
        start_index = next(
            (
                index
                for index, line in enumerate(lines)
                if "ScheduleurmColocation event=target_start" in line
            ),
            None,
        )
        end_index = next(
            (
                index
                for index, line in enumerate(lines)
                if "ScheduleurmColocation event=target_end" in line
                and start_index is not None
                and index > start_index
            ),
            None,
        )
        start_epoch = (
            _colocation_epoch(lines[start_index])
            if start_index is not None
            else None
        )
        end_epoch = (
            _colocation_epoch(lines[end_index])
            if end_index is not None
            else None
        )
        segment = (
            lines[start_index + 1 : end_index]
            if start_index is not None and end_index is not None
            else []
        )
        boundary_rule = "legacy_inline_markers"
    observations = []
    unit = ""
    for line in segment:
        match = re.search(
            r"ScheduleurmProgress\s+\w+\s+(\d+)/(\d+).*?rate=([0-9.eE+-]+)\s+(\w+)/s",
            line,
        )
        if match is None:
            continue
        observations.append(
            {
                "current": int(match.group(1)),
                "total": int(match.group(2)),
                "reported_rate": float(match.group(3)),
            }
        )
        unit = str(match.group(4))
    duration = (
        float(end_epoch) - float(start_epoch)
        if start_epoch is not None and end_epoch is not None
        else 0.0
    )
    completed_units = (
        max(0, int(observations[-1]["current"]) - int(observations[0]["current"]))
        if len(observations) >= 2
        else 0
    )
    conservative_rate = (
        float(completed_units) / duration
        if duration > 0.0 and completed_units > 0
        else 0.0
    )
    return {
        "ready": bool(
            duration > 0.0
            and len(observations) >= 2
            and completed_units > 0
            and conservative_rate > 0.0
        ),
        "start_epoch_s": start_epoch,
        "end_epoch_s": end_epoch,
        "duration_s": duration,
        "progress_observation_count": len(observations),
        "first_progress_unit": observations[0]["current"] if observations else None,
        "last_progress_unit": observations[-1]["current"] if observations else None,
        "completed_units_lower": completed_units,
        "conservative_rate_units_per_s": conservative_rate,
        "reported_rate_median_units_per_s": (
            _median(row["reported_rate"] for row in observations)
            if observations
            else 0.0
        ),
        "unit": unit,
        "boundary_source": boundary_rule,
        "start_line_count": int(start_line) if start_line is not None else None,
        "end_line_count": int(end_line) if end_line is not None else None,
        "boundary_rule": (
            "first-to-last resident progress counter inside remote target "
            "start/end line-count bounds divided by the full nanosecond overlap "
            "duration"
        ),
    }


def _colocation_epoch(line: str) -> float | None:
    match = re.search(r"\bepoch=([0-9]+(?:\.[0-9]+)?)", str(line or ""))
    if match is None:
        return None
    value = float(match.group(1))
    return value if value > 0.0 else None


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _oom_like(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        token in lower
        for token in (
            "out of memory",
            "resource exhausted",
            "cuda_error_out_of_memory",
            "cublas_status_alloc_failed",
        )
    )


def _reason(**values: Any) -> str:
    pieces = []
    for key in (
        "rc",
        "resident_rc",
        "target_rc",
        "resident_ready",
        "target_ready",
        "alive_start",
        "alive_end",
        "ready_count",
        "required_count",
        "overlap_ready",
    ):
        pieces.append(f"{key}={values.get(key)!r}")
    stderr = str(values.get("stderr") or "").strip()
    if stderr:
        pieces.append(f"stderr={stderr[-500:]!r}")
    return ";".join(pieces)


def _manifest_row(node: str, wave: int, scenario: LoadedTrajectorySpec) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "node": node,
        "wave": int(wave),
        "split_role": _split_role(wave),
        "resident_workload_key": scenario.resident_workload,
        "target_workload_key": scenario.target_workload,
        "resource_state": scenario.resource_state,
        "resident_mix": scenario.resident_mix,
        "resident_total_units": int(scenario.resident_total_units),
        "target_total_units": int(scenario.target_total_units),
        "status": "planned",
        "ready": False,
        "capacity_boundary": False,
    }


def _workload(workload_key: str) -> Any:
    rows = [row for row in WORKLOAD_SPECS if row.workload_key == workload_key]
    if len(rows) != 1:
        raise KeyError(f"expected one workload spec for {workload_key!r}")
    return rows[0]


def _split_role(wave: int) -> str:
    if int(wave) in TRAINING_WAVES:
        return "training"
    if int(wave) in CALIBRATION_WAVES:
        return "calibration"
    if int(wave) == HOLDOUT_WAVE:
        return "holdout"
    raise ValueError(f"unsupported wave {wave}")


def _write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Critical GPU Loaded Natural-Completion Campaign",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Node: `{report.get('node')}`",
        f"- Wave / split: `{report.get('wave')}` / `{report.get('split_role')}`",
        "",
        "| Scenario | Resident | Target | Ready | Boundary |",
        "|---|---|---|:---:|:---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{}` | `{}` | `{}` | {} | {} |".format(
                row.get("scenario_id"),
                row.get("resident_workload_key"),
                row.get("target_workload_key"),
                str(bool(row.get("ready"))).lower(),
                str(bool(row.get("capacity_boundary"))).lower(),
            )
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, choices=tuple(NODE_SPECS))
    parser.add_argument("--wave", required=True, type=int)
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--keep-remote-output", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_loaded_completion_campaign(
        node=args.node,
        wave=args.wave,
        scenario_ids=args.scenario,
        allow_launch=args.allow_launch,
        keep_remote_output=args.keep_remote_output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass": report["pass"],
                "ready_row_count": report.get("ready_row_count", 0),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["pass"] or not args.allow_launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
