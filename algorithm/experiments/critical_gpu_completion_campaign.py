"""Pre-registered natural-completion campaign for policy-reachable GPU actions.

This runner is deliberately outside ``scheduler.py``.  It launches only
controlled benchmark processes on explicitly selected idle GPUs, bypasses all
legacy packing limits, and admits a row only when task-native progress,
natural exit, stable service, backend identity, and real checkpoint allocation
are all observed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .remote_workload_selected_profile_probe import (
    RUN_ROOT,
    _run_remote_capture,
    build_remote_workload_selected_profile_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
TEMPLATE_ROOT = REPO_ROOT / "algorithm" / "experiments" / "templates"
PROTOCOL = "critical_gpu_phase_completion_v8"
CAMPAIGN_PREFIX = "critical_gpu_completion_v8"
STATIONARY_RATE_EVIDENCE = "stationary_rate_and_completion"
STAGED_TRAJECTORY_EVIDENCE = "staged_trajectory_completion"
TRAINING_WAVES = (1, 2, 3)
CALIBRATION_WAVES = tuple(range(4, 13))
HOLDOUT_WAVE = 13
GPU_IDLE_UTIL_LIMIT_PCT = 5
GPU_IDLE_MEMORY_LIMIT_MB = 512

@dataclass(frozen=True)
class NodeSpec:
    node: str
    node_bucket: str
    hardware_class: str
    gpus: tuple[int, ...]
    torch_python: str
    bapr_python: str
    workload_pythonpath: str
    hf_cache_dir: str
    output_root: str
    role: str = "representative"


@dataclass(frozen=True)
class WorkloadSpec:
    workload_key: str
    workload_env: str
    quadrant: str
    template: str
    profiles: tuple[int, ...]
    max_iters: int
    timeout_s: int
    unit: str
    stable_windows: int
    min_rate_samples: int
    stable_skip_samples: int
    stable_cycle_units: int = 0
    coordinated_start_barrier: bool = False
    admission_stagger_s: float = 0.0
    admission_stagger_axis: str = "local_index"
    admission_stagger_min_profile: int = 1
    extra_values: tuple[tuple[str, str], ...] = ()


NODE_SPECS: dict[str, NodeSpec] = {
    "jtl110gpu": NodeSpec(
        node="jtl110gpu",
        node_bucket="gpu_3080ti_12gb_dual",
        hardware_class="NVIDIA_RTX_3080Ti_12GB",
        gpus=(0, 1),
        torch_python="/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
        bapr_python="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python",
        workload_pythonpath=(
            "/home/erzhu419/.claude/scheduler/runtime_patches:"
            "/home/erzhu419/mine_code/RE-SAC"
        ),
        hf_cache_dir="/home/erzhu419/.cache/huggingface/hub",
        output_root="/tmp/scheduleurm_critical_gpu_completion",
    ),
    "jtl110gpu2": NodeSpec(
        node="jtl110gpu2",
        node_bucket="gpu_3080ti_12gb_dual",
        hardware_class="NVIDIA_RTX_3080Ti_12GB",
        gpus=(0, 1),
        torch_python="/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
        bapr_python="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python",
        workload_pythonpath=(
            "/home/erzhu419/scheduleurm_work/python_stdlib_patch:"
            "/home/erzhu419/.claude/scheduler/runtime_patches:"
            "/home/erzhu419/mine_code/RE-SAC"
        ),
        hf_cache_dir="/home/erzhu419/.cache/huggingface/hub",
        output_root="/tmp/scheduleurm_critical_gpu_completion",
        role="homogeneous_equivalence",
    ),
    "jtl311linux": NodeSpec(
        node="jtl311linux",
        node_bucket="gpu_2080_8gb_dual",
        hardware_class="NVIDIA_RTX_2080_8GB",
        gpus=(0, 1),
        torch_python="/home/erzhu419/.venvs/scheduleurm-torch-bench/bin/python",
        bapr_python="/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python",
        workload_pythonpath=(
            "/home/erzhu419/.claude/scheduler/runtime_patches:"
            "/home/erzhu419/mine_code/RE-SAC"
        ),
        hf_cache_dir="/home/erzhu419/.cache/huggingface/hub",
        output_root="/tmp/scheduleurm_critical_gpu_completion",
    ),
    "node007": NodeSpec(
        node="node007",
        node_bucket="gpu_2080ti_11gb_quad",
        hardware_class="NVIDIA_RTX_2080Ti_11GB",
        gpus=(0, 1, 2, 3),
        torch_python=(
            "/home/zhengliang01/scheduleurm_work/venvs/"
            "scheduleurm-gpu-eta-py310/bin/python"
        ),
        bapr_python=(
            "/home/zhengliang01/scheduleurm_work/conda_envs/"
            "csbapr-gpu-py310/bin/python"
        ),
        workload_pythonpath=(
            "/home/zhengliang01/scheduleurm_work/runtime_patches:"
            "/home/zhengliang01/scheduleurm_work/RE-SAC"
        ),
        hf_cache_dir=(
            "/home/zhengliang01/scheduleurm_work/model_cache/huggingface"
        ),
        output_root="/tmp/scheduleurm_critical_gpu_completion",
    ),
}


WORKLOAD_SPECS: tuple[WorkloadSpec, ...] = (
    WorkloadSpec(
        workload_key="gpu_heavy_jax_matmul",
        workload_env="jax_matmul_8192",
        quadrant="q01",
        template="gpu_matmul_progress.cmd.tpl",
        profiles=(1, 3),
        max_iters=40,
        timeout_s=1800,
        unit="step",
        stable_windows=3,
        min_rate_samples=6,
        stable_skip_samples=4,
        coordinated_start_barrier=True,
        admission_stagger_s=0.75,
        extra_values=(("matrix_size", "8192"), ("checkpoint_bytes", str(8 * 1024 * 1024))),
    ),
    WorkloadSpec(
        workload_key="gpu_cnn_torch_resnet50",
        workload_env="resnet50_synthetic_train_amp_b32_224",
        quadrant="q01",
        template="torch_cnn_resnet50.cmd.tpl",
        profiles=(1, 3),
        max_iters=60,
        timeout_s=3600,
        unit="step",
        stable_windows=3,
        min_rate_samples=6,
        stable_skip_samples=2,
        coordinated_start_barrier=True,
        admission_stagger_s=0.75,
    ),
    WorkloadSpec(
        workload_key="gpu_llm_distilgpt2",
        workload_env="distilgpt2_forward_fp16_b2_s128",
        quadrant="q01",
        template="torch_llm_distilgpt2.cmd.tpl",
        profiles=(1, 3, 10),
        max_iters=120,
        timeout_s=5400,
        unit="step",
        stable_windows=3,
        min_rate_samples=12,
        stable_skip_samples=2,
        coordinated_start_barrier=True,
        admission_stagger_s=0.75,
    ),
    WorkloadSpec(
        workload_key="hybrid_rl_resac_ant",
        workload_env="Ant-v2",
        quadrant="q11",
        template="resac_env_real_venv.cmd.tpl",
        profiles=(2, 5),
        # Four 5-iteration service cycles were insufficient for every child to
        # satisfy the unchanged cycle-stability gate at p5.  Eight complete
        # cycles preserve natural completion while giving the periodic RL
        # service process enough post-initialization support.
        max_iters=40,
        timeout_s=10800,
        unit="iter",
        stable_windows=2,
        min_rate_samples=10,
        stable_skip_samples=0,
        stable_cycle_units=5,
        admission_stagger_s=30.0,
        admission_stagger_axis="global_index",
        admission_stagger_min_profile=5,
        extra_values=(("mujoco_env", "Ant-v2"),),
    ),
)


NODE_WORKLOAD_PROFILE_OVERRIDES: dict[tuple[str, str], tuple[int, ...]] = {
    ("jtl110gpu", "gpu_llm_distilgpt2"): (1, 3, 8),
    ("jtl110gpu2", "gpu_llm_distilgpt2"): (1, 3, 8),
    ("jtl311linux", "gpu_llm_distilgpt2"): (1, 3, 5),
}


MEASUREMENT_CODE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "algorithm" / "experiments" / "critical_gpu_completion_series.py",
    REPO_ROOT / "algorithm" / "experiments" / "remote_workload_selected_profile_probe.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_wrapper.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_units.py",
    REPO_ROOT / "algorithm" / "experiments" / "gpu_progress_benchmark.py",
    REPO_ROOT / "algorithm" / "experiments" / "torch_cnn_progress_benchmark.py",
    REPO_ROOT / "algorithm" / "experiments" / "torch_llm_progress_benchmark.py",
    *(TEMPLATE_ROOT / spec.template for spec in WORKLOAD_SPECS),
)


def wave_role(wave: int) -> str:
    value = int(wave)
    if value == 0:
        return "smoke_excluded"
    if value in TRAINING_WAVES:
        return "training"
    if value in CALIBRATION_WAVES:
        return "calibration"
    if value == HOLDOUT_WAVE:
        return "holdout"
    raise ValueError("wave must be 0 (smoke) or 1..13 (pre-registered campaign)")


def measurement_code_manifest() -> dict[str, Any]:
    files: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(MEASUREMENT_CODE_PATHS, key=lambda item: str(item)):
        data = path.read_bytes()
        relative = str(path.relative_to(REPO_ROOT))
        digest = hashlib.sha256(data).hexdigest()
        files.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(data)
        aggregate.update(b"\0")
    return {
        "algorithm": "sha256(path\\0content\\0)",
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def workload_profiles(node: str, spec: WorkloadSpec) -> tuple[int, ...]:
    return NODE_WORKLOAD_PROFILE_OVERRIDES.get(
        (str(node), spec.workload_key),
        spec.profiles,
    )


def service_evidence_mode(spec: WorkloadSpec, profile: int) -> str:
    """Return the pre-registered evidence model for one candidate action."""

    if (
        not spec.coordinated_start_barrier
        and float(spec.admission_stagger_s) > 0.0
        and int(profile) >= int(spec.admission_stagger_min_profile)
    ):
        return STAGED_TRAJECTORY_EVIDENCE
    return STATIONARY_RATE_EVIDENCE


def campaign_cells(
    *,
    node: str,
    wave: int,
    workload_keys: Iterable[str] | None = None,
    profiles: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    node_spec = NODE_SPECS[node]
    selected_workloads = {str(x) for x in workload_keys or ()}
    selected_profiles = {int(x) for x in profiles or ()}
    cells: list[dict[str, Any]] = []
    for spec in WORKLOAD_SPECS:
        if selected_workloads and spec.workload_key not in selected_workloads:
            continue
        for profile in workload_profiles(node, spec):
            if selected_profiles and int(profile) not in selected_profiles:
                continue
            cells.append(
                {
                    "cell_id": _safe_id(
                        f"{node_spec.node}_{spec.workload_key}_p{profile}_r{int(wave):02d}"
                    ),
                    "node": node_spec.node,
                    "node_bucket": node_spec.node_bucket,
                    "hardware_class": node_spec.hardware_class,
                    "node_role": node_spec.role,
                    "gpus": list(node_spec.gpus),
                    "workload_key": spec.workload_key,
                    "workload_env": spec.workload_env,
                    "quadrant": spec.quadrant,
                    "profile": int(profile),
                    "profile_axis": "tasks_per_gpu",
                    "total_task_count": int(profile) * len(node_spec.gpus),
                    "wave": int(wave),
                    "split_role": wave_role(wave),
                    "max_iters": int(spec.max_iters),
                    "timeout_s": int(spec.timeout_s),
                    "service_evidence_mode": service_evidence_mode(spec, profile),
                }
            )
    return cells


def build_critical_gpu_completion_campaign(
    *,
    node: str,
    wave: int,
    allow_launch: bool = False,
    workload_keys: Iterable[str] | None = None,
    profiles: Iterable[int] | None = None,
    cleanup_remote_output: bool = True,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    selected_workloads = tuple(sorted({str(x) for x in workload_keys or ()}))
    selected_profiles = tuple(sorted({int(x) for x in profiles or ()}))
    node_spec = NODE_SPECS[node]
    role = wave_role(wave)
    cells = campaign_cells(
        node=node,
        wave=wave,
        workload_keys=selected_workloads,
        profiles=selected_profiles,
    )
    selection = ""
    if selected_workloads or selected_profiles:
        workload_slug = "-".join(selected_workloads) or "allworkloads"
        profile_slug = "-".join(f"p{value}" for value in selected_profiles) or "allprofiles"
        selection = f"_{workload_slug}_{profile_slug}"
    campaign_id = _safe_id(
        f"{CAMPAIGN_PREFIX}_{node}_r{int(wave):02d}{selection}_20260803"
    )
    output_path = ARTIFACT_ROOT / f"{campaign_id}.json"
    code_manifest = measurement_code_manifest()
    report: dict[str, Any] = {
        "gate": "critical_gpu_completion_campaign",
        "measurement_protocol": PROTOCOL,
        "campaign_id": campaign_id,
        "node": node,
        "node_spec": asdict(node_spec),
        "wave": int(wave),
        "split_role": role,
        "pre_registered_split": {
            "training": list(TRAINING_WAVES),
            "calibration": list(CALIBRATION_WAVES),
            "holdout": HOLDOUT_WAVE,
            "smoke_excluded": 0,
        },
        "legacy_scheduler_limits_bypassed": True,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "terminate_on_stable": False,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "coordinated_post_warmup_start_required_for_builtin_gpu_workloads": True,
        "admission_delay_included_in_completion_jct": True,
        "staged_trajectory_completion_evidence_required": True,
        "real_checkpoint_allocation_required": True,
        "measurement_code_manifest": code_manifest,
        "allow_launch": bool(allow_launch),
        "selection": {
            "workload_keys": list(selected_workloads),
            "profiles": list(selected_profiles),
            "filtered": bool(selected_workloads or selected_profiles),
        },
        "planned_cells": cells,
        "hardware_local_profile_registry": {
            spec.workload_key: list(workload_profiles(node, spec))
            for spec in WORKLOAD_SPECS
        },
        "rows": [],
        "status": "MANIFEST_ONLY" if not allow_launch else "RUNNING",
        "pass": False,
    }
    if not allow_launch:
        _write_report(report, output_path)
        return report

    env_gate = _node_environment_gate(node_spec)
    report["environment_gate"] = env_gate
    if not env_gate.get("ready"):
        report["status"] = "WAIT_ENVIRONMENT"
        report["blocker"] = env_gate.get("reason")
        _write_report(report, output_path)
        return report

    spec_by_key = {spec.workload_key: spec for spec in WORKLOAD_SPECS}
    capacity_blocked: set[str] = set()
    rows: list[dict[str, Any]] = []
    for cell in cells:
        workload_key = str(cell["workload_key"])
        if workload_key in capacity_blocked:
            rows.append({**cell, "status": "SKIPPED_AFTER_CAPACITY_BOUNDARY", "ready": False})
            continue
        preflight = _gpu_idle_snapshot(node_spec)
        if not preflight.get("ready"):
            rows.append(
                {
                    **cell,
                    "status": "WAIT_RESOURCE",
                    "ready": False,
                    "preflight": preflight,
                }
            )
            continue
        spec = spec_by_key[workload_key]
        row = _run_cell(
            campaign_id=campaign_id,
            node_spec=node_spec,
            spec=spec,
            cell=cell,
            preflight=preflight,
            cleanup_remote_output=cleanup_remote_output,
            expected_code_sha256=str(code_manifest["sha256"]),
        )
        rows.append(row)
        if row.get("capacity_boundary"):
            capacity_blocked.add(workload_key)
        report["rows"] = rows
        report["status"] = "RUNNING"
        _write_report(report, output_path)

    ready_rows = [row for row in rows if bool(row.get("ready"))]
    terminal_rows = [
        row for row in rows
        if row.get("status") not in {"WAIT_RESOURCE", "WAIT_ENVIRONMENT"}
    ]
    report["rows"] = rows
    report["ready_row_count"] = len(ready_rows)
    report["capacity_boundary_count"] = sum(
        bool(row.get("capacity_boundary")) for row in rows
    )
    report["wait_row_count"] = sum(
        str(row.get("status") or "").startswith("WAIT") for row in rows
    )
    final_code_manifest = measurement_code_manifest()
    report["final_measurement_code_manifest"] = final_code_manifest
    report["measurement_code_unchanged"] = bool(
        final_code_manifest["sha256"] == code_manifest["sha256"]
    )
    report["pass"] = bool(rows) and len(terminal_rows) == len(rows) and all(
        bool(row.get("ready")) or bool(row.get("capacity_boundary"))
        for row in rows
    ) and bool(report["measurement_code_unchanged"])
    report["status"] = "PASS" if report["pass"] else "INCOMPLETE"
    _write_report(report, output_path)
    return report


def _run_cell(
    *,
    campaign_id: str,
    node_spec: NodeSpec,
    spec: WorkloadSpec,
    cell: Mapping[str, Any],
    preflight: Mapping[str, Any],
    cleanup_remote_output: bool,
    expected_code_sha256: str,
    measurement_manifest_builder: Callable[[], dict[str, Any]] = measurement_code_manifest,
    measurement_protocol: str = PROTOCOL,
) -> dict[str, Any]:
    profile = int(cell["profile"])
    evidence_mode = str(cell["service_evidence_mode"])
    require_stationary_rate = evidence_mode == STATIONARY_RATE_EVIDENCE
    run_id = _safe_id(
        f"{campaign_id}_{spec.workload_key}_p{profile}_code{expected_code_sha256[:12]}"
    )
    remote_output = f"{node_spec.output_root.rstrip('/')}/{run_id}"
    values = dict(spec.extra_values)
    values.update(
        {
            "torch_python": node_spec.torch_python,
            "bapr_python": node_spec.bapr_python,
            "workload_pythonpath": node_spec.workload_pythonpath,
            "hf_cache_dir": node_spec.hf_cache_dir,
            "checkpoint_interval": str(max(1, int(spec.max_iters) // 2)),
            "coordinated_start_barrier": (
                "1" if spec.coordinated_start_barrier else "0"
            ),
            "admission_stagger_s": f"{spec.admission_stagger_s:.9g}",
            "admission_stagger_axis": spec.admission_stagger_axis,
            "admission_stagger_min_profile": str(spec.admission_stagger_min_profile),
        }
    )
    if spec.workload_key == "gpu_heavy_jax_matmul":
        values["torch_python"] = node_spec.bapr_python
    started = time.time()
    try:
        code_before = measurement_manifest_builder()
        if code_before["sha256"] != expected_code_sha256:
            raise RuntimeError("measurement code changed before cell launch")
        probe = build_remote_workload_selected_profile_probe(
            run_id=run_id,
            node=node_spec.node,
            gpus=list(node_spec.gpus),
            profiles=[profile],
            cwd=str(REPO_ROOT),
            cmd_template_file=str(TEMPLATE_ROOT / spec.template),
            output_root=remote_output,
            seed_base=830_000 + int(cell["wave"]) * 10_000,
            max_iters=spec.max_iters,
            timeout_s=spec.timeout_s,
            unit=spec.unit,
            terminate_on_stable=False,
            stable_windows=spec.stable_windows,
            min_rate_samples=spec.min_rate_samples,
            stable_cv=0.10,
            stable_rel_delta=0.08,
            stable_skip_samples=spec.stable_skip_samples,
            stable_cycle_units=spec.stable_cycle_units,
            coordinated_profile_launch=True,
            require_stable_rate=require_stationary_rate,
            require_completion_model=True,
            extra_template_values=values,
            deploy_bundle=True,
        )
        summary_path = RUN_ROOT / run_id / "reports" / f"profile_{profile}_per_gpu_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        artifact_audit = _remote_artifact_audit(node_spec.node, remote_output, run_id)
        backend_audit = _backend_audit(spec, summary)
        completion_audit = _completion_audit(spec, summary)
        expected_tasks = profile * len(node_spec.gpus)
        coordination_audit = _coordination_audit(
            spec,
            summary,
            expected_tasks=expected_tasks,
        )
        admission_audit = _admission_audit(spec, summary, profile=profile)
        trajectory_audit = _trajectory_progress_audit(
            spec,
            summary,
            profile=profile,
            evidence_mode=evidence_mode,
        )
        code_after = measurement_manifest_builder()
        code_identity_ready = bool(
            code_after["sha256"] == expected_code_sha256
            and code_before["sha256"] == expected_code_sha256
        )
        checkpoint_ready = bool(
            artifact_audit.get("ready")
            and int(artifact_audit.get("allocated_file_count") or 0) >= expected_tasks
        )
        capacity_boundary = _capacity_boundary(summary)
        service_evidence_ready = bool(
            summary.get("all_stable_rate_ready")
            if require_stationary_rate
            else trajectory_audit.get("ready")
        )
        ready = bool(
            probe.get("pass")
            and summary.get("measurement_valid")
            and service_evidence_ready
            and summary.get("all_completion_models_ready")
            and backend_audit.get("ready")
            and completion_audit.get("ready")
            and coordination_audit.get("ready")
            and admission_audit.get("ready")
            and checkpoint_ready
            and code_identity_ready
            and not capacity_boundary
        )
        row = {
            **dict(cell),
            "measurement_protocol": measurement_protocol,
            "run_id": run_id,
            "remote_output_root": remote_output,
            "summary_path": str(summary_path),
            "elapsed_wall_s": time.time() - started,
            "preflight": dict(preflight),
            "probe_result": probe,
            "summary": summary,
            "artifact_audit": artifact_audit,
            "backend_audit": backend_audit,
            "completion_audit": completion_audit,
            "coordination_audit": coordination_audit,
            "admission_audit": admission_audit,
            "trajectory_progress_audit": trajectory_audit,
            "service_evidence_mode": evidence_mode,
            "service_evidence_ready": service_evidence_ready,
            "measurement_code_sha256": expected_code_sha256,
            "measurement_code_identity_ready": code_identity_ready,
            "checkpoint_allocation_ready": checkpoint_ready,
            "capacity_boundary": capacity_boundary,
            "status": "READY" if ready else ("CAPACITY_BOUNDARY" if capacity_boundary else "FAILED"),
            "ready": ready,
        }
    except Exception as exc:
        row = {
            **dict(cell),
            "measurement_protocol": measurement_protocol,
            "run_id": run_id,
            "remote_output_root": remote_output,
            "elapsed_wall_s": time.time() - started,
            "preflight": dict(preflight),
            "status": "FAILED_EXCEPTION",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            "capacity_boundary": False,
            "measurement_code_sha256": expected_code_sha256,
            "ready": False,
        }
    finally:
        if cleanup_remote_output:
            cleanup = _remote_cleanup(node_spec.node, remote_output, run_id)
            if "row" in locals():
                row["remote_cleanup"] = cleanup
    return row


def _node_environment_gate(node_spec: NodeSpec) -> dict[str, Any]:
    shell = (
        "set -e; "
        f"test -x {shlex.quote(node_spec.torch_python)}; "
        f"test -x {shlex.quote(node_spec.bapr_python)}; "
        f"test -d {shlex.quote(node_spec.hf_cache_dir)}; "
        f"TORCH_SITE=$({shlex.quote(node_spec.torch_python)} -c "
        "'import site; print(site.getsitepackages()[0])'); "
        "for d in \"$TORCH_SITE\"/nvidia/*/lib; do "
        "[ -d \"$d\" ] && export LD_LIBRARY_PATH=\"$d:${LD_LIBRARY_PATH:-}\"; done; "
        f"export HF_HOME={shlex.quote(node_spec.hf_cache_dir)}; "
        f"PYTHONPATH={shlex.quote(node_spec.workload_pythonpath)} "
        f"{shlex.quote(node_spec.torch_python)} - <<'PY'\n"
        "import torch, transformers\n"
        "assert torch.cuda.is_available()\n"
        "from transformers import AutoConfig\n"
        f"AutoConfig.from_pretrained('distilgpt2', cache_dir={node_spec.hf_cache_dir!r}, local_files_only=True)\n"
        "print('ENV_READY')\n"
        "PY\n"
        f"PYTHONPATH={shlex.quote(node_spec.workload_pythonpath)} "
        f"{shlex.quote(node_spec.bapr_python)} - <<'PY'\n"
        "import jax\n"
        "assert any(getattr(device, 'platform', '') == 'gpu' for device in jax.devices())\n"
        "import jax_experiments.train\n"
        "print('RESAC_READY')\n"
        "PY"
    )
    prefix = ARTIFACT_ROOT / f"critical_gpu_env_gate_{node_spec.node}_20260803"
    rc, out, err = _run_remote_capture(node_spec.node, shell, prefix, timeout_s=600)
    ready = bool(rc == 0 and "ENV_READY" in out and "RESAC_READY" in out)
    return {
        "ready": ready,
        "returncode": int(rc),
        "stdout_tail": out[-2000:],
        "stderr_tail": err[-2000:],
        "reason": "" if ready else "isolated Torch/DistilGPT2 or RE-SAC CUDA environment is not ready",
    }


def _gpu_idle_snapshot(node_spec: NodeSpec) -> dict[str, Any]:
    shell = (
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits; echo __PROCESSES__; "
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory "
        "--format=csv,noheader,nounits 2>/dev/null || true"
    )
    prefix = ARTIFACT_ROOT / f"critical_gpu_idle_{node_spec.node}_{int(time.time())}"
    rc, out, err = _run_remote_capture(node_spec.node, shell, prefix, timeout_s=90)
    gpu_text, _, process_text = out.partition("__PROCESSES__")
    gpu_rows: list[dict[str, Any]] = []
    for line in gpu_text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        gpu_rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_mb": _as_int(parts[2]),
                "used_mb": _as_int(parts[3]),
                "util_pct": _as_int(parts[4]),
            }
        )
    selected = [row for row in gpu_rows if row["index"] in node_spec.gpus]
    process_rows = [line.strip() for line in process_text.splitlines() if line.strip()]
    ready = bool(
        rc == 0
        and len(selected) == len(node_spec.gpus)
        and not process_rows
        and all(row["util_pct"] <= GPU_IDLE_UTIL_LIMIT_PCT for row in selected)
        and all(row["used_mb"] <= GPU_IDLE_MEMORY_LIMIT_MB for row in selected)
    )
    return {
        "ready": ready,
        "returncode": int(rc),
        "selected_gpus": selected,
        "compute_process_rows": process_rows,
        "idle_util_limit_pct": GPU_IDLE_UTIL_LIMIT_PCT,
        "idle_memory_limit_mb": GPU_IDLE_MEMORY_LIMIT_MB,
        "stderr_tail": err[-1000:],
    }


def _backend_audit(spec: WorkloadSpec, summary: Mapping[str, Any]) -> dict[str, Any]:
    markers: list[str] = []
    for child in summary.get("rows") or []:
        log_path = Path(str(child.get("log_path") or ""))
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        for line in text.splitlines():
            if "BENCH_START" in line or "Starting training" in line or "JAX" in line:
                markers.append(line[:500])
        if "GPU_BACKEND_FALLBACK" in text or "BENCH_FALLBACK" in text:
            markers.append("FORBIDDEN_FALLBACK")
    expected = int(summary.get("running_count") or 0)
    if spec.workload_key == "gpu_heavy_jax_matmul":
        accepted = sum("BENCH_START" in line and ("backend=jax" in line or "backend=torch" in line) for line in markers)
    elif spec.workload_key == "gpu_cnn_torch_resnet50":
        accepted = sum("BENCH_START" in line and "backend=torch" in line for line in markers)
    elif spec.workload_key == "gpu_llm_distilgpt2":
        accepted = sum("BENCH_START" in line and "backend=torch_transformers" in line for line in markers)
    else:
        accepted = sum(
            int(child.get("returncode") or 0) == 0
            for child in summary.get("rows") or []
        )
    return {
        "ready": bool(expected > 0 and accepted == expected and "FORBIDDEN_FALLBACK" not in markers),
        "expected_task_count": expected,
        "accepted_task_count": accepted,
        "markers": markers[: max(20, expected * 2)],
    }


def _completion_audit(spec: WorkloadSpec, summary: Mapping[str, Any]) -> dict[str, Any]:
    models = [
        child.get("completion_model") or {}
        for child in summary.get("rows") or []
    ]
    natural = all(bool(model.get("natural_exit")) for model in models) if models else False
    ready = all(bool(model.get("completion_model_ready")) for model in models) if models else False
    phase_save_ready = True
    if spec.workload_key != "hybrid_rl_resac_ant":
        phase_save_ready = all(
            float(model.get("checkpoint_observed_s") or 0.0) > 0.0
            and float(model.get("save_observed_s") or 0.0) > 0.0
            for model in models
        )
    else:
        phase_save_ready = all(
            float(model.get("finalization_after_last_progress_s") or 0.0) > 0.0
            for model in models
        )
    return {
        "ready": bool(natural and ready and phase_save_ready),
        "task_count": len(models),
        "all_natural_exit": natural,
        "all_completion_models_ready": ready,
        "checkpoint_or_finalization_observed": phase_save_ready,
    }


def _coordination_audit(
    spec: WorkloadSpec,
    summary: Mapping[str, Any],
    *,
    expected_tasks: int,
) -> dict[str, Any]:
    if not spec.coordinated_start_barrier:
        return {
            "ready": True,
            "required": False,
            "expected_task_count": int(expected_tasks),
            "start_marker_count": 0,
            "end_marker_count": 0,
        }
    starts = 0
    ends = 0
    child_audits: list[dict[str, Any]] = []
    for child in summary.get("rows") or []:
        log_path = Path(str(child.get("log_path") or ""))
        text = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        has_start = "ScheduleurmPhase name=coordination_barrier event=start" in text
        has_end = "ScheduleurmPhase name=coordination_barrier event=end" in text
        starts += int(has_start)
        ends += int(has_end)
        child_audits.append(
            {
                "log_path": str(log_path),
                "start_marker": has_start,
                "end_marker": has_end,
            }
        )
    ready = bool(
        expected_tasks > 0
        and len(child_audits) == expected_tasks
        and starts == expected_tasks
        and ends == expected_tasks
    )
    return {
        "ready": ready,
        "required": True,
        "expected_task_count": int(expected_tasks),
        "observed_task_count": len(child_audits),
        "start_marker_count": starts,
        "end_marker_count": ends,
        "children": child_audits,
    }


def _admission_audit(
    spec: WorkloadSpec,
    summary: Mapping[str, Any],
    *,
    profile: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ready = True
    for child in summary.get("rows") or []:
        position = (
            int(child.get("global_index") or 0)
            if spec.admission_stagger_axis == "global_index"
            else int(child.get("idx") or 0)
        )
        expected = (
            float(spec.admission_stagger_s) * position
            if int(profile) >= int(spec.admission_stagger_min_profile)
            else 0.0
        )
        durations = (
            (child.get("completion_model") or {})
            .get("phase_durations_s", {})
            .get("admission_delay", [])
        )
        observed = sum(float(value) for value in durations or [])
        tolerance = max(0.25, 0.05 * expected)
        child_ready = bool(
            (expected == 0.0 and observed <= tolerance)
            or (expected > 0.0 and abs(observed - expected) <= tolerance)
        )
        ready = ready and child_ready
        rows.append(
            {
                "global_index": child.get("global_index"),
                "local_index": child.get("idx"),
                "expected_delay_s": expected,
                "observed_delay_s": observed,
                "tolerance_s": tolerance,
                "ready": child_ready,
            }
        )
    return {
        "ready": bool(rows) and ready,
        "stagger_s": float(spec.admission_stagger_s),
        "stagger_axis": spec.admission_stagger_axis,
        "minimum_profile": int(spec.admission_stagger_min_profile),
        "profile": int(profile),
        "delay_counted_in_startup_and_jct": True,
        "children": rows,
    }


def _trajectory_progress_audit(
    spec: WorkloadSpec,
    summary: Mapping[str, Any],
    *,
    profile: int,
    evidence_mode: str,
) -> dict[str, Any]:
    """Audit task-native support for a nonstationary candidate trajectory."""

    required = evidence_mode == STAGED_TRAJECTORY_EVIDENCE
    if not required:
        return {
            "ready": True,
            "required": False,
            "service_evidence_mode": evidence_mode,
        }

    cycle_units = max(1, int(spec.stable_cycle_units))
    expected_progress = int(spec.max_iters)
    expected_intervals = max(1, expected_progress - 1)
    expected_cycles = expected_progress // cycle_units
    children: list[dict[str, Any]] = []
    ready = bool(
        summary.get("measurement_valid")
        and summary.get("all_completion_models_ready")
        and summary.get("require_stable_rate") is False
    )
    for child in summary.get("rows") or []:
        model = child.get("completion_model") or {}
        raw_returncode = child.get("returncode")
        progress_count = int(model.get("progress_observation_count") or 0)
        interval_count = int(model.get("interval_sample_count") or 0)
        cycle_count = progress_count // cycle_units
        child_ready = bool(
            raw_returncode is not None
            and int(raw_returncode) == 0
            and float(child.get("rate") or 0.0) > 0.0
            and bool(child.get("completion_model_ready"))
            and bool(model.get("natural_exit"))
            and int(model.get("total_units") or 0) == expected_progress
            and progress_count >= expected_progress
            and interval_count >= expected_intervals
            and cycle_count >= expected_cycles
        )
        ready = ready and child_ready
        children.append(
            {
                "global_index": child.get("global_index"),
                "progress_observation_count": progress_count,
                "interval_sample_count": interval_count,
                "complete_cycle_count": cycle_count,
                "stable_rate_ready_diagnostic": bool(
                    child.get("stable_rate_ready")
                ),
                "ready": child_ready,
            }
        )
    expected_tasks = int(profile) * len(summary.get("gpus") or [])
    ready = bool(
        ready
        and expected_tasks > 0
        and len(children) == expected_tasks
        and int(summary.get("running_with_rate_count") or 0) == expected_tasks
        and int(summary.get("completion_model_ready_count") or 0)
        == expected_tasks
    )
    return {
        "ready": ready,
        "required": True,
        "service_evidence_mode": evidence_mode,
        "expected_task_count": expected_tasks,
        "observed_task_count": len(children),
        "expected_progress_observations_per_task": expected_progress,
        "expected_interval_samples_per_task": expected_intervals,
        "cycle_units": cycle_units,
        "expected_complete_cycles_per_task": expected_cycles,
        "completion_trajectory_includes_admission_and_drain": True,
        "children": children,
    }


def _remote_artifact_audit(node: str, remote_output: str, run_id: str) -> dict[str, Any]:
    shell = (
        "cd /tmp || exit 5; "
        f"test -d {shlex.quote(remote_output)} || exit 4; "
        f"find {shlex.quote(remote_output)} -type f -printf '%s\\t%b\\t%p\\n'"
    )
    prefix = ARTIFACT_ROOT / f"critical_gpu_checkpoint_audit_{run_id}"
    rc, out, err = _run_remote_capture(node, shell, prefix, timeout_s=300)
    files: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        files.append(
            {
                "size_bytes": _as_int(parts[0]),
                "allocated_blocks_512": _as_int(parts[1]),
                "path": parts[2],
            }
        )
    allocated = [row for row in files if row["size_bytes"] > 0 and row["allocated_blocks_512"] > 0]
    return {
        "ready": bool(rc == 0 and allocated),
        "returncode": int(rc),
        "file_count": len(files),
        "allocated_file_count": len(allocated),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "total_allocated_bytes": sum(row["allocated_blocks_512"] * 512 for row in files),
        "files": files,
        "stderr_tail": err[-1000:],
    }


def _remote_cleanup(node: str, remote_output: str, run_id: str) -> dict[str, Any]:
    prefix = ARTIFACT_ROOT / f"critical_gpu_cleanup_{run_id}"
    rc, out, err = _run_remote_capture(
        node,
        f"cd /tmp && rm -rf -- {shlex.quote(remote_output)}",
        prefix,
        timeout_s=600,
    )
    return {"ready": rc == 0, "returncode": int(rc), "stdout": out[-500:], "stderr": err[-500:]}


def _capacity_boundary(summary: Mapping[str, Any]) -> bool:
    if bool(summary.get("measurement_valid")):
        return False
    texts = []
    for child in summary.get("rows") or []:
        path = Path(str(child.get("log_path") or ""))
        if path.is_file():
            texts.append(path.read_text(encoding="utf-8", errors="replace").lower())
    joined = "\n".join(texts)
    return any(
        token in joined
        for token in (
            "out of memory",
            "cuda_error_out_of_memory",
            "resource exhausted",
            "cublas_status_alloc_failed",
            "cannot allocate memory",
        )
    )


def _write_report(report: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(_markdown(report, output_path), encoding="utf-8")


def _markdown(report: Mapping[str, Any], output_path: Path) -> str:
    lines = [
        "# Critical GPU completion campaign",
        "",
        f"- Artifact: `{output_path}`",
        f"- Protocol: `{report.get('measurement_protocol')}`",
        f"- Node: `{report.get('node')}`",
        f"- Wave / role: `{report.get('wave')}` / `{report.get('split_role')}`",
        f"- Status: `{report.get('status')}`",
        f"- Ready rows: `{report.get('ready_row_count', 0)}`",
        f"- Capacity boundaries: `{report.get('capacity_boundary_count', 0)}`",
        "",
        "| workload | profile | tasks | status | natural completion | checkpoint allocation |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in report.get("rows") or report.get("planned_cells") or []:
        lines.append(
            f"| `{row.get('workload_key')}` | {row.get('profile')} | "
            f"{row.get('total_task_count')} | `{row.get('status', 'planned')}` | "
            f"{bool((row.get('completion_audit') or {}).get('ready'))} | "
            f"{bool(row.get('checkpoint_allocation_ready'))} |"
        )
    return "\n".join(lines) + "\n"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:220]


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=sorted(NODE_SPECS), required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--profile", action="append", type=int, default=[])
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--keep-remote-output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_critical_gpu_completion_campaign(
        node=args.node,
        wave=args.wave,
        allow_launch=args.allow_launch,
        workload_keys=args.workload,
        profiles=args.profile,
        cleanup_remote_output=not args.keep_remote_output,
    )
    print(json.dumps({
        "campaign_id": report.get("campaign_id"),
        "status": report.get("status"),
        "pass": report.get("pass"),
        "ready_row_count": report.get("ready_row_count", 0),
    }, indent=2, sort_keys=True))
    return 0 if report.get("pass") or report.get("status") == "MANIFEST_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
