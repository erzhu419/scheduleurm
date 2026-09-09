"""Cross-node task-native ETA measurement manifest.

The default command builds a dry-run manifest and does not launch anything.
Use ``--allow-launch`` only when the named nodes are intentionally reserved for
benchmark probes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .remote_workload_selected_profile_probe import build_remote_workload_selected_profile_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
TEMPLATE_ROOT = REPO_ROOT / "algorithm" / "experiments" / "templates"


@dataclass(frozen=True)
class HardwareGroup:
    name: str
    nodes: tuple[str, ...]
    representative: str
    verify_node: str
    gpus: tuple[int, ...]
    note: str


@dataclass(frozen=True)
class WorkloadProbe:
    name: str
    workload_key: str
    workload_env: str
    resource_kind: str
    template: str
    cwd: str
    unit: str
    total_units: float
    profiles: tuple[int, ...]
    max_iters: int
    timeout_s: int
    launch_ready: bool = True
    stable_cycle_units: int = 0
    template_vars: Mapping[str, str] | None = None


HARDWARE_GROUPS = (
    HardwareGroup("gpu_3080ti_12gb_dual", ("jtl110gpu", "jtl110gpu2"), "jtl110gpu", "jtl110gpu2", (0, 1), "full measure one, equivalence on the other"),
    HardwareGroup("gpu_rtx2080_8gb_dual_cpu_fast", ("jtl311linux",), "jtl311linux", "jtl311linux", (0, 1), "full measure; CPU-fast/GPU-weaker row"),
    HardwareGroup("gpu_node007_4x12gb", ("node007",), "node007", "node007", (0, 1, 2, 3), "separate 4-GPU node007 row"),
    HardwareGroup("cpu_hpc_192c", ("node001", "node002", "node003", "node004", "node005", "node006"), "node001", "node002", tuple(), "full measure one 192-core CPU node; validate another"),
)


WORKLOAD_PROBES = (
    WorkloadProbe(
        name="cnn_resnet50",
        workload_key="gpu_cnn_torch_resnet50",
        workload_env="cnn",
        resource_kind="gpu_cnn",
        template="torch_cnn_resnet50.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=80.0,
        profiles=(1, 2, 4),
        max_iters=80,
        timeout_s=360,
    ),
    WorkloadProbe(
        name="llm_distilgpt2",
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        template="torch_llm_distilgpt2.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=64.0,
        profiles=(1, 2),
        max_iters=64,
        timeout_s=420,
    ),
    WorkloadProbe(
        name="resac_ant",
        workload_key="hybrid_rl_resac_ant",
        workload_env="ant",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        profiles=(1, 2, 4),
        max_iters=200,
        timeout_s=2400,
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Ant-v2"},
    ),
    WorkloadProbe(
        name="resac_halfcheetah",
        workload_key="hybrid_rl_resac_halfcheetah",
        workload_env="halfcheetah",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        profiles=(1, 2, 4),
        max_iters=200,
        timeout_s=2400,
        stable_cycle_units=5,
        template_vars={"mujoco_env": "HalfCheetah-v2"},
    ),
    WorkloadProbe(
        name="resac_hopper",
        workload_key="hybrid_rl_resac_hopper",
        workload_env="hopper",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        profiles=(1, 2, 4),
        max_iters=200,
        timeout_s=2400,
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Hopper-v2"},
    ),
    WorkloadProbe(
        name="resac_walker2d",
        workload_key="hybrid_rl_resac_walker2d",
        workload_env="walker2d",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        profiles=(1, 2, 4),
        max_iters=200,
        timeout_s=2400,
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Walker2d-v2"},
    ),
    WorkloadProbe(
        name="freqduet_cpu_surrogate",
        workload_key="freqduet_cpu_surrogate",
        workload_env="cpu",
        resource_kind="cpu_heavy",
        template="",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=200.0,
        profiles=(1, 8, 32, 96, 180),
        max_iters=200,
        timeout_s=600,
        launch_ready=False,
    ),
)


def build_cross_node_eta_matrix(
    *,
    allow_launch: bool = False,
    nodes: set[str] | None = None,
    workloads: set[str] | None = None,
    run_id: str = "cross_node_eta_matrix_20260629",
) -> dict[str, Any]:
    selected_groups = [g for g in HARDWARE_GROUPS if nodes is None or g.representative in nodes or any(n in nodes for n in g.nodes)]
    selected_workloads = [w for w in WORKLOAD_PROBES if workloads is None or w.name in workloads or w.workload_key in workloads]
    rows: list[dict[str, Any]] = []
    launched: list[dict[str, Any]] = []
    for group in selected_groups:
        for workload in selected_workloads:
            if group.name.startswith("cpu_") != (workload.resource_kind.startswith("cpu")):
                continue
            row = _manifest_row(group, workload, run_id=run_id)
            rows.append(row)
            if allow_launch and row.get("launch_ready"):
                launched.append(_launch_probe(row))
    coverage = _coverage(rows)
    manifest_ready = bool(rows) and all(row["eta_source_required"] == "tqdm/progress" for row in rows)
    launched_ready = all(bool(row.get("pass")) for row in launched) if allow_launch else True
    return {
        "gate": "cross_node_eta_matrix",
        "run_id": run_id,
        "allow_launch": bool(allow_launch),
        "hardware_groups": [group.__dict__ for group in HARDWARE_GROUPS],
        "workload_count": len(selected_workloads),
        "manifest_rows": rows,
        "launched_rows": launched,
        "stable_eta_rule": "task-native tqdm/ScheduleurmStableRate only; history fallback rows excluded from theorem-facing cache",
        "coverage": coverage,
        "pass": bool(manifest_ready and launched_ready),
        "manifest_ready": bool(manifest_ready),
        "launched_ready": bool(launched_ready),
        "status": "LAUNCHED_PASS" if allow_launch and launched_ready else ("LAUNCHED_OPEN" if allow_launch else "DRY_RUN_READY"),
    }


def _manifest_row(group: HardwareGroup, workload: WorkloadProbe, *, run_id: str) -> dict[str, Any]:
    profiles = list(workload.profiles)
    if group.name == "gpu_rtx2080_8gb_dual_cpu_fast" and workload.name == "llm_distilgpt2":
        profiles = [1]
    return {
        "row_id": f"{group.name}|{workload.name}",
        "hardware_class": group.name,
        "node_bucket": f"{group.representative}:{group.name}",
        "node": group.representative,
        "verify_node": group.verify_node,
        "gpus": list(group.gpus),
        "workload_key": workload.workload_key,
        "workload_env": workload.workload_env,
        "resource_kind": workload.resource_kind,
        "resource_states": ["empty", "half_loaded", "full_loaded", "high_vram_resident", "mixed_colocation"],
        "profiles": profiles,
        "template": str(TEMPLATE_ROOT / workload.template) if workload.template else "",
        "cwd": workload.cwd,
        "unit": workload.unit,
        "total_units": workload.total_units,
        "max_iters": workload.max_iters,
        "timeout_s": workload.timeout_s,
        "stable_cycle_units": int(workload.stable_cycle_units),
        "eta_source_required": "tqdm/progress",
        "launch_ready": bool(workload.launch_ready),
        "pending_reason": "" if workload.launch_ready else "CPU live launch uses the CPU selected-profile probe, not the GPU workload probe",
        "run_id": f"{run_id}_{group.name}_{workload.name}",
        "template_vars": dict(workload.template_vars or {}),
        "launch_command": (
            "python -m algorithm.experiments.cross_node_eta_matrix "
            f"--allow-launch --nodes {group.representative} --workloads {workload.name}"
        ),
    }


def _launch_probe(row: Mapping[str, Any]) -> dict[str, Any]:
    if not row.get("template"):
        return {"row_id": row["row_id"], "pass": False, "status": "CPU_PROBE_REQUIRES_CPU_SELECTED_PROFILE_RUNNER"}
    profiles = [int(x) for x in row["profiles"]]
    co_located = any(int(profile) > 1 for profile in profiles)
    effective_iters = int(row["max_iters"])
    if co_located:
        workload_key = str(row.get("workload_key") or "")
        if "cnn" in workload_key:
            effective_iters = max(effective_iters, 240)
        elif "llm" in workload_key:
            effective_iters = max(effective_iters, 96)
        else:
            effective_iters = max(effective_iters, 200)
    result = build_remote_workload_selected_profile_probe(
        run_id=str(row["run_id"]),
        node=str(row["node"]),
        gpus=[int(x) for x in row["gpus"]],
        profiles=profiles,
        cwd=str(row["cwd"]),
        cmd_template_file=str(row["template"]),
        output_root=f"/tmp/scheduleurm_eta_matrix/{row['run_id']}",
        max_iters=effective_iters,
        timeout_s=int(row["timeout_s"]),
        unit=str(row["unit"]),
        terminate_on_stable=not co_located,
        require_stable_rate=True,
        stable_windows=5 if co_located else 4,
        min_rate_samples=8 if co_located else 5,
        stable_cv=0.06 if co_located else 0.10,
        stable_rel_delta=0.05 if co_located else 0.06,
        stable_skip_samples=5 if co_located else 1,
        stable_cycle_units=int(row.get("stable_cycle_units") or 0),
        coordinated_profile_launch=True,
        extra_template_values=dict(row.get("template_vars") or {}),
    )
    return {"row_id": row["row_id"], **result}


def _coverage(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "node007_and_jtl311_separate": any("gpu_node007_4x12gb" in r["row_id"] for r in rows)
        and any("gpu_rtx2080_8gb_dual_cpu_fast" in r["row_id"] for r in rows),
        "jtl110_equivalence_planned": any(r.get("verify_node") == "jtl110gpu2" for r in rows),
        "env_specific_rl_planned": any(r.get("workload_env") in {"ant", "halfcheetah", "hopper", "walker2d"} for r in rows),
        "history_fallback_excluded": True,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cross-Node ETA Matrix Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allow launch: `{str(bool(report.get('allow_launch'))).lower()}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        "",
        "| Row | Node | Workload | Env | Profiles | Required ETA | Launch ready |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for row in report.get("manifest_rows") or []:
        lines.append(
            f"| `{row['row_id']}` | `{row['node']}` | `{row['workload_key']}` | "
            f"`{row['workload_env']}` | `{row['profiles']}` | `{row['eta_source_required']}` | "
            f"{str(bool(row.get('launch_ready'))).lower()} |"
        )
    lines.extend(["", "## Coverage", ""])
    for key, value in (report.get("coverage") or {}).items():
        lines.append(f"- `{key}`: `{str(bool(value)).lower()}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--run-id", default="cross_node_eta_matrix_20260629")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "cross_node_eta_matrix_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "cross_node_eta_matrix_20260629.md")
    args = parser.parse_args()
    nodes = {x.strip() for x in args.nodes.split(",") if x.strip()} or None
    workloads = {x.strip() for x in args.workloads.split(",") if x.strip()} or None
    report = build_cross_node_eta_matrix(
        allow_launch=bool(args.allow_launch),
        nodes=nodes,
        workloads=workloads,
        run_id=str(args.run_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
