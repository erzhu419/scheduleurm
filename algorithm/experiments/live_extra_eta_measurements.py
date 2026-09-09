"""Extra live ETA probes for hardware/env-aware service-cache v2.

These probes fill the gaps that are intentionally outside the first live v2
merge gate: homogeneous-node equivalence checks, additional RL environments,
and jtl311linux node-specific RL rows.  They still bypass the legacy scheduler
and run only controlled short benchmark windows with task-native progress.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache, records_from_summary_file

from .remote_workload_selected_profile_probe import build_remote_workload_selected_profile_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
TEMPLATE_ROOT = REPO_ROOT / "algorithm" / "experiments" / "templates"


@dataclass(frozen=True)
class ExtraProbeSpec:
    spec_id: str
    node: str
    gpus: tuple[int, ...]
    profiles: tuple[int, ...]
    workload_key: str
    workload_env: str
    resource_kind: str
    template: str
    cwd: str
    unit: str
    total_units: float
    max_iters: int
    timeout_s: int
    node_bucket: str
    hardware_class: str
    resource_state: str = "empty"
    stable_windows: int = 4
    min_rate_samples: int = 5
    stable_cv: float = 0.10
    stable_rel_delta: float = 0.06
    stable_skip_samples: int = 1
    stable_cycle_units: int = 0
    terminate_on_stable: bool = True
    require_stable_rate: bool = True
    template_vars: Mapping[str, str] | None = None

    @property
    def template_path(self) -> Path:
        return TEMPLATE_ROOT / self.template


EXTRA_PROBES = (
    ExtraProbeSpec(
        spec_id="jtl110gpu2_cnn_equiv",
        node="jtl110gpu2",
        gpus=(0, 1),
        profiles=(1, 2),
        workload_key="gpu_cnn_torch_resnet50",
        workload_env="cnn",
        resource_kind="gpu_cnn",
        template="torch_cnn_resnet50.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=80.0,
        max_iters=80,
        timeout_s=360,
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        hardware_class="gpu_3080ti_12gb_dual",
    ),
    ExtraProbeSpec(
        spec_id="jtl110gpu2_cnn_equiv_p2_long",
        node="jtl110gpu2",
        gpus=(0, 1),
        profiles=(2,),
        workload_key="gpu_cnn_torch_resnet50",
        workload_env="cnn",
        resource_kind="gpu_cnn",
        template="torch_cnn_resnet50.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=240.0,
        max_iters=240,
        timeout_s=900,
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        hardware_class="gpu_3080ti_12gb_dual",
        stable_windows=5,
        min_rate_samples=8,
        stable_cv=0.06,
        stable_rel_delta=0.05,
        stable_skip_samples=5,
        terminate_on_stable=False,
        require_stable_rate=True,
    ),
    ExtraProbeSpec(
        spec_id="jtl110gpu2_llm_equiv",
        node="jtl110gpu2",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        template="torch_llm_distilgpt2.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=64.0,
        max_iters=64,
        timeout_s=420,
        node_bucket="jtl110gpu2:gpu_3080ti_12gb_dual",
        hardware_class="gpu_3080ti_12gb_dual",
    ),
    ExtraProbeSpec(
        spec_id="jtl110gpu_resac_hopper_p1",
        node="jtl110gpu",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_hopper",
        workload_env="hopper",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        hardware_class="gpu_3080ti_12gb_dual",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Hopper-v2"},
    ),
    ExtraProbeSpec(
        spec_id="jtl110gpu_resac_walker2d_p1",
        node="jtl110gpu",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_walker2d",
        workload_env="walker2d",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        hardware_class="gpu_3080ti_12gb_dual",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Walker2d-v2"},
    ),
    ExtraProbeSpec(
        spec_id="jtl311_resac_halfcheetah_p1",
        node="jtl311linux",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_halfcheetah",
        workload_env="halfcheetah",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "HalfCheetah-v2"},
    ),
    ExtraProbeSpec(
        spec_id="jtl311_resac_ant_p1",
        node="jtl311linux",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_ant",
        workload_env="ant",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Ant-v2"},
    ),
    ExtraProbeSpec(
        spec_id="jtl311_resac_hopper_p1",
        node="jtl311linux",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_hopper",
        workload_env="hopper",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Hopper-v2"},
    ),
    ExtraProbeSpec(
        spec_id="jtl311_resac_walker2d_p1",
        node="jtl311linux",
        gpus=(0, 1),
        profiles=(1,),
        workload_key="hybrid_rl_resac_walker2d",
        workload_env="walker2d",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="jtl311linux:gpu_rtx2080_8gb_dual_cpu_fast",
        hardware_class="gpu_rtx2080_8gb_dual_cpu_fast",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Walker2d-v2"},
    ),
    ExtraProbeSpec(
        spec_id="node007_llm_p1",
        node="node007",
        gpus=(0, 1, 2, 3),
        profiles=(1,),
        workload_key="gpu_llm_distilgpt2",
        workload_env="llm",
        resource_kind="gpu_llm",
        template="torch_llm_distilgpt2.cmd.tpl",
        cwd=str(REPO_ROOT),
        unit="iter",
        total_units=96.0,
        max_iters=96,
        timeout_s=900,
        node_bucket="node007:gpu_node007_4x12gb",
        hardware_class="gpu_node007_4x12gb",
        stable_windows=5,
        min_rate_samples=8,
        stable_cv=0.08,
        stable_rel_delta=0.05,
        stable_skip_samples=5,
        terminate_on_stable=False,
    ),
    ExtraProbeSpec(
        spec_id="node007_resac_ant_p1",
        node="node007",
        gpus=(0, 1, 2, 3),
        profiles=(1,),
        workload_key="hybrid_rl_resac_ant",
        workload_env="ant",
        resource_kind="hybrid_rl",
        template="resac_env_real_venv.cmd.tpl",
        cwd="/home/erzhu419/mine_code/RE-SAC",
        unit="iter",
        total_units=200.0,
        max_iters=200,
        timeout_s=2400,
        node_bucket="node007:gpu_node007_4x12gb",
        hardware_class="gpu_node007_4x12gb",
        stable_cycle_units=5,
        template_vars={"mujoco_env": "Ant-v2"},
    ),
)


def build_live_extra_eta_measurements(
    *,
    allow_launch: bool = False,
    specs: Iterable[str] | None = None,
    run_prefix: str = "live_extra_eta_measurements_20260629",
) -> dict[str, Any]:
    selected_ids = {str(x) for x in specs or []}
    selected = [spec for spec in EXTRA_PROBES if not selected_ids or spec.spec_id in selected_ids]
    cache = ServiceRateCache()
    rows = []
    for spec in selected:
        row = _run_spec(spec, run_prefix=run_prefix) if allow_launch else _manifest_row(spec, run_prefix=run_prefix)
        rows.append(row)
        if allow_launch:
            _add_records_for_row(cache, row)
    admitted = [row for row in rows if row.get("measurement_valid")]
    boundaries = [row for row in rows if row.get("launched") and not row.get("measurement_valid")]
    pending = [row for row in rows if not row.get("launched")]
    pass_ready = bool(rows) and not pending and bool(admitted) and not boundaries
    return {
        "gate": "live_extra_eta_measurements",
        "run_prefix": run_prefix,
        "allow_launch": bool(allow_launch),
        "pass": bool(pass_ready) if allow_launch else bool(rows),
        "status": (
            "LIVE_EXTRA_ETA_READY"
            if allow_launch and pass_ready
            else ("LIVE_EXTRA_ETA_OPEN" if allow_launch else "DRY_RUN_READY")
        ),
        "row_count": len(rows),
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "pending_count": len(pending),
        "rows": rows,
        "admitted_rows": admitted,
        "boundary_rows": boundaries,
        "pending_rows": pending,
        "service_cache_snapshot": cache.snapshot(),
        "scope": (
            "Controlled extra task-native ETA rows for equivalent hardware checks "
            "and node/env-aware RL service. Legacy scheduler hard limits are bypassed; "
            "ordinary user tasks are never launched or migrated by this gate."
        ),
    }


def _manifest_row(spec: ExtraProbeSpec, *, run_prefix: str) -> dict[str, Any]:
    return {
        **_spec_base(spec, run_prefix=run_prefix),
        "launched": False,
        "measurement_valid": False,
        "reason": "dry_run_manifest",
    }


def _run_spec(spec: ExtraProbeSpec, *, run_prefix: str) -> dict[str, Any]:
    run_id = f"{run_prefix}_{spec.spec_id}"
    result = build_remote_workload_selected_profile_probe(
        run_id=run_id,
        node=spec.node,
        gpus=[int(x) for x in spec.gpus],
        profiles=[int(x) for x in spec.profiles],
        cwd=spec.cwd,
        cmd_template_file=str(spec.template_path),
        output_root=f"/tmp/scheduleurm_eta_extra/{run_id}",
        max_iters=int(spec.max_iters),
        timeout_s=int(spec.timeout_s),
        unit=spec.unit,
        terminate_on_stable=bool(spec.terminate_on_stable),
        require_stable_rate=bool(spec.require_stable_rate),
        stable_windows=int(spec.stable_windows),
        min_rate_samples=int(spec.min_rate_samples),
        stable_cv=float(spec.stable_cv),
        stable_rel_delta=float(spec.stable_rel_delta),
        stable_skip_samples=int(spec.stable_skip_samples),
        stable_cycle_units=int(spec.stable_cycle_units),
        coordinated_profile_launch=True,
        extra_template_values=dict(spec.template_vars or {}),
    )
    profile_rows = [_summary_row(spec, run_id=run_id, profile=int(profile)) for profile in spec.profiles]
    valid = all(bool(row.get("measurement_valid")) for row in profile_rows)
    return {
        **_spec_base(spec, run_prefix=run_prefix),
        "run_id": run_id,
        "launched": True,
        "measurement_valid": bool(result.get("pass")) and valid,
        "probe_result": result,
        "profile_rows": profile_rows,
        "reason": "" if bool(result.get("pass")) and valid else "one_or_more_profiles_unstable_or_missing",
    }


def _spec_base(spec: ExtraProbeSpec, *, run_prefix: str) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "run_prefix": run_prefix,
        "node": spec.node,
        "gpus": list(spec.gpus),
        "profiles": list(spec.profiles),
        "node_bucket": spec.node_bucket,
        "hardware_class": spec.hardware_class,
        "workload_key": spec.workload_key,
        "workload_env": spec.workload_env,
        "resource_kind": spec.resource_kind,
        "resource_state": spec.resource_state,
        "template": str(spec.template_path),
        "cwd": spec.cwd,
        "unit": spec.unit,
        "total_units": float(spec.total_units),
        "eta_source_required": "tqdm/progress",
        "hard_rule_mode": "clean_bench_direct_remote",
        "legacy_scheduler_limits_bypassed": True,
        "stable_gate": {
            "stable_windows": int(spec.stable_windows),
            "min_rate_samples": int(spec.min_rate_samples),
            "stable_cv": float(spec.stable_cv),
            "stable_rel_delta": float(spec.stable_rel_delta),
            "stable_skip_samples": int(spec.stable_skip_samples),
            "stable_cycle_units": int(spec.stable_cycle_units),
            "terminate_on_stable": bool(spec.terminate_on_stable),
            "require_stable_rate": bool(spec.require_stable_rate),
        },
    }


def _summary_row(spec: ExtraProbeSpec, *, run_id: str, profile: int) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "reports" / f"profile_{int(profile)}_per_gpu_summary.json"
    base = {
        "profile": int(profile),
        "summary_path": str(path),
        "path_exists": path.exists(),
    }
    if not path.exists():
        return {**base, "measurement_valid": False, "stable_rate_ready_count": 0, "running_count": 0}
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {
        **base,
        "measurement_valid": bool(summary.get("measurement_valid")),
        "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
        "running_count": int(summary.get("running_count") or 0),
        "aggregate_stable_rate_unit_s": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
        "mean_stable_rate_unit_s": float(summary.get("mean_stable_rate_unit_s") or 0.0),
        "status_counts": summary.get("status_counts") or {},
        "eta_source": "tqdm/progress",
    }


def _add_records_for_row(cache: ServiceRateCache, row: Mapping[str, Any]) -> None:
    for profile_row in row.get("profile_rows") or []:
        if not bool(profile_row.get("measurement_valid")):
            continue
        path = Path(str(profile_row.get("summary_path") or ""))
        if not path.exists():
            continue
        for record in records_from_summary_file(
            path,
            workload_key=str(row.get("workload_key")),
            command_fingerprint=f"live_extra_eta:{row.get('spec_id')}",
            resource_kind=str(row.get("resource_kind")),
            total_units=float(row.get("total_units") or 1.0),
            node_bucket=str(row.get("node_bucket")),
            workload_env=str(row.get("workload_env")),
            resource_state=str(row.get("resource_state")),
            eta_source="tqdm/progress",
            hardware_class=str(row.get("hardware_class")),
        ):
            cache.add(record, force_replace=True)


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live Extra ETA Measurements",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allow launch: `{str(bool(report.get('allow_launch'))).lower()}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        f"- Boundary rows: `{report.get('boundary_count')}`",
        "",
        "| Spec | Node | Workload | Env | Profiles | Valid | Rates |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        rates = []
        for profile_row in row.get("profile_rows") or []:
            rates.append(
                f"p{int(profile_row.get('profile') or 0)}={float(profile_row.get('aggregate_stable_rate_unit_s') or 0.0):.6g}"
            )
        lines.append(
            f"| `{row.get('spec_id')}` | `{row.get('node')}` | `{row.get('workload_key')}` | "
            f"`{row.get('workload_env')}` | `{row.get('profiles')}` | "
            f"{str(bool(row.get('measurement_valid'))).lower()} | `{', '.join(rates)}` |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--specs", default="")
    parser.add_argument("--run-prefix", default="live_extra_eta_measurements_20260629")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "live_extra_eta_measurements_20260629.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_live_extra_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "live_extra_eta_measurements_20260629.md")
    args = parser.parse_args()
    specs = [x.strip() for x in str(args.specs or "").split(",") if x.strip()]
    report = build_live_extra_eta_measurements(
        allow_launch=bool(args.allow_launch),
        specs=specs or None,
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
