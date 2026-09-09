"""Run controlled probes for rows from the full-factorial ETA design.

The runner is deliberately opt-in.  By default it only writes a manifest.  With
``--allow-launch`` it launches controlled benchmark probes through the same
task-native progress wrappers used by the existing live-v2 ETA measurements.
It never calls the legacy scheduler dispatch path.
"""
from __future__ import annotations

import argparse
import csv
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
DEFAULT_DESIGN = ARTIFACT_ROOT / "full_factorial_eta_design_rows_20260629.csv"


@dataclass(frozen=True)
class ProbeConfig:
    node: str
    gpus: tuple[int, ...]
    template: str
    cwd: str
    unit: str
    total_units: float
    max_iters: int
    timeout_s: int
    stable_cycle_units: int = 0
    template_values: Mapping[str, Any] | None = None


GPU_CONFIGS: dict[tuple[str, str], ProbeConfig] = {
    ("gpu_cnn_torch_resnet50", "jtl110gpu"): ProbeConfig(
        "jtl110gpu", (0, 1), "torch_cnn_resnet50.cmd.tpl", str(REPO_ROOT), "iter", 240.0, 240, 1200
    ),
    ("gpu_cnn_torch_resnet50", "jtl110gpu2"): ProbeConfig(
        "jtl110gpu2", (0, 1), "torch_cnn_resnet50.cmd.tpl", str(REPO_ROOT), "iter", 240.0, 240, 1200
    ),
    ("gpu_cnn_torch_resnet50", "jtl311linux"): ProbeConfig(
        "jtl311linux", (0, 1), "torch_cnn_resnet50.cmd.tpl", str(REPO_ROOT), "iter", 240.0, 240, 1200
    ),
    ("gpu_cnn_torch_resnet50", "node007"): ProbeConfig(
        "node007", (0, 1, 2, 3), "torch_cnn_resnet50.cmd.tpl", str(REPO_ROOT), "iter", 240.0, 240, 1200
    ),
    ("gpu_llm_distilgpt2", "jtl110gpu"): ProbeConfig(
        "jtl110gpu", (0, 1), "torch_llm_distilgpt2.cmd.tpl", str(REPO_ROOT), "iter", 128.0, 128, 1200
    ),
    ("gpu_llm_distilgpt2", "jtl110gpu2"): ProbeConfig(
        "jtl110gpu2", (0, 1), "torch_llm_distilgpt2.cmd.tpl", str(REPO_ROOT), "iter", 128.0, 128, 1200
    ),
    ("gpu_llm_distilgpt2", "jtl311linux"): ProbeConfig(
        "jtl311linux", (0, 1), "torch_llm_distilgpt2.cmd.tpl", str(REPO_ROOT), "iter", 128.0, 128, 1200
    ),
    ("gpu_llm_distilgpt2", "node007"): ProbeConfig(
        "node007", (0, 1, 2, 3), "torch_llm_distilgpt2.cmd.tpl", str(REPO_ROOT), "iter", 128.0, 128, 1200
    ),
    ("gpu_heavy_jax_matmul", "jtl110gpu"): ProbeConfig(
        "jtl110gpu", (0, 1), "gpu_matmul_progress.cmd.tpl", str(REPO_ROOT), "step", 160.0, 160, 1200,
        template_values={"matrix_size": "2048"},
    ),
    ("gpu_heavy_jax_matmul", "jtl110gpu2"): ProbeConfig(
        "jtl110gpu2", (0, 1), "gpu_matmul_progress.cmd.tpl", str(REPO_ROOT), "step", 160.0, 160, 1200,
        template_values={"matrix_size": "2048"},
    ),
    ("gpu_heavy_jax_matmul", "jtl311linux"): ProbeConfig(
        "jtl311linux", (0, 1), "gpu_matmul_progress.cmd.tpl", str(REPO_ROOT), "step", 160.0, 160, 1200,
        template_values={"matrix_size": "2048"},
    ),
    ("gpu_heavy_jax_matmul", "node007"): ProbeConfig(
        "node007", (0, 1, 2, 3), "gpu_matmul_progress.cmd.tpl", str(REPO_ROOT), "step", 160.0, 160, 1200,
        template_values={"matrix_size": "2048"},
    ),
}


for env, key in (
    ("ant", "hybrid_rl_resac_ant"),
    ("halfcheetah", "hybrid_rl_resac_halfcheetah"),
    ("hopper", "hybrid_rl_resac_hopper"),
    ("walker2d", "hybrid_rl_resac_walker2d"),
):
    mujoco = {
        "ant": "Ant-v2",
        "halfcheetah": "HalfCheetah-v2",
        "hopper": "Hopper-v2",
        "walker2d": "Walker2d-v2",
    }[env]
    for node, gpus in (
        ("jtl110gpu", (0, 1)),
        ("jtl110gpu2", (0, 1)),
        ("jtl311linux", (0, 1)),
        ("node007", (0, 1, 2, 3)),
    ):
        GPU_CONFIGS[(key, node)] = ProbeConfig(
            node,
            gpus,
            "resac_env_real_venv.cmd.tpl",
            "/home/erzhu419/mine_code/RE-SAC",
            "iter",
            200.0,
            200,
            2400,
            stable_cycle_units=5,
            template_values={"mujoco_env": mujoco},
        )

for node, gpus in (
    ("jtl110gpu", (0, 1)),
    ("jtl110gpu2", (0, 1)),
    ("jtl311linux", (0, 1)),
    ("node007", (0, 1, 2, 3)),
):
    GPU_CONFIGS[("hybrid_rl_bapr_ant", node)] = ProbeConfig(
        node,
        gpus,
        "bapr_ant_real.cmd.tpl",
        "/home/erzhu419/mine_code/BAPR",
        "iter",
        200.0,
        200,
        2400,
        stable_cycle_units=5,
    )


def build_full_factorial_eta_probe_runner(
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
    run_prefix: str = "full_factorial_eta_probe_20260630",
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
    boundaries = [
        row for row in rows
        if row.get("launched") and row.get("capacity_boundary_profiles")
    ]
    return {
        "gate": "full_factorial_eta_probe_runner",
        "allow_launch": bool(allow_launch),
        "run_prefix": run_prefix,
        "max_profiles": int(max_profiles),
        "tier": tier,
        "resource_state": resource_state,
        "selected_count": len(selected),
        "launched_count": len(launched),
        "admitted_count": len(admitted),
        "boundary_count": len(boundaries),
        "pass": bool(rows) and (not allow_launch or bool(launched)),
        "status": "PROBE_RUNNER_LAUNCHED" if allow_launch else "PROBE_RUNNER_MANIFEST",
        "rows": rows,
        "admitted_rows": admitted,
        "boundary_rows": boundaries,
        "service_cache_snapshot": cache.snapshot(),
        "scope": (
            "Controlled full-factorial ETA probes.  Launches bypass the legacy "
            "scheduler dispatch path and require task-native tqdm/progress.  "
            "Invalid profiles are retained as scoped capacity/instability "
            "boundaries for the exact node/workload/state."
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
    rows = list(csv.DictReader(design_csv.open(encoding="utf-8")))
    out = []
    for row in rows:
        if row.get("tier") != tier:
            continue
        if row.get("resource_state") != resource_state:
            continue
        if "pending" not in row.get("status", ""):
            continue
        if row.get("resource_kind") != "gpu":
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


def _manifest(row: Mapping[str, str], *, run_prefix: str, max_profiles: int = 0) -> dict[str, Any]:
    node = str(row.get("node") or "")
    workload_key = str(row.get("workload_key") or "")
    config = GPU_CONFIGS.get((workload_key, node))
    profiles = _parse_profiles(row.get("missing_profiles"))
    if int(max_profiles) > 0:
        profiles = profiles[: int(max_profiles)]
    run_id = _safe_id(f"{run_prefix}_{node}_{workload_key}_{row.get('workload_env')}_{row.get('resource_state')}")
    resident_supported = _resident_state_supported(row)
    launch_ready = config is not None and bool(profiles) and resident_supported
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
        "boundary_closed_profiles": _parse_profiles(row.get("boundary_closed_profiles")),
        "launch_ready": launch_ready,
        "pending_reason": "" if launch_ready else _pending_reason(config=config, profiles=profiles, resident_supported=resident_supported),
        "launched": False,
        "measurement_valid": False,
        "eta_source_required": "tqdm/progress",
        "legacy_scheduler_limits_bypassed": True,
        "config": None if config is None else {
            "node": config.node,
            "gpus": list(config.gpus),
            "template": config.template,
            "cwd": config.cwd,
            "max_iters": config.max_iters,
            "timeout_s": config.timeout_s,
            "stable_cycle_units": config.stable_cycle_units,
            "template_values": dict(config.template_values or {}),
        },
    }


def _resident_state_supported(row: Mapping[str, str]) -> bool:
    state = str(row.get("resource_state") or "empty")
    # Same-workload half/full rows are measured by launching the requested
    # profile concurrently on each target GPU.  That is the load state itself,
    # so no separate resident harness is required.
    return state in {"empty", "half_loaded", "full_loaded"}


def _pending_reason(*, config: ProbeConfig | None, profiles: list[int], resident_supported: bool) -> str:
    if config is None:
        return "no_gpu_probe_config"
    if not profiles:
        return "no_missing_profiles"
    if not resident_supported:
        return "resident_load_probe_not_implemented_for_state"
    return "not_launch_ready"


def _launch(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config_data = manifest.get("config") or {}
    config = ProbeConfig(
        node=str(config_data["node"]),
        gpus=tuple(int(x) for x in config_data["gpus"]),
        template=str(config_data["template"]),
        cwd=str(config_data["cwd"]),
        unit=str(config_data["unit"]) if "unit" in config_data else _unit_for(str(manifest.get("workload_key"))),
        total_units=_total_units_for(str(manifest.get("workload_key"))),
        max_iters=int(config_data["max_iters"]),
        timeout_s=int(config_data["timeout_s"]),
        stable_cycle_units=int(config_data.get("stable_cycle_units") or 0),
        template_values=dict(config_data.get("template_values") or {}),
    )
    profiles = [int(x) for x in manifest.get("profiles") or []]
    co_located = any(profile > 1 for profile in profiles)
    result = build_remote_workload_selected_profile_probe(
        run_id=str(manifest["run_id"]),
        node=config.node,
        gpus=list(config.gpus),
        profiles=profiles,
        cwd=config.cwd,
        cmd_template_file=str(TEMPLATE_ROOT / config.template),
        output_root=f"/tmp/scheduleurm_full_factorial_eta/{manifest['run_id']}",
        max_iters=config.max_iters,
        timeout_s=config.timeout_s,
        unit=config.unit,
        # Task-native stable-rate ETA is the measurement target.  Older runs
        # left co-located profiles alive until the hard timeout even after every
        # child had a stable progress window; that was reproducible but too slow
        # for full-factorial closure.  The wrapper still enforces min samples,
        # CV, relative-delta, and cycle-unit gates before exiting.
        terminate_on_stable=True,
        require_stable_rate=True,
        stable_windows=5,
        min_rate_samples=8,
        stable_cv=0.06 if co_located else 0.08,
        stable_rel_delta=0.05,
        stable_skip_samples=5,
        stable_cycle_units=config.stable_cycle_units,
        coordinated_profile_launch=True,
        extra_template_values=dict(config.template_values or {}),
    )
    profile_rows = [_profile_summary_row(str(manifest["run_id"]), profile) for profile in profiles]
    valid = all(bool(row.get("measurement_valid")) for row in profile_rows)
    capacity_boundary_profiles = sorted(_capacity_boundary_profiles(profile_rows))
    return {
        **dict(manifest),
        "launched": True,
        "measurement_valid": bool(result.get("pass")) and valid,
        "capacity_boundary_profiles": capacity_boundary_profiles,
        "probe_result": result,
        "profile_rows": profile_rows,
        "reason": "" if bool(result.get("pass")) and valid else "one_or_more_profiles_unstable_or_capacity_boundary",
    }


def _profile_summary_row(run_id: str, profile: int) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "reports" / f"profile_{int(profile)}_per_gpu_summary.json"
    base = {"profile": int(profile), "summary_path": str(path), "path_exists": path.exists()}
    if not path.exists():
        return {**base, "measurement_valid": False}
    summary = json.loads(path.read_text(encoding="utf-8"))
    measurement_valid = _summary_measurement_valid_for_eta(summary)
    return {
        **base,
        "measurement_valid": measurement_valid,
        "stable_rate_ready_count": int(summary.get("stable_rate_ready_count") or 0),
        "running_count": int(summary.get("running_count") or 0),
        "aggregate_stable_rate_unit_s": float(summary.get("aggregate_stable_rate_unit_s") or 0.0),
        "mean_stable_rate_unit_s": float(summary.get("mean_stable_rate_unit_s") or 0.0),
        "status_counts": summary.get("status_counts") or {},
        "eta_source": "tqdm/progress",
    }


def _summary_measurement_valid_for_eta(summary: Mapping[str, Any]) -> bool:
    if bool(summary.get("measurement_valid")):
        return True
    if summary.get("require_stable_rate") is not True:
        return False
    running_count = int(summary.get("running_count") or 0)
    if running_count <= 0:
        return False
    if summary.get("placement_valid") is False:
        return False
    if summary.get("all_stable_rate_ready") is not True:
        return False
    if int(summary.get("stable_rate_ready_count") or 0) < running_count:
        return False
    return float(summary.get("aggregate_stable_rate_unit_s") or 0.0) > 0.0


def _add_result_to_cache(cache: ServiceRateCache, result: Mapping[str, Any]) -> None:
    capacity_boundary_profiles = _capacity_boundary_profiles(result.get("profile_rows") or [])
    for profile_row in result.get("profile_rows") or []:
        path = Path(str(profile_row.get("summary_path") or ""))
        if not path.exists():
            continue
        for record in records_from_summary_file(
            path,
            workload_key=str(result.get("workload_key")),
            command_fingerprint="live_tqdm_service_cache_v2_full_factorial_20260630",
            resource_kind=str(result.get("family") or "gpu"),
            total_units=_total_units_for(str(result.get("workload_key"))),
            node_bucket=str(result.get("node_bucket")),
            workload_env=str(result.get("workload_env")),
            resource_state=str(result.get("resource_state")),
            resident_mix=str(result.get("resident_mix") or ""),
            eta_source="tqdm/progress",
            hardware_class=str(result.get("hardware_group")),
        ):
            if record.capacity_boundary and int(record.profile) not in capacity_boundary_profiles:
                continue
            cache.add(record, force_replace=True)


def _capacity_boundary_profiles(profile_rows: Iterable[Mapping[str, Any]]) -> set[int]:
    """Return profiles that form a monotone high-load failure tail.

    Capacity boundaries are allowed to close only the first profile in a
    high-profile tail where every measured profile at or above the boundary is
    unusable.  If a lower profile is unstable while a higher profile is valid,
    that row is a re-probe candidate, not a capacity cap.
    """

    rows = [
        (
            int(row.get("profile") or 0),
            bool(row.get("measurement_valid")),
            _profile_row_has_progress(row),
        )
        for row in profile_rows
        if int(row.get("profile") or 0) > 0
    ]
    if not rows:
        return set()
    ordered = sorted(rows)
    out: set[int] = set()
    for profile, valid, has_progress in ordered:
        if valid:
            continue
        has_valid_higher = any(other_profile > profile and other_valid for other_profile, other_valid, _ in ordered)
        if not has_valid_higher and has_progress:
            out.add(profile)
            break
    return out


def _profile_row_has_progress(row: Mapping[str, Any]) -> bool:
    if int(row.get("stable_rate_ready_count") or 0) > 0:
        return True
    if float(row.get("aggregate_stable_rate_unit_s") or 0.0) > 0.0:
        return True
    if float(row.get("mean_stable_rate_unit_s") or 0.0) > 0.0:
        return True
    return False


def _parse_profiles(text: Any) -> list[int]:
    raw = str(text or "").replace(";", ",")
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return sorted(dict.fromkeys(out))


def _safe_id(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_")[:180]


def _unit_for(workload_key: str) -> str:
    return "step" if "matmul" in workload_key else "iter"


def _total_units_for(workload_key: str) -> float:
    if "llm" in workload_key:
        return 128.0
    if "matmul" in workload_key:
        return 160.0
    if "cnn" in workload_key:
        return 240.0
    return 200.0


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full-Factorial ETA Probe Runner",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Allow launch: `{str(bool(report.get('allow_launch'))).lower()}`",
        f"- Selected rows: `{report.get('selected_count')}`",
        f"- Launched rows: `{report.get('launched_count')}`",
        f"- Admitted rows: `{report.get('admitted_count')}`",
        f"- Boundary rows: `{report.get('boundary_count')}`",
        "",
        "| Row | Node | Workload | Env | State | Profiles | Valid | Boundary profiles | Rates |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        rates = []
        for profile_row in row.get("profile_rows") or []:
            rates.append(
                f"p{int(profile_row.get('profile') or 0)}={float(profile_row.get('aggregate_stable_rate_unit_s') or 0.0):.6g}"
            )
        lines.append(
            f"| `{row.get('row_id')}` | `{row.get('node')}` | `{row.get('workload_key')}` | "
            f"`{row.get('workload_env')}` | `{row.get('resource_state')}` | `{row.get('profiles')}` | "
            f"{str(bool(row.get('measurement_valid'))).lower()} | `{row.get('capacity_boundary_profiles') or []}` | "
            f"`{', '.join(rates)}` |"
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
    parser.add_argument("--run-prefix", default="full_factorial_eta_probe_20260630")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "full_factorial_eta_probe_runner_20260630.json")
    parser.add_argument("--cache-output", type=Path, default=ARTIFACT_ROOT / "service_cache_v2_full_factorial_eta_probe_20260630.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "full_factorial_eta_probe_runner_20260630.md")
    args = parser.parse_args()
    report = build_full_factorial_eta_probe_runner(
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
