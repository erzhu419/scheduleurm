"""Parallel native FreqDuet/SUMO completion probes on node001-node006."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shlex
import tempfile
import time
from typing import Any, Iterable

from .phase_aware_cpu_completion_matrix import (
    _external_cpu_bucket,
    _parse_resource_state_log,
)
from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    MEASUREMENT_PROTOCOL,
)
from .progress_units import parse_completion_model
from .remote_workload_selected_profile_probe import (
    _run_remote,
    _run_remote_capture,
    _write_remote_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
RUN_ROOT = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"
REMOTE_ROOT = "/tmp/scheduleurm_native_cpu_wrapper_pkg"
REMOTE_EXP_DIR = f"{REMOTE_ROOT}/algorithm/experiments"
WRAPPER_MODULE = "algorithm.experiments.file_progress_completion_wrapper"


@dataclass(frozen=True)
class NativeCpuCell:
    cell_id: str
    node: str
    workload_key: str
    workload_env: str
    kind: str
    total_outer_units: int
    seed: int
    max_inner_steps: int = 0
    resource_state: str = "empty"
    allocation_workers: int = 1
    colocation_count: int = 1


def smoke_cells(*, seed_offset: int = 0) -> tuple[NativeCpuCell, ...]:
    offset = int(seed_offset)
    return (
        NativeCpuCell(
            f"freqduet_node001_seed{727001 + offset}",
            "node001",
            "freqduet_cpu_native",
            "freqduet",
            "freqduet",
            6,
            727001 + offset,
        ),
        NativeCpuCell(
            f"freqduet_node002_seed{727001 + offset}",
            "node002",
            "freqduet_cpu_native",
            "freqduet",
            "freqduet",
            6,
            727001 + offset,
        ),
        NativeCpuCell(
            f"freqduet_node003_seed{727001 + offset}",
            "node003",
            "freqduet_cpu_native",
            "freqduet",
            "freqduet",
            6,
            727001 + offset,
        ),
        NativeCpuCell(
            f"sumo_node004_seed{727004 + offset}",
            "node004",
            "sumo_eval_cpu_native",
            "sumo",
            "sumo",
            6,
            727004 + offset,
            600,
        ),
        NativeCpuCell(
            f"sumo_node005_seed{727004 + offset}",
            "node005",
            "sumo_eval_cpu_native",
            "sumo",
            "sumo",
            6,
            727004 + offset,
            600,
        ),
        NativeCpuCell(
            f"sumo_node006_seed{727004 + offset}",
            "node006",
            "sumo_eval_cpu_native",
            "sumo",
            "sumo",
            6,
            727004 + offset,
            600,
        ),
    )


def build_native_cpu_workload_completion_matrix(
    *,
    allow_launch: bool = False,
    run_id: str = "native_cpu_workload_smoke_20260727",
    nodes: set[str] | None = None,
    kinds: set[str] | None = None,
    max_parallel: int = 6,
    seed_offset: int = 0,
    init_cache_state: str = "unspecified",
    deploy_wrapper: bool = True,
) -> dict[str, Any]:
    cells = [
        cell
        for cell in smoke_cells(seed_offset=seed_offset)
        if (not nodes or cell.node in nodes) and (not kinds or cell.kind in kinds)
    ]
    started = time.time()
    if not allow_launch:
        rows = [
            {
                **asdict(cell),
                "calibration_cell_id": _calibration_cell_id(cell),
                "init_cache_state": str(init_cache_state),
                "launched": False,
                "status": "MANIFEST",
            }
            for cell in cells
        ]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(cells) or 1))) as pool:
            futures = {
                pool.submit(
                    _run_cell,
                    cell,
                    run_id=run_id,
                    init_cache_state=init_cache_state,
                    deploy_wrapper=deploy_wrapper,
                ): cell
                for cell in cells
            }
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            **asdict(cell),
                            "calibration_cell_id": _calibration_cell_id(cell),
                            "init_cache_state": str(init_cache_state),
                            "launched": True,
                            "status": "EXCEPTION",
                            "measurement_valid": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        rows.sort(key=lambda row: str(row.get("cell_id") or ""))
    ready = [row for row in rows if bool(row.get("measurement_valid"))]
    return {
        "gate": "native_cpu_workload_completion_matrix",
        "schema_version": 2,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "dispatch_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
        "run_id": run_id,
        "seed_offset": int(seed_offset),
        "init_cache_state": str(init_cache_state),
        "deploy_wrapper": bool(deploy_wrapper),
        "allow_launch": bool(allow_launch),
        "selected_count": len(cells),
        "admitted_count": len(ready),
        "all_completion_models_ready": bool(rows) and len(ready) == len(rows),
        "status": (
            "COMPLETE"
            if allow_launch and len(ready) == len(rows)
            else ("RUN_FINISHED_WITH_GAPS" if allow_launch else "MANIFEST")
        ),
        "elapsed_wall_s": time.time() - started,
        "rows": rows,
        "claim_boundary": (
            "This smoke wave validates native runtime, durable outer-loop progress, "
            "and end-to-end completion accounting. It is not yet a co-location "
            "service curve or a production-length FreqDuet/SUMO result."
        ),
    }


def _run_cell(
    cell: NativeCpuCell,
    *,
    run_id: str,
    init_cache_state: str = "unspecified",
    deploy_wrapper: bool = True,
) -> dict[str, Any]:
    cell_run_id = _safe_id(f"{run_id}_{cell.cell_id}")
    run_dir = RUN_ROOT / cell_run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if deploy_wrapper:
        _deploy_wrapper(cell.node, raw_dir)
    if cell.kind == "freqduet":
        python_path, progress_glob, artifact_globs, child_script = _freqduet_command(
            cell, run_id=cell_run_id
        )
    elif cell.kind == "sumo":
        python_path, progress_glob, artifact_globs, child_script = _sumo_command(
            cell, run_id=cell_run_id
        )
    else:
        raise ValueError(f"unsupported native CPU kind {cell.kind!r}")
    wrapper_args = [
        f"PYTHONPATH={shlex.quote(REMOTE_ROOT)}",
        shlex.quote(python_path),
        "-u",
        "-m",
        WRAPPER_MODULE,
        "--progress-file-glob",
        shlex.quote(progress_glob),
        "--total",
        str(cell.total_outer_units),
        "--unit",
        "episode",
        "--poll-s",
        "0.5",
        "--csv-has-header",
    ]
    for pattern in artifact_globs:
        wrapper_args.extend(["--artifact-glob", shlex.quote(pattern)])
    wrapper_args.extend(["--", "bash", "-lc", shlex.quote(child_script)])
    rc, out, err = _run_remote_capture(
        cell.node,
        " ".join(wrapper_args),
        raw_dir / "native_completion",
        timeout_s=7200,
    )
    log_path = raw_dir / "native_completion.log"
    log_path.write_text(
        (out or "") + ("\nSTDERR:\n" + err if err else ""),
        encoding="utf-8",
    )
    completion_model = parse_completion_model(out)
    resource_state = _parse_resource_state_log(log_path)
    run_window_external_core_equiv = (
        float(resource_state.get("external_cpu_core_equiv") or 0.0)
        if resource_state
        else None
    )
    dispatch_external_core_equiv = (
        float(resource_state.get("dispatch_external_cpu_core_equiv") or 0.0)
        if resource_state and resource_state.get("dispatch_state_ready")
        else None
    )
    dispatch_external_regime = (
        str(resource_state.get("dispatch_external_cpu_regime") or "")
        if resource_state and resource_state.get("dispatch_state_ready")
        else ""
    )
    measurement_protocol = (
        str(resource_state.get("measurement_protocol") or "")
        if resource_state
        else ""
    )
    dispatch_sample_count = (
        int(resource_state.get("dispatch_sample_count") or 0)
        if resource_state
        else 0
    )
    valid = bool(
        int(rc) == 0
        and completion_model
        and completion_model.get("completion_model_ready")
        and completion_model.get("natural_exit")
        and int(completion_model.get("child_returncode") or 0) == 0
        and measurement_protocol == MEASUREMENT_PROTOCOL
        and dispatch_sample_count >= DISPATCH_CPU_SAMPLE_COUNT
        and dispatch_external_regime
    )
    observed_effective_state = (
        "cpu_resident_external"
        if dispatch_external_regime != "cpu_external_idle"
        else cell.resource_state
    )
    run_window_effective_state = (
        "cpu_resident_external"
        if (
            run_window_external_core_equiv is not None
            and run_window_external_core_equiv > 2.0
        )
        else cell.resource_state
    )
    return {
        **asdict(cell),
        "calibration_cell_id": _calibration_cell_id(cell),
        "init_cache_state": str(init_cache_state),
        "run_id": cell_run_id,
        "launched": True,
        "returncode": int(rc),
        "status": "READY" if valid else "NOT_READY",
        "measurement_valid": valid,
        "completion_model_ready": bool(
            completion_model and completion_model.get("completion_model_ready")
        ),
        "completion_model": completion_model,
        "resource_telemetry_ready": resource_state is not None,
        "dispatch_state_ready": bool(
            resource_state and resource_state.get("dispatch_state_ready")
        ),
        "dispatch_sample_count": dispatch_sample_count,
        "dispatch_logical_cpu_count": (
            resource_state.get("dispatch_logical_cpu_count")
            if resource_state
            else None
        ),
        "dispatch_external_cpu_regime": dispatch_external_regime or None,
        "dispatch_external_cpu_fraction": (
            resource_state.get("dispatch_external_cpu_fraction")
            if resource_state
            else None
        ),
        "dispatch_external_cpu_fraction_p95": (
            resource_state.get("dispatch_external_cpu_fraction_p95")
            if resource_state
            else None
        ),
        "dispatch_external_cpu_core_equiv": dispatch_external_core_equiv,
        "dispatch_external_cpu_core_equiv_p95": (
            resource_state.get("dispatch_external_cpu_core_equiv_p95")
            if resource_state
            else None
        ),
        "dispatch_external_cpu_bucket": (
            _external_cpu_bucket(dispatch_external_core_equiv)
            if dispatch_external_core_equiv is not None
            else None
        ),
        "observed_resource_state": resource_state,
        "observed_effective_resource_state": observed_effective_state,
        "run_window_effective_resource_state": run_window_effective_state,
        "requested_resource_state_matches_observation": (
            observed_effective_state == cell.resource_state
        ),
        "run_window_external_cpu_core_equiv": run_window_external_core_equiv,
        "run_window_external_cpu_bucket": (
            _external_cpu_bucket(run_window_external_core_equiv)
            if run_window_external_core_equiv is not None
            else None
        ),
        # Compatibility aliases retain their historical run-window meaning.
        "external_cpu_core_equiv": run_window_external_core_equiv,
        "external_cpu_bucket": (
            _external_cpu_bucket(run_window_external_core_equiv)
            if run_window_external_core_equiv is not None
            else None
        ),
        "measurement_protocol": measurement_protocol or None,
        "eta_source": "task_native_durable_csv",
        "legacy_scheduler_limits_bypassed": True,
        "log_path": str(log_path),
        "stderr_tail": (err or "")[-1000:],
    }


def _deploy_wrapper(node: str, raw_dir: Path) -> None:
    _run_remote(
        node,
        (
            f"mkdir -p {shlex.quote(REMOTE_EXP_DIR)} && "
            f"touch {shlex.quote(REMOTE_ROOT + '/algorithm/__init__.py')} "
            f"{shlex.quote(REMOTE_EXP_DIR + '/__init__.py')}"
        ),
        raw_dir / "deploy_mkdir",
        timeout_s=45,
    )
    local_dir = REPO_ROOT / "algorithm" / "experiments"
    for name in ("progress_units.py", "file_progress_completion_wrapper.py"):
        _write_remote_text(
            node,
            f"{REMOTE_EXP_DIR}/{name}",
            (local_dir / name).read_text(encoding="utf-8"),
            raw_dir / f"deploy_{name}",
        )


def _freqduet_command(
    cell: NativeCpuCell, *, run_id: str
) -> tuple[str, str, tuple[str, ...], str]:
    root = "/home/zhengliang01/scheduleurm_work/TransitDuet/FreqDuet/freqduet"
    python_path = "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python"
    output = f"{root}/results_freqduet/{run_id}"
    progress_glob = f"{output}/logs/*/diagnostics.csv"
    artifact_globs = (f"{output}/logs/*/checkpoints/*.pt", f"{output}/logs/*/*.json")
    script = f"""
set -euo pipefail
cd {shlex.quote(root)}
export CUDA_VISIBLE_DEVICES=
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export TORCH_NUM_THREADS=1 FREQDUET_TORCH_THREADS=1
export PYGAME_HIDE_SUPPORT_PROMPT=1 SDL_VIDEODRIVER=dummy
unset FREQDUET_SUPPRESS_HEAVY_ARTIFACTS || true
test -x {shlex.quote(python_path)}
test -r configs_freqduet/F_freqduet_timetable_hiro.yaml
rm -rf {shlex.quote(output)}
mkdir -p {shlex.quote(output + '/logs')}
exec {shlex.quote(python_path)} -u runner_v3.py \
  --config configs_freqduet/F_freqduet_timetable_hiro.yaml \
  --episodes {int(cell.total_outer_units)} \
  --seed {int(cell.seed)} \
  --upper-warmup-eps 0 \
  --no-resume \
  --logs-dir {shlex.quote(output + '/logs')}
""".strip()
    return python_path, progress_glob, artifact_globs, script


def _sumo_command(
    cell: NativeCpuCell, *, run_id: str
) -> tuple[str, str, tuple[str, ...], str]:
    work = "/home/zhengliang01/scheduleurm_work"
    root = f"{work}/offline-sumo"
    python_path = f"{work}/conda_envs/sumo-lstm-py310/bin/python"
    runtime = f"{work}/sumo-1.25.0/site-packages"
    source = f"{work}/sumo-rl/_standalone_f543609/SUMO_ruiguang"
    output = f"{root}/experiment_output/{run_id}/{cell.node}"
    tasks = [
        {
            "kind": "baseline",
            "method": "nohold",
            "policy_kind": "nohold",
            "policy_seed": 42,
            "sumo_seed": int(cell.seed) + index,
            "episode_idx": index,
            "max_steps": int(cell.max_inner_steps),
        }
        for index in range(int(cell.total_outer_units))
    ]
    tasks_json = json.dumps(tasks, separators=(",", ":"))
    progress_glob = f"{output}/metrics.csv"
    artifact_globs = (f"{output}/metrics.csv", f"{output}/runtime_manifest.json")
    script = f"""
set -euo pipefail
ROOT={shlex.quote(root)}
PY={shlex.quote(python_path)}
RT={shlex.quote(runtime)}
SRC={shlex.quote(source)}
OUT={shlex.quote(output)}
export SUMO_HOME="$RT/sumo"
export PATH="$(dirname "$PY"):$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$RT:${{PYTHONPATH:-}}"
export LIBSUMO_AS_TRACI=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -x "$PY"
test -x "$SUMO_HOME/bin/sumo"
test -r "$SRC/online_control/control_sim_traci_period.sumocfg"
rm -rf "$OUT"
mkdir -p "$OUT"
printf '%s\\n' {shlex.quote(tasks_json)} > "$OUT/tasks.json"
"$PY" -u "$ROOT/scripts/write_run_manifest.py" \
  --output "$OUT/runtime_manifest.json" --expected_sumo 1.25.0 \
  --label {shlex.quote(run_id + '-' + cell.node)}
SB=$(mktemp -d "${{TMPDIR:-/tmp}}/sumo-{cell.node}.XXXXXX")
cleanup() {{ rm -rf -- "$SB"; }}
trap cleanup EXIT
ln -s "$ROOT" "$SB/offline-sumo"
P="$SB/sumo-rl/_standalone_f543609/SUMO_ruiguang"
mkdir -p "$P"
for d in b_network c_social_vehicle_rou d_bus_rou e_passenger_rou; do
  ln -s "$SRC/$d" "$P/$d"
done
cp -a "$SRC/online_control" "$P/"
"$PY" -u "$SB/offline-sumo/eval_revision_metrics.py" \
  --tasks_json "$OUT/tasks.json" --out_csv "$OUT/metrics.csv" \
  --n_workers 1 --expected_sumo_version 1.25.0
""".strip()
    return python_path, progress_glob, artifact_globs, script


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)[:180]


def _calibration_cell_id(cell: NativeCpuCell) -> str:
    return _safe_id(
        "|".join(
            (
                cell.workload_key,
                cell.workload_env,
                cell.node,
                cell.resource_state,
                f"aw{int(cell.allocation_workers)}",
                f"cc{int(cell.colocation_count)}",
                f"u{int(cell.total_outer_units)}",
            )
        )
    )


def _parse_set(raw: str) -> set[str] | None:
    values = {item.strip() for item in str(raw or "").split(",") if item.strip()}
    return values or None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--run-id", default="native_cpu_workload_smoke_20260727")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--kinds", default="")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--init-cache-state",
        choices=("cold", "warm", "unspecified"),
        default="unspecified",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "native_cpu_workload_smoke_20260727.json",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_native_cpu_workload_completion_matrix(
        allow_launch=bool(args.allow_launch),
        run_id=str(args.run_id),
        nodes=_parse_set(args.nodes),
        kinds=_parse_set(args.kinds),
        max_parallel=max(1, int(args.max_parallel)),
        seed_offset=int(args.seed_offset),
        init_cache_state=str(args.init_cache_state),
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not args.allow_launch or result.get("all_completion_models_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
