"""Resumable parallel campaign for native CPU ETA and split-conformal LCBs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

from .native_cpu_workload_completion_matrix import (
    ARTIFACT_ROOT,
    RUN_ROOT,
    _deploy_wrapper,
    build_native_cpu_workload_completion_matrix,
    smoke_cells,
)
from .file_progress_completion_wrapper import (
    DISPATCH_CPU_SAMPLE_COUNT,
    DISPATCH_CPU_SAMPLE_WINDOW_S,
    MEASUREMENT_PROTOCOL,
)
from .phase_aware_stochastic_lcb_gate import build_stochastic_completion_lcb_gate


def campaign_waves(*, tag: str) -> tuple[dict[str, Any], ...]:
    waves: list[dict[str, Any]] = []
    ordinal = 0
    for role, count in (("training", 3), ("calibration", 9), ("holdout", 1)):
        for replicate in range(1, count + 1):
            ordinal += 1
            stem = f"native_cpu_stochastic_{role}_r{replicate:02d}_{tag}"
            waves.append(
                {
                    "role": role,
                    "replicate": replicate,
                    "ordinal": ordinal,
                    "run_id": stem,
                    "seed_offset": ordinal * 1000,
                    "output": ARTIFACT_ROOT / f"{stem}.json",
                }
            )
    return tuple(waves)


def run_campaign(
    *,
    tag: str,
    allow_launch: bool,
    force: bool = False,
    max_parallel: int = 6,
    alpha: float = 0.10,
) -> dict[str, Any]:
    waves = campaign_waves(tag=tag)
    started = time.time()
    deployment_rows = (
        _deploy_campaign_wrappers(tag=tag, max_parallel=max_parallel)
        if allow_launch
        else []
    )
    deployment_ready = bool(
        not allow_launch or (deployment_rows and all(row["ready"] for row in deployment_rows))
    )
    completed: list[dict[str, Any]] = []
    if allow_launch and not deployment_ready:
        return {
            "campaign": "native_cpu_stochastic_calibration",
            "schema_version": 1,
            "tag": tag,
            "allow_launch": True,
            "force": bool(force),
            "miscoverage_alpha": float(alpha),
            "wave_count": len(waves),
            "completed_wave_count": 0,
            "all_waves_ready": False,
            "deployment_rows": deployment_rows,
            "deployment_ready": False,
            "gate_path": "",
            "gate_pass": False,
            "status": "DEPLOYMENT_GAP",
            "elapsed_wall_s": time.time() - started,
            "waves": [],
        }
    for wave in waves:
        output = Path(wave["output"])
        existing = _read_complete(output)
        if existing is not None and not force:
            payload = existing
            action = "REUSED"
        elif allow_launch:
            payload = build_native_cpu_workload_completion_matrix(
                allow_launch=True,
                run_id=str(wave["run_id"]),
                max_parallel=max_parallel,
                seed_offset=int(wave["seed_offset"]),
                init_cache_state="warm",
                deploy_wrapper=False,
            )
            _atomic_write_json(output, payload)
            action = "LAUNCHED"
        else:
            payload = build_native_cpu_workload_completion_matrix(
                allow_launch=False,
                run_id=str(wave["run_id"]),
                max_parallel=max_parallel,
                seed_offset=int(wave["seed_offset"]),
                init_cache_state="warm",
            )
            action = "MANIFEST"
        ready = bool(payload.get("all_completion_models_ready"))
        completed.append(
            {
                **{key: value for key, value in wave.items() if key != "output"},
                "output": str(output),
                "action": action,
                "ready": ready,
                "admitted_count": int(payload.get("admitted_count") or 0),
                "selected_count": int(payload.get("selected_count") or 0),
                "elapsed_wall_s": float(payload.get("elapsed_wall_s") or 0.0),
            }
        )
        print(
            json.dumps(
                {
                    "wave": wave["ordinal"],
                    "role": wave["role"],
                    "replicate": wave["replicate"],
                    "action": action,
                    "ready": ready,
                    "output": str(output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if allow_launch and not ready:
            break

    all_waves_ready = bool(completed) and len(completed) == len(waves) and all(
        bool(row["ready"]) for row in completed
    )
    gate_path = ARTIFACT_ROOT / f"native_cpu_stochastic_lcb_gate_{tag}.json"
    gate: dict[str, Any] | None = None
    if all_waves_ready:
        training_paths = [
            Path(row["output"]) for row in completed if row["role"] == "training"
        ]
        calibration_paths = [
            Path(row["output"]) for row in completed if row["role"] == "calibration"
        ]
        holdout_paths = [
            Path(row["output"]) for row in completed if row["role"] == "holdout"
        ]
        gate = build_stochastic_completion_lcb_gate(
            training_paths=training_paths,
            calibration_paths=calibration_paths,
            holdout_path=holdout_paths[0],
            miscoverage_alpha=alpha,
        )
        _atomic_write_json(gate_path, gate)
    status = (
        "PASS"
        if gate and gate.get("pass")
        else (
            "GATE_FAIL"
            if gate
            else ("MEASUREMENT_GAP" if allow_launch else "MANIFEST")
        )
    )
    return {
        "campaign": "native_cpu_stochastic_calibration",
        "schema_version": 1,
        "tag": tag,
        "allow_launch": bool(allow_launch),
        "force": bool(force),
        "miscoverage_alpha": float(alpha),
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "dispatch_cpu_sample_subwindow_s": DISPATCH_CPU_SAMPLE_WINDOW_S,
        "dispatch_cpu_sample_count": DISPATCH_CPU_SAMPLE_COUNT,
        "paired_seed_within_workload_wave": True,
        "wave_count": len(waves),
        "completed_wave_count": len(completed),
        "all_waves_ready": all_waves_ready,
        "deployment_rows": deployment_rows,
        "deployment_ready": deployment_ready,
        "gate_path": str(gate_path),
        "gate_pass": bool(gate and gate.get("pass")),
        "status": status,
        "elapsed_wall_s": time.time() - started,
        "waves": completed,
        "claim_boundary": (
            "All six node001-node006 lanes are used in every wave. Training, "
            "calibration, and holdout seeds are disjoint, and all admitted rows "
            "use a paired seed across same-workload nodes within each wave. They "
            "must use task-native durable progress in the same dispatch-time load "
            "and warm-cache cell. Full-run external load is retained as an outcome "
            "audit and is never used to select the conformal stratum."
        ),
    }


def _deploy_campaign_wrappers(*, tag: str, max_parallel: int) -> list[dict[str, Any]]:
    nodes = sorted({cell.node for cell in smoke_cells()})
    root = RUN_ROOT / f"native_cpu_stochastic_campaign_{tag}" / "deploy"
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(nodes)))) as pool:
        futures = {
            pool.submit(_deploy_wrapper, node, root / node): node for node in nodes
        }
        for future in as_completed(futures):
            node = futures[future]
            try:
                future.result()
                rows.append({"node": node, "ready": True})
            except Exception as exc:
                rows.append(
                    {
                        "node": node,
                        "ready": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return sorted(rows, key=lambda row: str(row["node"]))


def _read_complete(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("all_completion_models_ready") else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="20260727")
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "native_cpu_stochastic_campaign_20260727.json",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_campaign(
        tag=str(args.tag),
        allow_launch=bool(args.allow_launch),
        force=bool(args.force),
        max_parallel=max(1, int(args.max_parallel)),
        alpha=float(args.alpha),
    )
    _atomic_write_json(args.output, result)
    print(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if not args.allow_launch:
        return 0
    return 0 if result.get("gate_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
