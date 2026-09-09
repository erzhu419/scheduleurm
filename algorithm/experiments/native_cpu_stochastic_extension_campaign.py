"""State-stratified extension after an inspected CPU ETA holdout."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

from .native_cpu_workload_completion_matrix import (
    ARTIFACT_ROOT,
    RUN_ROOT,
    _deploy_wrapper,
    build_native_cpu_workload_completion_matrix,
    smoke_cells,
)
from .phase_aware_stochastic_lcb_gate import (
    build_stochastic_completion_lcb_gate,
)


def run_extension_campaign(
    *,
    previous_tag: str,
    extension_tag: str,
    allow_launch: bool,
    additional_calibration_count: int = 6,
    max_parallel: int = 6,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Add exact-state calibration and one new untouched holdout.

    The previously inspected holdout is explicitly downgraded to training. It
    is never reused as validation evidence.
    """

    started = time.time()
    additional_count = max(1, int(additional_calibration_count))
    base_training = sorted(
        ARTIFACT_ROOT.glob(
            f"native_cpu_stochastic_training_r*_{previous_tag}.json"
        )
    )
    inspected_holdout = (
        ARTIFACT_ROOT
        / f"native_cpu_stochastic_holdout_r01_{previous_tag}.json"
    )
    base_calibration = sorted(
        ARTIFACT_ROOT.glob(
            f"native_cpu_stochastic_calibration_r*_{previous_tag}.json"
        )
    )
    prerequisites = [*base_training, inspected_holdout, *base_calibration]
    missing = [str(path) for path in prerequisites if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "extension campaign prerequisites are missing: "
            + ", ".join(missing)
        )

    deployment_rows = (
        _deploy_campaign_wrappers(
            tag=extension_tag,
            max_parallel=max_parallel,
        )
        if allow_launch
        else []
    )
    deployment_ready = bool(
        not allow_launch
        or (
            deployment_rows
            and all(row.get("ready") for row in deployment_rows)
        )
    )
    waves = []
    for index in range(1, additional_count + 1):
        waves.append(
            {
                "role": "calibration",
                "replicate": index,
                "ordinal": index,
                "seed_offset": (13 + index) * 1000,
                "run_id": (
                    f"native_cpu_stochastic_extension_calibration_"
                    f"r{index:02d}_{extension_tag}"
                ),
                "output": (
                    ARTIFACT_ROOT
                    / (
                        "native_cpu_stochastic_extension_calibration_"
                        f"r{index:02d}_{extension_tag}.json"
                    )
                ),
            }
        )
    waves.append(
        {
            "role": "holdout",
            "replicate": 1,
            "ordinal": additional_count + 1,
            "seed_offset": (14 + additional_count) * 1000,
            "run_id": (
                f"native_cpu_stochastic_extension_holdout_r01_"
                f"{extension_tag}"
            ),
            "output": (
                ARTIFACT_ROOT
                / (
                    "native_cpu_stochastic_extension_holdout_r01_"
                    f"{extension_tag}.json"
                )
            ),
        }
    )

    completed = []
    if deployment_ready:
        for wave in waves:
            output = Path(wave["output"])
            existing = _read_complete(output)
            if existing is not None:
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
                _atomic_write_json(output, payload)
                action = "MANIFEST"
            ready = bool(payload.get("all_completion_models_ready"))
            completed.append(
                {
                    **{
                        key: value
                        for key, value in wave.items()
                        if key != "output"
                    },
                    "output": str(output),
                    "action": action,
                    "ready": ready,
                }
            )
            print(json.dumps(completed[-1], sort_keys=True), flush=True)
            if allow_launch and not ready:
                break

    all_waves_ready = bool(completed) and len(completed) == len(waves) and all(
        row.get("ready") for row in completed
    )
    gate = None
    gate_path = (
        ARTIFACT_ROOT
        / f"native_cpu_stochastic_lcb_gate_{extension_tag}.json"
    )
    if all_waves_ready:
        extension_calibration = [
            Path(row["output"])
            for row in completed
            if row["role"] == "calibration"
        ]
        fresh_holdout = next(
            Path(row["output"])
            for row in completed
            if row["role"] == "holdout"
        )
        gate = build_stochastic_completion_lcb_gate(
            training_paths=[*base_training, inspected_holdout],
            calibration_paths=[*base_calibration, *extension_calibration],
            holdout_path=fresh_holdout,
            miscoverage_alpha=alpha,
        )
        _atomic_write_json(gate_path, gate)

    return {
        "campaign": "native_cpu_stochastic_state_stratified_extension",
        "schema_version": 1,
        "previous_tag": previous_tag,
        "extension_tag": extension_tag,
        "allow_launch": bool(allow_launch),
        "additional_calibration_count": additional_count,
        "previous_holdout_reclassified_as": "training_only",
        "fresh_holdout_required": True,
        "wave_count": len(waves),
        "completed_wave_count": len(completed),
        "all_waves_ready": all_waves_ready,
        "deployment_rows": deployment_rows,
        "deployment_ready": deployment_ready,
        "gate_path": str(gate_path),
        "gate_pass": bool(gate and gate.get("pass")),
        "status": (
            "PASS"
            if gate and gate.get("pass")
            else (
                "GATE_FAIL"
                if gate is not None
                else (
                    "MEASUREMENT_GAP"
                    if allow_launch
                    else "MANIFEST"
                )
            )
        ),
        "elapsed_wall_s": time.time() - started,
        "waves": completed,
        "claim_boundary": (
            "The previous holdout was inspected after the first gate failed and "
            "is therefore used only as training. The reported certificate uses "
            "a newly launched, untouched holdout and state-stratified calibration."
        ),
    }


def _deploy_campaign_wrappers(
    *,
    tag: str,
    max_parallel: int,
) -> list[dict[str, Any]]:
    nodes = sorted({cell.node for cell in smoke_cells()})
    root = RUN_ROOT / f"native_cpu_stochastic_extension_{tag}" / "deploy"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(int(max_parallel), len(nodes)))
    ) as pool:
        futures = {
            pool.submit(_deploy_wrapper, node, root / node): node
            for node in nodes
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
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("all_completion_models_ready") else None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-tag", required=True)
    parser.add_argument("--extension-tag", required=True)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--additional-calibration-count", type=int, default=6)
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = run_extension_campaign(
        previous_tag=args.previous_tag,
        extension_tag=args.extension_tag,
        allow_launch=args.allow_launch,
        additional_calibration_count=args.additional_calibration_count,
        max_parallel=max(1, int(args.max_parallel)),
        alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not args.allow_launch:
        return 0
    return 0 if result.get("gate_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
