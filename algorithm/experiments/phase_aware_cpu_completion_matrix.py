"""Parallel natural-completion calibration for the 192-core CPU fabric.

This matrix is separate from the steady-rate full-factorial probes.  Every
launched child runs to natural completion through ``progress_wrapper`` and
therefore records startup, outer-loop, checkpoint, and final-save time.  The
six homogeneous nodes receive distinct cells; a later repeat rotates cells
across nodes to provide an independent hardware-equivalence holdout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable

from .remote_workload_selected_profile_probe import (
    RUN_ROOT,
    build_remote_workload_selected_profile_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "algorithm" / "experiments" / "templates" / "cpu_parallel_completion.cmd.tpl"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "md"
    / "experiment_artifacts"
    / "phase_aware_cpu_completion_matrix_20260727.json"
)
RESOURCE_STATE_PREFIX = "ScheduleurmResourceState "


@dataclass(frozen=True)
class CpuCompletionCell:
    cell_id: str
    workload_key: str
    workload_label: str
    node: str
    workers: int
    steps: int
    work_items: int
    checkpoint_interval: int
    checkpoint_bytes: int
    resource_state: str
    observed_external_load: float
    storage_class: str = "shared_home"


def default_cells(*, repeat_index: int = 0) -> tuple[CpuCompletionCell, ...]:
    """Return six distinct cells, rotated for cross-node holdout repeats."""

    specs = (
        ("light_p1", "light_control_local", "light", 1, 160, 8000, 40, 4 * 1024 * 1024, "empty", 0.0),
        ("cpu_p32", "cpu_heavy_local_bench", "cpu", 32, 100, 90000, 25, 8 * 1024 * 1024, "empty", 0.0),
        ("freqduet_p96", "freqduet_cpu_ablation_c17_32", "freqduet_surrogate", 96, 80, 130000, 20, 8 * 1024 * 1024, "empty", 0.0),
        ("sumo_p64_external", "sumo_eval_cpu", "sumo_surrogate", 64, 100, 70000, 25, 8 * 1024 * 1024, "cpu_resident_external", 48.1),
        ("cpu_p160", "cpu_heavy_local_bench", "cpu", 160, 80, 90000, 20, 16 * 1024 * 1024, "full_loaded", 0.0),
        ("freqduet_p128_external", "freqduet_cpu_ablation_c17_32", "freqduet_surrogate", 128, 80, 130000, 20, 16 * 1024 * 1024, "cpu_resident_external", 28.6),
    )
    nodes = ("node001", "node002", "node003", "node004", "node005", "node006")
    rotation = int(repeat_index) % len(nodes)
    assigned_nodes = nodes[rotation:] + nodes[:rotation]
    return tuple(
        CpuCompletionCell(
            cell_id=cell_id,
            workload_key=workload_key,
            workload_label=label,
            node=node,
            workers=workers,
            steps=steps,
            work_items=work_items,
            checkpoint_interval=checkpoint_interval,
            checkpoint_bytes=checkpoint_bytes,
            resource_state=resource_state,
            observed_external_load=observed_load,
        )
        for node, (
            cell_id,
            workload_key,
            label,
            workers,
            steps,
            work_items,
            checkpoint_interval,
            checkpoint_bytes,
            resource_state,
            observed_load,
        ) in zip(assigned_nodes, specs)
    )


def edge_cells(*, repeat_index: int = 0) -> tuple[CpuCompletionCell, ...]:
    """p1/high-allocation edge cells omitted from the first calibration wave."""

    specs = (
        ("light_p14", "light_control_local", "light", 14, 160, 8000, 40, 4 * 1024 * 1024, "full_loaded", 0.0),
        ("cpu_p1", "cpu_heavy_local_bench", "cpu", 1, 120, 90000, 30, 8 * 1024 * 1024, "empty", 0.0),
        ("freqduet_p1", "freqduet_cpu_ablation_c17_32", "freqduet_surrogate", 1, 120, 130000, 30, 8 * 1024 * 1024, "empty", 0.0),
        ("sumo_p128_external", "sumo_eval_cpu", "sumo_surrogate", 128, 80, 70000, 20, 16 * 1024 * 1024, "cpu_resident_external", 48.1),
        ("sumo_p1", "sumo_eval_cpu", "sumo_surrogate", 1, 120, 70000, 30, 8 * 1024 * 1024, "empty", 0.0),
        ("cpu_p1_external", "cpu_heavy_local_bench", "cpu", 1, 120, 90000, 30, 8 * 1024 * 1024, "cpu_resident_external", 28.6),
    )
    nodes = ("node001", "node002", "node003", "node004", "node005", "node006")
    rotation = int(repeat_index) % len(nodes)
    assigned_nodes = nodes[rotation:] + nodes[:rotation]
    return tuple(
        CpuCompletionCell(
            cell_id=cell_id,
            workload_key=workload_key,
            workload_label=label,
            node=node,
            workers=workers,
            steps=steps,
            work_items=work_items,
            checkpoint_interval=checkpoint_interval,
            checkpoint_bytes=checkpoint_bytes,
            resource_state=resource_state,
            observed_external_load=observed_load,
        )
        for node, (
            cell_id,
            workload_key,
            label,
            workers,
            steps,
            work_items,
            checkpoint_interval,
            checkpoint_bytes,
            resource_state,
            observed_load,
        ) in zip(assigned_nodes, specs)
    )


def allocation_curve_cells(
    *,
    workload: str,
    repeat_index: int = 0,
) -> tuple[CpuCompletionCell, ...]:
    definitions = {
        "cpu": (
            "cpu_heavy_local_bench",
            "cpu",
            90000,
            (2, 8, 16, 64, 96, 128),
        ),
        "freqduet": (
            "freqduet_cpu_ablation_c17_32",
            "freqduet_surrogate",
            130000,
            (2, 8, 16, 32, 64, 160),
        ),
        "sumo": (
            "sumo_eval_cpu",
            "sumo_surrogate",
            70000,
            (2, 8, 16, 32, 64, 96),
        ),
    }
    workload_key, label, work_items, worker_counts = definitions[workload]
    nodes = ("node001", "node002", "node003", "node004", "node005", "node006")
    rotation = int(repeat_index) % len(nodes)
    assigned_nodes = nodes[rotation:] + nodes[:rotation]
    rows = []
    for node, workers in zip(assigned_nodes, worker_counts):
        external_load = 48.1 if node == "node004" else (28.6 if node == "node006" else 0.0)
        state = "cpu_resident_external" if external_load > 0.0 else "empty"
        rows.append(
            CpuCompletionCell(
                cell_id=f"{workload}_p{workers}{'_external' if external_load else ''}",
                workload_key=workload_key,
                workload_label=label,
                node=node,
                workers=workers,
                steps=160,
                work_items=work_items,
                checkpoint_interval=40,
                checkpoint_bytes=8 * 1024 * 1024,
                resource_state=state,
                observed_external_load=external_load,
            )
        )
    return tuple(rows)


def allocation_curve_tail_cells(
    *,
    repeat_index: int = 0,
) -> tuple[CpuCompletionCell, ...]:
    """Complete the CPU worker ladder and repeat two cross-node anchors.

    The first ``cpu_curve`` wave measures workers 2, 8, 16, 64, 96, and 128.
    This disjoint wave adds workers 4, 32, 180, and 192; workers 64 and 128
    are repeated on different homogeneous nodes to audit node equivalence.
    Resource state is observational here and is assigned from the measured
    dispatch/run-window telemetry, never inferred from the node name.
    """

    worker_counts = (4, 32, 180, 192, 64, 128)
    nodes = ("node001", "node002", "node003", "node004", "node005", "node006")
    rotation = int(repeat_index) % len(nodes)
    assigned_nodes = nodes[rotation:] + nodes[:rotation]
    return tuple(
        CpuCompletionCell(
            cell_id=f"cpu_p{workers}_tail_wave",
            workload_key="cpu_heavy_local_bench",
            workload_label="cpu",
            node=node,
            workers=workers,
            steps=160,
            work_items=90_000,
            checkpoint_interval=40,
            checkpoint_bytes=8 * 1024 * 1024,
            resource_state="observed_current_state",
            observed_external_load=0.0,
        )
        for node, workers in zip(assigned_nodes, worker_counts)
    )


def jtl311_allocation_curve_cells(
    *,
    repeat_index: int = 0,
) -> tuple[CpuCompletionCell, ...]:
    """Natural-completion worker curve for the 8-core jtl311linux host."""

    return tuple(
        CpuCompletionCell(
            cell_id=f"jtl311_cpu_p{workers}",
            workload_key="cpu_heavy_local_bench",
            workload_label="cpu",
            node="jtl311linux",
            workers=workers,
            steps=160,
            work_items=1_800_000,
            checkpoint_interval=40,
            checkpoint_bytes=8 * 1024 * 1024,
            resource_state="empty",
            observed_external_load=0.0,
        )
        for workers in (2, 4, 8)
    )


def build_phase_aware_cpu_completion_matrix(
    *,
    allow_launch: bool = False,
    repeat_index: int = 0,
    matrix: str = "core",
    nodes: set[str] | None = None,
    max_parallel: int = 6,
    run_prefix: str = "phase_cpu_completion_20260727",
) -> dict[str, Any]:
    matrix_builders = {
        "core": default_cells,
        "edges": edge_cells,
        "cpu_curve": lambda **kwargs: allocation_curve_cells(workload="cpu", **kwargs),
        "cpu_curve_tail": allocation_curve_tail_cells,
        "freqduet_curve": lambda **kwargs: allocation_curve_cells(workload="freqduet", **kwargs),
        "sumo_curve": lambda **kwargs: allocation_curve_cells(workload="sumo", **kwargs),
        "jtl311_cpu_curve": jtl311_allocation_curve_cells,
    }
    if matrix not in matrix_builders:
        raise ValueError(f"unknown matrix {matrix!r}")
    cells = [
        cell
        for cell in matrix_builders[matrix](repeat_index=repeat_index)
        if not nodes or cell.node in nodes
    ]
    started = time.time()
    if not allow_launch:
        rows = [{**asdict(cell), "launched": False, "status": "MANIFEST"} for cell in cells]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=max(1, min(int(max_parallel), len(cells) or 1))) as pool:
            futures = {
                pool.submit(_run_cell, cell, repeat_index=repeat_index, run_prefix=run_prefix): cell
                for cell in cells
            }
            for future in as_completed(futures):
                cell = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append({
                        **asdict(cell),
                        "launched": True,
                        "status": "EXCEPTION",
                        "measurement_valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        rows.sort(key=lambda row: str(row.get("cell_id") or ""))
    ready = [row for row in rows if bool(row.get("completion_model_ready"))]
    hardware_group = (
        "gpu_rtx2080_8gb_dual_cpu_fast"
        if matrix == "jtl311_cpu_curve"
        else "cpu_hpc_192c"
    )
    node_equivalence_contract = (
        "jtl311linux is its own 8-core CPU-fast hardware class"
        if matrix == "jtl311_cpu_curve"
        else "node001-node006 are homogeneous 3x64-core nodes"
    )
    return {
        "gate": "phase_aware_cpu_completion_matrix",
        "schema_version": 1,
        "allow_launch": bool(allow_launch),
        "repeat_index": int(repeat_index),
        "matrix": matrix,
        "run_prefix": run_prefix,
        "hardware_group": hardware_group,
        "node_equivalence_contract": node_equivalence_contract,
        "selected_count": len(cells),
        "completion_model_ready_count": len(ready),
        "all_completion_models_ready": bool(rows) and len(ready) == len(rows),
        "elapsed_wall_s": time.time() - started,
        "status": "COMPLETE" if allow_launch and len(ready) == len(rows) else ("RUN_FINISHED_WITH_GAPS" if allow_launch else "MANIFEST"),
        "rows": rows,
        "claim_boundary": (
            "These are natural-completion calibration cells. FreqDuet and SUMO rows remain "
            "explicit surrogates until their native checkpointable commands are run. External "
            "resident load is observational and is never relabelled as an empty-state cell."
        ),
    }


def _run_cell(
    cell: CpuCompletionCell,
    *,
    repeat_index: int,
    run_prefix: str,
    deploy_bundle: bool = True,
) -> dict[str, Any]:
    run_id = _safe_id(f"{run_prefix}_r{repeat_index}_{cell.node}_{cell.cell_id}")
    output_root = _completion_output_root(cell.node, run_id)
    probe = build_remote_workload_selected_profile_probe(
        run_id=run_id,
        node=cell.node,
        gpus=[0],
        profiles=[1],
        cwd="/tmp",
        cmd_template_file=str(TEMPLATE),
        output_root=output_root,
        max_iters=cell.steps,
        timeout_s=7200,
        unit="step",
        terminate_on_stable=False,
        require_stable_rate=False,
        require_completion_model=True,
        stable_windows=5,
        min_rate_samples=8,
        stable_cv=0.08,
        stable_rel_delta=0.05,
        stable_skip_samples=5,
        coordinated_profile_launch=False,
        extra_template_values={
            "workers": str(cell.workers),
            "work_items": str(cell.work_items),
            "log_interval": "2",
            "checkpoint_interval": str(cell.checkpoint_interval),
            "checkpoint_bytes": str(cell.checkpoint_bytes),
        },
        deploy_bundle=deploy_bundle,
        survive_transport_disconnect=True,
    )
    summary_path = RUN_ROOT / run_id / "reports" / "profile_1_per_gpu_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    child_rows = summary.get("rows") or []
    child_row = child_rows[0] if child_rows and isinstance(child_rows[0], dict) else {}
    model = child_row.get("completion_model")
    observed_resource_state = _parse_resource_state_log(child_row.get("log_path"))
    external_cpu_core_equiv = (
        float(observed_resource_state.get("external_cpu_core_equiv") or 0.0)
        if observed_resource_state
        else None
    )
    return {
        **asdict(cell),
        # ``observed_external_load`` is retained for schema-v1 compatibility;
        # it is a dispatch-time snapshot, not the measured run-window load.
        "declared_external_load_snapshot": cell.observed_external_load,
        "run_id": run_id,
        "launched": True,
        "status": "READY" if bool(model and model.get("completion_model_ready")) else "NOT_READY",
        "measurement_valid": bool(summary.get("measurement_valid")),
        "completion_model_ready": bool(model and model.get("completion_model_ready")),
        "completion_model": model,
        "observed_resource_state": observed_resource_state,
        "resource_telemetry_ready": observed_resource_state is not None,
        "external_cpu_core_equiv": external_cpu_core_equiv,
        "external_cpu_bucket": (
            _external_cpu_bucket(external_cpu_core_equiv)
            if external_cpu_core_equiv is not None
            else None
        ),
        "child_log_path": child_row.get("log_path"),
        "summary_path": str(summary_path),
        "probe_result": probe,
    }


def _completion_output_root(node: str, run_id: str) -> str:
    if str(node) == "jtl311linux":
        return f"/home/erzhu419/scheduleurm_work/eta_completion/{run_id}"
    return f"/home/zhengliang01/scheduleurm_work/eta_completion/{run_id}"


def _parse_resource_state_log(raw_path: Any) -> dict[str, Any] | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.exists():
        return None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if not line.startswith(RESOURCE_STATE_PREFIX):
            continue
        try:
            payload = json.loads(line[len(RESOURCE_STATE_PREFIX):])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _external_cpu_bucket(core_equiv: float) -> str:
    value = max(0.0, float(core_equiv))
    if value <= 2.0:
        return "external_c0_2"
    if value <= 16.0:
        return "external_c3_16"
    if value <= 32.0:
        return "external_c17_32"
    if value <= 64.0:
        return "external_c33_64"
    return "external_c65p"


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in value)[:180]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _parse_nodes(raw: str) -> set[str] | None:
    values = {item.strip() for item in str(raw or "").split(",") if item.strip()}
    return values or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument(
        "--matrix",
        choices=(
            "core",
            "edges",
            "cpu_curve",
            "cpu_curve_tail",
            "freqduet_curve",
            "sumo_curve",
            "jtl311_cpu_curve",
        ),
        default="core",
    )
    parser.add_argument("--nodes", default="")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--run-prefix", default="phase_cpu_completion_20260727")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_phase_aware_cpu_completion_matrix(
        allow_launch=args.allow_launch,
        repeat_index=args.repeat_index,
        matrix=args.matrix,
        nodes=_parse_nodes(args.nodes),
        max_parallel=args.max_parallel,
        run_prefix=args.run_prefix,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not args.allow_launch or result.get("all_completion_models_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
