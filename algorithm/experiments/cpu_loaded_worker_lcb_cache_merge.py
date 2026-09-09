"""Merge controlled and organic loaded-state CPU LCB rows into cache v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_BASE = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_worker_lcb_20260727.json"
)
DEFAULT_CONTROLLED_GATE = (
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_lcb_gate_20260727.json"
)
DEFAULT_EXTERNAL_GATE = (
    ARTIFACT_ROOT
    / "native_cpu_external_state_worker_lcb_gate_20260727.json"
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_all_state_lcb_20260727.json"
)
NODE_BUCKETS = tuple(
    f"node{index:03d}:cpu_hpc_192c" for index in range(1, 7)
)
CONTROLLED_WORKER_BOUNDARIES = {
    "half_loaded": 128,
    "full_loaded": 16,
}


def build_cpu_loaded_worker_lcb_cache_merge(
    *,
    base_cache_path: Path,
    controlled_gate_path: Path,
    external_gate_path: Path,
) -> dict[str, Any]:
    controlled = _load_passed_gate(controlled_gate_path)
    external = _load_passed_gate(external_gate_path)
    cache = (
        ServiceRateCache.load(base_cache_path)
        if base_cache_path.exists()
        else ServiceRateCache()
    )
    inserted = []
    inserted_boundaries = []

    for row in controlled.get("rows") or []:
        for node_bucket in NODE_BUCKETS:
            record = _profile_record(
                row,
                node_bucket=node_bucket,
                source=controlled_gate_path,
                resident_mix=(
                    f"controlled_resident_workers="
                    f"{int(row.get('resident_workers') or 0)}"
                ),
            )
            cache.add(record, force_replace=True)
            inserted.append(record.snapshot())
    for resource_state, boundary_workers in CONTROLLED_WORKER_BOUNDARIES.items():
        resident_workers = 96 if resource_state == "half_loaded" else 180
        for node_bucket in NODE_BUCKETS:
            boundary = ProfileRecord(
                workload_key="cpu_heavy_local_bench",
                command_fingerprint=(
                    "cpu_parallel_loaded_declared_capacity_boundary_v1"
                ),
                resource_kind="cpu",
                node_bucket=node_bucket,
                profile=1,
                allocation_workers=boundary_workers,
                colocation_count=1,
                unit="step",
                total_units=160.0,
                aggregate_rate=0.0,
                per_task_rates=(),
                source=str(controlled_gate_path),
                capacity_boundary=True,
                workload_env="cpu",
                resource_state=resource_state,
                resident_mix=(
                    f"controlled_resident_workers={resident_workers}"
                ),
                eta_source="declared_192_core_capacity_rule",
                stable_rate_ready=False,
                hardware_class="cpu_hpc_192c",
            )
            cache.add(boundary, force_replace=True)
            inserted_boundaries.append(boundary.snapshot())

    for row in external.get("rows") or []:
        measured_nodes = tuple(
            str(node) for node in row.get("measured_nodes") or ()
        )
        if not measured_nodes:
            raise ValueError("external-state row has no measured node")
        for node in measured_nodes:
            record = _profile_record(
                row,
                node_bucket=f"{node}:cpu_hpc_192c",
                source=external_gate_path,
                resident_mix="organic_external_measured",
            )
            cache.add(record, force_replace=True)
            inserted.append(record.snapshot())

    return {
        "gate": "cpu_loaded_worker_lcb_cache_merge",
        "status": "PASS",
        "base_cache_path": str(base_cache_path),
        "controlled_gate_path": str(controlled_gate_path),
        "external_gate_path": str(external_gate_path),
        "inserted_count": len(inserted),
        "inserted_boundary_count": len(inserted_boundaries),
        "controlled_rows_replicated_to_homogeneous_nodes": True,
        "organic_rows_restricted_to_measured_nodes": True,
        "inserted_rows": inserted,
        "inserted_boundaries": inserted_boundaries,
        "service_cache_snapshot": cache.snapshot(),
        "claim_boundary": (
            "Controlled 96/180-resident rows transfer across the declared "
            "homogeneous node001-node006 hardware class. Organic light/moderate "
            "rows remain attached only to their measured nodes and observed "
            "dispatch regimes. Half/full profiles above the remaining physical "
            "core budget are closed by declared state-scoped capacity boundaries."
        ),
    }


def _load_passed_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("pass"):
        raise ValueError(f"lower-service gate did not pass: {path}")
    return payload


def _profile_record(
    row: Mapping[str, Any],
    *,
    node_bucket: str,
    source: Path,
    resident_mix: str,
) -> ProfileRecord:
    if not row.get("lower_service_valid_on_holdout"):
        raise ValueError("gate contains an uncovered loaded-state row")
    workers = int(row.get("allocation_workers") or 0)
    lower_rate = float(row.get("lower_service_units_per_s") or 0.0)
    resource_state = str(row.get("resource_state") or "")
    if workers <= 0 or lower_rate <= 0.0 or not resource_state:
        raise ValueError("loaded-state row is incomplete")
    return ProfileRecord(
        workload_key="cpu_heavy_local_bench",
        command_fingerprint=(
            "cpu_parallel_loaded_natural_completion_split_conformal_v1"
        ),
        resource_kind="cpu",
        node_bucket=node_bucket,
        profile=1,
        allocation_workers=workers,
        colocation_count=1,
        unit="step",
        total_units=160.0,
        aggregate_rate=lower_rate,
        per_task_rates=(lower_rate,),
        source=str(source),
        workload_env="cpu",
        resource_state=resource_state,
        resident_mix=resident_mix,
        eta_source=(
            "task_native_tqdm_natural_completion_split_conformal_lcb"
        ),
        stable_rate_ready=True,
        hardware_class="cpu_hpc_192c",
        completion_model_ready=True,
        completion_model_sample_count=(
            int(row.get("training_sample_count") or 0)
            + int(row.get("calibration_wave_count") or 0)
            + 1
        ),
        startup_overhead_s=float(row.get("startup_overhead_s") or 0.0),
        completion_unit_s=float(row.get("completion_unit_s") or 0.0),
        finalization_overhead_s=float(
            row.get("finalization_overhead_s") or 0.0
        ),
        checkpoint_observed_s=float(
            row.get("checkpoint_observed_s") or 0.0
        ),
        save_observed_s=float(row.get("save_observed_s") or 0.0),
        completion_total_wall_s=float(row.get("point_eta_s") or 0.0),
        completion_group_total_units=160.0,
        completion_model_relative_error=float(
            row.get("point_relative_error") or 0.0
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--controlled-gate",
        type=Path,
        default=DEFAULT_CONTROLLED_GATE,
    )
    parser.add_argument(
        "--external-gate",
        type=Path,
        default=DEFAULT_EXTERNAL_GATE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=(
            ARTIFACT_ROOT / "cpu_loaded_worker_lcb_cache_merge_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_cpu_loaded_worker_lcb_cache_merge(
        base_cache_path=args.base_cache,
        controlled_gate_path=args.controlled_gate,
        external_gate_path=args.external_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report["service_cache_snapshot"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_without_snapshot = {
        key: value
        for key, value in report.items()
        if key != "service_cache_snapshot"
    }
    args.report_output.write_text(
        json.dumps(report_without_snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
