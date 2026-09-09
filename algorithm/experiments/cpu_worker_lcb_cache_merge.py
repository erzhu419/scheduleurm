"""Merge certified CPU worker lower-service/completion rows into cache v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_BASE = ARTIFACT_ROOT / "service_cache_v2_live_merged_20260629.json"
DEFAULT_GATE = (
    ARTIFACT_ROOT / "native_cpu_allocation_worker_lcb_gate_20260727.json"
)
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "service_cache_v2_live_merged_cpu_worker_lcb_20260727.json"
)
NODE_BUCKETS = tuple(
    f"node{index:03d}:cpu_hpc_192c" for index in range(1, 7)
)


def build_cpu_worker_lcb_cache_merge(
    *,
    base_cache_path: Path,
    gate_path: Path,
) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("pass"):
        raise ValueError("worker lower-service gate must pass before cache merge")
    cache = (
        ServiceRateCache.load(base_cache_path)
        if base_cache_path.exists()
        else ServiceRateCache()
    )
    inserted = []
    for row in gate.get("rows") or []:
        if not row.get("lower_service_valid_on_holdout"):
            raise ValueError("gate contains an uncovered worker profile")
        workers = int(row.get("allocation_workers") or 0)
        lower_rate = float(row.get("lower_service_units_per_s") or 0.0)
        if workers <= 0 or lower_rate <= 0.0:
            raise ValueError("gate contains a nonpositive worker/rate row")
        for node_bucket in NODE_BUCKETS:
            record = ProfileRecord(
                workload_key="cpu_heavy_local_bench",
                command_fingerprint=(
                    "cpu_parallel_natural_completion_split_conformal_v1"
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
                source=str(gate_path),
                workload_env="cpu",
                resource_state="empty",
                resident_mix="",
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
                startup_overhead_s=float(
                    row.get("startup_overhead_s") or 0.0
                ),
                completion_unit_s=float(
                    row.get("completion_unit_s") or 0.0
                ),
                finalization_overhead_s=float(
                    row.get("finalization_overhead_s") or 0.0
                ),
                checkpoint_observed_s=float(
                    row.get("checkpoint_observed_s") or 0.0
                ),
                save_observed_s=float(
                    row.get("save_observed_s") or 0.0
                ),
                completion_total_wall_s=float(
                    row.get("point_eta_s") or 0.0
                ),
                completion_group_total_units=160.0,
                completion_model_relative_error=float(
                    row.get("point_relative_error") or 0.0
                ),
            )
            cache.add(record, force_replace=True)
            inserted.append(record.snapshot())
    return {
        "gate": "cpu_worker_lcb_cache_merge",
        "status": "PASS",
        "base_cache_path": str(base_cache_path),
        "worker_gate_path": str(gate_path),
        "inserted_count": len(inserted),
        "inserted_rows": inserted,
        "service_cache_snapshot": cache.snapshot(),
        "claim_boundary": (
            "Rows are replicated across node001-node006 only under the declared "
            "homogeneous-hardware contract. Node-specific loaded states remain "
            "separate and are not inferred from these empty-state records."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=(
            ARTIFACT_ROOT / "cpu_worker_lcb_cache_merge_20260727.json"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_cpu_worker_lcb_cache_merge(
        base_cache_path=args.base_cache,
        gate_path=args.gate,
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
