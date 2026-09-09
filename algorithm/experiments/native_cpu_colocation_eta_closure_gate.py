"""Closure gate for native FreqDuet/SUMO CPU co-location completion ETA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.native_cpu_colocation_lcb_cache_merge import (
    DEFAULT_GATES,
    DEFAULT_OUTPUT as DEFAULT_CACHE,
    DEFAULT_REPORT as DEFAULT_MERGE_REPORT,
    EXPECTED_NODES,
    EXPECTED_PROFILES,
    EXPECTED_WORKLOADS,
)
from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = (
    ARTIFACT_ROOT
    / "native_cpu_colocation_eta_closure_gate_20260727.json"
)


def build_native_cpu_colocation_eta_closure_gate(
    *,
    gate_paths: Sequence[Path],
    merge_report_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    gates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in gate_paths
    ]
    merge = json.loads(merge_report_path.read_text(encoding="utf-8"))
    cache = ServiceRateCache.load(cache_path)
    expected_keys = {
        (workload_key, node, profile)
        for workload_key in EXPECTED_WORKLOADS
        for node in EXPECTED_NODES
        for profile in EXPECTED_PROFILES
    }
    observed_rows: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    duplicates = []
    for gate in gates:
        for row in gate.get("rows") or ():
            key = (
                str(row.get("workload_key") or ""),
                str(row.get("node") or ""),
                int(row.get("colocation_count") or 0),
            )
            if key in observed_rows:
                duplicates.append(key)
            observed_rows[key] = row

    missing = sorted(expected_keys - set(observed_rows))
    unexpected = sorted(set(observed_rows) - expected_keys)
    exact_rows = []
    exact_errors = []
    for workload_key, node, profile in sorted(expected_keys):
        source = observed_rows.get((workload_key, node, profile))
        if source is None:
            continue
        state = str(
            source.get("observed_effective_resource_state") or ""
        )
        resident_mix = (
            "organic_external_measured"
            if state.endswith("_external")
            else ""
        )
        result = cache.lookup_statewise_exact(
            workload_key,
            workload_env=EXPECTED_WORKLOADS[workload_key],
            node_bucket=f"{node}:cpu_hpc_192c",
            resource_state=state,
            allocation_workers=1,
            colocation_count=profile,
            resident_mix=resident_mix,
        )
        if not result.is_exact or result.record is None:
            exact_errors.append(
                f"missing exact cache tuple {workload_key}/{node}/p{profile}/"
                f"{state}/{resident_mix}"
            )
            continue
        record = result.record
        source_rate = float(
            source.get("simultaneous_lower_service_units_per_s") or 0.0
        )
        record_ready = bool(
            record.stable_rate_ready
            and record.completion_model_ready
            and record.completion_total_wall_s > 0.0
            and record.completion_unit_s > 0.0
            and "history" not in record.eta_source.lower()
            and math.isclose(
                record.aggregate_rate,
                source_rate,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        if not record_ready:
            exact_errors.append(
                f"unready exact cache tuple {workload_key}/{node}/p{profile}"
            )
        exact_rows.append(
            {
                "workload_key": workload_key,
                "workload_env": EXPECTED_WORKLOADS[workload_key],
                "node": node,
                "node_bucket": record.node_bucket,
                "resource_state": state,
                "resident_mix": resident_mix,
                "allocation_workers": record.allocation_workers,
                "colocation_count": record.colocation_count,
                "point_eta_s": record.completion_total_wall_s,
                "lower_service_units_per_s": record.aggregate_rate,
                "completion_model_sample_count": (
                    record.completion_model_sample_count
                ),
                "eta_source": record.eta_source,
                "ready": record_ready,
            }
        )

    checks = {
        "all_source_gates_pass": bool(gates)
        and all(gate.get("pass") for gate in gates),
        "all_simultaneous_bounds_ready": bool(gates)
        and all(
            (gate.get("simultaneous_wave_max_bound") or {}).get("ready")
            for gate in gates
        ),
        "exact_source_matrix": (
            not duplicates
            and not missing
            and not unexpected
            and len(observed_rows) == len(expected_keys)
        ),
        "merge_pass": bool(merge.get("pass"))
        and merge.get("status") == "PASS",
        "merge_inserted_exact_row_count": int(
            merge.get("inserted_count") or 0
        )
        == len(expected_keys),
        "all_cache_exact_tuples_ready": (
            not exact_errors
            and len(exact_rows) == len(expected_keys)
            and all(row["ready"] for row in exact_rows)
        ),
        "no_history_fallback": bool(exact_rows)
        and all(
            "history" not in str(row["eta_source"]).lower()
            for row in exact_rows
        ),
        "legacy_native_profile_index_unpopulated": all(
            cache.get(workload_key, profile) is None
            for workload_key in EXPECTED_WORKLOADS
            for profile in EXPECTED_PROFILES
        ),
    }
    passed = all(checks.values())
    artifacts = {
        "merge_report": _artifact(merge_report_path),
        "cache": _artifact(cache_path),
        "source_gates": [_artifact(path) for path in gate_paths],
    }
    return {
        "gate": "native_cpu_colocation_eta_closure_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "checks": checks,
        "expected_row_count": len(expected_keys),
        "exact_row_count": len(exact_rows),
        "duplicates": duplicates,
        "missing": missing,
        "unexpected": unexpected,
        "exact_errors": exact_errors,
        "profiles": list(EXPECTED_PROFILES),
        "nodes": list(EXPECTED_NODES),
        "workload_keys": sorted(EXPECTED_WORKLOADS),
        "exact_rows": exact_rows,
        "artifacts": artifacts,
        "claim_boundary": (
            "PASS certifies natural-completion ETA and simultaneous "
            "holdout-valid lower service for the exact native FreqDuet/SUMO "
            "2-workload x 6-node x {1,2,4}-colocation matrix. It does not "
            "certify legacy workload keys, unmeasured profiles, other CPU "
            "hardware classes, migration behavior, or future workloads."
        ),
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gates",
        default=",".join(str(path) for path in DEFAULT_GATES),
    )
    parser.add_argument(
        "--merge-report",
        type=Path,
        default=DEFAULT_MERGE_REPORT,
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    gate_paths = tuple(
        Path(item.strip())
        for item in str(args.gates).split(",")
        if item.strip()
    )
    report = build_native_cpu_colocation_eta_closure_gate(
        gate_paths=gate_paths,
        merge_report_path=args.merge_report,
        cache_path=args.cache,
    )
    _atomic_write_json(args.output, report)
    print(args.output)
    return 0 if report.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
