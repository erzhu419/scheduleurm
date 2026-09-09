"""Top-level machine-readable closure gate for the 192-core CPU ETA matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_EMPTY_GATE = (
    ARTIFACT_ROOT / "native_cpu_allocation_worker_lcb_gate_20260727.json"
)
DEFAULT_CONTROLLED_GATE = (
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_lcb_gate_20260727.json"
)
DEFAULT_EXTERNAL_GATE = (
    ARTIFACT_ROOT
    / "native_cpu_external_state_worker_lcb_gate_20260727.json"
)
DEFAULT_MERGE_REPORT = (
    ARTIFACT_ROOT / "cpu_loaded_worker_lcb_cache_merge_20260727.json"
)
DEFAULT_DESIGN = (
    ARTIFACT_ROOT
    / "full_factorial_eta_design_cpu_native_all_state_lcb_20260727.json"
)
DEFAULT_NATIVE_CLOSURE = (
    ARTIFACT_ROOT
    / "native_cpu_colocation_eta_closure_gate_20260727.json"
)
FORMAL_STATES = {
    ("empty", "none"),
    ("half_loaded", "controlled_resident_workers=96"),
    ("full_loaded", "controlled_resident_workers=180"),
}


def build_cpu_eta_closure_gate(
    *,
    empty_gate_path: Path,
    controlled_gate_path: Path,
    external_gate_path: Path,
    merge_report_path: Path,
    design_path: Path,
    native_closure_path: Path | None = None,
) -> dict[str, Any]:
    paths = {
        "empty_gate": empty_gate_path,
        "controlled_gate": controlled_gate_path,
        "external_gate": external_gate_path,
        "merge_report": merge_report_path,
        "factorial_design": design_path,
    }
    if native_closure_path is not None:
        paths["native_colocation_closure"] = native_closure_path
    payloads = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in paths.items()
    }
    artifacts = {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for key, path in paths.items()
    }
    empty = payloads["empty_gate"]
    controlled = payloads["controlled_gate"]
    external = payloads["external_gate"]
    merge = payloads["merge_report"]
    design = payloads["factorial_design"]
    native_closure = payloads.get("native_colocation_closure")

    formal_rows = [
        row
        for row in design.get("eta_rows") or []
        if row.get("hardware_group") == "cpu_hpc_192c"
        and row.get("workload_key") == "cpu_heavy_local_bench"
        and (row.get("resource_state"), row.get("resident_mix"))
        in FORMAL_STATES
    ]
    formal_by_state = {
        (str(row["resource_state"]), str(row["resident_mix"])): row
        for row in formal_rows
    }
    design_ready = bool(
        set(formal_by_state) == FORMAL_STATES
        and all(not str(row.get("missing_profiles") or "") for row in formal_rows)
        and all(
            not str(row.get("eta_missing_profiles") or "")
            and not str(row.get("service_only_profiles") or "")
            for row in formal_rows
        )
        and all(
            row.get("status")
            in {"measured", "measured_to_capacity_boundary"}
            for row in formal_rows
        )
        and all(
            row.get("eta_status")
            in {"eta_measured", "eta_measured_to_capacity_boundary"}
            for row in formal_rows
        )
    )
    inserted_rows = merge.get("inserted_rows") or []
    no_history_fallback = bool(inserted_rows) and all(
        "history" not in str(row.get("eta_source") or "").lower()
        and bool(row.get("stable_rate_ready"))
        and bool(row.get("completion_model_ready"))
        for row in inserted_rows
    )
    checks = {
        "empty_worker_gate_pass": bool(empty.get("pass")),
        "controlled_worker_gate_pass": bool(controlled.get("pass")),
        "external_worker_gate_pass": bool(external.get("pass")),
        "empty_expected_cell_count": len(empty.get("rows") or []) == 11,
        "controlled_expected_cell_count": (
            len(controlled.get("rows") or []) == 12
        ),
        "external_expected_cell_count": len(external.get("rows") or []) == 16,
        "all_holdout_lower_service_valid": all(
            bool(row.get("lower_service_valid_on_holdout"))
            for gate in (empty, controlled, external)
            for row in gate.get("rows") or []
        ),
        "no_missing_duplicate_or_wrong_node": all(
            not gate.get(field)
            for gate in (empty, controlled, external)
            for field in (
                "missing_cells",
                "duplicate_cells",
                "node_mismatches",
            )
        ),
        "cache_merge_pass": merge.get("status") == "PASS",
        "loaded_cache_inserted_rows": int(merge.get("inserted_count") or 0)
        == 88,
        "state_scoped_capacity_boundaries": int(
            merge.get("inserted_boundary_count") or 0
        )
        == 12,
        "no_history_fallback_in_loaded_rows": no_history_fallback,
        "node001_formal_states_closed": design_ready,
    }
    if native_closure is not None:
        checks["native_colocation_2x6x3_closure_pass"] = bool(
            native_closure.get("pass")
            and native_closure.get("status") == "PASS"
            and int(native_closure.get("exact_row_count") or 0) == 36
        )
    passed = all(checks.values())
    return {
        "gate": "cpu_eta_closure_gate",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "checks": checks,
        "artifacts": artifacts,
        "formal_factorial_rows": formal_rows,
        "metrics": {
            "empty_point_max_relative_error": empty.get(
                "point_max_relative_error"
            ),
            "controlled_point_max_relative_error": controlled.get(
                "point_max_relative_error"
            ),
            "external_point_max_relative_error": external.get(
                "point_max_relative_error"
            ),
            "empty_simultaneous_ratio_margin": empty.get(
                "simultaneous_ratio_margin"
            ),
            "controlled_simultaneous_ratio_margin": controlled.get(
                "simultaneous_ratio_margin"
            ),
            "external_simultaneous_ratio_margin": external.get(
                "simultaneous_ratio_margin"
            ),
        },
        "claim_boundary": (
            "PASS closes the declared node001-node006 192-core allocation-"
            "worker family and the measured organic node004/node006 states. "
            "When the native closure artifact is supplied, it also closes the "
            "exact FreqDuet/SUMO 2-workload x 6-node x {1,2,4}-colocation "
            "family. It does not close legacy ablation keys, jtl311linux, "
            "jtl110cpu/jtl110cpu2, unknown environments, GPU states, migration "
            "costs, or arbitrary future workloads."
        ),
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
    parser.add_argument("--empty-gate", type=Path, default=DEFAULT_EMPTY_GATE)
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
    parser.add_argument(
        "--merge-report",
        type=Path,
        default=DEFAULT_MERGE_REPORT,
    )
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--native-closure",
        type=Path,
        default=DEFAULT_NATIVE_CLOSURE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACT_ROOT / "cpu_eta_closure_gate_20260727.json",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = build_cpu_eta_closure_gate(
        empty_gate_path=args.empty_gate,
        controlled_gate_path=args.controlled_gate,
        external_gate_path=args.external_gate,
        merge_report_path=args.merge_report,
        design_path=args.design,
        native_closure_path=args.native_closure,
    )
    _atomic_write_json(args.output, result)
    print(args.output)
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
