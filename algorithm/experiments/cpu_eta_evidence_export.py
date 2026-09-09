"""Export one auditable CPU ETA/lower-service table without mixing axes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_NATIVE_GATES = (
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p1_extended_lcb_gate_20260727formal_v6_state_targeted_p1_ext9.json",
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p2_lcb_gate_20260727formal_v6_state_targeted_p2.json",
    ARTIFACT_ROOT
    / "native_cpu_colocation_state_targeted_p4_merged_lcb_gate_20260727formal_v6_state_targeted_p4.json",
)
DEFAULT_WORKER_GATE = (
    ARTIFACT_ROOT
    / "native_cpu_allocation_worker_lcb_gate_20260727.json"
)
DEFAULT_LOADED_GATES = (
    ARTIFACT_ROOT
    / "native_cpu_controlled_resident_worker_lcb_gate_20260727.json",
    ARTIFACT_ROOT
    / "native_cpu_external_state_worker_lcb_gate_20260727.json",
)


def build_cpu_eta_evidence_rows(
    *,
    native_gate_paths: Sequence[Path],
    worker_gate_path: Path,
    loaded_gate_paths: Sequence[Path] = (),
) -> list[dict[str, Any]]:
    rows = []
    for path in native_gate_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows") or []:
            rows.append(
                {
                    "evidence_family": "native_colocation",
                    "node": row.get("node"),
                    "workload_key": row.get("workload_key"),
                    "workload_env": row.get("workload_env"),
                    "resource_state": row.get(
                        "observed_effective_resource_state"
                    ),
                    "dispatch_regime": row.get(
                        "dispatch_external_cpu_regime"
                    ),
                    "profile_axis": "colocation_count",
                    "allocation_workers": int(
                        row.get("allocation_workers") or 1
                    ),
                    "colocation_count": int(
                        row.get("colocation_count") or 1
                    ),
                    "point_eta_s": float(
                        row.get("point_predicted_makespan_s") or 0.0
                    ),
                    "holdout_actual_s": float(
                        row.get("actual_makespan_s") or 0.0
                    ),
                    "conservative_eta_s": float(
                        row.get(
                            "simultaneous_conservative_upper_makespan_s"
                        )
                        or 0.0
                    ),
                    "lower_service_units_per_s": float(
                        row.get(
                            "simultaneous_lower_service_units_per_s"
                        )
                        or 0.0
                    ),
                    "lower_service_valid": bool(
                        row.get(
                            "simultaneous_lower_service_valid_on_holdout"
                        )
                    ),
                    "training_samples": int(
                        row.get("training_sample_count") or 0
                    ),
                    "calibration_samples": int(
                        row.get("calibration_sample_count") or 0
                    ),
                    "holdout_samples": 1,
                    "eta_contract": (
                        "native startup+outer-loop+checkpoint/finalization"
                    ),
                    "source": str(path),
                }
            )

    worker = json.loads(worker_gate_path.read_text(encoding="utf-8"))
    for row in worker.get("rows") or []:
        rows.append(
            {
                "evidence_family": "allocation_worker_curve",
                "node": "node001-node006 homogeneous class",
                "workload_key": "cpu_heavy_local_bench",
                "workload_env": "cpu",
                "resource_state": row.get("resource_state"),
                "dispatch_regime": row.get(
                    "dispatch_external_cpu_regime"
                ),
                "profile_axis": "allocation_workers",
                "allocation_workers": int(
                    row.get("allocation_workers") or 1
                ),
                "colocation_count": 1,
                "point_eta_s": float(row.get("point_eta_s") or 0.0),
                "holdout_actual_s": float(
                    row.get("holdout_actual_completion_s") or 0.0
                ),
                "conservative_eta_s": float(
                    row.get("simultaneous_conservative_eta_s") or 0.0
                ),
                "lower_service_units_per_s": float(
                    row.get("lower_service_units_per_s") or 0.0
                ),
                "lower_service_valid": bool(
                    row.get("lower_service_valid_on_holdout")
                ),
                "training_samples": int(
                    row.get("training_sample_count") or 0
                ),
                "calibration_samples": int(
                    row.get("calibration_wave_count") or 0
                ),
                "holdout_samples": 1,
                "eta_contract": (
                    "startup+outer-loop+periodic-checkpoint+final-save"
                ),
                "source": str(worker_gate_path),
            }
        )
    for loaded_gate_path in loaded_gate_paths:
        loaded = json.loads(loaded_gate_path.read_text(encoding="utf-8"))
        if not loaded.get("pass"):
            raise ValueError(
                f"loaded-state lower-service gate did not pass: "
                f"{loaded_gate_path}"
            )
        controlled = (
            loaded.get("gate")
            == "native_cpu_controlled_resident_worker_lcb_gate"
        )
        for row in loaded.get("rows") or []:
            measured_nodes = row.get("measured_nodes") or ()
            rows.append(
                {
                    "evidence_family": (
                        "controlled_loaded_worker_curve"
                        if controlled
                        else "organic_external_worker_curve"
                    ),
                    "node": (
                        "node001-node006 homogeneous class"
                        if controlled
                        else ",".join(str(node) for node in measured_nodes)
                    ),
                    "workload_key": "cpu_heavy_local_bench",
                    "workload_env": "cpu",
                    "resource_state": row.get("resource_state"),
                    "dispatch_regime": row.get(
                        "dispatch_external_cpu_regime"
                    ),
                    "profile_axis": "allocation_workers",
                    "allocation_workers": int(
                        row.get("allocation_workers") or 1
                    ),
                    "colocation_count": 1,
                    "point_eta_s": float(row.get("point_eta_s") or 0.0),
                    "holdout_actual_s": float(
                        row.get("holdout_actual_completion_s") or 0.0
                    ),
                    "conservative_eta_s": float(
                        row.get("simultaneous_conservative_eta_s") or 0.0
                    ),
                    "lower_service_units_per_s": float(
                        row.get("lower_service_units_per_s") or 0.0
                    ),
                    "lower_service_valid": bool(
                        row.get("lower_service_valid_on_holdout")
                    ),
                    "training_samples": int(
                        row.get("training_sample_count") or 0
                    ),
                    "calibration_samples": int(
                        row.get("calibration_wave_count") or 0
                    ),
                    "holdout_samples": 1,
                    "eta_contract": (
                        "startup+outer-loop+periodic-checkpoint+final-save"
                    ),
                    "source": str(loaded_gate_path),
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["evidence_family"]),
            str(row["node"]),
            str(row["workload_key"]),
            int(row["colocation_count"]),
            int(row["allocation_workers"]),
        )
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# CPU ETA and lower-service evidence table",
        "",
        (
            "The two profile axes are intentionally separate: "
            "`colocation_count` counts independent tasks, whereas "
            "`allocation_workers` is the CPU allocation of one task."
        ),
        "",
        "| evidence | workload/node | axis/profile | state | point ETA | holdout | conservative ETA | lower service | valid |",
        "|---|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        axis = str(row["profile_axis"])
        profile = (
            row["colocation_count"]
            if axis == "colocation_count"
            else row["allocation_workers"]
        )
        lines.append(
            "| {evidence_family} | {workload_key} / {node} | "
            "{axis}=p{profile} | {resource_state} | {point_eta_s:.3f} | "
            "{holdout_actual_s:.3f} | {conservative_eta_s:.3f} | "
            "{lower_service_units_per_s:.6f} | {valid} |".format(
                **row,
                axis=axis,
                profile=profile,
                valid="yes" if row["lower_service_valid"] else "no",
            )
        )
    lines.extend(
        [
            "",
            (
                "Only rows with task-native progress, natural completion, "
                "state-certified dispatch, and fresh holdout coverage are "
                "theorem-facing."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-gates",
        default=",".join(str(path) for path in DEFAULT_NATIVE_GATES),
    )
    parser.add_argument(
        "--worker-gate",
        type=Path,
        default=DEFAULT_WORKER_GATE,
    )
    parser.add_argument(
        "--loaded-gates",
        default=",".join(str(path) for path in DEFAULT_LOADED_GATES),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=(
            ARTIFACT_ROOT / "cpu_eta_theorem_evidence_table_20260727.csv"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=(
            REPO_ROOT / "md" / "cpu_eta_theorem_evidence_20260727.md"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    rows = build_cpu_eta_evidence_rows(
        native_gate_paths=tuple(
            Path(item.strip())
            for item in str(args.native_gates).split(",")
            if item.strip()
        ),
        worker_gate_path=args.worker_gate,
        loaded_gate_paths=tuple(
            Path(item.strip())
            for item in str(args.loaded_gates).split(",")
            if item.strip()
        ),
    )
    _write_csv(args.csv_output, rows)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(rows), encoding="utf-8")
    print(args.csv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
