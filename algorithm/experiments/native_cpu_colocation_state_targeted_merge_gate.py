"""Merge disjoint state-targeted profile campaigns into one exact family gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from .file_progress_completion_wrapper import MEASUREMENT_PROTOCOL
from .native_cpu_colocation_state_targeted_campaign import (
    ARTIFACT_ROOT,
    _atomic_write_json,
    _file_sha256_map,
    _safe_id,
)
from .phase_aware_colocation_stochastic_lcb_gate import (
    build_colocation_stochastic_lcb_gate,
)


def merge_state_targeted_profile_campaigns(
    *,
    campaign_paths: Sequence[Path],
    tag: str,
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    """Merge complete, disjoint cell families and recompute the final gate."""

    started = time.time()
    safe_tag = _safe_id(tag)
    loaded = [
        _load_component(Path(path).resolve())
        for path in campaign_paths
    ]
    if len(loaded) < 2:
        raise ValueError("at least two component campaigns are required")
    profiles = {component["profile"] for component in loaded}
    if len(profiles) != 1:
        raise ValueError(f"component profile mismatch: {sorted(profiles)}")
    profile = next(iter(profiles))
    expected_cell_ids = []
    seen_cells = set()
    targets = {}
    for component in loaded:
        overlap = seen_cells & set(component["expected_cell_ids"])
        if overlap:
            raise ValueError(f"component cell overlap: {sorted(overlap)}")
        seen_cells.update(component["expected_cell_ids"])
        expected_cell_ids.extend(component["expected_cell_ids"])
        target_overlap = set(targets) & set(component["target_dispatch_regimes"])
        if target_overlap:
            raise ValueError(
                f"component node target overlap: {sorted(target_overlap)}"
            )
        targets.update(component["target_dispatch_regimes"])

    paths_by_role = {"training": [], "calibration": [], "holdout": []}
    role_counts = {"training": 3, "calibration": 9, "holdout": 1}
    for role, repeat_count in role_counts.items():
        for replicate in range(1, repeat_count + 1):
            source_payloads = [
                json.loads(
                    component[f"{role}_paths"][replicate - 1].read_text(
                        encoding="utf-8"
                    )
                )
                for component in loaded
            ]
            rows = []
            provenance = []
            observed_cells = set()
            for component, payload in zip(loaded, source_payloads):
                source_path = component[f"{role}_paths"][replicate - 1]
                source_hash = _sha256_file(source_path)
                for raw_row in payload.get("rows") or []:
                    row = dict(raw_row)
                    cell_id = str(
                        row.get("calibration_cell_id") or ""
                    ).strip()
                    if not cell_id or cell_id in observed_cells:
                        raise ValueError(
                            f"invalid or duplicate merged cell {cell_id!r}"
                        )
                    observed_cells.add(cell_id)
                    rows.append(row)
                    provenance.append(
                        {
                            "calibration_cell_id": cell_id,
                            "source_campaign": str(component["path"]),
                            "source_artifact": str(source_path),
                            "source_artifact_sha256": source_hash,
                        }
                    )
            if observed_cells != seen_cells:
                raise ValueError(
                    f"merged {role} r{replicate:02d} cell set mismatch"
                )
            output = ARTIFACT_ROOT / (
                f"native_cpu_colocation_state_targeted_p{profile}_merged_"
                f"{role}_r{replicate:02d}_{safe_tag}.json"
            )
            _atomic_write_json(
                output,
                {
                    "gate": "native_cpu_colocation_completion_matrix",
                    "schema_version": 1,
                    "measurement_protocol": MEASUREMENT_PROTOCOL,
                    "merge_protocol": "disjoint_exact_cell_family_merge_v1",
                    "profile": profile,
                    "profile_axis": "colocation_count",
                    "role": role,
                    "replicate": replicate,
                    "status": "COMPLETE",
                    "all_completion_models_ready": True,
                    "selected_count": len(rows),
                    "admitted_count": len(rows),
                    "expected_calibration_cell_ids": sorted(seen_cells),
                    "target_dispatch_regimes": dict(sorted(targets.items())),
                    "performance_outcome_fields_used_for_selection": [],
                    "source_provenance": sorted(
                        provenance,
                        key=lambda row: row["calibration_cell_id"],
                    ),
                    "rows": sorted(
                        rows,
                        key=lambda row: str(
                            row.get("calibration_cell_id") or ""
                        ),
                    ),
                },
            )
            paths_by_role[role].append(output)

    gate = build_colocation_stochastic_lcb_gate(
        training_paths=paths_by_role["training"],
        calibration_paths=paths_by_role["calibration"],
        holdout_path=paths_by_role["holdout"][0],
        miscoverage_alpha=float(miscoverage_alpha),
        expected_cell_ids=sorted(seen_cells),
    )
    gate_path = ARTIFACT_ROOT / (
        f"native_cpu_colocation_state_targeted_p{profile}_merged_lcb_gate_"
        f"{safe_tag}.json"
    )
    _atomic_write_json(gate_path, gate)
    evidence_paths = [
        *paths_by_role["training"],
        *paths_by_role["calibration"],
        *paths_by_role["holdout"],
    ]
    passed = bool(gate.get("pass"))
    return {
        "campaign": "native_cpu_colocation_state_targeted_merge_gate",
        "schema_version": 1,
        "tag": safe_tag,
        "measurement_protocol": MEASUREMENT_PROTOCOL,
        "merge_protocol": "disjoint_exact_cell_family_merge_v1",
        "profile": profile,
        "component_campaigns": [
            {
                "path": str(component["path"]),
                "sha256": component["sha256"],
                "status": component["status"],
                "pass": component["pass"],
                "expected_cell_ids": component["expected_cell_ids"],
            }
            for component in loaded
        ],
        "component_cell_sets_disjoint": True,
        "target_dispatch_regimes": dict(sorted(targets.items())),
        "expected_cell_ids": sorted(seen_cells),
        "expected_cell_count": len(seen_cells),
        "training_paths": [
            str(path) for path in paths_by_role["training"]
        ],
        "calibration_paths": [
            str(path) for path in paths_by_role["calibration"]
        ],
        "holdout_path": str(paths_by_role["holdout"][0]),
        "gate_path": str(gate_path),
        "lcb_gate_path": str(gate_path),
        "gate": gate,
        "gate_pass": passed,
        "evidence_sha256": _file_sha256_map(evidence_paths),
        "status": "PASS" if passed else "GATE_FAIL",
        "pass": passed,
        "elapsed_wall_s": time.time() - started,
        "claim_boundary": (
            "The merge performs no row selection and no model averaging. It "
            "combines pre-registered disjoint exact-cell components replicate "
            "by replicate, preserves source hashes, and recomputes both cellwise "
            "and wave-max simultaneous split-conformal bounds over the complete "
            "family. A component diagnostic failure is not silently relabeled; "
            "the final claim is based only on this recomputed family gate."
        ),
    }


def _load_component(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    errors = []
    if payload.get("campaign") != (
        "native_cpu_colocation_state_targeted_profile_campaign"
    ):
        errors.append("campaign_identity_mismatch")
    if payload.get("measurement_protocol") != MEASUREMENT_PROTOCOL:
        errors.append("measurement_protocol_mismatch")
    expected = tuple(
        str(value) for value in payload.get("expected_cell_ids") or []
    )
    if not expected:
        errors.append("expected_cell_set_empty")
    result = {
        "path": path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "profile": int(payload.get("profile") or 0),
        "status": str(payload.get("status") or ""),
        "pass": bool(payload.get("pass")),
        "expected_cell_ids": expected,
        "target_dispatch_regimes": {
            str(key): str(value)
            for key, value in (
                payload.get("target_dispatch_regimes") or {}
            ).items()
        },
    }
    for role, expected_count in (
        ("training", 3),
        ("calibration", 9),
        ("holdout", 1),
    ):
        key = f"{role}_paths" if role != "holdout" else "holdout_path"
        raw_paths = (
            payload.get(key) or []
            if role != "holdout"
            else [payload.get(key)]
        )
        paths = tuple(
            Path(value).resolve()
            for value in raw_paths
            if value
        )
        if len(paths) != expected_count:
            errors.append(f"{role}_artifact_count_mismatch")
        for evidence_path in paths:
            if not evidence_path.is_file():
                errors.append(f"evidence_missing:{evidence_path}")
        result[f"{role}_paths"] = paths
    if errors:
        raise ValueError(f"invalid component {path}: {errors}")
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = merge_state_targeted_profile_campaigns(
        campaign_paths=args.campaign,
        tag=args.tag,
        miscoverage_alpha=args.alpha,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
