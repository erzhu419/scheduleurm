"""Matched-wave equivalence gate for the homogeneous 2x3080Ti GPU bucket.

This gate decides whether measurements from ``jtl110gpu2`` may be pooled with
the representative ``jtl110gpu`` hardware bucket.  It does not construct a
stochastic lower confidence bound (LCB): the representative node remains the
only input to the main multi-wave LCB gate.  The second node supplies one
matched-wave hardware-equivalence audit over the nine pre-registered actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    CAMPAIGN_PREFIX,
    NODE_SPECS,
    PROTOCOL,
    STAGED_TRAJECTORY_EVIDENCE,
)
from .critical_gpu_stochastic_lcb_gate import (
    CAMPAIGN_DATE,
    EXPECTED_CELL_COUNT,
    RESOURCE_STATE,
    _audit_wave,
    _expected_cells,
)


REPRESENTATIVE_NODE = "jtl110gpu"
EQUIVALENCE_NODE = "jtl110gpu2"
DEFAULT_MATCHED_WAVE = 1

# Pre-registered before observing a complete matched pair.  Every cell and
# both functionals must satisfy the broad guardrail; the cross-cell robust
# center must also satisfy the tighter interval.
CELL_RATIO_FACTOR = 4.0 / 3.0
CELL_RATIO_LOWER = 1.0 / CELL_RATIO_FACTOR
CELL_RATIO_UPPER = CELL_RATIO_FACTOR
ROBUST_CENTER_FACTOR = 1.10
ROBUST_CENTER_LOWER = 1.0 / ROBUST_CENTER_FACTOR
ROBUST_CENTER_UPPER = ROBUST_CENTER_FACTOR


def default_artifact_paths(
    *,
    wave: int = DEFAULT_MATCHED_WAVE,
    artifact_root: Path = ARTIFACT_ROOT,
) -> tuple[Path, Path]:
    """Return the exact current-protocol paths for one matched wave."""

    value = int(wave)
    return (
        Path(artifact_root) / _artifact_name(REPRESENTATIVE_NODE, value),
        Path(artifact_root) / _artifact_name(EQUIVALENCE_NODE, value),
    )


def build_critical_gpu_homogeneous_equivalence_gate(
    *,
    representative_path: Path | None = None,
    equivalence_path: Path | None = None,
    wave: int = DEFAULT_MATCHED_WAVE,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Audit a matched representative/equivalence campaign pair.

    Missing or still-running measurements are WAIT conditions.  Existing
    artifacts that violate the campaign contract are FAIL conditions.  PASS
    requires all nine matched cells, every strict measurement audit, every
    cell-level bidirectional ratio, and both robust-center ratio checks.
    """

    matched_wave = int(wave)
    errors: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    if matched_wave == 0:
        errors.append(
            _issue(
                "SMOKE_WAVE_FORBIDDEN",
                detail="wave 0 is excluded from hardware-bucket equivalence",
            )
        )
    elif matched_wave < 1 or matched_wave > 13:
        errors.append(
            _issue(
                "UNREGISTERED_WAVE",
                detail="matched wave must be one of the formal waves 1..13",
            )
        )

    default_representative, default_equivalence = default_artifact_paths(
        wave=matched_wave,
        artifact_root=artifact_root,
    )
    paths = {
        REPRESENTATIVE_NODE: Path(representative_path or default_representative),
        EQUIVALENCE_NODE: Path(equivalence_path or default_equivalence),
    }
    payloads: dict[str, dict[str, Any]] = {}
    source_artifacts: list[dict[str, Any]] = []
    missing_nodes: list[str] = []
    for node, path in paths.items():
        expected_name = _artifact_name(node, matched_wave)
        if not path.is_file():
            missing_nodes.append(node)
            waits.append(
                _issue(
                    "MISSING_ARTIFACT",
                    source=node,
                    detail=str(path),
                )
            )
            continue
        if path.name != expected_name:
            errors.append(
                _issue(
                    "ARTIFACT_PATH_RULE_MISMATCH",
                    source=node,
                    detail=f"expected {expected_name!r}, got {path.name!r}",
                )
            )
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                _issue(
                    "UNREADABLE_ARTIFACT",
                    source=node,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        if not isinstance(payload, dict):
            errors.append(
                _issue(
                    "INVALID_ARTIFACT_ROOT",
                    source=node,
                    detail="JSON root must be an object",
                )
            )
            continue
        payloads[node] = payload
        source_artifacts.append(
            {
                "node": node,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": payload.get("status"),
                "pass": bool(payload.get("pass")),
            }
        )

    representative_spec = NODE_SPECS[REPRESENTATIVE_NODE]
    equivalence_spec = NODE_SPECS[EQUIVALENCE_NODE]
    if (
        representative_spec.node_bucket != equivalence_spec.node_bucket
        or representative_spec.hardware_class != equivalence_spec.hardware_class
    ):
        errors.append(
            _issue(
                "DECLARED_HARDWARE_BUCKET_MISMATCH",
                detail="the campaign registry no longer declares a homogeneous pair",
            )
        )

    expected_by_node = {
        node: _expected_cells(node)
        for node in (REPRESENTATIVE_NODE, EQUIVALENCE_NODE)
    }
    expected_identity_sets = {
        node: {_logical_cell_identity(row) for row in cells.values()}
        for node, cells in expected_by_node.items()
    }
    if any(len(cells) != EXPECTED_CELL_COUNT for cells in expected_by_node.values()):
        errors.append(
            _issue(
                "EXPECTED_CELL_COUNT_MISMATCH",
                detail=f"each node must expose exactly {EXPECTED_CELL_COUNT} cells",
            )
        )
    if expected_identity_sets[REPRESENTATIVE_NODE] != expected_identity_sets[EQUIVALENCE_NODE]:
        errors.append(
            _issue(
                "REGISTERED_CELL_SET_MISMATCH",
                detail="registered workload_env/profile sets differ across nodes",
            )
        )

    observations: dict[str, dict[str, dict[str, Any]]] = {}
    node_audits: list[dict[str, Any]] = []
    required_roles = {
        REPRESENTATIVE_NODE: "representative",
        EQUIVALENCE_NODE: "homogeneous_equivalence",
    }
    for node, payload in payloads.items():
        node_spec_payload = payload.get("node_spec") or {}
        expected_role = required_roles[node]
        if not isinstance(node_spec_payload, Mapping):
            errors.append(
                _issue(
                    "INVALID_NODE_SPEC",
                    source=node,
                    detail="node_spec must be an object",
                )
            )
            node_spec_payload = {}
        if node_spec_payload.get("role") != expected_role:
            errors.append(
                _issue(
                    "NODE_ROLE_MISMATCH",
                    source=node,
                    detail=(
                        f"expected node_spec.role={expected_role!r}, "
                        f"got {node_spec_payload.get('role')!r}"
                    ),
                )
            )
        for collection_name in ("planned_cells", "rows"):
            for index, row in enumerate(payload.get(collection_name) or []):
                if not isinstance(row, Mapping):
                    errors.append(
                        _issue(
                            "INVALID_CELL_ROW",
                            source=node,
                            cell=f"row_index={index}",
                            detail=f"{collection_name} row must be an object",
                        )
                    )
                    continue
                if row.get("node_role") != expected_role:
                    errors.append(
                        _issue(
                            "NODE_ROLE_MISMATCH",
                            source=node,
                            cell=_raw_cell_label(row, index=index),
                            detail=(
                                f"{collection_name} row must have "
                                f"node_role={expected_role!r}"
                            ),
                        )
                    )

        try:
            audit, node_waits, node_errors, node_observations = _audit_wave(
                node=node,
                wave=matched_wave,
                payload=payload,
                expected_cells=expected_by_node[node],
                admitted_cell_ids=set(expected_by_node[node]),
                excluded_cell_ids=set(),
            )
        except Exception as exc:  # malformed external artifact is a contract failure
            errors.append(
                _issue(
                    "ARTIFACT_AUDIT_ERROR",
                    source=node,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        node_audits.append({"source": node, **audit})
        waits.extend(_tag_issues(node_waits, source=node))
        errors.extend(_tag_issues(node_errors, source=node))
        observations[node] = node_observations

    same_protocol = bool(
        len(payloads) == 2
        and all(
            payload.get("measurement_protocol") == PROTOCOL
            for payload in payloads.values()
        )
    )
    code_hashes = {
        str((payload.get("measurement_code_manifest") or {}).get("sha256") or "")
        for payload in payloads.values()
    }
    same_code = bool(len(payloads) == 2 and len(code_hashes) == 1 and "" not in code_hashes)
    if len(payloads) == 2 and not same_code:
        errors.append(
            _issue(
                "MEASUREMENT_CODE_IDENTITY_MISMATCH",
                detail=f"matched nodes used different code hashes: {sorted(code_hashes)!r}",
            )
        )
    same_wave = bool(
        len(payloads) == 2
        and all(
            _int_or(payload.get("wave"), -1) == matched_wave
            for payload in payloads.values()
        )
    )
    observed_identity_sets = {
        node: {_logical_cell_identity(row) for row in values.values()}
        for node, values in observations.items()
    }
    same_nine_cells = bool(
        len(observed_identity_sets) == 2
        and all(len(values) == EXPECTED_CELL_COUNT for values in observed_identity_sets.values())
        and observed_identity_sets[REPRESENTATIVE_NODE]
        == observed_identity_sets[EQUIVALENCE_NODE]
        == expected_identity_sets[REPRESENTATIVE_NODE]
    )

    comparison: dict[str, Any] = {
        "constructed": False,
        "all_cell_ratio_guardrails_pass": False,
        "robust_center_pass": False,
        "rows": [],
    }
    measurements_ready = bool(
        not errors
        and not waits
        and same_protocol
        and same_code
        and same_wave
        and same_nine_cells
    )
    if measurements_ready:
        comparison = _build_comparison(
            representative_payload=payloads[REPRESENTATIVE_NODE],
            equivalence_payload=payloads[EQUIVALENCE_NODE],
            representative_observations=observations[REPRESENTATIVE_NODE],
            equivalence_observations=observations[EQUIVALENCE_NODE],
        )

    if errors:
        status = "FAIL_CONTRACT"
    elif waits:
        status = (
            "WAIT_MISSING_ARTIFACT"
            if missing_nodes
            else "WAIT_INCOMPLETE_ARTIFACT"
        )
    elif not comparison.get("constructed"):
        status = "FAIL_CONTRACT"
        errors.append(
            _issue(
                "COMPARISON_NOT_CONSTRUCTED",
                detail="strict observations were unavailable after artifact audit",
            )
        )
    elif not (
        comparison.get("all_cell_ratio_guardrails_pass")
        and comparison.get("robust_center_pass")
    ):
        status = "FAIL_EQUIVALENCE"
    else:
        status = "PASS"

    passed = status == "PASS"
    return {
        "gate": "critical_gpu_homogeneous_equivalence_gate",
        "schema_version": 1,
        "status": status,
        "pass": passed,
        "hardware_bucket_pooling_ready": passed,
        "representative_node": REPRESENTATIVE_NODE,
        "equivalence_node": EQUIVALENCE_NODE,
        "equivalence_node_participates_in_main_lcb": False,
        "matched_wave": matched_wave,
        "same_wave_required": True,
        "measurement_protocol": PROTOCOL,
        "same_measurement_protocol": same_protocol,
        "same_measurement_code": same_code,
        "measurement_code_sha256": next(iter(code_hashes)) if same_code else None,
        "node_bucket": representative_spec.node_bucket,
        "hardware_class": representative_spec.hardware_class,
        "resource_state": RESOURCE_STATE,
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "same_nine_workload_env_profile_cells": same_nine_cells,
        "pre_registered_acceptance": {
            "cell_ratio_definition": "equivalence_node / representative_node",
            "cell_ratio_lower": CELL_RATIO_LOWER,
            "cell_ratio_upper": CELL_RATIO_UPPER,
            "cell_ratio_factor": CELL_RATIO_FACTOR,
            "cell_outlier_allowance": 0,
            "robust_center_statistic": "median of nine matched ratios",
            "robust_center_lower": ROBUST_CENTER_LOWER,
            "robust_center_upper": ROBUST_CENTER_UPPER,
            "robust_center_factor": ROBUST_CENTER_FACTOR,
            "functionals": [
                "drain_completion_s",
                "empirical_aggregate_lower_service_units_per_s",
            ],
        },
        "missing_nodes": missing_nodes,
        "wait_reasons": waits,
        "validation_errors": errors,
        "measurements_ready": measurements_ready,
        "node_audits": node_audits,
        "source_artifacts": source_artifacts,
        "comparison": comparison,
        "claim_boundary": (
            "PASS certifies only that the declared jtl110gpu and jtl110gpu2 "
            "empty-state, matched-wave measurements support pooling the two "
            "registered 2xRTX-3080Ti nodes into one hardware bucket for the nine "
            "declared workload_env/profile cells. jtl110gpu2 remains an "
            "independent homogeneous-equivalence audit and is not an observation "
            "in the representative node's stochastic LCB fit. The empirical "
            "aggregate lower-service quantity is the minimum of task-native "
            "stable aggregate service and naturally completed aggregate service; "
            "it is an equivalence functional, not a new confidence bound. This "
            "gate does not certify loaded states, unmeasured actions, future "
            "software or hardware changes, or arbitrary future states."
        ),
    }


def _build_comparison(
    *,
    representative_payload: Mapping[str, Any],
    equivalence_payload: Mapping[str, Any],
    representative_observations: Mapping[str, Mapping[str, Any]],
    equivalence_observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    representative_rows = _rows_by_identity(representative_payload.get("rows") or [])
    equivalence_rows = _rows_by_identity(equivalence_payload.get("rows") or [])
    representative_by_identity = {
        _logical_cell_identity(row): row
        for row in representative_observations.values()
    }
    equivalence_by_identity = {
        _logical_cell_identity(row): row
        for row in equivalence_observations.values()
    }
    identities = sorted(representative_by_identity)
    rows: list[dict[str, Any]] = []
    for identity in identities:
        representative = representative_by_identity[identity]
        equivalent = equivalence_by_identity[identity]
        representative_source = representative_rows[identity]
        equivalence_source = equivalence_rows[identity]
        representative_drain = float(representative["drain_completion_s"])
        equivalence_drain = float(equivalent["drain_completion_s"])
        representative_service = _empirical_aggregate_lower_service(
            representative,
            representative_source,
        )
        equivalence_service = _empirical_aggregate_lower_service(
            equivalent,
            equivalence_source,
        )
        drain_ratio = _positive_ratio(equivalence_drain, representative_drain)
        service_ratio = _positive_ratio(equivalence_service, representative_service)
        drain_pass = _within(drain_ratio, CELL_RATIO_LOWER, CELL_RATIO_UPPER)
        service_pass = _within(service_ratio, CELL_RATIO_LOWER, CELL_RATIO_UPPER)
        rows.append(
            {
                "workload_env": identity[0],
                "profile": identity[1],
                "representative_drain_completion_s": representative_drain,
                "equivalence_drain_completion_s": equivalence_drain,
                "drain_completion_ratio": drain_ratio,
                "representative_empirical_aggregate_lower_service_units_per_s": representative_service,
                "equivalence_empirical_aggregate_lower_service_units_per_s": equivalence_service,
                "aggregate_lower_service_ratio": service_ratio,
                "drain_ratio_guardrail_pass": drain_pass,
                "aggregate_lower_service_ratio_guardrail_pass": service_pass,
                "cell_equivalence_pass": drain_pass and service_pass,
            }
        )

    drain_ratios = [float(row["drain_completion_ratio"]) for row in rows]
    service_ratios = [float(row["aggregate_lower_service_ratio"]) for row in rows]
    drain_center = median(drain_ratios)
    service_center = median(service_ratios)
    drain_center_pass = _within(
        drain_center,
        ROBUST_CENTER_LOWER,
        ROBUST_CENTER_UPPER,
    )
    service_center_pass = _within(
        service_center,
        ROBUST_CENTER_LOWER,
        ROBUST_CENTER_UPPER,
    )
    return {
        "constructed": len(rows) == EXPECTED_CELL_COUNT,
        "matched_cell_count": len(rows),
        "all_cell_ratio_guardrails_pass": bool(rows)
        and all(row["cell_equivalence_pass"] for row in rows),
        "robust_center_pass": drain_center_pass and service_center_pass,
        "robust_centers": {
            "drain_completion_ratio_median": drain_center,
            "drain_completion_ratio_median_pass": drain_center_pass,
            "aggregate_lower_service_ratio_median": service_center,
            "aggregate_lower_service_ratio_median_pass": service_center_pass,
        },
        "maximum_multiplicative_deviation": {
            "drain_completion": max(_multiplicative_deviation(value) for value in drain_ratios),
            "aggregate_lower_service": max(
                _multiplicative_deviation(value) for value in service_ratios
            ),
        },
        "rows": rows,
    }


def _empirical_aggregate_lower_service(
    observation: Mapping[str, Any],
    source_row: Mapping[str, Any],
) -> float:
    aggregate_units = _positive_float(observation.get("aggregate_total_units"))
    drain = _positive_float(observation.get("drain_completion_s"))
    completion_service = aggregate_units / drain if aggregate_units and drain else 0.0
    if (
        source_row.get("service_evidence_mode")
        == STAGED_TRAJECTORY_EVIDENCE
    ):
        return completion_service
    children = (source_row.get("summary") or {}).get("rows") or []
    stable_rates = [_positive_float(child.get("stable_rate")) for child in children]
    if not stable_rates or any(value <= 0.0 for value in stable_rates):
        return 0.0
    task_native_service = sum(stable_rates)
    return min(completion_service, task_native_service)


def _rows_by_identity(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        identity = _logical_cell_identity(row)
        if identity in result:
            raise ValueError(f"duplicate logical cell {identity!r}")
        result[identity] = row
    return result


def _logical_cell_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    workload_env = str(row.get("workload_env") or "").strip()
    profile = int(row.get("profile") or 0)
    if not workload_env or profile <= 0:
        raise ValueError("workload_env and positive profile are required")
    return workload_env, profile


def _raw_cell_label(row: Mapping[str, Any], *, index: int) -> str:
    try:
        env, profile = _logical_cell_identity(row)
        return f"workload_env={env}|profile={profile}"
    except (TypeError, ValueError):
        return f"row_index={index}"


def _tag_issues(
    issues: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    return [{"source": source, **dict(issue)} for issue in issues]


def _artifact_name(node: str, wave: int) -> str:
    return f"{CAMPAIGN_PREFIX}_{node}_r{int(wave):02d}_{CAMPAIGN_DATE}.json"


def _issue(
    code: str,
    *,
    source: str | None = None,
    cell: str | None = None,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "code": code,
        "source": source,
        "cell": cell,
        "detail": detail,
    }


def _positive_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) and result > 0.0 else 0.0


def _int_or(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _positive_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        return math.inf
    return numerator / denominator


def _within(value: float, lower: float, upper: float) -> bool:
    return bool(math.isfinite(value) and lower <= value <= upper)


def _multiplicative_deviation(value: float) -> float:
    return max(value, 1.0 / value) if value > 0.0 and math.isfinite(value) else math.inf


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_gate_outputs(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    _atomic_write(
        json_path,
        json.dumps(dict(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(markdown_path, _markdown(report, json_path=json_path))


def _markdown(report: Mapping[str, Any], *, json_path: Path) -> str:
    acceptance = report.get("pre_registered_acceptance") or {}
    comparison = report.get("comparison") or {}
    lines = [
        "# Critical GPU homogeneous-equivalence gate",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{report.get('status')}`",
        f"- Matched wave: `r{int(report.get('matched_wave') or 0):02d}`",
        f"- Representative: `{report.get('representative_node')}`",
        f"- Equivalence node: `{report.get('equivalence_node')}`",
        f"- Hardware-bucket pooling ready: `{str(bool(report.get('hardware_bucket_pooling_ready'))).lower()}`",
        f"- Equivalence observations enter main LCB: `{str(bool(report.get('equivalence_node_participates_in_main_lcb'))).lower()}`",
        (
            "- Cell ratio interval: "
            f"`[{float(acceptance.get('cell_ratio_lower') or 0.0):.6f}, "
            f"{float(acceptance.get('cell_ratio_upper') or 0.0):.6f}]`"
        ),
        (
            "- Robust median interval: "
            f"`[{float(acceptance.get('robust_center_lower') or 0.0):.6f}, "
            f"{float(acceptance.get('robust_center_upper') or 0.0):.6f}]`"
        ),
        "",
    ]
    if report.get("wait_reasons"):
        lines.extend(["## Wait reasons", ""])
        for issue in report["wait_reasons"]:
            lines.append(
                f"- `{issue.get('code')}` source={issue.get('source')}: "
                f"{issue.get('detail')}"
            )
        lines.append("")
    if report.get("validation_errors"):
        lines.extend(["## Validation errors", ""])
        for issue in report["validation_errors"]:
            lines.append(
                f"- `{issue.get('code')}` source={issue.get('source')} "
                f"cell={issue.get('cell')}: {issue.get('detail')}"
            )
        lines.append("")
    rows = comparison.get("rows") or []
    if rows:
        lines.extend(
            [
                "## Matched cells",
                "",
                "| workload environment | profile | drain ratio | aggregate lower-service ratio | pass |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['workload_env']}` | {row['profile']} | "
                f"{row['drain_completion_ratio']:.6f} | "
                f"{row['aggregate_lower_service_ratio']:.6f} | "
                f"{'yes' if row['cell_equivalence_pass'] else 'no'} |"
            )
        lines.append("")
    lines.extend([str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", type=int, default=DEFAULT_MATCHED_WAVE)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--representative-artifact", type=Path)
    parser.add_argument("--equivalence-artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output = args.output or (
        args.artifact_root
        / f"critical_gpu_homogeneous_equivalence_gate_r{args.wave:02d}_{CAMPAIGN_DATE}.json"
    )
    markdown = args.markdown_output or output.with_suffix(".md")
    report = build_critical_gpu_homogeneous_equivalence_gate(
        representative_path=args.representative_artifact,
        equivalence_path=args.equivalence_artifact,
        wave=args.wave,
        artifact_root=args.artifact_root,
    )
    write_gate_outputs(report, json_path=output, markdown_path=markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pass": report["pass"],
                "output": str(output),
                "markdown_output": str(markdown),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["pass"]:
        return 0
    return 3 if str(report["status"]).startswith("WAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
