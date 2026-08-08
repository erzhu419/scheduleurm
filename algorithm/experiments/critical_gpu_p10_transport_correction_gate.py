"""Composite nine-action LCB gate with a pre-registered p10 transport repair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL as SOURCE_PROTOCOL,
    TRAINING_WAVES,
    wave_role,
)
from .critical_gpu_p10_transport_completion_campaign import (
    CAMPAIGN_DATE as CORRECTION_DATE,
    CAMPAIGN_PREFIX as CORRECTION_PREFIX,
    EXPECTED_WAVES,
    MIN_PROGRESS_OBSERVATIONS,
    NODE,
    PROFILE,
    PROTOCOL as CORRECTION_PROTOCOL,
    WORKLOAD_KEY,
    campaign_artifact as correction_campaign_artifact,
    source_campaign_artifact,
)
from .critical_gpu_stochastic_lcb_gate import (
    RESOURCE_STATE,
    _audit_measurement_row,
    _audit_wave,
    _build_certificate,
    _expected_cells,
    _issue,
    write_gate_outputs,
)


def build_critical_gpu_p10_transport_correction_gate(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    source_paths: Mapping[int, Path] | None = None,
    correction_paths: Mapping[int, Path] | None = None,
    miscoverage_alpha: float = 0.10,
) -> dict[str, Any]:
    alpha = float(miscoverage_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage_alpha must lie in (0, 1)")
    source = {
        wave: Path(path)
        for wave, path in (
            source_paths
            or {
                wave: source_campaign_artifact(wave, artifact_root=artifact_root)
                for wave in EXPECTED_WAVES
            }
        ).items()
    }
    corrections = {
        wave: Path(path)
        for wave, path in (
            correction_paths
            or {
                wave: correction_campaign_artifact(wave, artifact_root=artifact_root)
                for wave in EXPECTED_WAVES
            }
        ).items()
    }
    expected_cells = _expected_cells(NODE)
    p10_ids = [
        cell_id
        for cell_id, cell in expected_cells.items()
        if cell["workload_key"] == WORKLOAD_KEY
        and int(cell["profile"]) == PROFILE
    ]
    if len(expected_cells) != 9 or len(p10_ids) != 1:
        raise RuntimeError("node007 correction gate requires exactly nine cells and one p10")
    p10_id = p10_ids[0]

    observations: dict[tuple[int, str], dict[str, Any]] = {}
    validation_errors: list[dict[str, Any]] = []
    wait_reasons: list[dict[str, Any]] = []
    wave_audits: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    correction_artifacts: list[dict[str, Any]] = []
    source_code_hashes: set[str] = set()
    correction_code_hashes: set[str] = set()

    for wave in EXPECTED_WAVES:
        source_payload, source_raw = _load_artifact(
            source.get(wave),
            wave=wave,
            kind="source_v8",
            wait_reasons=wait_reasons,
            validation_errors=validation_errors,
        )
        correction_payload, correction_raw = _load_artifact(
            corrections.get(wave),
            wave=wave,
            kind="p10_transport_correction",
            wait_reasons=wait_reasons,
            validation_errors=validation_errors,
        )
        if source_payload is None or correction_payload is None:
            continue

        source_sha = hashlib.sha256(source_raw).hexdigest()
        correction_sha = hashlib.sha256(correction_raw).hexdigest()
        source_manifest_sha = str(
            (source_payload.get("measurement_code_manifest") or {}).get("sha256")
            or ""
        )
        correction_manifest_sha = str(
            (correction_payload.get("measurement_code_manifest") or {}).get("sha256")
            or ""
        )
        source_code_hashes.add(source_manifest_sha)
        correction_code_hashes.add(correction_manifest_sha)
        source_artifacts.append(
            {
                "wave": wave,
                "path": str(source[wave]),
                "sha256": source_sha,
                "measurement_code_sha256": source_manifest_sha,
            }
        )
        correction_artifacts.append(
            {
                "wave": wave,
                "path": str(corrections[wave]),
                "sha256": correction_sha,
                "measurement_code_sha256": correction_manifest_sha,
            }
        )

        source_audit, source_waits, source_errors, source_observations = _audit_wave(
            node=NODE,
            wave=wave,
            payload=source_payload,
            expected_cells=expected_cells,
        )
        wait_reasons.extend(source_waits)
        accepted_transport_errors = [
            error
            for error in source_errors
            if _known_p10_transport_error(error, p10_id=p10_id)
        ]
        unexpected_source_errors = [
            error
            for error in source_errors
            if not _known_p10_transport_error(error, p10_id=p10_id)
        ]
        validation_errors.extend(unexpected_source_errors)
        if p10_id in source_observations:
            validation_errors.append(
                _issue(
                    "SOURCE_P10_NOT_TRANSPORT_DEFECTIVE",
                    wave=wave,
                    cell=p10_id,
                    detail="source p10 unexpectedly passed the unchanged 12-sample audit",
                )
            )
        if not accepted_transport_errors:
            validation_errors.append(
                _issue(
                    "SOURCE_TRANSPORT_DEFECT_NOT_REPRODUCED",
                    wave=wave,
                    cell=p10_id,
                    detail="no registered prefix-truncation progress error was observed",
                )
            )
        unaffected = {
            cell_id: observation
            for cell_id, observation in source_observations.items()
            if cell_id != p10_id
        }
        if len(unaffected) != 8:
            validation_errors.append(
                _issue(
                    "UNAFFECTED_SOURCE_CELL_COUNT_INVALID",
                    wave=wave,
                    detail=f"expected 8 unaffected observations, observed {len(unaffected)}",
                )
            )
        observations.update(
            {(wave, cell_id): observation for cell_id, observation in unaffected.items()}
        )

        correction_errors, correction_observation, correction_audit = (
            _audit_correction_wave(
                wave=wave,
                payload=correction_payload,
                expected=expected_cells[p10_id],
                expected_source_path=source[wave],
                expected_source_sha256=source_sha,
            )
        )
        validation_errors.extend(correction_errors)
        if correction_observation is not None and not correction_errors:
            observations[(wave, p10_id)] = correction_observation
        wave_audits.append(
            {
                "wave": wave,
                "split_role": wave_role(wave),
                "source_v8_audit": source_audit,
                "accepted_source_transport_error_count": len(accepted_transport_errors),
                "unexpected_source_error_count": len(unexpected_source_errors),
                "unaffected_source_observation_count": len(unaffected),
                "correction_audit": correction_audit,
                "composite_observation_count": sum(
                    (wave, cell_id) in observations for cell_id in expected_cells
                ),
            }
        )

    if source_code_hashes and ("" in source_code_hashes or len(source_code_hashes) != 1):
        validation_errors.append(
            _issue(
                "SOURCE_MEASUREMENT_CODE_IDENTITY_MISMATCH",
                detail=f"observed source hashes {sorted(source_code_hashes)!r}",
            )
        )
    if correction_code_hashes and (
        "" in correction_code_hashes or len(correction_code_hashes) != 1
    ):
        validation_errors.append(
            _issue(
                "CORRECTION_MEASUREMENT_CODE_IDENTITY_MISMATCH",
                detail=f"observed correction hashes {sorted(correction_code_hashes)!r}",
            )
        )

    expected_observation_count = len(EXPECTED_WAVES) * len(expected_cells)
    all_measurements_ready = bool(
        not wait_reasons
        and not validation_errors
        and len(observations) == expected_observation_count
    )
    certificate: dict[str, Any] = {
        "constructed": False,
        "finite_sample_rank_ready": False,
        "rows": [],
    }
    if all_measurements_ready:
        certificate = _build_certificate(
            observations=observations,
            expected_cells=expected_cells,
            alpha=alpha,
        )
        if not certificate.get("finite_sample_rank_ready"):
            validation_errors.append(
                _issue("CONFORMAL_RANK_UNAVAILABLE", detail="finite-sample rank unavailable")
            )
        elif not certificate.get("all_holdout_bounds_valid"):
            validation_errors.append(
                _issue("HOLDOUT_NOT_COVERED", detail="composite holdout bound violated")
            )

    if validation_errors:
        status = "FAIL_VALIDATION"
    elif wait_reasons:
        status = "WAIT_MISSING_CORRECTION_WAVES"
    elif certificate.get("constructed") and certificate.get("all_holdout_bounds_valid"):
        status = "PASS"
    else:
        status = "FAIL_CERTIFICATE"
    passed = status == "PASS"
    node_spec = NODE_SPECS[NODE]
    return {
        "gate": "critical_gpu_p10_transport_correction_gate",
        "schema_version": 1,
        "status": status,
        "pass": passed,
        "certificate_ready": passed,
        "node": NODE,
        "node_bucket": node_spec.node_bucket,
        "hardware_class": node_spec.hardware_class,
        "resource_state": RESOURCE_STATE,
        "source_measurement_protocol": SOURCE_PROTOCOL,
        "correction_measurement_protocol": CORRECTION_PROTOCOL,
        "correction_scope": "pre_registered_measurement_transport_integrity",
        "performance_conditioned_selection": False,
        "expected_cell_count": len(expected_cells),
        "expected_wave_count": len(EXPECTED_WAVES),
        "expected_observation_count": expected_observation_count,
        "ready_observation_count": len(observations),
        "training_waves": list(TRAINING_WAVES),
        "calibration_waves": list(CALIBRATION_WAVES),
        "holdout_wave": HOLDOUT_WAVE,
        "miscoverage_alpha": alpha,
        "all_measurements_ready": all_measurements_ready,
        "same_source_measurement_code_all_waves": bool(
            len(source_code_hashes) == 1 and "" not in source_code_hashes
        ),
        "same_correction_measurement_code_all_waves": bool(
            len(correction_code_hashes) == 1 and "" not in correction_code_hashes
        ),
        "source_measurement_code_sha256": (
            next(iter(source_code_hashes)) if len(source_code_hashes) == 1 else None
        ),
        "correction_measurement_code_sha256": (
            next(iter(correction_code_hashes))
            if len(correction_code_hashes) == 1
            else None
        ),
        "wait_reasons": wait_reasons,
        "validation_errors": validation_errors,
        "expected_cells": [expected_cells[key] for key in sorted(expected_cells)],
        "wave_audits": wave_audits,
        "source_artifacts": source_artifacts,
        "correction_artifacts": correction_artifacts,
        "certificate": certificate,
        "claim_boundary": (
            "This hardware-local empty-state certificate retains the eight valid "
            "node007 v8 actions in every pre-registered wave and replaces only the "
            "DistilGPT2 p10 observation whose aggregate detached log was prefix-"
            "truncated by the transport output cap. The replacement uses the same "
            "train/calibration/holdout wave roles, natural completion, checkpoint "
            "allocation, and unchanged 12-observation child threshold, with exact "
            "chunked transfer verified by remote size and SHA-256. The correction "
            "was fixed by action identity and transport defect, not selected by "
            "measured performance. It does not add new candidate actions, loaded "
            "resource states, hardware classes, or arbitrary future workloads."
        ),
    }


def _audit_correction_wave(
    *,
    wave: int,
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_source_path: Path,
    expected_source_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def require(condition: bool, code: str, detail: str) -> None:
        if not condition:
            errors.append(_issue(code, wave=wave, cell=str(expected["service_cell_id"]), detail=detail))

    require(payload.get("gate") == "critical_gpu_p10_transport_correction_campaign", "CORRECTION_CONTRACT_MISMATCH", "gate differs")
    require(payload.get("measurement_protocol") == CORRECTION_PROTOCOL, "CORRECTION_CONTRACT_MISMATCH", "protocol differs")
    require(payload.get("source_measurement_protocol") == SOURCE_PROTOCOL, "CORRECTION_CONTRACT_MISMATCH", "source protocol differs")
    require(payload.get("campaign_id") == f"{CORRECTION_PREFIX}_{NODE}_r{wave:02d}_{CORRECTION_DATE}", "CORRECTION_CONTRACT_MISMATCH", "campaign id differs")
    require(payload.get("node") == NODE, "CORRECTION_CONTRACT_MISMATCH", "node differs")
    require(int(payload.get("wave") or 0) == wave, "CORRECTION_CONTRACT_MISMATCH", "wave differs")
    require(payload.get("split_role") == wave_role(wave), "CORRECTION_CONTRACT_MISMATCH", "split role differs")
    require(payload.get("correction_scope") == "pre_registered_measurement_transport_integrity", "CORRECTION_CONTRACT_MISMATCH", "scope differs")
    require(payload.get("performance_conditioned_selection") is False, "PERFORMANCE_CONDITIONED_CORRECTION", "correction was performance conditioned")
    require(payload.get("legacy_scheduler_limits_bypassed") is True, "CORRECTION_CONTRACT_MISMATCH", "legacy limits were not bypassed")
    require(payload.get("natural_completion_required") is True, "CORRECTION_CONTRACT_MISMATCH", "natural completion is not required")
    require(payload.get("task_native_progress_required") is True, "CORRECTION_CONTRACT_MISMATCH", "task-native progress is not required")
    require(int(payload.get("minimum_progress_observations_per_child") or 0) == MIN_PROGRESS_OBSERVATIONS, "PROGRESS_THRESHOLD_CHANGED", "12-sample threshold changed")
    require(payload.get("exact_chunked_transfer_required") is True, "TRANSPORT_INTEGRITY_INVALID", "chunked transfer is not required")
    require(payload.get("remote_size_and_sha256_required") is True, "TRANSPORT_INTEGRITY_INVALID", "size/hash check is not required")
    require(payload.get("status") == "PASS" and payload.get("pass") is True, "CORRECTION_NOT_PASSING", "correction campaign is not PASS")
    require(payload.get("measurement_code_unchanged") is True, "CORRECTION_CODE_IDENTITY_INVALID", "measurement code changed")
    selection = payload.get("selection") or {}
    require(selection == {"workload_key": WORKLOAD_KEY, "profile": PROFILE, "pre_registered": True, "performance_conditioned": False}, "CORRECTION_SELECTION_INVALID", "selection differs")
    source_audit = payload.get("source_v8_artifact") or {}
    require(source_audit.get("ready") is True, "SOURCE_TRANSPORT_DEFECT_INVALID", "source defect audit is not ready")
    require(Path(str(source_audit.get("path") or "")) == expected_source_path, "SOURCE_ARTIFACT_MISMATCH", "source path differs")
    require(source_audit.get("sha256") == expected_source_sha256, "SOURCE_ARTIFACT_MISMATCH", "source SHA-256 differs")
    planned = payload.get("planned_cells") or []
    rows = payload.get("rows") or []
    require(len(planned) == 1 and len(rows) == 1, "CORRECTION_CELL_SET_INVALID", "expected one planned and measured row")
    row = rows[0] if len(rows) == 1 else {}
    transport = row.get("transport_integrity_audit") or {}
    require(transport.get("ready") is True, "TRANSPORT_INTEGRITY_INVALID", "transport audit is not ready")
    require(transport.get("transport") == "chunked_base64_64k_exact", "TRANSPORT_INTEGRITY_INVALID", "transport method differs")
    require(transport.get("combined_log_exact_match") is True, "TRANSPORT_INTEGRITY_INVALID", "combined log is not byte exact")
    require(int(transport.get("minimum_progress_observation_count") or 0) >= MIN_PROGRESS_OBSERVATIONS, "PROGRESS_SOURCE_INVALID", "minimum progress support is below 12")
    require(int(transport.get("minimum_interval_sample_count") or 0) >= MIN_PROGRESS_OBSERVATIONS, "PROGRESS_SOURCE_INVALID", "minimum interval support is below 12")
    row_errors, observation = _audit_measurement_row(
        row=row,
        expected=expected,
        wave=wave,
        expected_protocol=CORRECTION_PROTOCOL,
    )
    errors.extend(row_errors)
    return errors, observation if not errors else None, {
        "status": payload.get("status"),
        "pass": bool(payload.get("pass")),
        "planned_cell_count": len(planned),
        "row_count": len(rows),
        "transport_integrity_ready": bool(transport.get("ready")),
        "minimum_progress_observation_count": transport.get("minimum_progress_observation_count"),
        "minimum_interval_sample_count": transport.get("minimum_interval_sample_count"),
        "validation_error_count": len(errors),
    }


def _known_p10_transport_error(error: Mapping[str, Any], *, p10_id: str) -> bool:
    return bool(
        error.get("cell") == p10_id
        and error.get("code") == "PROGRESS_SOURCE_INVALID"
        and re.fullmatch(
            r"child \d+ has too few (progress observations|outer-loop intervals)",
            str(error.get("detail") or ""),
        )
    )


def _load_artifact(
    path: Path | None,
    *,
    wave: int,
    kind: str,
    wait_reasons: list[dict[str, Any]],
    validation_errors: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bytes]:
    if path is None or not path.is_file():
        wait_reasons.append(_issue("MISSING_ARTIFACT", wave=wave, detail=f"{kind}: {path}"))
        return None, b""
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        validation_errors.append(_issue("UNREADABLE_ARTIFACT", wave=wave, detail=f"{kind}: {exc}"))
        return None, b""
    if not isinstance(payload, dict):
        validation_errors.append(_issue("INVALID_ARTIFACT_ROOT", wave=wave, detail=f"{kind}: root is not an object"))
        return None, b""
    return payload, raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output = args.output or (
        args.artifact_root
        / f"critical_gpu_stochastic_lcb_gate_v9_{NODE}_{CORRECTION_DATE}.json"
    )
    markdown = args.markdown_output or output.with_suffix(".md")
    report = build_critical_gpu_p10_transport_correction_gate(
        artifact_root=args.artifact_root,
        miscoverage_alpha=args.alpha,
    )
    write_gate_outputs(report, json_path=output, markdown_path=markdown)
    print(json.dumps({"status": report["status"], "pass": report["pass"], "output": str(output)}, indent=2, sort_keys=True))
    if report["pass"]:
        return 0
    return 3 if str(report["status"]).startswith("WAIT") else 2


if __name__ == "__main__":
    raise SystemExit(main())
