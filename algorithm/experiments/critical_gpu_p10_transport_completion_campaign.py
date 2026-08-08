"""Pre-registered node007 p10 transport-integrity correction campaign.

The original nine-action v8 campaign naturally completed the DistilGPT2 p10
action, but its detached aggregate log exceeded the login bridge output cap.
This protocol remeasures only that policy-reachable action with exact chunked
file transfer.  It is a transport correction, not a performance-conditioned
search over actions or waves.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .critical_gpu_completion_campaign import (
    ARTIFACT_ROOT,
    CALIBRATION_WAVES,
    HOLDOUT_WAVE,
    NODE_SPECS,
    PROTOCOL as SOURCE_PROTOCOL,
    REPO_ROOT,
    RUN_ROOT,
    TEMPLATE_ROOT,
    TRAINING_WAVES,
    WORKLOAD_SPECS,
    _gpu_idle_snapshot,
    _node_environment_gate,
    _run_cell,
    _safe_id,
    _write_report,
    campaign_cells,
    wave_role,
)


PROTOCOL = "critical_gpu_p10_transport_correction_v2"
CAMPAIGN_PREFIX = "critical_gpu_p10_transport_v2"
CAMPAIGN_DATE = "20260808"
NODE = "node007"
WORKLOAD_KEY = "gpu_llm_distilgpt2"
PROFILE = 10
EXPECTED_TASK_COUNT = 40
MIN_PROGRESS_OBSERVATIONS = 12
EXPECTED_WAVES = (*TRAINING_WAVES, *CALIBRATION_WAVES, HOLDOUT_WAVE)

MEASUREMENT_CODE_PATHS = (
    Path(__file__).resolve(),
    REPO_ROOT / "algorithm" / "experiments" / "critical_gpu_completion_campaign.py",
    REPO_ROOT / "algorithm" / "experiments" / "remote_workload_selected_profile_probe.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_wrapper.py",
    REPO_ROOT / "algorithm" / "experiments" / "progress_units.py",
    REPO_ROOT / "algorithm" / "experiments" / "torch_llm_progress_benchmark.py",
    TEMPLATE_ROOT / "torch_llm_distilgpt2.cmd.tpl",
)


def measurement_code_manifest() -> dict[str, Any]:
    files: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(MEASUREMENT_CODE_PATHS, key=lambda item: str(item)):
        data = path.read_bytes()
        relative = str(path.relative_to(REPO_ROOT))
        files.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(data)
        aggregate.update(b"\0")
    return {
        "algorithm": "sha256(path\\0content\\0)",
        "sha256": aggregate.hexdigest(),
        "files": files,
    }


def campaign_artifact(wave: int, *, artifact_root: Path = ARTIFACT_ROOT) -> Path:
    return Path(artifact_root) / (
        f"{CAMPAIGN_PREFIX}_{NODE}_r{int(wave):02d}_{CAMPAIGN_DATE}.json"
    )


def source_campaign_artifact(
    wave: int,
    *,
    artifact_root: Path = ARTIFACT_ROOT,
) -> Path:
    return Path(artifact_root) / (
        f"critical_gpu_completion_v8_{NODE}_r{int(wave):02d}_20260803.json"
    )


def correction_cell(wave: int) -> dict[str, Any]:
    cells = campaign_cells(
        node=NODE,
        wave=int(wave),
        workload_keys=[WORKLOAD_KEY],
        profiles=[PROFILE],
    )
    if len(cells) != 1:
        raise RuntimeError(f"expected one registered p10 cell, observed {len(cells)}")
    return cells[0]


def build_critical_gpu_p10_transport_completion_campaign(
    *,
    wave: int,
    allow_launch: bool = False,
    artifact_root: Path = ARTIFACT_ROOT,
    cleanup_remote_output: bool = True,
) -> dict[str, Any]:
    role = wave_role(int(wave))
    if int(wave) not in EXPECTED_WAVES:
        raise ValueError("transport correction uses pre-registered waves 1..13")
    node_spec = NODE_SPECS[NODE]
    cell = correction_cell(int(wave))
    campaign_id = f"{CAMPAIGN_PREFIX}_{NODE}_r{int(wave):02d}_{CAMPAIGN_DATE}"
    output_path = campaign_artifact(int(wave), artifact_root=artifact_root)
    source_path = source_campaign_artifact(int(wave), artifact_root=artifact_root)
    source_audit = _source_v8_audit(source_path, wave=int(wave))
    code_manifest = measurement_code_manifest()
    report: dict[str, Any] = {
        "gate": "critical_gpu_p10_transport_correction_campaign",
        "measurement_protocol": PROTOCOL,
        "source_measurement_protocol": SOURCE_PROTOCOL,
        "campaign_id": campaign_id,
        "node": NODE,
        "node_spec": asdict(node_spec),
        "wave": int(wave),
        "split_role": role,
        "pre_registered_split": {
            "training": list(TRAINING_WAVES),
            "calibration": list(CALIBRATION_WAVES),
            "holdout": HOLDOUT_WAVE,
            "smoke_excluded": 0,
        },
        "correction_scope": "pre_registered_measurement_transport_integrity",
        "performance_conditioned_selection": False,
        "legacy_scheduler_limits_bypassed": True,
        "algorithm_path": "controlled_theorem_measurement_direct",
        "terminate_on_stable": False,
        "natural_completion_required": True,
        "task_native_progress_required": True,
        "minimum_progress_observations_per_child": MIN_PROGRESS_OBSERVATIONS,
        "exact_chunked_transfer_required": True,
        "remote_size_and_sha256_required": True,
        "coordinated_post_warmup_start_required_for_builtin_gpu_workloads": True,
        "admission_delay_included_in_completion_jct": True,
        "staged_trajectory_completion_evidence_required": True,
        "real_checkpoint_allocation_required": True,
        "source_v8_artifact": source_audit,
        "measurement_code_manifest": code_manifest,
        "allow_launch": bool(allow_launch),
        "selection": {
            "workload_key": WORKLOAD_KEY,
            "profile": PROFILE,
            "pre_registered": True,
            "performance_conditioned": False,
        },
        "planned_cells": [cell],
        "rows": [],
        "status": "MANIFEST_ONLY" if not allow_launch else "RUNNING",
        "pass": False,
    }
    if not source_audit.get("ready"):
        report["status"] = "FAIL_SOURCE_CONTRACT"
        _write_report(report, output_path)
        return report
    if not allow_launch:
        _write_report(report, output_path)
        return report

    env_gate = _node_environment_gate(node_spec)
    report["environment_gate"] = env_gate
    if not env_gate.get("ready"):
        report["status"] = "WAIT_ENVIRONMENT"
        report["blocker"] = env_gate.get("reason")
        _write_report(report, output_path)
        return report
    preflight = _gpu_idle_snapshot(node_spec)
    if not preflight.get("ready"):
        report["status"] = "WAIT_RESOURCE"
        report["rows"] = [{**cell, "status": "WAIT_RESOURCE", "ready": False, "preflight": preflight}]
        _write_report(report, output_path)
        return report

    spec = next(item for item in WORKLOAD_SPECS if item.workload_key == WORKLOAD_KEY)
    row = _run_cell(
        campaign_id=campaign_id,
        node_spec=node_spec,
        spec=spec,
        cell=cell,
        preflight=preflight,
        cleanup_remote_output=cleanup_remote_output,
        expected_code_sha256=str(code_manifest["sha256"]),
        measurement_manifest_builder=measurement_code_manifest,
        measurement_protocol=PROTOCOL,
    )
    transport_audit = _transport_integrity_audit(
        run_id=str(row.get("run_id") or ""),
        summary=row.get("summary") or {},
    )
    row["transport_integrity_audit"] = transport_audit
    if not transport_audit.get("ready"):
        row["ready"] = False
        row["status"] = "FAILED_TRANSPORT_INTEGRITY"
    final_manifest = measurement_code_manifest()
    report["rows"] = [row]
    report["ready_row_count"] = int(bool(row.get("ready")))
    report["final_measurement_code_manifest"] = final_manifest
    report["measurement_code_unchanged"] = bool(
        final_manifest["sha256"] == code_manifest["sha256"]
    )
    report["pass"] = bool(
        row.get("ready")
        and transport_audit.get("ready")
        and report["measurement_code_unchanged"]
    )
    report["status"] = "PASS" if report["pass"] else "INCOMPLETE"
    _write_report(report, output_path)
    return report


def _source_v8_audit(path: Path, *, wave: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "ready": False,
        "expected_transport_defect": "aggregate_log_prefix_truncated_by_transport_cap",
    }
    if not path.is_file():
        result["reason"] = "missing_source_v8_artifact"
        return result
    raw = path.read_bytes()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["reason"] = f"unreadable_source: {exc}"
        return result
    rows = [
        row
        for row in payload.get("rows") or []
        if row.get("workload_key") == WORKLOAD_KEY
        and int(row.get("profile") or 0) == PROFILE
    ]
    deficient_children = []
    if len(rows) == 1:
        deficient_children = [
            int(child.get("global_index") or 0)
            for child in (rows[0].get("summary") or {}).get("rows") or []
            if int((child.get("completion_model") or {}).get("progress_observation_count") or 0)
            < MIN_PROGRESS_OBSERVATIONS
        ]
    result.update(
        {
            "source_gate": payload.get("gate"),
            "source_protocol": payload.get("measurement_protocol"),
            "source_status": payload.get("status"),
            "source_pass": bool(payload.get("pass")),
            "source_p10_row_count": len(rows),
            "source_deficient_child_count": len(deficient_children),
            "source_deficient_child_indices": deficient_children,
        }
    )
    result["ready"] = bool(
        payload.get("gate") == "critical_gpu_completion_campaign"
        and payload.get("measurement_protocol") == SOURCE_PROTOCOL
        and payload.get("node") == NODE
        and int(payload.get("wave") or 0) == int(wave)
        and payload.get("status") == "PASS"
        and payload.get("pass") is True
        and len(rows) == 1
        and rows[0].get("ready") is True
        and len(deficient_children) > 0
    )
    if not result["ready"]:
        result["reason"] = "source_v8_contract_or_expected_transport_defect_missing"
    return result


def _transport_integrity_audit(
    *,
    run_id: str,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    raw_dir = RUN_ROOT / run_id / "raw" / f"profile_{PROFILE}_per_gpu"
    reconstructed = sorted(raw_dir.glob("coordinated_profile_run_fetch_*.reconstructed.log"))
    metadata = sorted(raw_dir.glob("coordinated_profile_run_fetch_*_meta.stdout"))
    combined_path = raw_dir / "coordinated_profile_run.combined.log"
    declared_size = None
    declared_sha256 = None
    if metadata:
        match = re.search(
            r"__SCHEDULEURM_FILE__\s+(\d+)\s+([0-9a-fA-F]{64})",
            metadata[-1].read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            declared_size = int(match.group(1))
            declared_sha256 = match.group(2).lower()
    payload = reconstructed[-1].read_bytes() if reconstructed else b""
    actual_sha256 = hashlib.sha256(payload).hexdigest() if reconstructed else None
    combined = combined_path.read_bytes() if combined_path.is_file() else b""
    children = (summary.get("rows") or []) if isinstance(summary, Mapping) else []
    progress_counts = [
        int((child.get("completion_model") or {}).get("progress_observation_count") or 0)
        for child in children
    ]
    interval_counts = [
        int((child.get("completion_model") or {}).get("interval_sample_count") or 0)
        for child in children
    ]
    ready = bool(
        len(reconstructed) == 1
        and len(metadata) == 1
        and declared_size is not None
        and declared_sha256 is not None
        and len(payload) == declared_size
        and actual_sha256 == declared_sha256
        and combined == payload
        and payload.count(b"__SCHEDULEURM_LOG_BEGIN__ ") == EXPECTED_TASK_COUNT
        and payload.count(b"__SCHEDULEURM_LOG_END__ ") == EXPECTED_TASK_COUNT
        and payload.count(b"__SCHEDULEURM_RC__ ") == EXPECTED_TASK_COUNT
        and len(children) == EXPECTED_TASK_COUNT
        and min(progress_counts, default=0) >= MIN_PROGRESS_OBSERVATIONS
        and min(interval_counts, default=0) >= MIN_PROGRESS_OBSERVATIONS
    )
    return {
        "ready": ready,
        "transport": "chunked_base64_64k_exact",
        "remote_size_bytes": declared_size,
        "remote_sha256": declared_sha256,
        "reconstructed_size_bytes": len(payload),
        "reconstructed_sha256": actual_sha256,
        "combined_log_exact_match": bool(combined and combined == payload),
        "log_begin_count": payload.count(b"__SCHEDULEURM_LOG_BEGIN__ "),
        "log_end_count": payload.count(b"__SCHEDULEURM_LOG_END__ "),
        "returncode_marker_count": payload.count(b"__SCHEDULEURM_RC__ "),
        "child_count": len(children),
        "minimum_progress_observation_count": min(progress_counts, default=0),
        "minimum_interval_sample_count": min(interval_counts, default=0),
        "minimum_required_count": MIN_PROGRESS_OBSERVATIONS,
        "reconstructed_path": str(reconstructed[-1]) if reconstructed else None,
        "metadata_path": str(metadata[-1]) if metadata else None,
    }


def completed_wave(wave: int, *, artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any] | None:
    path = campaign_artifact(wave, artifact_root=artifact_root)
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = report.get("rows") or []
    if not (
        report.get("measurement_protocol") == PROTOCOL
        and report.get("status") == "PASS"
        and report.get("pass") is True
        and len(rows) == 1
        and rows[0].get("ready") is True
        and (rows[0].get("transport_integrity_audit") or {}).get("ready") is True
    ):
        return None
    return report


def run_series(
    *,
    first_wave: int = 1,
    last_wave: int = 13,
    allow_launch: bool = False,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    if not (1 <= int(first_wave) <= int(last_wave) <= 13):
        raise ValueError("series must satisfy 1 <= first_wave <= last_wave <= 13")
    waves: list[dict[str, Any]] = []
    for wave in range(int(first_wave), int(last_wave) + 1):
        prior = completed_wave(wave, artifact_root=artifact_root)
        if prior is not None:
            waves.append({"wave": wave, "status": "SKIPPED_ALREADY_PASS"})
            continue
        if not allow_launch:
            waves.append({"wave": wave, "status": "PENDING_LAUNCH"})
            continue
        report = build_critical_gpu_p10_transport_completion_campaign(
            wave=wave,
            allow_launch=True,
            artifact_root=artifact_root,
        )
        waves.append(
            {
                "wave": wave,
                "status": report.get("status"),
                "campaign_id": report.get("campaign_id"),
            }
        )
        print(json.dumps(waves[-1], sort_keys=True), flush=True)
        if report.get("status") != "PASS":
            break
    expected = int(last_wave) - int(first_wave) + 1
    return {
        "gate": "critical_gpu_p10_transport_completion_series",
        "measurement_protocol": PROTOCOL,
        "first_wave": int(first_wave),
        "last_wave": int(last_wave),
        "allow_launch": bool(allow_launch),
        "waves": waves,
        "pass": sum(
            row["status"] in {"PASS", "SKIPPED_ALREADY_PASS"}
            for row in waves
        ) == expected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-wave", type=int, default=1)
    parser.add_argument("--last-wave", type=int, default=13)
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_series(
        first_wave=args.first_wave,
        last_wave=args.last_wave,
        allow_launch=args.allow_launch,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] or not args.allow_launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
