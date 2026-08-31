"""Certify normalized drift slack for all measured GPU execution classes.

The certificate consumes the hash-linked empty-state service-cache merge.  It
does not estimate service again: each class is scaled by one admitted,
hardware-local lower-service action, so the heterogeneous physical units become
dimensionless before the drift budget is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.experiments.critical_gpu_all_hardware_lcb_cache_merge import (
    DEFAULT_CACHE_OUTPUT,
    DEFAULT_REPORT,
)
from algorithm.experiments.critical_gpu_completion_campaign import NODE_SPECS
from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_gpu_all_hardware_statewise_slack_20260810.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "critical_gpu_all_hardware_statewise_slack_20260810.md"
EXPECTED_NODES = ("jtl110gpu", "jtl110gpu2", "node007", "jtl311linux")
EXPECTED_WORKLOADS = frozenset(
    {
        "gpu_cnn_torch_resnet50",
        "gpu_heavy_jax_matmul",
        "gpu_llm_distilgpt2",
        "hybrid_rl_resac_ant",
    }
)
LOAD_FRACTION = 0.80
NODE007_CORRECTION_PROTOCOL = "critical_gpu_p10_transport_correction_v2"


def build_critical_gpu_all_hardware_statewise_slack_certificate(
    *,
    cache_path: Path = DEFAULT_CACHE_OUTPUT,
    merge_report_path: Path = DEFAULT_REPORT,
    load_fraction: float = LOAD_FRACTION,
) -> dict[str, Any]:
    load = float(load_fraction)
    if not 0.0 < load < 1.0:
        raise ValueError("load_fraction must lie in (0, 1)")

    cache_source = Path(cache_path).resolve()
    report_source = Path(merge_report_path).resolve()
    common = {
        "gate": "critical_gpu_all_hardware_statewise_slack_certificate",
        "schema_version": 1,
        "status": "WAIT_SOURCES",
        "pass": False,
        "load_fraction": load,
        "cache_path": str(cache_source),
        "merge_report_path": str(report_source),
        "expected_nodes": list(EXPECTED_NODES),
        "nodes": [],
        "errors": [],
    }
    missing = [
        str(path)
        for path in (cache_source, report_source)
        if not path.is_file()
    ]
    if missing:
        return {**common, "wait_reasons": [{"code": "MISSING_SOURCE", "paths": missing}]}

    try:
        cache_bytes = cache_source.read_bytes()
        report_bytes = report_source.read_bytes()
        cache = ServiceRateCache.from_snapshot(json.loads(cache_bytes.decode("utf-8")))
        merge = json.loads(report_bytes.decode("utf-8"))
        provenance = _validated_provenance(
            merge,
            cache_sha256=hashlib.sha256(cache_bytes).hexdigest(),
            report_path=report_source,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            **common,
            "status": "FAIL_SOURCE",
            "errors": [{"code": "INVALID_SOURCE", "detail": f"{type(exc).__name__}: {exc}"}],
        }

    errors: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    for node in EXPECTED_NODES:
        source = provenance[node]
        gate = source["payload"]
        execution_class = str(source["execution_class"])
        admitted = gate.get("certificate", {}).get("rows") or []
        excluded = gate.get("training_capacity_excluded_cells") or []
        if len(admitted) + len(excluded) != 9:
            errors.append(
                {
                    "code": "REGISTERED_SUPPORT_COUNT_MISMATCH",
                    "node": node,
                    "admitted": len(admitted),
                    "excluded": len(excluded),
                }
            )
            continue

        by_workload: dict[str, list[Any]] = {}
        for row in admitted:
            workload_key = str(row.get("workload_key") or "")
            lookup = cache.lookup_statewise_exact(
                workload_key,
                workload_env=str(row.get("workload_env") or ""),
                node_bucket=execution_class,
                resource_state="empty",
                allocation_workers=1,
                colocation_count=int(row.get("profile") or 0),
                resident_mix="",
            )
            if not lookup.is_exact or lookup.record is None:
                errors.append(
                    {
                        "code": "MISSING_HASH_LINKED_EXACT_ACTION",
                        "node": node,
                        "workload_key": workload_key,
                        "profile": int(row.get("profile") or 0),
                    }
                )
                continue
            record = lookup.record
            if not (
                record.stable_rate_ready
                and record.completion_model_ready
                and record.completion_model_sample_count >= 12
                and record.aggregate_rate > 0.0
                and "hist" not in record.eta_source.lower()
            ):
                errors.append(
                    {
                        "code": "ACTION_NOT_THEOREM_READY",
                        "node": node,
                        "workload_key": workload_key,
                        "profile": record.profile,
                    }
                )
                continue
            by_workload.setdefault(workload_key, []).append(record)

        if set(by_workload) != EXPECTED_WORKLOADS:
            errors.append(
                {
                    "code": "WORKLOAD_CLASS_COVERAGE_MISMATCH",
                    "node": node,
                    "observed": sorted(by_workload),
                    "expected": sorted(EXPECTED_WORKLOADS),
                }
            )
            continue

        selected = []
        for workload_key in sorted(by_workload):
            best = max(by_workload[workload_key], key=lambda record: record.aggregate_rate)
            physical_lower_service = len(NODE_SPECS[node].gpus) * float(best.aggregate_rate)
            selected.append(
                {
                    "workload_key": workload_key,
                    "workload_env": best.workload_env,
                    "selected_profile": int(best.profile),
                    "service_unit": best.unit,
                    "physical_lower_service_units_per_s": physical_lower_service,
                    "diagonal_normalization_scale_units_per_s": physical_lower_service,
                    "normalized_dedicated_service": 1.0,
                    "eta_source": best.eta_source,
                    "completion_model_sample_count": int(best.completion_model_sample_count),
                }
            )

        class_count = len(selected)
        stationary_weight = 1.0 / class_count
        normalized_service = stationary_weight
        normalized_arrival = load * normalized_service
        delta = normalized_service - normalized_arrival
        error_budget = {
            "L_rho": 0.0,
            "epsilon_est": 0.0,
            "beta": 0.0,
            "alpha_1": 0.0,
            "reason": (
                "exact measured dedicated actions; one-sided admitted lower service; "
                "unpenalized exact finite MaxWeight oracle"
            ),
        }
        eta = delta - sum(float(error_budget[key]) for key in ("L_rho", "epsilon_est", "beta", "alpha_1"))
        alpha = float(source["miscoverage_alpha"])
        node_rows.append(
            {
                "node": node,
                "operational_execution_class": execution_class,
                "source_gate_path": str(source["path"]),
                "source_gate_sha256": str(source["sha256"]),
                "class_count": class_count,
                "classes": selected,
                "candidate_actions": [
                    {
                        "action": f"dedicate_all_gpus_to_{row['workload_key']}",
                        "normalized_service_vector": {
                            other["workload_key"]: (
                                1.0
                                if other["workload_key"] == row["workload_key"]
                                else 0.0
                            )
                            for other in selected
                        },
                    }
                    for row in selected
                ],
                "stationary_mix": {
                    f"dedicate_all_gpus_to_{row['workload_key']}": stationary_weight
                    for row in selected
                },
                "normalized_stationary_service_per_class": normalized_service,
                "normalized_bernoulli_arrival_mean_per_class": normalized_arrival,
                "delta": delta,
                "error_budget": error_budget,
                "eta": eta,
                "bounded_drift_constant_B": 0.5 * class_count * 2.0,
                "miscoverage_alpha": alpha,
                "nominal_lcb_coverage": 1.0 - alpha,
                "ready": class_count == len(EXPECTED_WORKLOADS) and eta > 0.0,
            }
        )

    all_ready = bool(
        not errors
        and len(node_rows) == len(EXPECTED_NODES)
        and all(row["ready"] for row in node_rows)
    )
    simultaneous_coverage = max(
        0.0,
        1.0 - sum(float(row["miscoverage_alpha"]) for row in node_rows),
    )
    return {
        **common,
        "status": "PASS" if all_ready else "FAIL",
        "pass": all_ready,
        "cache_sha256": hashlib.sha256(cache_bytes).hexdigest(),
        "merge_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "stochastic_model": (
            "four independent Bernoulli normalized-work arrival streams per "
            "hardware-local queueing system with exact finite dedicated actions"
        ),
        "normalization": (
            "diagonal class-specific scale equal to the admitted hardware-local "
            "physical lower service of the selected dedicated action"
        ),
        "nodes": node_rows,
        "errors": errors,
        "minimum_eta": min((row["eta"] for row in node_rows), default=0.0),
        "simultaneous_four_node_union_bound_coverage": simultaneous_coverage,
        "claim_boundary": (
            "PASS certifies four separate hardware-local, four-class stochastic "
            "models at the declared load, conditional on the hash-linked one-sided "
            "LCB events. Diagonal normalization resolves heterogeneous service units. "
            "The simultaneous four-node value is only the displayed union-bound lower "
            "bound; the primary statistical statement remains per-node. This does not "
            "estimate organic production arrivals, certify loaded co-location dynamics, "
            "or pool rates across physical nodes."
        ),
    }


def _validated_provenance(
    merge: Mapping[str, Any],
    *,
    cache_sha256: str,
    report_path: Path,
) -> dict[str, dict[str, Any]]:
    if not (
        merge.get("gate") == "critical_gpu_all_hardware_lcb_cache_merge"
        and merge.get("status") == "PASS"
        and merge.get("pass") is True
        and merge.get("all_four_gpu_nodes_ready") is True
    ):
        raise ValueError(f"{report_path}: all-hardware merge is not passing")
    expected_cache_hash = str(merge.get("cache_output_sha256") or "")
    if not expected_cache_hash or expected_cache_hash != cache_sha256:
        raise ValueError(f"{report_path}: cache hash mismatch")

    available_path = _resolve_linked_path(
        merge.get("available_report_path"),
        anchor=report_path,
    )
    available_bytes = available_path.read_bytes()
    if hashlib.sha256(available_bytes).hexdigest() != str(merge.get("available_report_sha256") or ""):
        raise ValueError(f"{report_path}: available report hash mismatch")
    available = json.loads(available_bytes.decode("utf-8"))
    if not (
        available.get("gate") == "critical_gpu_available_lcb_cache_merge"
        and available.get("status") == "PASS"
        and available.get("pass") is True
    ):
        raise ValueError(f"{available_path}: available merge is not passing")

    execution_classes = dict(available.get("node_to_operational_execution_class") or {})
    execution_classes[str(merge.get("new_physical_node") or "")] = str(
        merge.get("new_operational_execution_class") or ""
    )
    audits = {
        str(row.get("node") or ""): row
        for row in available.get("source_gate_audits") or []
    }
    audits[str(merge.get("new_physical_node") or "")] = {
        "node": merge.get("new_physical_node"),
        "path": merge.get("jtl311_gate_path"),
        "sha256": merge.get("jtl311_gate_sha256"),
    }
    if set(audits) != set(EXPECTED_NODES):
        raise ValueError(f"source gate set mismatch: {sorted(audits)!r}")

    out: dict[str, dict[str, Any]] = {}
    for node in EXPECTED_NODES:
        audit = audits[node]
        path = _resolve_linked_path(audit.get("path"), anchor=available_path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != str(audit.get("sha256") or ""):
            raise ValueError(f"{path}: source gate hash mismatch")
        payload = json.loads(raw.decode("utf-8"))
        if not _source_gate_contract_ready(payload, node=node):
            raise ValueError(f"{path}: source gate is not passing for {node}")
        alpha = float(payload.get("miscoverage_alpha"))
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"{path}: invalid miscoverage alpha")
        execution_class = str(execution_classes.get(node) or "")
        if not execution_class:
            raise ValueError(f"{report_path}: no execution class for {node}")
        out[node] = {
            "path": path,
            "sha256": digest,
            "payload": payload,
            "execution_class": execution_class,
            "miscoverage_alpha": alpha,
        }
    return out


def _source_gate_contract_ready(payload: Mapping[str, Any], *, node: str) -> bool:
    common_ready = bool(
        payload.get("status") == "PASS"
        and payload.get("pass") is True
        and payload.get("certificate_ready") is True
        and payload.get("node") == node
    )
    if not common_ready:
        return False
    if payload.get("gate") == "critical_gpu_stochastic_lcb_gate":
        return True
    return bool(
        node == "node007"
        and payload.get("gate") == "critical_gpu_p10_transport_correction_gate"
        and payload.get("correction_measurement_protocol")
        == NODE007_CORRECTION_PROTOCOL
        and payload.get("correction_scope")
        == "pre_registered_measurement_transport_integrity"
        and payload.get("performance_conditioned_selection") is False
        and payload.get("all_measurements_ready") is True
        and payload.get("same_correction_measurement_code_all_waves") is True
        and payload.get("same_source_measurement_code_all_waves") is True
    )


def _resolve_linked_path(raw: object, *, anchor: Path) -> Path:
    path = Path(str(raw or "")).expanduser()
    if path.is_file():
        return path.resolve()
    packaged = anchor.parent / path.name
    if packaged.is_file():
        return packaged.resolve()
    raise ValueError(f"missing hash-linked source {path} (also tried {packaged})")


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU All-Hardware Statewise Slack Certificate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Load fraction: `{report.get('load_fraction')}`",
        f"- Minimum eta: `{report.get('minimum_eta')}`",
        f"- Four-node union-bound coverage: `{report.get('simultaneous_four_node_union_bound_coverage')}`",
        "",
        "| Node | Classes | delta | eta | B | Per-node coverage | Ready |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report.get("nodes") or []:
        lines.append(
            "| `{}` | {} | {:.9g} | {:.9g} | {:.9g} | {:.6g} | {} |".format(
                row.get("node"),
                int(row.get("class_count") or 0),
                float(row.get("delta") or 0.0),
                float(row.get("eta") or 0.0),
                float(row.get("bounded_drift_constant_B") or 0.0),
                float(row.get("nominal_lcb_coverage") or 0.0),
                str(bool(row.get("ready"))).lower(),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any],
    *,
    output: Path,
    markdown_output: Path,
) -> None:
    for path, text in (
        (output, json.dumps(report, indent=2, sort_keys=True) + "\n"),
        (markdown_output, markdown_report(report)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_OUTPUT)
    parser.add_argument("--merge-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--load-fraction", type=float, default=LOAD_FRACTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_all_hardware_statewise_slack_certificate(
        cache_path=args.cache,
        merge_report_path=args.merge_report,
        load_fraction=args.load_fraction,
    )
    write_outputs(report, output=args.output, markdown_output=args.markdown_output)
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, sort_keys=True))
    return 0 if report["pass"] else (3 if str(report["status"]).startswith("WAIT") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
