"""Constructive normalized drift-slack certificate for measured GPU actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.experiments.critical_gpu_available_lcb_cache_merge import (
    AVAILABLE_NODES,
    DEFAULT_CACHE_OUTPUT,
    EFFECTIVE_NODE_BUCKETS,
)
from algorithm.experiments.critical_gpu_completion_campaign import (
    TRAINING_WAVES,
    campaign_cells,
)
from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "critical_gpu_statewise_slack_20260808.json"
DEFAULT_MARKDOWN = REPO_ROOT / "md" / "critical_gpu_statewise_slack_20260808.md"
LOAD_FRACTION = 0.80
PER_NODE_MIS_COVERAGE = 0.10


def build_critical_gpu_statewise_slack_certificate(
    *,
    cache_path: Path = DEFAULT_CACHE_OUTPUT,
    load_fraction: float = LOAD_FRACTION,
) -> dict[str, Any]:
    load = float(load_fraction)
    if not 0.0 < load < 1.0:
        raise ValueError("load_fraction must lie in (0, 1)")
    source_path = Path(cache_path).resolve()
    cache = ServiceRateCache.load(source_path)
    node_rows = []
    errors = []
    for node in AVAILABLE_NODES:
        cells = campaign_cells(node=node, wave=TRAINING_WAVES[0])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for cell in cells:
            grouped.setdefault(str(cell["workload_key"]), []).append(cell)
        selected = []
        for workload_key, action_cells in sorted(grouped.items()):
            records = []
            for cell in action_cells:
                lookup = cache.lookup_statewise_exact(
                    workload_key,
                    workload_env=str(cell["workload_env"]),
                    node_bucket=EFFECTIVE_NODE_BUCKETS[node],
                    resource_state="empty",
                    allocation_workers=1,
                    colocation_count=int(cell["profile"]),
                    resident_mix="",
                )
                if not lookup.is_exact or lookup.record is None:
                    errors.append(
                        {
                            "code": "MISSING_EXACT_ACTION",
                            "node": node,
                            "workload_key": workload_key,
                            "profile": int(cell["profile"]),
                        }
                    )
                    continue
                records.append(lookup.record)
            if len(records) != len(action_cells):
                continue
            best = max(records, key=lambda record: float(record.aggregate_rate))
            node_service = len(cells[0]["gpus"]) * float(best.aggregate_rate)
            if node_service <= 0.0:
                errors.append(
                    {
                        "code": "NONPOSITIVE_DEDICATED_SERVICE",
                        "node": node,
                        "workload_key": workload_key,
                    }
                )
                continue
            selected.append(
                {
                    "workload_key": workload_key,
                    "workload_env": best.workload_env,
                    "selected_profile": int(best.profile),
                    "service_unit": best.unit,
                    "physical_lower_service_units_per_s": node_service,
                    "diagonal_normalization_scale_units_per_s": node_service,
                    "normalized_dedicated_service": 1.0,
                    "eta_source": best.eta_source,
                    "completion_model_sample_count": int(
                        best.completion_model_sample_count
                    ),
                }
            )
        class_count = len(selected)
        mixture_weight = 1.0 / class_count if class_count else 0.0
        normalized_mixture_service = mixture_weight
        normalized_arrival_mean = load * normalized_mixture_service
        delta = normalized_mixture_service - normalized_arrival_mean
        error_budget = {
            "L_rho": 0.0,
            "epsilon_est": 0.0,
            "beta": 0.0,
            "alpha_1": 0.0,
            "reason": (
                "exact measured dedicated actions; one-sided admitted lower "
                "service; unpenalized exact finite MaxWeight oracle"
            ),
        }
        eta = delta - sum(
            float(error_budget[key])
            for key in ("L_rho", "epsilon_est", "beta", "alpha_1")
        )
        # Independent Bernoulli arrivals in normalized work units have A_max=1;
        # each dedicated normalized action has mu_max=1 for one class.
        drift_constant_B = 0.5 * class_count * (1.0**2 + 1.0**2)
        node_rows.append(
            {
                "node": node,
                "operational_execution_class": EFFECTIVE_NODE_BUCKETS[node],
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
                    f"dedicate_all_gpus_to_{row['workload_key']}": mixture_weight
                    for row in selected
                },
                "normalized_stationary_service_per_class": normalized_mixture_service,
                "normalized_bernoulli_arrival_mean_per_class": normalized_arrival_mean,
                "load_fraction": load,
                "delta": delta,
                "error_budget": error_budget,
                "eta": eta,
                "bounded_drift_constant_B": drift_constant_B,
                "per_node_nominal_lcb_coverage": 1.0 - PER_NODE_MIS_COVERAGE,
                "ready": bool(
                    class_count == 4
                    and len(selected) == len(grouped)
                    and eta > 0.0
                    and all(
                        row["completion_model_sample_count"] == 12
                        and "hist" not in str(row["eta_source"]).lower()
                        for row in selected
                    )
                ),
            }
        )

    all_ready = bool(
        not errors
        and len(node_rows) == len(AVAILABLE_NODES)
        and all(row["ready"] for row in node_rows)
    )
    global_union_bound_coverage = max(
        0.0,
        1.0 - len(AVAILABLE_NODES) * PER_NODE_MIS_COVERAGE,
    )
    return {
        "gate": "critical_gpu_statewise_slack_certificate",
        "schema_version": 1,
        "status": "PASS" if all_ready else "FAIL",
        "pass": all_ready,
        "cache": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "stochastic_model": (
            "four independent Bernoulli normalized-work arrival streams per "
            "hardware-local queueing system with exact finite dedicated actions"
        ),
        "normalization": (
            "diagonal class-specific scale equal to the admitted physical "
            "lower service of the selected dedicated action"
        ),
        "load_fraction": load,
        "per_node_nominal_lcb_coverage": 1.0 - PER_NODE_MIS_COVERAGE,
        "simultaneous_three_node_union_bound_coverage": global_union_bound_coverage,
        "nodes": node_rows,
        "errors": errors,
        "minimum_eta": min((row["eta"] for row in node_rows), default=0.0),
        "claim_boundary": (
            "PASS is a constructive theorem-condition certificate for three "
            "separate hardware-local, four-class stochastic models at the "
            "declared 0.8 load. The stationary mix and Bernoulli arrivals are "
            "explicit; diagonal normalization resolves heterogeneous step/iter "
            "units; exact measured actions give Lrho=0 and the admitted one-sided "
            "lower service gives epsilon_est=0 on each gate's coverage event. "
            "It is not an estimate of organic production arrival rates, does not "
            "certify mixed-workload co-location actions, and does not include the "
            "pending jtl311linux class. Per-node coverage is 0.9; a joint claim "
            "over all three nodes has only the stated 0.7 union-bound lower bound."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Critical GPU Statewise Slack Certificate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Load fraction: `{report.get('load_fraction')}`",
        f"- Minimum eta: `{report.get('minimum_eta')}`",
        "",
        "| Node | Classes | delta | error budget | eta | B | Ready |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report.get("nodes") or []:
        budget = sum(
            float((row.get("error_budget") or {}).get(key) or 0.0)
            for key in ("L_rho", "epsilon_est", "beta", "alpha_1")
        )
        lines.append(
            "| `{}` | {} | {:.9g} | {:.9g} | {:.9g} | {:.9g} | {} |".format(
                row.get("node"),
                int(row.get("class_count") or 0),
                float(row.get("delta") or 0.0),
                budget,
                float(row.get("eta") or 0.0),
                float(row.get("bounded_drift_constant_B") or 0.0),
                str(bool(row.get("ready"))).lower(),
            )
        )
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_OUTPUT)
    parser.add_argument("--load-fraction", type=float, default=LOAD_FRACTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build_critical_gpu_statewise_slack_certificate(
        cache_path=args.cache,
        load_fraction=args.load_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "pass": report["pass"], "minimum_eta": report["minimum_eta"]}, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
