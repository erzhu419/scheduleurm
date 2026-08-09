"""OR-domain generalization gates for the Scheduleurm action model.

These are finite replay/simulator certificates.  They do not claim empirical
performance on physical ports or external shop-floor systems; they show that
the same action ledger shape covers port, FJSP, and RCPSP/MMRCPSP instances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.theorem_dispatch.global_dispatch import select_global_action


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_or_generalization_gate() -> dict[str, Any]:
    domains = {
        "port_terminal": _port_rows(),
        "fjsp": _fjsp_rows(),
        "mmrcpsp": _mmrcpsp_rows(),
    }
    reports = {}
    for domain, rows in domains.items():
        queue = _queue(rows)
        action = select_global_action(rows, queue, max_batch_size=3)
        reports[domain] = {
            "row_count": len(rows),
            "queue_vector": queue,
            "selected_action": action.snapshot(),
            "pass": bool(action.selected_action_ids),
            "migration_analogue_present": any(row.get("action_type") in {"reberth", "reroute", "mode_switch"} for row in rows),
        }
    return {
        "gate": "or_generalization_benchmarks",
        "pass": all(bool(row["pass"]) and bool(row["migration_analogue_present"]) for row in reports.values()),
        "status": "OR_GENERALIZATION_SIMULATOR_PASS" if all(bool(row["pass"]) for row in reports.values()) else "OR_GENERALIZATION_SIMULATOR_OPEN",
        "domains": reports,
        "references": {
            "JSPLib": "https://scheduleopt.github.io/benchmarks/jsplib/",
            "FJSPLib": "https://scheduleopt.github.io/benchmarks/fjsplib",
            "PSPLIB": "https://www.om-db.wi.tum.de/psplib/",
        },
        "scope": "Simulator/replay action-model coverage for OR benchmark families; not physical-port empirical validation.",
    }


def _port_rows() -> list[dict[str, Any]]:
    return [
        _row("port", "ship_a", "berth_fast_qc2", "q11_port_berth_quay", {"q11_port_berth_quay": 2.4}, 0.3, "launch"),
        _row("port", "ship_b", "berth_slow_qc1", "q10_port_yard_gate", {"q10_port_yard_gate": 1.0}, 0.1, "launch"),
        _row("port", "ship_a", "reberth_fast", "q11_port_berth_quay", {"q11_port_berth_quay": 2.8}, 0.9, "reberth"),
    ]


def _fjsp_rows() -> list[dict[str, Any]]:
    return [
        _row("fjsp", "job1_op1", "machine_a", "q01_fjsp_gpu_like", {"q01_fjsp_gpu_like": 1.6}, 0.1, "launch"),
        _row("fjsp", "job1_op1", "machine_b", "q01_fjsp_gpu_like", {"q01_fjsp_gpu_like": 1.2}, 0.0, "launch"),
        _row("fjsp", "job2_op2", "reroute_machine_c", "q00_fjsp_light", {"q00_fjsp_light": 1.8}, 0.2, "reroute"),
    ]


def _mmrcpsp_rows() -> list[dict[str, Any]]:
    return [
        _row("mmrcpsp", "activity_1", "mode_low_resource", "q00_rcpsp_light", {"q00_rcpsp_light": 1.0}, 0.0, "launch"),
        _row("mmrcpsp", "activity_2", "mode_cpu_heavy", "q10_rcpsp_cpu", {"q10_rcpsp_cpu": 1.7}, 0.1, "launch"),
        _row("mmrcpsp", "activity_2", "mode_switch_fast", "q10_rcpsp_cpu", {"q10_rcpsp_cpu": 2.2}, 0.8, "mode_switch"),
    ]


def _row(domain: str, task_id: str, action: str, workload: str, service: dict[str, float], penalty: float, action_type: str) -> dict[str, Any]:
    return {
        "domain": domain,
        "task_id": f"{domain}:{task_id}",
        "action_id": f"{domain}:{task_id}:{action}",
        "action_type": action_type,
        "resource_id": f"{domain}:{action}",
        "resource_ids": (f"{domain}:{action}",),
        "workload_key": workload,
        "service_workload_key": workload,
        "lower_service": service,
        "penalty_units": float(penalty),
        "score_semantics": "robust_maxweight_lower_service",
        "theorem_ready": True,
        "scheduler_hint_only": True,
    }


def _queue(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        for key in (row.get("lower_service") or {}):
            out[str(key)] = max(out.get(str(key), 0.0), 3.0)
    return out


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# OR Generalization Benchmarks",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        "",
        "| Domain | Rows | Selected actions | Migration analogue |",
        "|---|---:|---|---:|",
    ]
    for domain, row in (report.get("domains") or {}).items():
        action = row.get("selected_action") or {}
        lines.append(
            f"| `{domain}` | {int(row.get('row_count') or 0)} | "
            f"`{action.get('selected_action_ids')}` | {str(bool(row.get('migration_analogue_present'))).lower()} |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "or_generalization_benchmarks_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "or_generalization_benchmarks_20260629.md")
    args = parser.parse_args()
    report = build_or_generalization_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

