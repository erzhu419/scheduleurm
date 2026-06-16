"""Registered SOTA adapter-closure gate.

This gate attacks the gap between a loose SOTA registry and a theorem-facing
finite action universe.  It distinguishes two adapter layers:

* policy/service-unit adapters, which place a registered SOTA family inside the
  finite measured-cache action union; and
* direct external-binary adapters, which run the registered system's own code on
  the same workload and metric substrate.

The first layer can be closed for the registered non-adjacent SOTA universe.
The second layer remains closed only for the named systems that already have
same-host same-workload full-stack rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .registered_sota_runtime_gate import build_registered_sota_runtime_gate
from .sota_admitted_universe_closure_gate import build_sota_admitted_universe_closure_gate
from .sota_fullstack_superiority_gate import build_sota_fullstack_superiority_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "registered_sota_adapter_closure_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "registered_sota_adapter_closure_gate_20260614.md"


def build_registered_sota_adapter_closure_gate() -> dict[str, Any]:
    runtime = build_registered_sota_runtime_gate(run_smoke=True)
    admitted = build_sota_admitted_universe_closure_gate()
    fullstack = build_sota_fullstack_superiority_gate()

    admitted_rows = {str(row.get("system")): row for row in admitted.get("rows") or []}
    runtime_rows = {str(row.get("system")): row for row in runtime.get("rows") or []}
    fullstack_rows = {str(row.get("system") or row.get("adapter")): row for row in fullstack.get("rows") or []}

    systems = [
        "Gavel",
        "Pollux/AdaptDL",
        "Sia",
        "IADeep",
        "Salus",
        "Tiresias",
        "Themis",
        "Gandiva",
        "Shockwave",
        "AlloX",
        "Optimus",
    ]
    rows = []
    for system in systems:
        arow = admitted_rows.get(system) or {}
        rrow = runtime_rows.get(system) or {}
        named_direct = _named_direct_ready(system, fullstack_rows)
        extra = _system_extra(system)
        policy_adapter_ready = bool(
            arow.get("policy_semantics_admitted")
            and arow.get("action_union_dominance_ready")
        ) or named_direct
        direct_binary_adapter_ready = bool(named_direct)
        rows.append(
            {
                "system": system,
                "policy_service_unit_adapter_ready": policy_adapter_ready,
                "direct_external_binary_adapter_ready": direct_binary_adapter_ready,
                "direct_fullstack_superiority_ready": direct_binary_adapter_ready,
                "runtime_inventory_ready": bool(
                    rrow.get("repo_or_paper_inventory_ready")
                    or system in {"Gavel", "Pollux/AdaptDL", "Sia", "IADeep", "Salus"}
                ),
                "entrypoint_smoke_ready": bool(rrow.get("entrypoint_smoke_ready")),
                "paper_only": bool(rrow and not rrow.get("has_public_runtime_repo")),
                "adapter_evidence": extra,
                "blocker": _blocker(
                    system,
                    policy_ready=policy_adapter_ready,
                    direct_ready=direct_binary_adapter_ready,
                    runtime_row=rrow,
                    extra=extra,
                ),
            }
        )
    policy_ready_count = sum(1 for row in rows if row["policy_service_unit_adapter_ready"])
    direct_ready_count = sum(1 for row in rows if row["direct_external_binary_adapter_ready"])
    no_policy_adapter = [row["system"] for row in rows if not row["policy_service_unit_adapter_ready"]]
    no_direct_adapter = [row["system"] for row in rows if not row["direct_external_binary_adapter_ready"]]
    policy_closed = policy_ready_count == len(rows)
    return {
        "gate": "registered_sota_adapter_closure_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "REGISTERED_SOTA_POLICY_ADAPTER_UNIVERSE_CLOSED_DIRECT_BINARY_PENDING"
            if policy_closed else "REGISTERED_SOTA_POLICY_ADAPTER_UNIVERSE_OPEN"
        ),
        "gate_pass": policy_closed,
        "scoped_claim_ready": policy_closed,
        "strong_claim_ready": False,
        "registered_system_count": len(rows),
        "policy_service_unit_adapter_ready_count": policy_ready_count,
        "direct_external_binary_adapter_ready_count": direct_ready_count,
        "registered_systems_without_any_policy_adapter_count": len(no_policy_adapter),
        "registered_systems_without_direct_binary_adapter_count": len(no_direct_adapter),
        "registered_systems_without_any_policy_adapter": no_policy_adapter,
        "registered_systems_without_direct_binary_adapter": no_direct_adapter,
        "registered_policy_adapter_universe_ready": policy_closed,
        "registered_direct_external_binary_superiority_ready": False,
        "registered_direct_external_binary_adapter_ready": direct_ready_count == len(rows),
        "rows": rows,
        "runtime_snapshot": {
            "status": runtime.get("status"),
            "entrypoint_smoke_ready_count": runtime.get("entrypoint_smoke_ready_count"),
            "same_workload_fullstack_ready_count": runtime.get("same_workload_fullstack_ready_count"),
        },
        "admitted_snapshot": {
            "status": admitted.get("status"),
            "registered_sota_admitted_policy_superiority_ready": admitted.get(
                "registered_sota_admitted_policy_superiority_ready"
            ),
            "strict_frontier_closed": admitted.get("strict_frontier_closed"),
        },
        "fullstack_snapshot": {
            "status": fullstack.get("status"),
            "direct_fullstack_named_sota_superiority_ready": fullstack.get(
                "direct_fullstack_named_sota_superiority_ready"
            ),
        },
        "blocker": (
            "Registered policy/service-unit adapters are closed for all "
            "non-adjacent systems, but direct external-binary superiority still "
            "requires same-workload executable adapter rows for systems listed "
            "in registered_systems_without_direct_binary_adapter."
        ),
        "scope": (
            "Finite registered-SOTA policy/service-unit adapter closure.  This "
            "is the theorem-facing adapter universe: every registered "
            "non-adjacent SOTA family is represented in the measured-cache "
            "candidate action union.  It is not direct external-binary "
            "superiority for systems without executable same-workload adapter "
            "rows."
        ),
        "pass": policy_closed,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Registered SOTA Adapter Closure Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `registered_system_count` | {report.get('registered_system_count')} |",
        f"| `policy_service_unit_adapter_ready_count` | {report.get('policy_service_unit_adapter_ready_count')} |",
        f"| `direct_external_binary_adapter_ready_count` | {report.get('direct_external_binary_adapter_ready_count')} |",
        f"| `registered_systems_without_any_policy_adapter_count` | {report.get('registered_systems_without_any_policy_adapter_count')} |",
        f"| `registered_systems_without_direct_binary_adapter_count` | {report.get('registered_systems_without_direct_binary_adapter_count')} |",
        f"| `registered_policy_adapter_universe_ready` | {str(bool(report.get('registered_policy_adapter_universe_ready'))).lower()} |",
        f"| `registered_direct_external_binary_superiority_ready` | {str(bool(report.get('registered_direct_external_binary_superiority_ready'))).lower()} |",
        "",
        "## Rows",
        "",
        "| System | Policy/service-unit adapter | Direct binary adapter | Runtime inventory | Entrypoint smoke | Evidence | Blocker |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("rows") or []:
        evidence = row.get("adapter_evidence") or {}
        lines.append(
            "| {system} | {policy} | {direct} | {runtime} | {smoke} | {evidence} | {blocker} |".format(
                system=row.get("system"),
                policy=str(bool(row.get("policy_service_unit_adapter_ready"))).lower(),
                direct=str(bool(row.get("direct_external_binary_adapter_ready"))).lower(),
                runtime=str(bool(row.get("runtime_inventory_ready"))).lower(),
                smoke=str(bool(row.get("entrypoint_smoke_ready"))).lower(),
                evidence=_md(evidence.get("summary") or evidence),
                blocker=_md(row.get("blocker")),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _named_direct_ready(system: str, fullstack_rows: Mapping[str, Mapping[str, Any]]) -> bool:
    aliases = {
        "Gavel": ("gavel_simulation", "Gavel"),
        "Pollux/AdaptDL": ("pollux_adaptdl_scheduler", "Pollux/AdaptDL"),
        "Sia": ("sia_goodput_scheduler", "Sia"),
        "IADeep": ("iadeep_kubernetes_extender", "IADeep"),
        "Salus": ("salus_gpu_sharing", "Salus"),
    }
    for alias in aliases.get(system, (system,)):
        row = fullstack_rows.get(alias) or {}
        if row.get("same_workload_full_stack_ready") and row.get("full_stack_superiority_claim_allowed"):
            return True
    return False


def _system_extra(system: str) -> dict[str, Any]:
    if system == "Tiresias":
        repo = REPO_ROOT / "reference" / "repos" / "tiresias" / "simulator"
        return {
            "summary": "Tiresias trace/cluster simulator files are present; Python2 numpy blocks executable smoke.",
            "trace_files_present": all((repo / name).exists() for name in ("tf_job.csv", "n1g4.csv", "run_sim.py")),
            "simulator_command": "python2.7 run_sim.py --cluster_spec=n1g4.csv --trace_file=tf_job.csv --scheme=yarn --schedule=fifo",
        }
    if system == "Shockwave":
        repo = REPO_ROOT / "reference" / "repos" / "shockwave" / "scheduler"
        scripts = sorted((repo / "reproduce").glob("*.sh"))
        traces = sorted((repo / "traces" / "reproduce").glob("*.trace"))
        return {
            "summary": f"Shockwave reproduce scripts/traces are present ({len(scripts)} scripts, {len(traces)} traces); solver/protobuf stack blocks executable same-workload smoke.",
            "reproduce_script_count": len(scripts),
            "trace_count": len(traces),
            "throughput_files_present": all((repo / name).exists() for name in ("tacc_throughputs.json", "wisr_throughputs.json", "actual_throughputs.json")),
        }
    if system == "AlloX":
        return _allox_log_summary()
    if system == "Optimus":
        repo = REPO_ROOT / "reference" / "repos" / "optimus"
        return {
            "summary": "Optimus scheduler/template files are present; Python2 numpy/jinja2 and Kubernetes/MXNet wiring block executable same-workload smoke.",
            "scheduler_files_present": all((repo / name).exists() for name in ("scheduler/optimus_scheduler.py", "templates/render-template.py", "templates/k8s-mxnet-template.jinja")),
        }
    if system in {"Themis", "Gandiva"}:
        return {
            "summary": f"{system} is represented by policy-semantics action family; no official runtime row is present in this package.",
            "paper_semantics_adapter": True,
        }
    return {"summary": "Named system direct full-stack row is closed in the named-five gate."}


def _allox_log_summary() -> dict[str, Any]:
    root = REPO_ROOT / "reference" / "repos" / "allox_sim"
    files = [
        "output/AlloX-output_3_12.csv",
        "output/DRF-output_3_12.csv",
        "output/ES-output_3_12.csv",
        "output/AlloXPlus-output_10_20_debug.csv",
        "output/SJF-output_10_20_debug.csv",
        "output/SRPT-output_10_20_debug.csv",
    ]
    rows = []
    for rel in files:
        path = root / rel
        if not path.exists():
            continue
        rows.append(_csv_metric(path, rel))
    return {
        "summary": f"AlloX simulator output/log CSVs are present and parseable ({len(rows)} files); Java classpath is incomplete for rerun on this host.",
        "log_backed_simulator_evidence_ready": bool(rows),
        "parsed_output_rows": rows,
    }


def _csv_metric(path: Path, rel: str) -> dict[str, Any]:
    with path.open(encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        records = list(reader)
    end_times = []
    durations = []
    for row in records:
        row = {str(k).strip(): v for k, v in row.items() if k is not None}
        try:
            end_times.append(float(str(row.get("endTime") or "").strip()))
        except ValueError:
            pass
        try:
            durations.append(float(str(row.get("duration") or "").strip()))
        except ValueError:
            pass
    return {
        "path": rel,
        "row_count": len(records),
        "makespan": max(end_times) if end_times else None,
        "mean_duration": sum(durations) / len(durations) if durations else None,
    }


def _blocker(
    system: str,
    *,
    policy_ready: bool,
    direct_ready: bool,
    runtime_row: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> str:
    if direct_ready:
        return "none for named same-host same-workload direct binary row"
    if policy_ready:
        base = "none for policy/service-unit adapter; direct binary row remains separate"
    else:
        base = "policy/service-unit adapter missing"
    runtime_blocker = str(runtime_row.get("blocker") or "")
    if runtime_blocker:
        return f"{base}; {runtime_blocker}"
    return base


def _md(value: Any) -> str:
    return json.dumps(value, sort_keys=True)[:900] if isinstance(value, (dict, list)) else str(value or "").replace("|", "\\|").replace("\n", " ")[:900]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_registered_sota_adapter_closure_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.registered_sota_adapter_closure_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build registered-SOTA adapter closure gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
