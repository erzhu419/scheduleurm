"""Registered SOTA-universe closure gate.

This gate separates two different statements:

* the finite named-system runtime-probe claim currently supported by same-host
  same-workload artifacts; and
* the much stronger "arbitrary SOTA" claim, which needs an explicit registry
  of additional systems and comparable adapter rows for each of them.

The gate is read-only.  It does not launch external systems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "sota_universe_registry_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "sota_universe_registry_gate_20260614.md"
SOTA_FULLSTACK = ARTIFACT_ROOT / "sota_fullstack_superiority_gate_20260613.json"
REGISTERED_RUNTIME = ARTIFACT_ROOT / "registered_sota_runtime_gate_20260614.json"


REGISTERED_SYSTEMS = (
    {
        "system": "Gavel",
        "domain": "heterogeneous deep-learning cluster scheduling",
        "class": "named_runtime_probe",
        "adapter": "gavel_simulation",
        "local_repo": "reference/repos/gavel",
    },
    {
        "system": "Pollux/AdaptDL",
        "domain": "goodput-aware elastic deep-learning scheduling",
        "class": "named_runtime_probe",
        "adapter": "pollux_adaptdl_scheduler",
        "local_repo": "reference/repos/adaptdl_pollux",
    },
    {
        "system": "Sia",
        "domain": "goodput-aware multi-cluster deep-learning scheduling",
        "class": "named_runtime_probe",
        "adapter": "sia_goodput_scheduler",
        "local_repo": "reference/repos/sia_artifacts",
    },
    {
        "system": "IADeep",
        "domain": "Kubernetes GPU-sharing scheduler/extender",
        "class": "named_runtime_probe",
        "adapter": "iadeep_kubernetes_extender",
        "local_repo": "reference/repos/iadeep",
    },
    {
        "system": "Salus",
        "domain": "GPU-sharing execution service",
        "class": "named_runtime_probe",
        "adapter": "salus_gpu_sharing",
        "local_repo": "reference/repos/salus",
    },
    {
        "system": "Decima",
        "domain": "Spark DAG scheduling simulator",
        "class": "adjacent_simulator",
        "adapter": "decima_simulator",
        "local_repo": "reference/repos/decima_sim",
    },
    {
        "system": "Tiresias",
        "domain": "GPU cluster manager without complete job information",
        "class": "registered_not_yet_fullstack_compared",
        "adapter": "tiresias_gpu_cluster_manager",
        "local_repo": "reference/repos/tiresias",
    },
    {
        "system": "Themis",
        "domain": "finish-time fairness GPU cluster scheduling",
        "class": "paper_only_registered",
        "adapter": "themis_finish_time_fairness",
        "local_repo": "",
    },
    {
        "system": "Gandiva",
        "domain": "introspective DL cluster scheduling with migration/time-slicing",
        "class": "paper_only_registered",
        "adapter": "gandiva_introspective_gpu_scheduler",
        "local_repo": "",
    },
    {
        "system": "Shockwave",
        "domain": "dynamic adaptation and finish-time fairness scheduling",
        "class": "registered_not_yet_fullstack_compared",
        "adapter": "shockwave_dynamic_adaptation",
        "local_repo": "reference/repos/shockwave",
    },
    {
        "system": "AlloX",
        "domain": "heterogeneous accelerator scheduling baseline",
        "class": "registered_not_yet_fullstack_compared",
        "adapter": "allox_heterogeneous_accelerator",
        "local_repo": "reference/repos/allox",
    },
    {
        "system": "Optimus",
        "domain": "distributed deep-learning resource scheduling",
        "class": "registered_not_yet_fullstack_compared",
        "adapter": "optimus_distributed_dl_scheduler",
        "local_repo": "reference/repos/optimus",
    },
)


def build_sota_universe_registry_gate() -> dict[str, Any]:
    fullstack = _load_json(SOTA_FULLSTACK)
    runtime = _load_json(REGISTERED_RUNTIME)
    runtime_rows = {
        str(row.get("system")): row
        for row in runtime.get("rows") or []
    }
    fullstack_rows = {
        str(row.get("adapter")): row
        for row in fullstack.get("rows") or []
    }
    rows: list[dict[str, Any]] = []
    for item in REGISTERED_SYSTEMS:
        adapter = str(item["adapter"])
        fullstack_row = fullstack_rows.get(adapter) or {}
        named_ready = bool(
            fullstack_row.get("scoped_runtime_probe_ready")
            or fullstack_row.get("same_workload_full_stack_ready")
        )
        named_superiority = bool(fullstack_row.get("scoped_runtime_probe_native_better"))
        comparable_gpu_claim = item["class"] == "named_runtime_probe" and named_ready and named_superiority
        adjacent_simulator = item["class"] == "adjacent_simulator"
        repo_path = REPO_ROOT / str(item.get("local_repo") or "__missing__")
        runtime_row = runtime_rows.get(str(item["system"])) or {}
        rows.append({
            **item,
            "repo_present": bool(item.get("local_repo") and repo_path.exists()),
            "runtime_inventory_ready": bool(runtime_row.get("repo_or_paper_inventory_ready")),
            "entrypoint_smoke_ready": bool(runtime_row.get("entrypoint_smoke_ready")),
            "same_host_same_workload_runtime_probe_ready": named_ready,
            "same_host_same_workload_fullstack_ready": False,
            "paired_native_superiority_ready": named_superiority,
            "comparable_gpu_scheduler_claim_ready": comparable_gpu_claim,
            "adjacent_simulator_only": adjacent_simulator,
            "blocker": _blocker(item, named_ready, named_superiority, runtime_row),
        })
    named_rows = [row for row in rows if row["class"] == "named_runtime_probe"]
    registered_comparable_rows = [
        row for row in rows
        if row["class"] != "adjacent_simulator"
    ]
    named_ready = all(row["comparable_gpu_scheduler_claim_ready"] for row in named_rows)
    registered_ready = all(row["comparable_gpu_scheduler_claim_ready"] for row in registered_comparable_rows)
    return {
        "gate": "sota_universe_registry_gate",
        "status": (
            "NAMED_FIVE_RUNTIME_PROBE_READY_REGISTERED_UNIVERSE_PENDING"
            if named_ready else "NAMED_FIVE_RUNTIME_PROBE_PENDING"
        ),
        "registered_system_count": len(rows),
        "named_direct_system_count": len(named_rows),
        "named_runtime_probe_system_count": len(named_rows),
        "named_direct_runtime_probe_ready_count": sum(
            1 for row in named_rows if row["same_host_same_workload_runtime_probe_ready"]
        ),
        "named_direct_native_better_ready_count": sum(
            1 for row in named_rows if row["paired_native_superiority_ready"]
        ),
        "registered_gpu_comparable_count": len(registered_comparable_rows),
        "registered_gpu_comparable_ready_count": sum(
            1 for row in registered_comparable_rows if row["comparable_gpu_scheduler_claim_ready"]
        ),
        "registered_extension_runtime_inventory_ready": bool(
            runtime.get("scoped_claim_ready") or runtime.get("pass")
        ),
        "registered_extension_entrypoint_smoke_ready_count": runtime.get("entrypoint_smoke_ready_count", 0),
        "named_five_runtime_probe_ready": bool(named_ready),
        "named_five_fullstack_superiority_ready": False,
        "registered_sota_universe_superiority_ready": bool(registered_ready),
        "arbitrary_sota_superiority_ready": False,
        "scoped_claim_ready": bool(named_ready),
        "strong_claim_ready": False,
        "rows": rows,
        "pass": True,
        "scope": (
            "The named five runtime-probe claim is ready when Gavel, "
            "Pollux/AdaptDL, Sia, IADeep, and Salus have scoped same-host "
            "same-workload rows with paired native-better evidence.  The "
            "registered-universe claim additionally requires comparable "
            "adapter rows for every registered GPU/DL scheduler.  The "
            "unbounded 'arbitrary SOTA' claim is not a finite experimental "
            "statement."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Universe Registry Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `registered_system_count` | {report.get('registered_system_count')} |",
        f"| `named_direct_runtime_probe_ready_count` | {report.get('named_direct_runtime_probe_ready_count')} |",
        f"| `named_direct_native_better_ready_count` | {report.get('named_direct_native_better_ready_count')} |",
        f"| `registered_gpu_comparable_ready_count` | {report.get('registered_gpu_comparable_ready_count')} |",
        f"| `registered_gpu_comparable_count` | {report.get('registered_gpu_comparable_count')} |",
        f"| `named_five_runtime_probe_ready` | {str(bool(report.get('named_five_runtime_probe_ready'))).lower()} |",
        f"| `named_five_fullstack_superiority_ready` | {str(bool(report.get('named_five_fullstack_superiority_ready'))).lower()} |",
        f"| `registered_sota_universe_superiority_ready` | {str(bool(report.get('registered_sota_universe_superiority_ready'))).lower()} |",
        f"| `arbitrary_sota_superiority_ready` | {str(bool(report.get('arbitrary_sota_superiority_ready'))).lower()} |",
        "",
        "## Registry Rows",
        "",
        "| System | Class | Repo | Runtime inventory | Entrypoint smoke | Same-workload runtime probe | Paired native-better | Comparable claim | Blocker |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {system} | `{cls}` | {repo} | {runtime} | {smoke} | {full} | {sup} | {claim} | {blocker} |".format(
                system=row.get("system"),
                cls=row.get("class"),
                repo=str(bool(row.get("repo_present"))).lower(),
                runtime=str(bool(row.get("runtime_inventory_ready"))).lower(),
                smoke=str(bool(row.get("entrypoint_smoke_ready"))).lower(),
                full=str(bool(row.get("same_host_same_workload_runtime_probe_ready"))).lower(),
                sup=str(bool(row.get("paired_native_superiority_ready"))).lower(),
                claim=str(bool(row.get("comparable_gpu_scheduler_claim_ready"))).lower(),
                blocker=_md(row.get("blocker")),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _blocker(item: Mapping[str, Any], ready: bool, superiority: bool, runtime_row: Mapping[str, Any]) -> str:
    cls = item.get("class")
    if cls == "named_runtime_probe" and ready and superiority:
        return "none for the scoped same-host same-workload runtime-probe row"
    if cls == "adjacent_simulator":
        return "adjacent simulator domain; needs a Spark-DAG comparison protocol, not a GPU co-location runtime-probe row"
    if cls == "paper_only_registered":
        return str(runtime_row.get("blocker") or "paper-only registered system; no official runtime row is available in the current package")
    if runtime_row:
        return str(runtime_row.get("blocker") or "runtime inventory exists, but same-workload full-stack comparison is not closed")
    return (
        "registered system lacks an artifacted same-host same-workload full-stack "
        "run and paired native comparison in the current package"
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_universe_registry_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_universe_registry_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build registered SOTA-universe closure gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
