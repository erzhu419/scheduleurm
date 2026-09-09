"""Aggregate gate for broad SOTA/future/deployment claim boundaries.

The paper's strongest safe claim is deliberately narrower than five tempting
universal claims:

1. arbitrary SOTA systems;
2. arbitrary future workloads;
3. multi-node original deployments;
4. Decima-style Spark-DAG scheduling;
5. production-wide organic launched-completion traces.

This module runs the five executable boundary gates and emits one aggregate
certificate.  A scoped gate pass means the paper has a reproducible certificate
and a correct blocker for the broad claim.  It does not turn the corresponding
universal claim into a theorem.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .decima_same_domain_benchmark_gate import build_decima_same_domain_benchmark_gate
from .future_workload_protocol_gate import build_future_workload_protocol_gate
from .multinode_original_deployment_gate import build_multinode_original_deployment_gate
from .organic_production_canary_recorder_gate import DEFAULT_TRACE_PATH as DEFAULT_ORGANIC_TRACE
from .production_load_certificate import _file_fingerprint
from .production_wide_organic_trace_gate import build_production_wide_organic_trace_gate
from .registered_sota_adapter_closure_gate import build_registered_sota_adapter_closure_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


def build_universal_claim_closure_gate(*, probe_remotes: bool = False) -> dict[str, Any]:
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = state_dir / "queue.json"
    archive = state_dir / "queue_archive.jsonl"
    trace = Path(DEFAULT_ORGANIC_TRACE).expanduser()
    return copy.deepcopy(_cached_universal_claim_closure_gate(
        bool(probe_remotes),
        str(trace),
        *_file_fingerprint(trace),
        str(queue),
        *_file_fingerprint(queue),
        str(archive),
        *_file_fingerprint(archive),
    ))


@lru_cache(maxsize=8)
def _cached_universal_claim_closure_gate(
    probe_remotes: bool,
    trace_path: str,
    trace_mtime_ns: int,
    trace_size: int,
    queue_path: str,
    queue_mtime_ns: int,
    queue_size: int,
    archive_path: str,
    archive_mtime_ns: int,
    archive_size: int,
) -> dict[str, Any]:
    gates = {
        "arbitrary_sota_universe": build_registered_sota_adapter_closure_gate(),
        "arbitrary_future_workload": build_future_workload_protocol_gate(),
        "multinode_original_deployment": build_multinode_original_deployment_gate(
            probe_remotes=probe_remotes
        ),
        "decima_spark_dag": build_decima_same_domain_benchmark_gate(),
        "production_wide_organic_trace": build_production_wide_organic_trace_gate(),
    }
    rows = [_row(name, report) for name, report in gates.items()]
    scoped_ready_all = all(row["scoped_claim_ready"] for row in rows)
    strong_ready_all = all(row["strong_claim_ready"] for row in rows)
    return {
        "gate": "universal_claim_closure_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "UNIVERSAL_STRONG_CLAIMS_CLOSED"
            if strong_ready_all else "SCOPED_BOUNDARY_GATES_READY_UNIVERSAL_STRONG_PENDING"
        ),
        "gate_pass": scoped_ready_all,
        "scoped_claim_ready": scoped_ready_all,
        "strong_claim_ready": strong_ready_all,
        "scoped_ready_count": sum(1 for row in rows if row["scoped_claim_ready"]),
        "strong_ready_count": sum(1 for row in rows if row["strong_claim_ready"]),
        "gate_count": len(rows),
        "rows": rows,
        "reports": gates,
        "safe_paper_claim": (
            "Named same-host same-workload runtime-probe gates, measured/admitted future "
            "workload protocol, conservative deployment/fabric boundaries, "
            "registered-SOTA policy/service-unit adapter closure, Decima "
            "same-domain Spark-DAG execution audit, and production organic "
            "history/recorder gate."
        ),
        "forbidden_claim": (
            "Do not claim arbitrary SOTA superiority, arbitrary future workload "
            "positive service, direct external-binary superiority for registered "
            "systems without executable adapter rows, original multi-node "
            "full-stack superiority, Decima GPU co-location superiority, or "
            "production-wide organic launched completion unless the corresponding "
            "strong_ready field is true."
        ),
        "scope": (
            "Aggregate reviewer gate for the five broad-claim directions.  Passing "
            "this gate means the paper has executable certificates and explicit "
            "blockers for each direction.  It is not a universal theorem."
        ),
        "pass": scoped_ready_all,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Universal Claim Closure Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `scoped_ready_count` | {report.get('scoped_ready_count')} / {report.get('gate_count')} |",
        f"| `strong_ready_count` | {report.get('strong_ready_count')} / {report.get('gate_count')} |",
        "",
        "## Gate Matrix",
        "",
        "| Direction | Scoped ready | Strong ready | Status | Blocker |",
        "|---|---:|---:|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{name}` | {scoped} | {strong} | `{status}` | {blocker} |".format(
                name=row.get("name"),
                scoped=str(bool(row.get("scoped_claim_ready"))).lower(),
                strong=str(bool(row.get("strong_claim_ready"))).lower(),
                status=row.get("status"),
                blocker=str(row.get("blocker") or "none").replace("\n", " "),
            )
        )
    lines.extend([
        "",
        "## Safe Paper Claim",
        "",
        str(report.get("safe_paper_claim") or ""),
        "",
        "## Forbidden Claim",
        "",
        str(report.get("forbidden_claim") or ""),
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _row(name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "gate": report.get("gate"),
        "status": report.get("status"),
        "scoped_claim_ready": bool(
            report.get("scoped_claim_ready")
            if "scoped_claim_ready" in report else report.get("pass")
        ),
        "strong_claim_ready": bool(report.get("strong_claim_ready")),
        "pass": bool(report.get("pass")),
        "blocker": report.get("blocker") or _infer_blocker(report),
    }


def _infer_blocker(report: Mapping[str, Any]) -> str:
    for key in (
        "forbidden_claim",
        "remaining_blocker",
        "blockers",
        "scope",
    ):
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_universal_claim_closure_gate(probe_remotes=args.probe_remotes)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.universal_claim_closure_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build aggregate universal-claim boundary gate")
    build.add_argument("--probe-remotes", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "universal_claim_closure_gate_20260614.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "universal_claim_closure_gate_20260614.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
