"""Non-future broad-claim closure gate.

This aggregate gate answers the reviewer question: after excluding arbitrary
future workloads, which broad directions are closed enough for manuscript
language?  It deliberately separates finite, artifacted closures from
unbounded/external-binary claims that remain out of scope.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .decima_same_domain_benchmark_gate import build_decima_same_domain_benchmark_gate
from .decima_spark_dag_gate import build_decima_spark_dag_gate
from .multinode_original_deployment_gate import build_multinode_original_deployment_gate
from .organic_history_completion_gate import build_organic_history_completion_gate
from .production_wide_organic_trace_gate import build_production_wide_organic_trace_gate
from .registered_sota_adapter_closure_gate import build_registered_sota_adapter_closure_gate
from .sota_admitted_universe_closure_gate import build_sota_admitted_universe_closure_gate
from .sota_fullstack_superiority_gate import build_sota_fullstack_superiority_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "non_future_claim_closure_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "non_future_claim_closure_gate_20260614.md"
FULLSTACK_ARTIFACT = ARTIFACT_ROOT / "sota_fullstack_superiority_gate_20260613.json"
ADMITTED_SOTA_ARTIFACT = ARTIFACT_ROOT / "sota_admitted_universe_closure_gate_20260614.json"
REGISTERED_ADAPTER_ARTIFACT = ARTIFACT_ROOT / "registered_sota_adapter_closure_gate_20260614.json"
ORGANIC_HISTORY_ARTIFACT = ARTIFACT_ROOT / "organic_history_completion_gate_20260614.json"
PRODUCTION_WIDE_ARTIFACT = ARTIFACT_ROOT / "production_wide_organic_trace_gate_20260614.json"
MULTINODE_ORIGINAL_ARTIFACT = ARTIFACT_ROOT / "multinode_original_deployment_gate_20260614.json"
DECIMA_SPARK_ARTIFACT = ARTIFACT_ROOT / "decima_spark_dag_gate_20260614.json"
DECIMA_SAME_DOMAIN_ARTIFACT = ARTIFACT_ROOT / "decima_same_domain_benchmark_gate_20260614.json"


def build_non_future_claim_closure_gate(
    *,
    probe_remotes: bool = False,
    refresh_subgates: bool = False,
    refresh_rolling_production: bool = False,
) -> dict[str, Any]:
    fullstack = _cached_or_build(
        FULLSTACK_ARTIFACT,
        build_sota_fullstack_superiority_gate,
        refresh=refresh_subgates,
    )
    sota = _cached_or_build(
        ADMITTED_SOTA_ARTIFACT,
        build_sota_admitted_universe_closure_gate,
        refresh=refresh_subgates,
    )
    registered_adapter = _cached_or_build(
        REGISTERED_ADAPTER_ARTIFACT,
        build_registered_sota_adapter_closure_gate,
        refresh=refresh_subgates,
    )
    production_history = _cached_or_build(
        ORGANIC_HISTORY_ARTIFACT,
        build_organic_history_completion_gate,
        refresh=refresh_subgates,
    )
    production_wide = (
        build_production_wide_organic_trace_gate()
        if refresh_rolling_production
        else _load_json(PRODUCTION_WIDE_ARTIFACT)
    )
    if not production_wide:
        production_wide = build_production_wide_organic_trace_gate()
    multinode = (
        build_multinode_original_deployment_gate(probe_remotes=probe_remotes)
        if refresh_subgates
        else _load_json(MULTINODE_ORIGINAL_ARTIFACT)
    )
    if not multinode:
        multinode = build_multinode_original_deployment_gate(probe_remotes=probe_remotes)
    decima = _cached_or_build(
        DECIMA_SPARK_ARTIFACT,
        lambda: build_decima_spark_dag_gate(run_smoke=True),
        refresh=refresh_subgates,
    )
    decima_same_domain = _cached_or_build(
        DECIMA_SAME_DOMAIN_ARTIFACT,
        build_decima_same_domain_benchmark_gate,
        refresh=refresh_subgates,
    )

    rows = [
        {
            "name": "named_external_runtime_probe",
            "closed_claim": "Named five same-host same-workload external runtime probes are artifacted with paired native-better rows.",
            "ready": bool(fullstack.get("named_same_host_runtime_probe_ready")),
            "forbidden_extension": "Do not extend to arbitrary or unregistered SOTA systems.",
            "status": fullstack.get("status"),
        },
        {
            "name": "registered_sota_policy_semantics_universe",
            "closed_claim": "Every registered non-adjacent SOTA family has a policy/service-unit adapter in the finite measured-cache action union and the measured-cache external-policy frontier is closed.",
            "ready": bool(
                sota.get("registered_sota_admitted_policy_superiority_ready")
                and registered_adapter.get("registered_policy_adapter_universe_ready")
            ),
            "forbidden_extension": "Do not call this direct external-binary superiority for registered systems lacking executable same-workload adapters.",
            "status": registered_adapter.get("status"),
        },
        {
            "name": "production_wide_organic_completion",
            "closed_claim": "Strict theorem-facing organic scheduler history has enough launched/completed production rows, zero unadmitted rows, and production-wide wrapper is closed.",
            "ready": bool(
                production_history.get("large_scale_organic_history_completion_ready")
                and production_wide.get("large_scale_organic_launched_completion_ready")
            ),
            "forbidden_extension": "Do not claim raw history, attempted-only rows, or future unknown arrivals are automatically theorem-grade.",
            "status": production_wide.get("status"),
        },
        {
            "name": "scheduleurm_multinode_fullstack_history",
            "closed_claim": "Scheduleurm-native strict production history spans multiple nodes and GPU nodes with launched/completed theorem-admitted rows.",
            "ready": bool(multinode.get("scheduleurm_multinode_history_completion_ready")),
            "forbidden_extension": "Do not claim every external SOTA scheduler has been run through its original multi-node control plane.",
            "status": multinode.get("status"),
        },
        {
            "name": "decima_spark_dag_bridge",
            "closed_claim": "Decima Spark-DAG simulator smoke, metric bridge, and paired same-domain executable benchmark are closed as an adjacent-domain audit.",
            "ready": bool(
                decima.get("spark_dag_metric_bridge_ready")
                and decima_same_domain.get("spark_dag_same_domain_benchmark_ready")
            ),
            "forbidden_extension": "Do not mix Decima Spark-DAG evidence into GPU co-location full-stack superiority or hide mixed same-domain performance rows.",
            "status": decima_same_domain.get("status"),
        },
    ]
    ready = all(bool(row["ready"]) for row in rows)
    return {
        "gate": "non_future_claim_closure_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "NON_FUTURE_SCOPED_CLAIMS_CLOSED_STRONG_EXTENSIONS_FALSE"
            if ready else "NON_FUTURE_SCOPED_CLAIMS_PENDING"
        ),
        "gate_pass": ready,
        "scoped_claim_ready": ready,
        "non_future_scoped_closure_ready": ready,
        "strong_claim_ready": False,
        "excluded_direction": "arbitrary_future_workload",
        "row_count": len(rows),
        "ready_count": sum(1 for row in rows if row["ready"]),
        "rows": rows,
        "snapshots": {
            "sota_fullstack": {
                "status": fullstack.get("status"),
                "named_same_host_runtime_probe_ready": fullstack.get(
                    "named_same_host_runtime_probe_ready"
                ),
                "direct_fullstack_sota_superiority_ready": fullstack.get(
                    "direct_fullstack_sota_superiority_ready"
                ),
                "arbitrary_sota_superiority_ready": fullstack.get(
                    "arbitrary_sota_superiority_ready"
                ),
            },
            "sota_admitted_universe": {
                "status": sota.get("status"),
                "registered_sota_admitted_policy_superiority_ready": sota.get(
                    "registered_sota_admitted_policy_superiority_ready"
                ),
                "registered_sota_universe_superiority_ready": sota.get(
                    "registered_sota_universe_superiority_ready"
                ),
            },
            "registered_sota_adapter": {
                "status": registered_adapter.get("status"),
                "registered_policy_adapter_universe_ready": registered_adapter.get(
                    "registered_policy_adapter_universe_ready"
                ),
                "direct_external_binary_adapter_ready_count": registered_adapter.get(
                    "direct_external_binary_adapter_ready_count"
                ),
                "registered_system_count": registered_adapter.get(
                    "registered_system_count"
                ),
                "registered_direct_external_binary_superiority_ready": registered_adapter.get(
                    "registered_direct_external_binary_superiority_ready"
                ),
            },
            "production_history": {
                "status": production_history.get("status"),
                "strict_organic_launched_count": production_history.get(
                    "strict_organic_launched_count"
                ),
                "strict_organic_completed_count": production_history.get(
                    "strict_organic_completed_count"
                ),
                "strict_unadmitted_launched_count": production_history.get(
                    "strict_unadmitted_launched_count"
                ),
            },
            "production_wide": {
                "status": production_wide.get("status"),
                "production_wide_live_trace_closed": production_wide.get(
                    "production_wide_live_trace_closed"
                ),
                "production_wide_history_completion_closed": production_wide.get(
                    "production_wide_history_completion_closed"
                ),
            },
            "multinode": {
                "status": multinode.get("status"),
                "scheduleurm_multinode_history_completion_ready": multinode.get(
                    "scheduleurm_multinode_history_completion_ready"
                ),
                "external_sota_original_multinode_superiority_ready": multinode.get(
                    "external_sota_original_multinode_superiority_ready"
                ),
            },
            "decima": {
                "status": decima.get("status"),
                "spark_dag_metric_bridge_ready": decima.get("spark_dag_metric_bridge_ready"),
                "direct_fullstack_gpu_sota_claim_ready": decima.get(
                    "direct_fullstack_gpu_sota_claim_ready"
                ),
            },
            "decima_same_domain": {
                "status": decima_same_domain.get("status"),
                "spark_dag_same_domain_benchmark_ready": decima_same_domain.get(
                    "spark_dag_same_domain_benchmark_ready"
                ),
                "spark_dag_same_domain_performance_superiority_ready": decima_same_domain.get(
                    "spark_dag_same_domain_performance_superiority_ready"
                ),
                "direct_fullstack_gpu_sota_claim_ready": decima_same_domain.get(
                    "direct_fullstack_gpu_sota_claim_ready"
                ),
            },
        },
        "forbidden_claims": [
            "arbitrary future workload positive-service theorem readiness",
            "arbitrary or unregistered SOTA superiority",
            "direct external-binary superiority for registered systems without executable same-workload adapter rows",
            "external SOTA original multi-node control-plane superiority",
            "Decima Spark-DAG superiority as if it were a GPU co-location scheduler",
        ],
        "scope": (
            "Aggregate certificate for non-future broad directions.  It closes "
            "finite named runtime-probe, registered policy-semantics, strict "
            "production-history, Scheduleurm-native multi-node, and Decima "
            "adjacent-domain bridge claims.  It intentionally excludes arbitrary "
            "future workloads and keeps stronger unbounded/external-binary "
            "extensions as forbidden unless their own gates close."
        ),
        "pass": ready,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Non-Future Claim Closure Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `excluded_direction` | `{report.get('excluded_direction')}` |",
        f"| `ready_count` | {report.get('ready_count')} / {report.get('row_count')} |",
        "",
        "## Closure Rows",
        "",
        "| Direction | Ready | Closed claim | Forbidden extension | Status |",
        "|---|---:|---|---|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{name}` | {ready} | {claim} | {forbidden} | `{status}` |".format(
                name=row.get("name"),
                ready=str(bool(row.get("ready"))).lower(),
                claim=_md(row.get("closed_claim")),
                forbidden=_md(row.get("forbidden_extension")),
                status=row.get("status"),
            )
        )
    lines.extend([
        "",
        "## Forbidden Claims",
        "",
    ])
    for claim in report.get("forbidden_claims") or []:
        lines.append(f"- {claim}")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cached_or_build(
    path: Path,
    builder: Any,
    *,
    refresh: bool,
) -> dict[str, Any]:
    if not refresh:
        cached = _load_json(path)
        if cached:
            return cached
    return dict(builder())


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_non_future_claim_closure_gate(
        probe_remotes=args.probe_remotes,
        refresh_subgates=args.refresh_subgates,
        refresh_rolling_production=args.refresh_rolling_production,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m algorithm.experiments.non_future_claim_closure_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build non-future broad-claim closure gate")
    build.add_argument("--probe-remotes", action="store_true")
    build.add_argument("--refresh-subgates", action="store_true")
    build.add_argument("--refresh-rolling-production", action="store_true")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
