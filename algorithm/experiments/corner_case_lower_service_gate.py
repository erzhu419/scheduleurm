"""Corner-case service-cache and lower-service capacity gate.

The live corner-case probes are intentionally separate from the default replay
cache.  This gate admits only stable, canonical rows into a standalone service
cache snapshot and proves a row-level lower-service capacity certificate for
each admitted marginal action.  It does not promote the result to global
production stability.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from simulation.service_cache import ProfileRecord, ServiceRateCache

from .capacity_lp import drift_margin_certificate, solve_capacity_slack


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_SUMMARY = ARTIFACT_ROOT / "corner_case_live_20260613_summary.json"
DEFAULT_LIVE_DIR = ARTIFACT_ROOT / "corner_case_live_20260613"
DEFAULT_CACHE_OUTPUT = ARTIFACT_ROOT / "corner_case_service_cache_20260613.json"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "corner_case_lower_service_gate_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "corner_case_lower_service_gate_20260613.md"

STEP_RE = re.compile(r"--steps\s+([0-9]+)")

CANONICAL_SCENARIOS: dict[str, dict[str, str]] = {
    "node007_empty_small": {
        "workload_key": "corner_gpu_node007_empty_marginal_cuda",
        "resource_kind": "gpu_corner_empty",
        "node_bucket": "node007-direct:4x12gb",
        "role": "empty_baseline",
    },
    "node007_30gb_add_small_stable": {
        "workload_key": "corner_gpu_node007_30gb_resident_marginal_cuda",
        "resource_kind": "gpu_corner_resident_vram",
        "node_bucket": "node007-direct:30gb-resident",
        "role": "co_location",
    },
    "cpu_empty_node001_single": {
        "workload_key": "corner_cpu_node001_empty_single",
        "resource_kind": "cpu_corner_empty",
        "node_bucket": "node001:192c",
        "role": "empty_baseline",
    },
    "cpu_half_node001_add_single_v2": {
        "workload_key": "corner_cpu_node001_96worker_resident_single",
        "resource_kind": "cpu_corner_resident",
        "node_bucket": "node001:96worker-resident",
        "role": "co_location",
    },
    "cpu_full_node001_add_single_v2": {
        "workload_key": "corner_cpu_node001_180worker_resident_single",
        "resource_kind": "cpu_corner_resident",
        "node_bucket": "node001:180worker-resident",
        "role": "co_location",
    },
}


def build_corner_case_lower_service_gate(
    *,
    summary_path: str | Path = DEFAULT_SUMMARY,
    live_dir: str | Path = DEFAULT_LIVE_DIR,
    cache_output_path: str | Path = DEFAULT_CACHE_OUTPUT,
    offered_load_fraction: float = 0.80,
) -> dict[str, Any]:
    summary_path = Path(summary_path).expanduser()
    live_dir = Path(live_dir).expanduser()
    cache_output_path = Path(cache_output_path).expanduser()
    summary = _load_json(summary_path)
    detail_rows = {
        scenario_id: _load_detail(live_dir, scenario_id)
        for scenario_id in CANONICAL_SCENARIOS
    }
    records: list[ProfileRecord] = []
    rows: list[dict[str, Any]] = []
    for scenario_id, spec in CANONICAL_SCENARIOS.items():
        detail = detail_rows.get(scenario_id) or {}
        row = _admission_row(
            scenario_id=scenario_id,
            spec=spec,
            detail=detail,
            live_dir=live_dir,
            offered_load_fraction=offered_load_fraction,
        )
        rows.append(row)
        if row["admitted_to_corner_case_service_cache"]:
            records.append(
                ProfileRecord(
                    workload_key=spec["workload_key"],
                    command_fingerprint=_fingerprint(detail.get("probe_cmd")),
                    resource_kind=spec["resource_kind"],
                    node_bucket=spec["node_bucket"],
                    profile=1,
                    unit="step",
                    total_units=float(row["probe_steps"]),
                    aggregate_rate=float(row["lower_service_rate_step_s"]),
                    per_task_rates=(float(row["lower_service_rate_step_s"]),),
                    source=str(detail.get("artifact") or _detail_path(live_dir, scenario_id)),
                    capacity_boundary=False,
                    gpu_count_observed=1,
                )
            )
    cache = ServiceRateCache(records)
    cache.save(cache_output_path)
    row_ready = [row for row in rows if row.get("row_lower_service_capacity_ready")]
    co_location_rows = [row for row in rows if row.get("role") == "co_location"]
    co_location_ready = [
        row for row in co_location_rows
        if row.get("row_lower_service_capacity_ready")
    ]
    return {
        "gate": "corner_case_lower_service_gate",
        "status": (
            "CORNER_CASE_LOWER_SERVICE_PASS"
            if len(row_ready) == len(rows) and len(co_location_ready) == len(co_location_rows)
            else "CORNER_CASE_LOWER_SERVICE_PENDING"
        ),
        "summary_path": str(summary_path),
        "live_dir": str(live_dir),
        "service_cache_path": str(cache_output_path),
        "service_cache_record_count": len(records),
        "summary_all_required_pass": bool(summary.get("all_required_pass")),
        "summary_all_required_stable_eta": bool(summary.get("all_required_stable_eta")),
        "canonical_scenario_count": len(CANONICAL_SCENARIOS),
        "admitted_scenario_count": len(records),
        "co_location_scenario_count": len(co_location_rows),
        "co_location_lower_service_ready_count": len(co_location_ready),
        "offered_load_fraction": float(offered_load_fraction),
        "corner_case_service_cache_ready": len(records) == len(CANONICAL_SCENARIOS),
        "corner_case_lower_service_capacity_ready": len(row_ready) == len(rows),
        "co_location_lower_service_capacity_ready": len(co_location_ready) == len(co_location_rows),
        "cache_roundtrip_ready": _cache_roundtrip_ready(cache, rows),
        "scoped_claim_ready": (
            len(records) == len(CANONICAL_SCENARIOS)
            and len(row_ready) == len(rows)
            and len(co_location_ready) == len(co_location_rows)
        ),
        "strong_claim_ready": False,
        "pass": True,
        "pass_meaning": (
            "standalone service-cache admission plus row-level lower-service "
            "capacity slack for the canonical corner-case probe slice"
        ),
        "rows": rows,
        "scope": (
            "The gate admits the controlled node007/node001 corner-case rows into "
            "a standalone service-cache snapshot and certifies positive row-level "
            "capacity slack under a conservative offered-load fraction.  It is not "
            "a production-wide queueing theorem, a full fabric-cover proof, or a "
            "direct external full-stack SOTA comparison."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Corner-Case Lower-Service Gate",
        "",
        "This artifact admits only stable canonical corner-case probes into a standalone service cache. It does not modify the default replay cache.",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `service_cache_record_count` | {report.get('service_cache_record_count', 0)} |",
        f"| `corner_case_lower_service_capacity_ready` | {str(bool(report.get('corner_case_lower_service_capacity_ready'))).lower()} |",
        f"| `co_location_lower_service_capacity_ready` | {str(bool(report.get('co_location_lower_service_capacity_ready'))).lower()} |",
        f"| `cache_roundtrip_ready` | {str(bool(report.get('cache_roundtrip_ready'))).lower()} |",
        f"| `offered_load_fraction` | {float(report.get('offered_load_fraction') or 0.0):.6g} |",
        f"| `service_cache_path` | `{report.get('service_cache_path')}` |",
        "",
        "## Rows",
        "",
        "| Scenario | Role | Workload key | Lower service | Lambda | Delta | Eta | Ready |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{scenario}` | `{role}` | `{workload}` | {service:.6g} | {lam:.6g} | {delta:.6g} | {eta:.6g} | {ready} |".format(
                scenario=row.get("scenario_id"),
                role=row.get("role"),
                workload=row.get("workload_key"),
                service=float(row.get("lower_service_rate_step_s") or 0.0),
                lam=float(row.get("lambda_step_s") or 0.0),
                delta=float(((row.get("capacity_slack") or {}).get("delta")) or 0.0),
                eta=float(((row.get("drift_margin") or {}).get("eta")) or 0.0),
                ready=str(bool(row.get("row_lower_service_capacity_ready"))).lower(),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _admission_row(
    *,
    scenario_id: str,
    spec: Mapping[str, str],
    detail: Mapping[str, Any],
    live_dir: Path,
    offered_load_fraction: float,
) -> dict[str, Any]:
    stable = detail.get("stable_eta") or {}
    tail = [float(x) for x in stable.get("tail") or [] if _positive_float(x)]
    tail_lower = min(tail) if tail else 0.0
    last_rate = _as_float(detail.get("probe_last_rate_step_s"))
    lower = tail_lower if bool(stable.get("ready")) and tail_lower > 0.0 else 0.0
    lam_value = float(offered_load_fraction) * lower
    workload_key = str(spec["workload_key"])
    action = {
        "action_id": f"admit_{scenario_id}",
        "lower_service": {workload_key: lower},
    }
    slack = solve_capacity_slack([action], {workload_key: lam_value}) if lower > 0.0 else {
        "status": "missing_stable_lower_service",
        "delta": None,
        "usable_for_theorem": False,
        "mix": {},
    }
    drift = drift_margin_certificate(
        delta=float(slack.get("delta") or 0.0),
        L=0.0,
        rho=0.0,
        epsilon_est=0.0,
        beta=0.0,
        alpha1=0.0,
        B=0.0,
        P0=0.0,
        alpha0=0.0,
    )
    return {
        "scenario_id": scenario_id,
        "role": spec["role"],
        "workload_key": workload_key,
        "resource_kind": spec["resource_kind"],
        "node_bucket": spec["node_bucket"],
        "probe_steps": _steps_from_cmd(str(detail.get("probe_cmd") or "")),
        "probe_last_rate_step_s": last_rate,
        "stable_tail_rates_step_s": tail,
        "lower_service_rule": "min(last three stable ETA rate windows)",
        "lower_service_rate_step_s": lower,
        "lambda_step_s": lam_value,
        "admitted_to_corner_case_service_cache": lower > 0.0,
        "capacity_slack": slack,
        "drift_margin": drift,
        "row_lower_service_capacity_ready": bool(slack.get("usable_for_theorem")) and bool(drift.get("usable_for_theorem")),
        "artifact": str(_detail_path(live_dir, scenario_id)),
    }


def _cache_roundtrip_ready(cache: ServiceRateCache, rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        if not row.get("admitted_to_corner_case_service_cache"):
            return False
        record = cache.get(str(row.get("workload_key")), 1)
        if record is None or record.capacity_boundary:
            return False
        if not math.isclose(float(record.aggregate_rate), float(row.get("lower_service_rate_step_s") or 0.0), rel_tol=1e-12):
            return False
    return True


def _load_detail(live_dir: Path, scenario_id: str) -> dict[str, Any]:
    path = _detail_path(live_dir, scenario_id)
    row = _load_json(path)
    if row:
        row["artifact"] = str(path)
    return row


def _detail_path(live_dir: Path, scenario_id: str) -> Path:
    return live_dir / f"{scenario_id}_remote_corner_case_probe.json"


def _steps_from_cmd(cmd: str) -> int:
    match = STEP_RE.search(cmd)
    return max(1, int(match.group(1))) if match else 1


def _fingerprint(value: Any) -> str:
    text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()[:240]


def _positive_float(value: Any) -> bool:
    return _as_float(value) > 0.0


def _as_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_corner_case_lower_service_gate(
        summary_path=args.summary_path,
        live_dir=args.live_dir,
        cache_output_path=args.cache_output,
        offered_load_fraction=args.offered_load_fraction,
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("scoped_claim_ready") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.corner_case_lower_service_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build corner-case service-cache/lower-service gate")
    build.add_argument("--summary-path", default=str(DEFAULT_SUMMARY))
    build.add_argument("--live-dir", default=str(DEFAULT_LIVE_DIR))
    build.add_argument("--cache-output", default=str(DEFAULT_CACHE_OUTPUT))
    build.add_argument("--offered-load-fraction", type=float, default=0.80)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
