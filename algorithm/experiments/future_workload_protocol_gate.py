"""Future-workload protocol closure gate.

The theorem-facing claim for future jobs is not "every future workload is
already certified."  It is an admission protocol: exact measured jobs enter the
theorem stream, while unknown workloads are forced into probe/admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .future_admitted_fabric_cover_gate import build_future_admitted_fabric_cover_gate
from .future_production_admission_contract import build_future_production_admission_contract
from .production_load_certificate import load_scheduler_records


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "future_workload_protocol_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "future_workload_protocol_gate_20260614.md"


SYNTHETIC_FUTURE_ROWS = (
    {
        "id": "future_known_resac",
        "status": "queued",
        "project": "BAPR",
        "signature": "BAPR RE-SAC Ant-v5 training",
        "cmd": "python train.py --env Ant-v5 --algo resac",
        "est_vram_mb": 12000,
    },
    {
        "id": "future_known_cnn",
        "status": "queued",
        "project": "future_cnn",
        "signature": "torch_cnn_progress_benchmark resnet50 image classification",
        "cmd": "python torch_cnn_progress_benchmark.py --model resnet50",
        "est_vram_mb": 8000,
    },
    {
        "id": "future_known_llm",
        "status": "queued",
        "project": "future_llm_measured",
        "signature": "torch_llm_progress_benchmark decoder stack measured family",
        "cmd": "python torch_llm_progress_benchmark.py --model distilgpt2",
        "est_vram_mb": 12000,
    },
    {
        "id": "future_known_cpu",
        "status": "queued",
        "project": "future_cpu_heavy",
        "signature": "cpu_heavy_local prime_sieve benchmark",
        "cmd": "python cpu_progress_benchmark.py --kind prime_sieve --workers 32",
        "est_vram_mb": 0,
    },
    {
        "id": "future_known_control",
        "status": "queued",
        "project": "future_control",
        "signature": "light_control sleep_20ms control-plane probe",
        "cmd": "python light_control.py --sleep_20ms",
        "est_vram_mb": 0,
    },
    {
        "id": "future_unknown_llm_new_arch",
        "status": "queued",
        "project": "future_llm",
        "signature": "new transformer architecture not in service cache",
        "cmd": "python train_future_model.py --model unknown-70b",
        "est_vram_mb": 42000,
    },
    {
        "id": "future_unknown_cpu_pipeline",
        "status": "queued",
        "project": "future_data_pipeline",
        "signature": "unseen cpu dataloader preprocessing pipeline",
        "cmd": "python preprocess_unseen_dataset.py --workers 192",
        "est_vram_mb": 0,
    },
    {
        "id": "future_unknown_cuda_kernel",
        "status": "queued",
        "project": "future_custom_cuda",
        "signature": "unseen fused custom cuda kernel with unknown memory allocator",
        "cmd": "python train_custom_cuda_kernel.py --kernel fused_unknown",
        "est_vram_mb": 16000,
    },
    {
        "id": "future_unknown_spark_dag",
        "status": "queued",
        "project": "future_spark_dag",
        "signature": "unseen spark dag scheduling job outside gpu co-location service cache",
        "cmd": "spark-submit future_query.py --scheduler decima_like",
        "est_vram_mb": 0,
    },
    {
        "id": "future_unknown_numa_pipeline",
        "status": "queued",
        "project": "future_numa_pipeline",
        "signature": "unseen multi-node numa data pipeline with remote storage pressure",
        "cmd": "python train_pipeline.py --numa --remote-storage --workers 384",
        "est_vram_mb": 0,
    },
)


def build_future_workload_protocol_gate() -> dict[str, Any]:
    production_admission = build_future_production_admission_contract(
        records=load_scheduler_records(),
        admission_mode="strict",
    )
    synthetic_admission = build_future_production_admission_contract(
        records=SYNTHETIC_FUTURE_ROWS,
        admission_mode="strict",
    )
    fabric_cover = build_future_admitted_fabric_cover_gate()
    synthetic_routes = {
        str(row.get("task_id")): str(row.get("route"))
        for row in synthetic_admission.get("rows") or []
    }
    unknown_ids = [
        task["id"] for task in SYNTHETIC_FUTURE_ROWS
        if str(task["id"]).startswith("future_unknown_")
    ]
    known_ids = [
        task["id"] for task in SYNTHETIC_FUTURE_ROWS
        if str(task["id"]).startswith("future_known_")
    ]
    unknown_probe_ready = all(
        synthetic_routes.get(task_id) == "PROBE_REQUIRED"
        for task_id in unknown_ids
    )
    known_admission_or_probe = synthetic_routes.get("future_known_resac") in {
        "ADMIT_THEOREM_TRACE",
        "PROBE_REQUIRED",
    } and all(synthetic_routes.get(task_id) in {"ADMIT_THEOREM_TRACE", "PROBE_REQUIRED"} for task_id in known_ids)
    protocol_ready = bool(
        production_admission.get("theorem_admission_default_trace_mode_ready")
        and synthetic_admission.get("theorem_admission_default_trace_mode_ready")
        and fabric_cover.get("future_admitted_measured_state_ready")
        and unknown_probe_ready
        and known_admission_or_probe
    )
    return {
        "gate": "future_workload_protocol_gate",
        "status": (
            "FUTURE_WORKLOAD_PROTOCOL_READY_ARBITRARY_FUTURE_FALSE"
            if protocol_ready else "FUTURE_WORKLOAD_PROTOCOL_PENDING"
        ),
        "future_workload_protocol_ready": bool(protocol_ready),
        "future_admitted_measured_state_ready": bool(
            fabric_cover.get("future_admitted_measured_state_ready")
        ),
        "arbitrary_future_workload_theorem_ready": False,
        "scoped_claim_ready": bool(protocol_ready),
        "strong_claim_ready": False,
        "unknown_future_jobs_probe_required_ready": bool(unknown_probe_ready),
        "production_admission_snapshot": {
            "active_production_count": production_admission.get("active_production_count"),
            "admitted_traceable_count": production_admission.get("admitted_traceable_count"),
            "probe_required_count": production_admission.get("probe_required_count"),
            "future_jobs_all_theorem_grade_without_probe": production_admission.get(
                "future_jobs_all_theorem_grade_without_probe"
            ),
        },
        "synthetic_routes": synthetic_routes,
        "synthetic_known_count": len(known_ids),
        "synthetic_unknown_count": len(unknown_ids),
        "synthetic_rows": synthetic_admission.get("rows") or [],
        "fabric_cover_snapshot": {
            "workload_count": fabric_cover.get("workload_count"),
            "future_admitted_measured_state_ready": fabric_cover.get(
                "future_admitted_measured_state_ready"
            ),
            "future_all_state_fabric_cover_ready": fabric_cover.get(
                "future_all_state_fabric_cover_ready"
            ),
        },
        "pass": bool(protocol_ready),
        "scope": (
            "Future-workload readiness is an admission protocol.  Exact measured "
            "positive profiles may enter theorem traces with identity projection; "
            "unmeasured future workloads must be routed to probe/admission before "
            "they can support positive-service theorem claims."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    prod = report.get("production_admission_snapshot") or {}
    cover = report.get("fabric_cover_snapshot") or {}
    lines = [
        "# Future Workload Protocol Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `future_workload_protocol_ready` | {str(bool(report.get('future_workload_protocol_ready'))).lower()} |",
        f"| `future_admitted_measured_state_ready` | {str(bool(report.get('future_admitted_measured_state_ready'))).lower()} |",
        f"| `unknown_future_jobs_probe_required_ready` | {str(bool(report.get('unknown_future_jobs_probe_required_ready'))).lower()} |",
        f"| `arbitrary_future_workload_theorem_ready` | {str(bool(report.get('arbitrary_future_workload_theorem_ready'))).lower()} |",
        f"| `production_active_count` | {prod.get('active_production_count')} |",
        f"| `production_admitted_traceable_count` | {prod.get('admitted_traceable_count')} |",
        f"| `production_probe_required_count` | {prod.get('probe_required_count')} |",
        f"| `fabric_workload_count` | {cover.get('workload_count')} |",
        "",
        "## Synthetic Future Routes",
        "",
        "| Task | Route |",
        "|---|---|",
    ]
    for task, route in sorted((report.get("synthetic_routes") or {}).items()):
        lines.append(f"| `{task}` | `{route}` |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_future_workload_protocol_gate()
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.future_workload_protocol_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build future-workload protocol gate")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
