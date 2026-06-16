"""Sia/AdaptDL MIP same-workload full-stack evidence gate.

This gate is scoped to direct runtime evidence for the Sia physical-cluster
AdaptDL artifact on the jtl110gpu K3s/NVIDIA test substrate.  It certifies that
Sia's Kubernetes controller/allocator path completed the same CUDA workload
binary used by the native Scheduleurm-controlled Docker pair.  It does not
claim direct full-stack superiority over every SOTA system.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
SIA_ARTIFACT_DIR = ARTIFACT_ROOT / "sia_fullstack_20260613"
POLLUX_ARTIFACT_DIR = ARTIFACT_ROOT / "pollux_fullstack_20260613"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "sia_fullstack_same_workload_gate_20260613.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "sia_fullstack_same_workload_gate_20260613.md"


def build_sia_fullstack_same_workload_gate(
    *,
    artifact_dir: Path = SIA_ARTIFACT_DIR,
    native_artifact_dir: Path = POLLUX_ARTIFACT_DIR,
) -> dict[str, Any]:
    short = _paired_case(
        artifact_dir=artifact_dir,
        native_artifact_dir=native_artifact_dir,
        label="cuda_probe_short",
        sia_result="sia_cuda_probe.json",
        sia_job="sia_cuda_probe_adaptdljob.json",
        sia_wall=None,
        native_result="native_cuda_probe.json",
        native_wall="native_cuda_probe_wall.json",
    )
    long = _paired_case(
        artifact_dir=artifact_dir,
        native_artifact_dir=native_artifact_dir,
        label="cuda_probe_long",
        sia_result="sia_cuda_probe_long.json",
        sia_job="sia_cuda_probe_long_adaptdljob.json",
        sia_wall="sia_cuda_probe_long_wall.json",
        native_result="native_cuda_probe_long.json",
        native_wall="native_cuda_probe_long_wall.json",
    )
    cases = [short, long]
    all_ready = all(bool(row.get("ready")) for row in cases)
    all_native_faster_jct = all(
        row.get("native_wall_s") is not None
        and row.get("sia_crd_jct_s") is not None
        and float(row["native_wall_s"]) < float(row["sia_crd_jct_s"])
        for row in cases
    )
    all_same_service_scale = all(
        row.get("workload_steps_per_s_ratio_native_over_sia") is not None
        and 0.90 <= float(row["workload_steps_per_s_ratio_native_over_sia"]) <= 1.20
        for row in cases
    )
    return {
        "gate": "sia_fullstack_same_workload_gate",
        "artifact_dir": str(artifact_dir),
        "native_artifact_dir": str(native_artifact_dir),
        "sia_adapter": "sia_goodput_scheduler_mip",
        "same_workload_binary": "scheduleurm/cuda-probe:local",
        "workload": "cuda_saxpy_loop",
        "host": "jtl110gpu",
        "gpu": "NVIDIA GeForce RTX 3080 Ti",
        "sia_fullstack_completed": bool(all_ready),
        "native_pair_completed": bool(all_ready),
        "same_service_scale_ready": bool(all_ready and all_same_service_scale),
        "scoped_sia_fullstack_same_workload_ready": bool(all_ready),
        "scoped_sia_same_workload_superiority_ready": bool(
            all_ready and all_native_faster_jct and all_same_service_scale
        ),
        "direct_fullstack_sota_superiority_ready": False,
        "strong_all_sota_claim_ready": False,
        "cases": cases,
        "compatibility_fixes": [
            {
                "component": "Sia/AdaptDL scheduler image",
                "fix": "pin scheduler runtime dependencies to NumPy < 1.24 and pandas < 2.0",
                "reason": "upstream Sia/AdaptDL artifact uses deprecated NumPy/Pandas APIs",
                "semantics_changed": False,
            },
            {
                "component": "Sia/AdaptDL supervisor",
                "fix": "cast ADAPTDL_SUPERVISOR_SERVICE_PORT to int",
                "reason": "Kubernetes service environment variables are strings and aiohttp requires an integer port",
                "semantics_changed": False,
            },
            {
                "component": "Sia physical-cluster mapping",
                "fix": "map the K3s node name huiwei-super-server to the existing rtx cluster type",
                "reason": "the artifact assumes a hard-coded Phoebe cluster node inventory",
                "semantics_changed": False,
            },
            {
                "component": "Sia MIP policy adapter",
                "fix": "lift a scalar single-cluster speedup function into the artifact's {gpu_type: fn} map",
                "reason": "single-GPU-type Kubernetes deployments do not need a multi-cluster speedup dictionary",
                "semantics_changed": False,
            },
        ],
        "scope": (
            "Scoped same-host same-binary full-stack evidence for the Sia "
            "physical-cluster AdaptDL/MIP artifact.  It supports a claim that "
            "the Sia runtime can be run and compared on this cluster, and that "
            "the native Scheduleurm-controlled Docker path has lower JCT on "
            "these two CUDA probe cases.  It does not prove direct full-stack "
            "superiority over Gavel, Pollux, IADeep, Salus, or arbitrary future "
            "workloads."
        ),
        "pass": True,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sia Full-Stack Same-Workload Gate",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `sia_fullstack_completed` | {str(bool(report.get('sia_fullstack_completed'))).lower()} |",
        f"| `native_pair_completed` | {str(bool(report.get('native_pair_completed'))).lower()} |",
        f"| `same_service_scale_ready` | {str(bool(report.get('same_service_scale_ready'))).lower()} |",
        f"| `scoped_sia_same_workload_superiority_ready` | {str(bool(report.get('scoped_sia_same_workload_superiority_ready'))).lower()} |",
        f"| `direct_fullstack_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_sota_superiority_ready'))).lower()} |",
        "",
        "## Paired Cases",
        "",
        "| Case | Sia phase | Sia CRD JCT (s) | Native wall (s) | Native/Sia JCT | Sia steps/s | Native steps/s | Native/Sia service |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("cases") or []:
        lines.append(
            "| `{label}` | {phase} | {sjct:.6f} | {nwall:.6f} | {jctr:.6f} | {ss:.6f} | {ns:.6f} | {sr:.6f} |".format(
                label=row.get("label"),
                phase=row.get("sia_phase"),
                sjct=_num(row.get("sia_crd_jct_s")),
                nwall=_num(row.get("native_wall_s")),
                jctr=_num(row.get("jct_ratio_native_over_sia_crd")),
                ss=_num(row.get("sia_steps_per_s")),
                ns=_num(row.get("native_steps_per_s")),
                sr=_num(row.get("workload_steps_per_s_ratio_native_over_sia")),
            )
        )
    lines.extend([
        "",
        "## Compatibility Fixes",
        "",
        "| Component | Fix | Semantics changed | Reason |",
        "|---|---|---:|---|",
    ])
    for row in report.get("compatibility_fixes") or []:
        lines.append(
            "| {component} | {fix} | {changed} | {reason} |".format(
                component=_md(row.get("component")),
                fix=_md(row.get("fix")),
                changed=str(bool(row.get("semantics_changed"))).lower(),
                reason=_md(row.get("reason")),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _paired_case(
    *,
    artifact_dir: Path,
    native_artifact_dir: Path,
    label: str,
    sia_result: str,
    sia_job: str,
    sia_wall: str | None,
    native_result: str,
    native_wall: str,
) -> dict[str, Any]:
    sia = _load_json(artifact_dir / sia_result)
    job = _load_json(artifact_dir / sia_job)
    sia_wall_data = _load_json(artifact_dir / sia_wall) if sia_wall else {}
    native = _load_json(native_artifact_dir / native_result)
    native_wall_data = _load_json(native_artifact_dir / native_wall)
    status = job.get("status") or {}
    phase = str(status.get("phase") or sia_wall_data.get("phase") or "")
    creation = (job.get("metadata") or {}).get("creationTimestamp") or sia_wall_data.get("creationTimestamp")
    completion = (
        status.get("completionTimestamp")
        or status.get("completionTime")
        or sia_wall_data.get("completionTimestamp")
    )
    sia_crd_jct_s = _seconds_between(creation, completion)
    native_wall_s = _float(native_wall_data.get("wall_s"))
    sia_steps = _float(sia.get("steps_per_s"))
    native_steps = _float(native.get("steps_per_s"))
    return {
        "label": label,
        "ready": bool(phase == "Succeeded" and sia and native and native_wall_s is not None),
        "sia_phase": phase,
        "sia_creation_timestamp": creation,
        "sia_completion_timestamp": completion,
        "sia_crd_jct_s": sia_crd_jct_s,
        "sia_observed_wall_s": _float(sia_wall_data.get("wall_observed_s")),
        "native_wall_s": native_wall_s,
        "jct_ratio_native_over_sia_crd": _ratio(native_wall_s, sia_crd_jct_s),
        "sia_total_s": _float(sia.get("total_s")),
        "native_total_s": _float(native.get("total_s")),
        "sia_steps_per_s": sia_steps,
        "native_steps_per_s": native_steps,
        "workload_steps_per_s_ratio_native_over_sia": _ratio(native_steps, sia_steps),
        "sia_result_artifact": str(artifact_dir / sia_result),
        "sia_job_artifact": str(artifact_dir / sia_job),
        "native_result_artifact": str(native_artifact_dir / native_result),
        "native_wall_artifact": str(native_artifact_dir / native_wall),
    }


def _seconds_between(start: Any, end: Any) -> float | None:
    if not start or not end:
        return None
    try:
        return (_parse_time(str(end)) - _parse_time(str(start))).total_seconds()
    except Exception:
        return None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _num(value: Any) -> float:
    parsed = _float(value)
    return float("nan") if parsed is None else parsed


def _load_json(path: Path) -> dict[str, Any]:
    if not path or not path.exists():
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
    report = build_sia_fullstack_same_workload_gate(
        artifact_dir=Path(args.artifact_dir),
        native_artifact_dir=Path(args.native_artifact_dir),
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sia_fullstack_same_workload_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Sia/AdaptDL MIP same-workload full-stack evidence gate")
    build.add_argument("--artifact-dir", default=str(SIA_ARTIFACT_DIR))
    build.add_argument("--native-artifact-dir", default=str(POLLUX_ARTIFACT_DIR))
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
