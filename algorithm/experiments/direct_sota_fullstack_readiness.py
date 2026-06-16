"""Direct external-SOTA full-stack readiness certificate.

This audit is intentionally read-only.  It does not install packages, start
Docker/Kubernetes, edit cloned repositories, or touch the live scheduler.  Its
purpose is to separate three claims that are easy to conflate in a paper:

* an external repository and entrypoint are present;
* the entrypoint can smoke-test locally;
* the external system can run as a same-workload, full-stack baseline for
  Scheduleurm.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .direct_sota_baseline_scaffold import (
    ADAPTERS,
    ARTIFACT_ROOT,
    REPO_ROOT,
    REFERENCE_ROOT,
    build_direct_sota_scaffold,
)


GAVEL_NATIVE_SMOKE_ARTIFACT = ARTIFACT_ROOT / "gavel_direct_native_smoke_20260612.json"
GAVEL_NATIVE_PERFORMANCE_ARTIFACT = ARTIFACT_ROOT / "gavel_native_performance_microbaseline_20260612.json"


@dataclass(frozen=True)
class FullStackRequirement:
    adapter: str
    required_tools: tuple[str, ...]
    same_workload_assets: tuple[str, ...]
    stack_assets: tuple[str, ...]
    extra_blockers: tuple[str, ...]


REQUIREMENTS = {
    "gavel_simulation": FullStackRequirement(
        adapter="gavel_simulation",
        required_tools=("python3",),
        same_workload_assets=(
            "scheduleurm_to_gavel_trace_converter",
            "scheduleurm_to_gavel_throughput_table",
        ),
        stack_assets=("scheduler/scripts/sweeps/run_sweep_static.py",),
        extra_blockers=(
            "Scheduleurm trace/throughput seeds exist, but same-workload Gavel schema/service-unit validation is still required",
        ),
    ),
    "pollux_adaptdl_scheduler": FullStackRequirement(
        adapter="pollux_adaptdl_scheduler",
        required_tools=("python3", "kubectl", "docker"),
        same_workload_assets=("adaptdljob_manifest_generator",),
        stack_assets=(
            "helm/adaptdl-sched/templates/adaptdl-crd.yaml",
            "helm/adaptdl-sched/templates/adaptdl-sched.yaml",
        ),
        extra_blockers=(
            "AdaptDLJob manifest seeds exist, but a Kubernetes cluster, container image, and isolated test namespace are still required",
        ),
    ),
    "sia_goodput_scheduler": FullStackRequirement(
        adapter="sia_goodput_scheduler",
        required_tools=("python3", "kubectl", "docker"),
        same_workload_assets=("adaptdljob_manifest_generator",),
        stack_assets=(
            "sia-simulator/sia.py",
            "sia-simulator/ftf_simulator.py",
            "sia-simulator/requirements.txt",
            "adaptdl/sched/setup.py",
            "adaptdl/benchmark/run_workload.py",
        ),
        extra_blockers=(
            "Sia official artifact is cloned, but simulator execution needs the official cvxpy CBC/GLPK and pymoo environment; physical-cluster execution needs AdaptDL on Kubernetes with container images and cluster-specific GPU-type mapping",
        ),
    ),
    "iadeep_kubernetes_extender": FullStackRequirement(
        adapter="iadeep_kubernetes_extender",
        required_tools=("go", "kubectl", "docker"),
        same_workload_assets=("iadeep_pod_manifest_generator",),
        stack_assets=(
            "iadeep-scheduler-extender/go.mod",
            "iadeep-device-plugin/go.mod",
            "iadeep-scheduler-extender/config/iadeep-scheduler-policy-config.yaml",
        ),
        extra_blockers=(
            "IADeep pod manifest seeds exist, but a Kubernetes cluster with NVIDIA device-plugin/runtime wiring is still required",
        ),
    ),
    "salus_gpu_sharing": FullStackRequirement(
        adapter="salus_gpu_sharing",
        required_tools=("docker", "nvidia-smi"),
        same_workload_assets=("salus_benchmark_driver_scheduleurm_adapter",),
        stack_assets=("Dockerfile", "CMakeLists.txt", "benchmarks/driver/__main__.py"),
        extra_blockers=(
            "Salus benchmark seed exists, but Salus server/runtime execution is still required",
        ),
    ),
    "decima_simulator": FullStackRequirement(
        adapter="decima_simulator",
        required_tools=("python3",),
        same_workload_assets=("scheduleurm_to_decima_dag_converter",),
        stack_assets=("compute_baselines.py", "spark_env"),
        extra_blockers=(
            "Decima is a Spark-DAG simulator, so even a runnable entrypoint is not a GPU co-location system baseline",
        ),
    ),
}


def build_direct_sota_fullstack_readiness(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    run_smoke: bool = True,
) -> dict[str, Any]:
    root = Path(reference_root).expanduser()
    scaffold = build_direct_sota_scaffold(
        reference_root=root,
        run_smoke=run_smoke,
        tasksets=(
            "q00_light_control",
            "q01_gpu_bound_compute",
            "q10_cpu_host_bound",
            "q11_cpu_gpu_coupled",
            "hybrid_research_portfolio",
        ),
    )
    gavel_smoke_artifact = _latest_artifact(
        "gavel_direct_native_smoke_*.json",
        GAVEL_NATIVE_SMOKE_ARTIFACT,
    )
    gavel_performance_artifact = _latest_artifact(
        "gavel_native_performance_microbaseline_*.json",
        GAVEL_NATIVE_PERFORMANCE_ARTIFACT,
    )
    native_smokes = {
        "gavel_simulation": _load_json_if_exists(gavel_smoke_artifact),
    }
    native_performance = {
        "gavel_simulation": _load_json_if_exists(gavel_performance_artifact),
    }
    adapter_rows = []
    for adapter in ADAPTERS:
        scaffold_row = _find_scaffold_row(scaffold, adapter.name)
        requirement = REQUIREMENTS[adapter.name]
        adapter_rows.append(
            _readiness_row(
                adapter_name=adapter.name,
                repo=root / adapter.repo_dir,
                requirement=requirement,
                scaffold_row=scaffold_row,
                native_smoke=native_smokes.get(adapter.name),
                native_performance=native_performance.get(adapter.name),
            )
        )
    same_workload_ready = [
        row for row in adapter_rows
        if row.get("same_workload_full_stack_ready")
    ]
    full_stack_ready = [
        row for row in adapter_rows
        if row.get("entrypoint_smoke_pass") and row.get("same_workload_full_stack_ready")
    ]
    return {
        "gate": "direct_sota_fullstack_readiness",
        "reference_root": str(root),
        "run_smoke": bool(run_smoke),
        "tool_inventory": _tool_inventory(),
        "adapters": adapter_rows,
        "entrypoint_smoke_pass_count": sum(
            1 for row in adapter_rows if row.get("entrypoint_smoke_pass")
        ),
        "same_workload_full_stack_ready_count": len(same_workload_ready),
        "direct_full_stack_same_workload_ready": bool(full_stack_ready),
        "policy_semantics_fallback_available": bool(scaffold.get("pass")),
        "pass": True,
        "scope": (
            "read-only readiness audit for direct external-system baselines. "
            "It certifies local repo/entrypoint/stack blockers; it does not claim "
            "direct full-stack superiority unless direct_full_stack_same_workload_ready is true."
        ),
        "scaffold_summary": {
            "direct_full_stack_ready": scaffold.get("direct_full_stack_ready"),
            "any_direct_adapter_smoke_pass": scaffold.get("any_direct_adapter_smoke_pass"),
            "fallback_policy_semantics_pass": scaffold.get("pass"),
        },
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Direct SOTA Full-Stack Readiness",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `entrypoint_smoke_pass_count` | {report.get('entrypoint_smoke_pass_count', 0)} |",
        f"| `same_workload_full_stack_ready_count` | {report.get('same_workload_full_stack_ready_count', 0)} |",
        f"| `direct_full_stack_same_workload_ready` | {str(bool(report.get('direct_full_stack_same_workload_ready'))).lower()} |",
        f"| `policy_semantics_fallback_available` | {str(bool(report.get('policy_semantics_fallback_available'))).lower()} |",
        "",
        "## Tool Inventory",
        "",
        "| Tool | Available | Path |",
        "|---|---:|---|",
    ]
    for tool, row in sorted((report.get("tool_inventory") or {}).items()):
        lines.append(
            f"| `{tool}` | {str(bool(row.get('available'))).lower()} | `{row.get('path') or ''}` |"
        )
    lines.extend([
        "",
        "## Adapter Readiness",
        "",
        "| Adapter | Repo | Smoke | Native generated | Native trace | Native microbaseline | Missing tools | Missing workload assets | Full-stack ready | Blockers |",
        "|---|---:|---:|---:|---|---:|---|---|---:|---|",
    ])
    for row in report.get("adapters") or []:
        lines.append(
            "| `{name}` | {repo} | {smoke} | {generated} | `{native}` | {micro} | {tools} | {assets} | {ready} | {blockers} |".format(
                name=row.get("adapter"),
                repo=str(bool(row.get("repo_exists"))).lower(),
                smoke=str(bool(row.get("entrypoint_smoke_pass"))).lower(),
                generated=str(bool((row.get("native_smoke") or {}).get("native_generated_jobs_pass"))).lower(),
                native=(row.get("native_smoke") or {}).get("native_trace_status") or "",
                micro=str(bool((row.get("native_performance") or {}).get("native_gavel_simulator_microbaseline_ready"))).lower(),
                tools=", ".join(f"`{x}`" for x in row.get("missing_required_tools") or []) or "none",
                assets=", ".join(f"`{x}`" for x in row.get("missing_same_workload_assets") or []) or "none",
                ready=str(bool(row.get("same_workload_full_stack_ready"))).lower(),
                blockers="; ".join(row.get("blockers") or []) or "none",
            )
        )
    lines.extend([
        "",
        "## Scope",
        "",
        str(report.get("scope") or ""),
        "",
    ])
    return "\n".join(lines)


def _readiness_row(
    *,
    adapter_name: str,
    repo: Path,
    requirement: FullStackRequirement,
    scaffold_row: Mapping[str, Any],
    native_smoke: Mapping[str, Any] | None = None,
    native_performance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    missing_tools = [tool for tool in requirement.required_tools if shutil.which(tool) is None]
    missing_stack_assets = [asset for asset in requirement.stack_assets if not (repo / asset).exists()]
    missing_workload_assets = [
        asset for asset in requirement.same_workload_assets
        if not _repo_asset_exists(asset)
    ]
    blockers = []
    if not repo.exists():
        blockers.append("repository not cloned")
    if missing_tools:
        blockers.append("missing required tools: " + ", ".join(missing_tools))
    if missing_stack_assets:
        blockers.append("missing stack assets: " + ", ".join(missing_stack_assets))
    if missing_workload_assets:
        blockers.append("missing Scheduleurm same-workload adapters: " + ", ".join(missing_workload_assets))
    native_smoke_summary = _native_smoke_summary(native_smoke)
    native_performance_summary = _native_performance_summary(native_performance)
    if adapter_name == "gavel_simulation":
        if not native_smoke:
            blockers.append("Gavel isolated native smoke artifact is missing")
        elif not native_smoke.get("direct_native_trace_pass"):
            blockers.append(
                "Gavel native trace smoke is not a direct baseline: "
                + str(native_smoke_summary.get("native_trace_status") or "unknown")
            )
        elif native_smoke.get("scheduleurm_native_trace_seed_pass"):
            blockers.append(
                "Gavel native trace and Scheduleurm trace-seed smoke pass, but service-unit equivalence is not yet a direct performance baseline"
            )
        if native_performance_summary.get("native_gavel_simulator_microbaseline_ready"):
            blockers.append(
                "Gavel native simulator microbaseline rows exist, but they still do not certify Scheduleurm measured-service-unit equivalence"
            )
    blockers.extend(requirement.extra_blockers)
    entrypoint_smoke_pass = bool(scaffold_row.get("entrypoint_smoke_pass"))
    smoke_status = scaffold_row.get("smoke_status")
    if adapter_name == "gavel_simulation" and native_smoke_summary.get("entrypoint_help_pass"):
        entrypoint_smoke_pass = True
        if smoke_status != "PASS":
            smoke_status = "PASS_ISOLATED_NATIVE"
    same_workload_ready = (
        repo.exists()
        and not missing_tools
        and not missing_stack_assets
        and not missing_workload_assets
        and not requirement.extra_blockers
    )
    return {
        "adapter": adapter_name,
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "entrypoint_smoke_pass": entrypoint_smoke_pass,
        "smoke_status": smoke_status,
        "missing_required_tools": missing_tools,
        "missing_stack_assets": missing_stack_assets,
        "missing_same_workload_assets": missing_workload_assets,
        "native_smoke": native_smoke_summary,
        "native_performance": native_performance_summary,
        "same_workload_full_stack_ready": same_workload_ready,
        "blockers": blockers,
    }


def _repo_asset_exists(asset: str) -> bool:
    roots = (
        REPO_ROOT / "algorithm",
        REPO_ROOT / "simulation",
        REPO_ROOT / "scripts",
    )
    needles = {
        asset,
        f"{asset}.py",
        f"{asset}.json",
        f"{asset}.yaml",
        f"{asset}.yml",
    }
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name in needles:
                return True
    return False


def _find_scaffold_row(scaffold: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in scaffold.get("adapters") or []:
        if row.get("name") == name:
            return row
    return {}


def _load_json_if_exists(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"artifact": str(path), "error": "json_decode_error"}


def _native_smoke_summary(native_smoke: Mapping[str, Any] | None) -> dict[str, Any]:
    if not native_smoke:
        return {}
    trace = native_smoke.get("native_trace_smoke") or {}
    return {
        "artifact": str((native_smoke or {}).get("artifact") or _latest_artifact(
            "gavel_direct_native_smoke_*.json",
            GAVEL_NATIVE_SMOKE_ARTIFACT,
        )),
        "dependency_import_pass": bool(native_smoke.get("dependency_import_pass")),
        "protobuf_stub_generation_pass": bool(native_smoke.get("protobuf_stub_generation_pass")),
        "entrypoint_help_pass": bool(native_smoke.get("entrypoint_help_pass")),
        "native_generated_jobs_pass": bool(native_smoke.get("native_generated_jobs_pass")),
        "direct_native_trace_pass": bool(native_smoke.get("direct_native_trace_pass")),
        "scheduleurm_native_trace_seed_pass": bool(native_smoke.get("scheduleurm_native_trace_seed_pass")),
        "native_trace_status": trace.get("status") or trace.get("classifier"),
    }


def _native_performance_summary(native_performance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not native_performance:
        return {}
    return {
        "artifact": str((native_performance or {}).get("artifact") or _latest_artifact(
            "gavel_native_performance_microbaseline_*.json",
            GAVEL_NATIVE_PERFORMANCE_ARTIFACT,
        )),
        "usable_native_performance_sample_count": int(
            native_performance.get("usable_native_performance_sample_count") or 0
        ),
        "native_gavel_simulator_microbaseline_ready": bool(
            native_performance.get("native_gavel_simulator_microbaseline_ready")
        ),
        "same_workload_trace_schema_ready": bool(
            native_performance.get("same_workload_trace_schema_ready")
        ),
        "service_unit_equivalence_ready": bool(
            native_performance.get("service_unit_equivalence_ready")
        ),
        "direct_full_stack_performance_ready": bool(
            native_performance.get("direct_full_stack_performance_ready")
        ),
    }


def _tool_inventory() -> dict[str, dict[str, Any]]:
    tools = sorted({tool for req in REQUIREMENTS.values() for tool in req.required_tools})
    out: dict[str, dict[str, Any]] = {}
    for tool in tools:
        path = shutil.which(tool)
        version = _tool_version(tool) if path else ""
        out[tool] = {
            "available": path is not None,
            "path": path,
            "version": version,
        }
    return out


def _tool_version(tool: str) -> str:
    commands = {
        "python3": ("python3", "--version"),
        "go": ("go", "version"),
        "docker": ("docker", "--version"),
        "kubectl": ("kubectl", "version", "--client=true"),
        "nvidia-smi": ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
    }
    cmd = commands.get(tool, (tool, "--version"))
    try:
        proc = subprocess.run(
            list(cmd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or proc.stderr).strip()[:1000]


def _latest_artifact(pattern: str, fallback: Path) -> Path:
    matches = sorted(ARTIFACT_ROOT.glob(pattern), key=lambda path: path.name)
    return matches[-1] if matches else fallback


def _write_json(path: str | Path, report: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_direct_sota_fullstack_readiness(
        reference_root=args.reference_root,
        run_smoke=not args.no_smoke,
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
        prog="python -m algorithm.experiments.direct_sota_fullstack_readiness"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build read-only direct-SOTA full-stack readiness certificate")
    build.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    build.add_argument("--no-smoke", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "direct_sota_fullstack_readiness.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "direct_sota_fullstack_readiness.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
