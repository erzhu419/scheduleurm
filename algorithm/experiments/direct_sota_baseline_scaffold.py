"""Direct external-SOTA baseline scaffold with measured-cache fallback."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache
from simulation.sota_baselines import compare_against_sota_suite
from simulation.tasksets import taskset_by_name


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "repos"
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


@dataclass(frozen=True)
class ExternalAdapter:
    name: str
    representative_systems: tuple[str, ...]
    repo_dir: str
    command_cwd: str
    direct_command: tuple[str, ...]
    required_files: tuple[str, ...]
    blockers: tuple[str, ...]
    fallback_tasksets: tuple[str, ...]


ADAPTERS = (
    ExternalAdapter(
        name="gavel_simulation",
        representative_systems=("Gavel",),
        repo_dir="gavel",
        command_cwd="",
        direct_command=("python3", "scheduler/scripts/sweeps/run_sweep_static.py", "-h"),
        required_files=("scheduler/scripts/sweeps/run_sweep_static.py", "scheduler/requirements.txt"),
        blockers=("requires native validation of the Scheduleurm Gavel trace/throughput seeds",),
        fallback_tasksets=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled", "hybrid_research_portfolio"),
    ),
    ExternalAdapter(
        name="pollux_adaptdl_scheduler",
        representative_systems=("Pollux", "AdaptDL"),
        repo_dir="adaptdl_pollux",
        command_cwd="sched",
        direct_command=("python3", "setup.py", "--help"),
        required_files=("sched/adaptdl_sched/__main__.py", "sched/setup.py", "adaptdl/adaptdl/goodput.py"),
        blockers=("requires Kubernetes/AdaptDL job CRD, container image, and isolated namespace validation",),
        fallback_tasksets=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled", "hybrid_research_portfolio"),
    ),
    ExternalAdapter(
        name="iadeep_kubernetes_extender",
        representative_systems=("IADeep",),
        repo_dir="iadeep",
        command_cwd="iadeep-scheduler-extender",
        direct_command=("go", "test", "./..."),
        required_files=("iadeep-scheduler-extender/go.mod", "iadeep-scheduler-extender/cmd/main.go"),
        blockers=("requires Kubernetes 1.18+, device plugin, etcd, Docker/NVIDIA runtime, and pod-manifest validation",),
        fallback_tasksets=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled"),
    ),
    ExternalAdapter(
        name="salus_gpu_sharing",
        representative_systems=("Salus",),
        repo_dir="salus",
        command_cwd="",
        direct_command=("python3", "-m", "benchmarks.driver", "--help"),
        required_files=("CMakeLists.txt", "benchmarks/driver/__main__.py", "requirements.txt"),
        blockers=("requires Salus server, customized TensorFlow/runtime stack, and benchmark seed validation",),
        fallback_tasksets=("q01_gpu_bound_compute", "q11_cpu_gpu_coupled"),
    ),
    ExternalAdapter(
        name="decima_simulator",
        representative_systems=("Decima",),
        repo_dir="decima_sim",
        command_cwd="",
        direct_command=("python3", "compute_baselines.py", "--help"),
        required_files=("compute_baselines.py", "spark_env", "param.py"),
        blockers=("Spark-DAG simulator baseline, not a GPU co-location scheduler",),
        fallback_tasksets=("q00_light_control", "q10_cpu_host_bound"),
    ),
)


def build_direct_sota_scaffold(
    *,
    reference_root: str | Path = REFERENCE_ROOT,
    run_smoke: bool = False,
    tasksets: Iterable[str] = ("hybrid_research_portfolio",),
) -> dict[str, Any]:
    root = Path(reference_root).expanduser()
    adapters = [_adapter_row(adapter, root=root, run_smoke=run_smoke) for adapter in ADAPTERS]
    fallback = _fallback_comparisons(tasksets)
    return {
        "gate": "direct_sota_binary_baseline_scaffold",
        "reference_root": str(root),
        "run_smoke": bool(run_smoke),
        "adapters": adapters,
        "direct_full_stack_ready": all(row.get("direct_runnable") for row in adapters),
        "any_direct_adapter_smoke_pass": any(row.get("smoke_status") == "PASS" for row in adapters),
        "fallback_policy_semantics": fallback,
        "pass": bool(fallback.get("all_fallbacks_replayable")),
        "scope": (
            "direct external-system adapters are discovered and checked for local "
            "entrypoints. Full-stack superiority claims require the direct adapter "
            "status to become runnable with validated same-workload adapter seeds. Until then "
            "the paper uses policy-semantics replay baselines on the same measured "
            "Scheduleurm service cache."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Direct SOTA Baseline Scaffold",
        "",
        "## Adapter Status",
        "",
        "| Adapter | Systems | Repo | Command cwd | Files | Entrypoint smoke | Full-stack baseline | Blocker |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in report.get("adapters") or []:
        lines.append(
            "| `{name}` | {systems} | `{repo}` | `{cwd}` | {files} | `{smoke}` | {runnable} | {blocker} |".format(
                name=row.get("name"),
                systems=", ".join(row.get("representative_systems") or []),
                repo=row.get("repo_path"),
                cwd=row.get("command_cwd") or ".",
                files=row.get("required_file_count_present"),
                smoke=row.get("smoke_status"),
                runnable=str(bool(row.get("direct_runnable"))).lower(),
                blocker="; ".join(row.get("blockers") or []),
            )
        )
    fallback = report.get("fallback_policy_semantics") or {}
    lines.extend([
        "",
        "## Fallback Replay",
        "",
        "| Taskset | Replayable | Candidate nondominated | Best makespan baseline | Best flow baseline |",
        "|---|---:|---:|---|---|",
    ])
    for row in fallback.get("tasksets") or []:
        lines.append(
            "| `{taskset}` | {replayable} | {nondom} | `{best_ms}` | `{best_flow}` |".format(
                taskset=row.get("taskset"),
                replayable=str(bool(row.get("replayable"))).lower(),
                nondom=str(bool((row.get("comparison") or {}).get("candidate_not_pareto_dominated"))).lower(),
                best_ms=((row.get("comparison") or {}).get("best_baseline_by_makespan") or {}).get("name"),
                best_flow=((row.get("comparison") or {}).get("best_baseline_by_mean_flow") or {}).get("name"),
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


def _adapter_row(adapter: ExternalAdapter, *, root: Path, run_smoke: bool) -> dict[str, Any]:
    repo = root / adapter.repo_dir
    command_cwd = repo / adapter.command_cwd if adapter.command_cwd else repo
    files = [str(path) for path in adapter.required_files if (repo / path).exists()]
    git = _git_info(repo)
    smoke = _smoke(adapter, repo) if run_smoke and repo.exists() else {
        "status": "NOT_RUN",
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "",
        "cwd": str(command_cwd),
    }
    missing = [path for path in adapter.required_files if not (repo / path).exists()]
    blockers = list(adapter.blockers)
    if missing:
        blockers.append("missing required files: " + ", ".join(missing))
    if not repo.exists():
        blockers.append("repository not cloned")
    if repo.exists() and not command_cwd.exists():
        blockers.append(f"command cwd does not exist: {adapter.command_cwd or '.'}")
    direct_runnable = (
        repo.exists()
        and not missing
        and command_cwd.exists()
        and run_smoke
        and smoke.get("status") == "PASS"
        and not adapter.blockers
    )
    return {
        "name": adapter.name,
        "representative_systems": list(adapter.representative_systems),
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "git": git,
        "direct_command": list(adapter.direct_command),
        "command_cwd": adapter.command_cwd or ".",
        "command_cwd_exists": command_cwd.exists(),
        "required_file_count_present": len(files),
        "required_file_count": len(adapter.required_files),
        "missing_required_files": missing,
        "smoke_status": smoke.get("status"),
        "smoke": smoke,
        "entrypoint_smoke_pass": smoke.get("status") == "PASS",
        "direct_runnable": direct_runnable,
        "blockers": blockers,
        "fallback_tasksets": list(adapter.fallback_tasksets),
    }


def _fallback_comparisons(tasksets: Iterable[str]) -> dict[str, Any]:
    cache = build_default_cache()
    rows = []
    for name in tasksets:
        taskset = taskset_by_name(str(name))
        missing = taskset.missing_measurements(cache)
        if missing:
            rows.append({
                "taskset": taskset.name,
                "replayable": False,
                "missing_measurements": missing,
                "comparison": {},
            })
            continue
        comparison = compare_against_sota_suite(cache, taskset.workload_specs(), trials=41, seed=17)
        rows.append({
            "taskset": taskset.name,
            "replayable": True,
            "missing_measurements": {},
            "comparison": comparison,
        })
    return {
        "tasksets": rows,
        "all_fallbacks_replayable": all(row.get("replayable") for row in rows),
        "all_candidates_nondominated": all(
            bool((row.get("comparison") or {}).get("candidate_not_pareto_dominated"))
            for row in rows if row.get("replayable")
        ),
    }


def _smoke(adapter: ExternalAdapter, repo: Path) -> dict[str, Any]:
    cwd = repo / adapter.command_cwd if adapter.command_cwd else repo
    try:
        proc = subprocess.run(
            list(adapter.direct_command),
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        return {
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "cwd": str(cwd),
        }
    except Exception as exc:
        return {
            "status": "ERROR",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "cwd": str(cwd),
        }


def _git_info(repo: Path) -> dict[str, Any]:
    if not repo.exists():
        return {"exists": False}
    return {
        "exists": True,
        "remote": _run_git(repo, "remote", "get-url", "origin"),
        "commit": _run_git(repo, "rev-parse", "HEAD"),
        "dirty": bool(_run_git(repo, "status", "--short")),
    }


def _run_git(repo: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_direct_sota_scaffold(
        reference_root=args.reference_root,
        run_smoke=args.run_smoke,
        tasksets=[x.strip() for x in args.tasksets.split(",") if x.strip()],
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.direct_sota_baseline_scaffold")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Inspect direct SOTA adapters and run replay fallback")
    build.add_argument("--reference-root", default=str(REFERENCE_ROOT))
    build.add_argument("--run-smoke", action="store_true")
    build.add_argument("--tasksets", default="hybrid_research_portfolio")
    build.add_argument("--output", default=str(ARTIFACT_ROOT / "direct_sota_baseline_scaffold.json"))
    build.add_argument("--markdown-output", default=str(REPO_ROOT / "md" / "direct_sota_baseline_scaffold.md"))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
