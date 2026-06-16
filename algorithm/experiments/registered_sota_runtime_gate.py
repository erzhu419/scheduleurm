"""Runtime readiness gate for registered SOTA schedulers beyond the named five.

The named-five same-host full-stack gate is already closed elsewhere.  This
gate attacks the broader registered-SOTA claim by turning each additional
system into a concrete runtime inventory row.  It is intentionally read-only:
no dependency installation, no Kubernetes mutation, and no CUDA/cuDNN changes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "registered_sota_runtime_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "registered_sota_runtime_gate_20260614.md"


def build_registered_sota_runtime_gate(*, run_smoke: bool = True) -> dict[str, Any]:
    rows = [
        _tiresias_row(run_smoke=run_smoke),
        _shockwave_row(run_smoke=run_smoke),
        _allox_row(run_smoke=run_smoke),
        _optimus_row(run_smoke=run_smoke),
        _paper_only_row(
            system="Themis",
            domain="finish-time fairness GPU cluster scheduling",
            paper_url="https://www.usenix.org/conference/nsdi20/presentation/mahajan",
            blocker="No official runtime repository was found in the current registry search; treat as paper-semantics baseline until an artifacted implementation is added.",
        ),
        _paper_only_row(
            system="Gandiva",
            domain="introspective deep-learning cluster scheduling",
            paper_url="https://www.microsoft.com/en-us/research/project/gandiva-scheduler-for-dnns/",
            blocker="No official public runtime repository was found in the current registry search; the system also relies on framework/runtime co-design, so direct comparison needs an artifacted implementation or faithful reimplementation protocol.",
        ),
    ]
    repo_inventory_ready = all(bool(row.get("repo_or_paper_inventory_ready")) for row in rows)
    smoke_ready_count = sum(1 for row in rows if row.get("entrypoint_smoke_ready"))
    runnable_system_count = sum(1 for row in rows if row.get("has_public_runtime_repo"))
    fullstack_ready_count = sum(1 for row in rows if row.get("same_workload_fullstack_ready"))
    return {
        "gate": "registered_sota_runtime_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "REGISTERED_SOTA_RUNTIME_INVENTORY_READY_FULLSTACK_PENDING"
            if repo_inventory_ready else "REGISTERED_SOTA_RUNTIME_INVENTORY_PENDING"
        ),
        "gate_pass": repo_inventory_ready,
        "scoped_claim_ready": repo_inventory_ready,
        "strong_claim_ready": False,
        "registered_extension_system_count": len(rows),
        "runnable_public_runtime_system_count": runnable_system_count,
        "entrypoint_smoke_ready_count": smoke_ready_count,
        "same_workload_fullstack_ready_count": fullstack_ready_count,
        "registered_extension_fullstack_superiority_ready": False,
        "rows": rows,
        "blocker": (
            "Registered SOTA-universe superiority still needs same-workload "
            "executable adapter/runtime rows for each runnable system and an explicit policy for "
            "paper-only systems.  This gate closes inventory and blocker "
            "discovery, not performance superiority."
        ),
        "pass": repo_inventory_ready,
        "scope": (
            "Read-only runtime inventory for registered SOTA systems beyond the "
            "named five.  It distinguishes runnable public repositories from "
            "paper-only systems and records dependency blockers without changing "
            "the host environment."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Registered SOTA Runtime Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `scoped_claim_ready` | {str(bool(report.get('scoped_claim_ready'))).lower()} |",
        f"| `strong_claim_ready` | {str(bool(report.get('strong_claim_ready'))).lower()} |",
        f"| `registered_extension_system_count` | {report.get('registered_extension_system_count')} |",
        f"| `runnable_public_runtime_system_count` | {report.get('runnable_public_runtime_system_count')} |",
        f"| `entrypoint_smoke_ready_count` | {report.get('entrypoint_smoke_ready_count')} |",
        f"| `same_workload_fullstack_ready_count` | {report.get('same_workload_fullstack_ready_count')} |",
        "",
        "## Rows",
        "",
        "| System | Public runtime | Repo/Paper inventory | Entrypoint smoke | Full-stack row | Blocker |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| {system} | {runtime} | {inventory} | {smoke} | {fullstack} | {blocker} |".format(
                system=row.get("system"),
                runtime=str(bool(row.get("has_public_runtime_repo"))).lower(),
                inventory=str(bool(row.get("repo_or_paper_inventory_ready"))).lower(),
                smoke=str(bool(row.get("entrypoint_smoke_ready"))).lower(),
                fullstack=str(bool(row.get("same_workload_fullstack_ready"))).lower(),
                blocker=_md(row.get("blocker")),
            )
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _tiresias_row(*, run_smoke: bool) -> dict[str, Any]:
    repo = REPO_ROOT / "reference" / "repos" / "tiresias"
    simulator = repo / "simulator"
    probes = {
        "python2_numpy": _probe(["python2.7", "-c", "import numpy; print('numpy-ok')"], cwd=simulator) if run_smoke else _not_run(),
        "run_sim_help": _probe(["python2.7", "run_sim.py", "--help"], cwd=simulator) if run_smoke else _not_run(),
    }
    smoke_ready = all(probe.get("returncode") == 0 for probe in probes.values())
    return _runtime_row(
        system="Tiresias",
        domain="GPU cluster manager / trace-driven simulator",
        repo=repo,
        has_public_runtime_repo=True,
        entrypoint_smoke_ready=smoke_ready,
        probes=probes,
        blocker="" if smoke_ready else "Python2 simulator is present, but the current host lacks the Python2 runtime dependencies needed for the entrypoint smoke, starting with numpy.",
    )


def _shockwave_row(*, run_smoke: bool) -> dict[str, Any]:
    repo = REPO_ROOT / "reference" / "repos" / "shockwave"
    scheduler = repo / "scheduler"
    probes = {
        "grpc_tools": _probe(["python3", "-c", "import grpc_tools.protoc; print('grpc-tools-ok')"], cwd=scheduler) if run_smoke else _not_run(),
        "gurobipy": _probe(["python3", "-c", "import gurobipy; print('gurobi-ok')"], cwd=scheduler) if run_smoke else _not_run(),
        "scheduler_help": _probe(["python3", "scheduler.py", "--help"], cwd=scheduler) if run_smoke else _not_run(),
    }
    smoke_ready = all(probe.get("returncode") == 0 for probe in probes.values())
    return _runtime_row(
        system="Shockwave",
        domain="dynamic adaptation / Gavel-derived scheduler runtime and simulator",
        repo=repo,
        has_public_runtime_repo=True,
        entrypoint_smoke_ready=smoke_ready,
        probes=probes,
        blocker="" if smoke_ready else "Shockwave public runtime is cloned, but current-host smoke is blocked by generated protobuf stubs and/or Gurobi dependency.",
    )


def _allox_row(*, run_smoke: bool) -> dict[str, Any]:
    repo = REPO_ROOT / "reference" / "repos" / "allox"
    kube_repo = REPO_ROOT / "reference" / "repos" / "kubernetes-allox"
    sim_repo = REPO_ROOT / "reference" / "repos" / "allox_sim"
    probes = {
        "java": _probe(["java", "-version"], cwd=sim_repo) if run_smoke else _not_run(),
        "javac": _probe(["javac", "-version"], cwd=sim_repo) if run_smoke else _not_run(),
        "sim_run_script": _probe(["bash", "run.sh"], cwd=sim_repo) if run_smoke else _not_run(),
    }
    smoke_ready = all(probe.get("returncode") == 0 for probe in probes.values())
    row = _runtime_row(
        system="AlloX",
        domain="interchangeable CPU/GPU hybrid-cluster allocation",
        repo=repo,
        has_public_runtime_repo=True,
        entrypoint_smoke_ready=smoke_ready,
        probes=probes,
        blocker="" if smoke_ready else "AlloX pointer repo plus Kubernetes fork and simulator are cloned. Java is present when the java probe passes, but the shipped simulator bin is incomplete on this host and javac is unavailable to rebuild the missing inner-class outputs; the run script also expects experiment arguments.",
    )
    row["auxiliary_repos"] = {
        "kubernetes_allox": str(kube_repo),
        "kubernetes_allox_present": kube_repo.exists(),
        "allox_sim": str(sim_repo),
        "allox_sim_present": sim_repo.exists(),
    }
    row["repo_or_paper_inventory_ready"] = bool(repo.exists() and kube_repo.exists() and sim_repo.exists())
    return row


def _optimus_row(*, run_smoke: bool) -> dict[str, Any]:
    repo = REPO_ROOT / "reference" / "repos" / "optimus"
    probes = {
        "python2_numpy": _probe(["python2.7", "-c", "import numpy; print('numpy-ok')"], cwd=repo) if run_smoke else _not_run(),
        "python2_jinja2": _probe(["python2.7", "-c", "import jinja2; print('jinja2-ok')"], cwd=repo) if run_smoke else _not_run(),
        "scheduler_import": _probe(
            [
                "python2.7",
                "-c",
                "import sys; sys.path.insert(0, 'scheduler'); import optimus_scheduler; print('optimus-import-ok')",
            ],
            cwd=repo,
        ) if run_smoke else _not_run(),
        "template_check": _probe(["python2.7", "templates/check-jinja.py", "templates/k8s-mxnet-template.jinja"], cwd=repo) if run_smoke else _not_run(),
    }
    smoke_ready = all(probe.get("returncode") == 0 for probe in probes.values())
    return _runtime_row(
        system="Optimus",
        domain="Kubernetes/MXNet dynamic resource scheduler",
        repo=repo,
        has_public_runtime_repo=True,
        entrypoint_smoke_ready=smoke_ready,
        probes=probes,
        blocker="" if smoke_ready else "Optimus public runtime is cloned, but current-host smoke is blocked by Python2 dependencies such as numpy and jinja2; full comparison also needs Kubernetes/MXNet wiring.",
    )


def _paper_only_row(*, system: str, domain: str, paper_url: str, blocker: str) -> dict[str, Any]:
    return {
        "system": system,
        "domain": domain,
        "repo": "",
        "paper_url": paper_url,
        "repo_present": False,
        "has_public_runtime_repo": False,
        "repo_or_paper_inventory_ready": True,
        "entrypoint_smoke_ready": False,
        "same_workload_fullstack_ready": False,
        "paired_native_superiority_ready": False,
        "probes": {},
        "blocker": blocker,
    }


def _runtime_row(
    *,
    system: str,
    domain: str,
    repo: Path,
    has_public_runtime_repo: bool,
    entrypoint_smoke_ready: bool,
    probes: Mapping[str, Any],
    blocker: str,
) -> dict[str, Any]:
    return {
        "system": system,
        "domain": domain,
        "repo": str(repo),
        "repo_present": repo.exists(),
        "git_head": _git_head(repo),
        "has_public_runtime_repo": has_public_runtime_repo,
        "repo_or_paper_inventory_ready": bool(repo.exists()),
        "entrypoint_smoke_ready": bool(entrypoint_smoke_ready),
        "same_workload_fullstack_ready": False,
        "paired_native_superiority_ready": False,
        "probes": dict(probes),
        "blocker": blocker,
    }


def _probe(cmd: list[str], *, cwd: Path, timeout: int = 20) -> dict[str, Any]:
    if not cwd.exists():
        return {"returncode": None, "stdout_tail": "", "stderr_tail": "cwd missing", "cmd": cmd}
    if shutil.which(cmd[0]) is None:
        return {"returncode": None, "stdout_tail": "", "stderr_tail": f"command not found: {cmd[0]}", "cmd": cmd}
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "stdout_tail": "", "stderr_tail": str(exc), "cmd": cmd}
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "cmd": cmd,
    }


def _not_run() -> dict[str, Any]:
    return {"returncode": None, "stdout_tail": "", "stderr_tail": "smoke disabled", "cmd": []}


def _git_head(repo: Path) -> str:
    if not repo.exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_registered_sota_runtime_gate(run_smoke=not args.no_smoke)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("pass") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.registered_sota_runtime_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build registered SOTA runtime readiness gate")
    build.add_argument("--no-smoke", action="store_true")
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
