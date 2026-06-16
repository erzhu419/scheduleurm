"""Native external-SOTA execution attempts for reviewer audit.

This gate records what can be executed from the cloned external repositories in
the current machine without mutating the production scheduler or requiring a
Kubernetes/Docker cluster.  It is deliberately not a success gate for direct
full-stack superiority; it is an evidence ledger.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .gavel_direct_native_smoke import build_gavel_direct_native_smoke
from .gavel_native_performance_microbaseline import build_gavel_native_performance_microbaseline


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
REFERENCE_ROOT = REPO_ROOT / "reference" / "repos"
POLLUX_COMPAT_ROOT = Path("/tmp/scheduleurm_pollux_compat")
POLLUX_PYMOO05_ROOT = Path("/tmp/scheduleurm_pollux_pymoo05")
POLLUX_NP126_ROOT = Path("/tmp/scheduleurm_pollux_np126")


def build_sota_native_execution_attempts(*, run_commands: bool = True) -> dict[str, Any]:
    gavel_smoke = build_gavel_direct_native_smoke() if run_commands else {}
    gavel_micro = build_gavel_native_performance_microbaseline() if run_commands else {}
    pollux = _pollux_policy_attempt() if run_commands else _not_run("disabled")
    sia = _run(
        ["python3", "sia.py", "--help"],
        cwd=REFERENCE_ROOT / "sia_artifacts" / "sia-simulator",
        timeout=20,
    ) if run_commands else _not_run("disabled")
    decima = _run(
        ["python3", "compute_baselines.py", "--help"],
        cwd=REFERENCE_ROOT / "decima_sim",
        timeout=20,
    ) if run_commands else _not_run("disabled")
    salus = _run(
        ["python3", "-m", "benchmarks.driver", "--help"],
        cwd=REFERENCE_ROOT / "salus",
        timeout=20,
    ) if run_commands else _not_run("disabled")
    rows = [
        {
            "adapter": "gavel_simulation",
            "native_execution_scope": "isolated generated-jobs, native trace, and bounded same-trace simulator microbaseline",
            "native_execution_ready": bool(gavel_smoke.get("pass")) and bool(gavel_micro.get("native_gavel_simulator_microbaseline_ready")),
            "direct_full_stack_ready": False,
            "details": {
                "direct_native_smoke_pass": bool(gavel_smoke.get("pass")),
                "native_generated_jobs_pass": bool(gavel_smoke.get("native_generated_jobs_pass")),
                "direct_native_trace_pass": bool(gavel_smoke.get("direct_native_trace_pass")),
                "scheduleurm_native_trace_seed_pass": bool(gavel_smoke.get("scheduleurm_native_trace_seed_pass")),
                "native_microbaseline_ready": bool(gavel_micro.get("native_gavel_simulator_microbaseline_ready")),
                "usable_native_performance_sample_count": int(gavel_micro.get("usable_native_performance_sample_count") or 0),
                "service_unit_equivalence_ready": bool(gavel_micro.get("service_unit_equivalence_ready")),
            },
            "blocker": "service-unit equivalence and live full-stack Gavel cluster execution are not certified",
        },
        {
            "adapter": "pollux_adaptdl_scheduler",
            "native_execution_scope": "local Pollux policy-layer pytest with compatibility paths, no Kubernetes cluster",
            "native_execution_ready": bool(
                pollux.get("full_policy_optimizer_tests_ready")
                or pollux.get("policy_layer_import_and_partial_tests_ready")
            ),
            "direct_full_stack_ready": False,
            "details": pollux,
            "blocker": (
                "local Pollux optimizer policy tests pass; direct full stack still "
                "requires Kubernetes/Docker/AdaptDLJob execution"
                if pollux.get("full_policy_optimizer_tests_ready")
                else "full optimizer tests still fail under current pymoo/numpy tensor-layout compatibility; full stack also requires Kubernetes/Docker/AdaptDLJob execution"
            ),
        },
        {
            "adapter": "sia_goodput_scheduler",
            "native_execution_scope": "official Sia simulator entrypoint probe, no AdaptDL/Kubernetes cluster",
            "native_execution_ready": bool(sia.get("returncode") == 0),
            "direct_full_stack_ready": False,
            "details": sia,
            "blocker": (
                "Sia simulator entrypoint runs locally; direct full stack still requires AdaptDL/Kubernetes, container images, and same-workload service-unit/JCT calibration"
                if sia.get("returncode") == 0
                else "Sia official artifact is cloned, but simulator entrypoint needs the official cvxpy CBC/GLPK and pymoo environment; physical run also requires AdaptDL/Kubernetes"
            ),
        },
        {
            "adapter": "decima_simulator",
            "native_execution_scope": "local simulator entrypoint help",
            "native_execution_ready": bool(decima.get("returncode") == 0),
            "direct_full_stack_ready": False,
            "details": decima,
            "blocker": "Decima is a Spark-DAG simulator, not a GPU co-location scheduler baseline for Scheduleurm",
        },
        {
            "adapter": "salus_gpu_sharing",
            "native_execution_scope": "local Python driver entrypoint probe",
            "native_execution_ready": bool(salus.get("returncode") == 0),
            "direct_full_stack_ready": False,
            "details": salus,
            "blocker": "Salus driver requires old future-fstrings/TensorFlow/C++ runtime and server stack; direct full-stack run is not available here",
        },
    ]
    return {
        "gate": "sota_native_execution_attempts",
        "status": "SOTA_NATIVE_EXECUTION_ATTEMPTS_RECORDED",
        "rows": rows,
        "native_execution_ready_count": sum(1 for row in rows if row["native_execution_ready"]),
        "direct_full_stack_ready_count": sum(1 for row in rows if row["direct_full_stack_ready"]),
        "direct_fullstack_sota_superiority_ready": False,
        "scoped_claim_ready": True,
        "strong_claim_ready": False,
        "pass": True,
        "scope": (
            "Native execution ledger for cloned external repositories on the "
            "current host.  Passing native rows do not imply direct full-stack "
            "SOTA superiority unless direct_full_stack_ready is true."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# SOTA Native Execution Attempts",
        "",
        "## Summary",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `native_execution_ready_count` | {report.get('native_execution_ready_count', 0)} |",
        f"| `direct_full_stack_ready_count` | {report.get('direct_full_stack_ready_count', 0)} |",
        f"| `direct_fullstack_sota_superiority_ready` | {str(bool(report.get('direct_fullstack_sota_superiority_ready'))).lower()} |",
        "",
        "## Rows",
        "",
        "| Adapter | Native scope | Native ready | Direct full-stack ready | Blocker |",
        "|---|---|---:|---:|---|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            "| `{adapter}` | {scope} | {native} | {direct} | {blocker} |".format(
                adapter=row.get("adapter"),
                scope=_md(row.get("native_execution_scope")),
                native=str(bool(row.get("native_execution_ready"))).lower(),
                direct=str(bool(row.get("direct_full_stack_ready"))).lower(),
                blocker=_md(row.get("blocker")),
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


def _pollux_policy_attempt() -> dict[str, Any]:
    _ensure_pollux_compat()
    path_parts = [str(POLLUX_COMPAT_ROOT)]
    if POLLUX_NP126_ROOT.exists():
        path_parts.append(str(POLLUX_NP126_ROOT))
    if POLLUX_PYMOO05_ROOT.exists():
        path_parts.append(str(POLLUX_PYMOO05_ROOT))
    path_parts.extend([".", "../adaptdl"])
    result = _run(
        [
            "python3",
            "-m",
            "pytest",
            "-q",
            "adaptdl_sched/policy/pollux_test.py",
            "adaptdl_sched/policy/speedup_test.py",
        ],
        cwd=REFERENCE_ROOT / "adaptdl_pollux" / "sched",
        timeout=60,
        env={"PYTHONPATH": ":".join(path_parts)},
    )
    stdout = str(result.get("stdout_tail") or "") + "\n" + str(result.get("stderr_tail") or "")
    full_ready = bool(result.get("returncode") == 0)
    return {
        **result,
        "compat_root": str(POLLUX_COMPAT_ROOT),
        "pymoo05_root_exists": POLLUX_PYMOO05_ROOT.exists(),
        "numpy126_root_exists": POLLUX_NP126_ROOT.exists(),
        "policy_layer_import_and_partial_tests_ready": full_ready or " passed" in stdout,
        "full_policy_optimizer_tests_ready": full_ready,
        "observed_blocker": _pollux_blocker(stdout),
    }


def _ensure_pollux_compat() -> None:
    crossover = POLLUX_COMPAT_ROOT / "pymoo" / "operators" / "crossover"
    crossover.mkdir(parents=True, exist_ok=True)
    for path in [
        POLLUX_COMPAT_ROOT / "pymoo",
        POLLUX_COMPAT_ROOT / "pymoo" / "operators",
        crossover,
    ]:
        (path / "__init__.py").write_text(
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n",
            encoding="utf-8",
        )
    (crossover / "util.py").write_text(
        "import numpy as np\n\n"
        "def crossover_mask(X, M):\n"
        "    _X = np.copy(X)\n"
        "    _X[0][M] = X[1][M]\n"
        "    _X[1][M] = X[0][M]\n"
        "    return _X\n",
        encoding="utf-8",
    )
    (POLLUX_COMPAT_ROOT / "sitecustomize.py").write_text(
        "try:\n"
        "    import numpy as _np\n"
        "    if not hasattr(_np, 'int'):\n"
        "        _np.int = int\n"
        "    if not hasattr(_np, 'float'):\n"
        "        _np.float = float\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )


def _pollux_blocker(output: str) -> str:
    if "broadcast_to" in output and "axis remapping" in output:
        return "pymoo/numpy tensor-layout compatibility failure in Pollux crossover"
    if "pymoo.operators.crossover.util" in output:
        return "pymoo 0.5 import path missing"
    if "np.int" in output:
        return "NumPy deprecated scalar alias compatibility"
    return output[-500:] if output else ""


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    merged_env = None
    if env is not None:
        import os

        merged_env = dict(os.environ)
        merged_env.update(env)
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env=merged_env,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }
    except Exception as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": repr(exc),
            "timed_out": False,
        }


def _not_run(reason: str) -> dict[str, Any]:
    return {"returncode": None, "stdout_tail": "", "stderr_tail": reason, "timed_out": False}


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, report: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_sota_native_execution_attempts(run_commands=not args.no_run)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.sota_native_execution_attempts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Record native external-SOTA execution attempts")
    build.add_argument("--no-run", action="store_true")
    build.add_argument(
        "--output",
        default=str(ARTIFACT_ROOT / "sota_native_execution_attempts_20260613.json"),
    )
    build.add_argument(
        "--markdown-output",
        default=str(REPO_ROOT / "md" / "sota_native_execution_attempts_20260613.md"),
    )
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
