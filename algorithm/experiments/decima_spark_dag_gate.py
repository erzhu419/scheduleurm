"""Decima Spark-DAG compatibility gate.

Decima is a Spark-DAG scheduling simulator, not a GPU co-location scheduler.
This gate records what is runnable from the cloned Decima simulator and keeps
the comparison scope separate from the GPU full-stack SOTA gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DECIMA_REPO = REPO_ROOT / "reference" / "repos" / "decima_sim"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "decima_spark_dag_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "decima_spark_dag_gate_20260614.md"


def build_decima_spark_dag_gate(*, run_smoke: bool = True) -> dict[str, Any]:
    repo_ready = DECIMA_REPO.exists()
    required = [
        "compute_baselines.py",
        "test.py",
        "spark_env/env.py",
        "spark_env/job_generator.py",
        "spark_env/job_dag.py",
        "spark_env/task.py",
    ]
    file_rows = [
        {"path": path, "present": (DECIMA_REPO / path).exists()}
        for path in required
    ]
    import_smoke = _run_import_smoke() if run_smoke and repo_ready else _not_run("disabled_or_missing_repo")
    baseline_help = _run(["python3", "compute_baselines.py", "--help"], timeout=20) if run_smoke and repo_ready else _not_run("disabled_or_missing_repo")
    learned_help = _run(["python3", "test.py", "--help"], timeout=20) if run_smoke and repo_ready else _not_run("disabled_or_missing_repo")
    heuristic_benchmark = _run_heuristic_benchmark() if run_smoke and repo_ready else _not_run("disabled_or_missing_repo")
    simulator_smoke_ready = bool(
        repo_ready
        and all(row["present"] for row in file_rows)
        and import_smoke.get("returncode") == 0
        and baseline_help.get("returncode") == 0
    )
    spark_dag_heuristic_benchmark_ready = bool(heuristic_benchmark.get("returncode") == 0)
    learned_policy_ready = bool(learned_help.get("returncode") == 0)
    comparable_gpu_claim_ready = False
    return {
        "gate": "decima_spark_dag_gate",
        "status": (
            "DECIMA_SIMULATOR_SMOKE_READY_GPU_COMPARISON_FALSE"
            if simulator_smoke_ready else "DECIMA_SIMULATOR_SMOKE_PENDING"
        ),
        "repo": str(DECIMA_REPO),
        "repo_ready": repo_ready,
        "file_rows": file_rows,
        "simulator_import_smoke": import_smoke,
        "baseline_entrypoint_smoke": baseline_help,
        "heuristic_spark_dag_benchmark": heuristic_benchmark,
        "learned_policy_entrypoint_smoke": learned_help,
        "spark_dag_simulator_smoke_ready": simulator_smoke_ready,
        "spark_dag_heuristic_benchmark_ready": spark_dag_heuristic_benchmark_ready,
        "learned_policy_runtime_ready": learned_policy_ready,
        "spark_dag_metric_bridge_ready": bool(simulator_smoke_ready and spark_dag_heuristic_benchmark_ready),
        "scheduleurm_gpu_colocation_comparable_ready": comparable_gpu_claim_ready,
        "direct_fullstack_gpu_sota_claim_ready": False,
        "scoped_claim_ready": bool(simulator_smoke_ready),
        "strong_claim_ready": False,
        "blockers": _blockers(
            repo_ready=repo_ready,
            file_rows=file_rows,
            import_smoke=import_smoke,
            baseline_help=baseline_help,
            learned_help=learned_help,
        ),
        "pass": True,
        "scope": (
            "Decima evidence is Spark-DAG simulator evidence.  A Decima claim "
            "needs a Spark-DAG benchmark and metric bridge.  It should not be "
            "mixed into the GPU co-location full-stack superiority gate."
        ),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Decima Spark-DAG Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `repo_ready` | {str(bool(report.get('repo_ready'))).lower()} |",
        f"| `spark_dag_simulator_smoke_ready` | {str(bool(report.get('spark_dag_simulator_smoke_ready'))).lower()} |",
        f"| `spark_dag_heuristic_benchmark_ready` | {str(bool(report.get('spark_dag_heuristic_benchmark_ready'))).lower()} |",
        f"| `spark_dag_metric_bridge_ready` | {str(bool(report.get('spark_dag_metric_bridge_ready'))).lower()} |",
        f"| `learned_policy_runtime_ready` | {str(bool(report.get('learned_policy_runtime_ready'))).lower()} |",
        f"| `scheduleurm_gpu_colocation_comparable_ready` | {str(bool(report.get('scheduleurm_gpu_colocation_comparable_ready'))).lower()} |",
        f"| `direct_fullstack_gpu_sota_claim_ready` | {str(bool(report.get('direct_fullstack_gpu_sota_claim_ready'))).lower()} |",
        "",
        "## Required Files",
        "",
        "| Path | Present |",
        "|---|---:|",
    ]
    for row in report.get("file_rows") or []:
        lines.append(f"| `{row.get('path')}` | {str(bool(row.get('present'))).lower()} |")
    lines.extend(["", "## Smoke Results", "", "| Probe | Return code | Note |", "|---|---:|---|"])
    for name, key in (
        ("import", "simulator_import_smoke"),
        ("baseline_help", "baseline_entrypoint_smoke"),
        ("heuristic_benchmark", "heuristic_spark_dag_benchmark"),
        ("learned_policy_help", "learned_policy_entrypoint_smoke"),
    ):
        row = report.get(key) or {}
        lines.append(f"| `{name}` | `{row.get('returncode')}` | {_md(row.get('stderr_tail') or row.get('stdout_tail') or row.get('note'))} |")
    lines.extend(["", "## Blockers", "", "| Blocker |", "|---|"])
    blockers = report.get("blockers") or []
    if not blockers:
        lines.append("| none |")
    for blocker in blockers:
        lines.append(f"| {_md(blocker)} |")
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _run_import_smoke() -> dict[str, Any]:
    code = (
        "import sys; "
        "sys.path.insert(0,'.'); "
        "from spark_env.job_generator import generate_jobs; "
        "from spark_env.env import Environment; "
        "print('DECIMA_IMPORT_SMOKE_READY')"
    )
    return _run(["python3", "-c", code], timeout=20)


def _run_heuristic_benchmark() -> dict[str, Any]:
    code = r"""
import json, sys, time
sys.argv = [
    'decima_heuristic_smoke',
    '--exec_cap', '4',
    '--num_init_dags', '1',
    '--num_stream_dags', '2',
    '--canvs_visualization', '0',
    '--num_exp', '1',
]
sys.path.insert(0, '.')
from spark_env.env import Environment
from heuristic_agent import DynamicPartitionAgent

env = Environment()
env.seed(1)
env.reset()
agent = DynamicPartitionAgent()
obs = env.observe()
total_reward = 0.0
steps = 0
start = time.time()
done = False
while not done and steps < 10000:
    node, use_exec = agent.get_action(obs)
    obs, reward, done = env.step(node, use_exec)
    total_reward += float(reward)
    steps += 1
print(json.dumps({
    'done': done,
    'steps': steps,
    'total_reward': total_reward,
    'wall_s': round(time.time() - start, 6),
    'finished_jobs': len(env.finished_job_dags),
}, sort_keys=True))
raise SystemExit(0 if done else 2)
"""
    return _run(["python3", "-c", code], timeout=30)


def _run(cmd: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=DECIMA_REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _not_run(note: str) -> dict[str, Any]:
    return {"returncode": None, "stdout_tail": "", "stderr_tail": "", "note": note}


def _blockers(
    *,
    repo_ready: bool,
    file_rows: list[Mapping[str, Any]],
    import_smoke: Mapping[str, Any],
    baseline_help: Mapping[str, Any],
    learned_help: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not repo_ready:
        blockers.append("Decima repository is not cloned")
    missing = [row["path"] for row in file_rows if not row.get("present")]
    if missing:
        blockers.append("missing Decima files: " + ", ".join(str(x) for x in missing))
    if import_smoke.get("returncode") != 0:
        blockers.append("Decima Spark environment import smoke failed")
    if baseline_help.get("returncode") != 0:
        blockers.append("Decima baseline entrypoint smoke failed")
    if learned_help.get("returncode") != 0:
        blockers.append(
            "Decima learned-policy entrypoint is not runnable in the local environment; TensorFlow is the expected blocker if absent"
        )
    blockers.append(
        "Decima is a Spark-DAG simulator; a Scheduleurm-vs-Decima claim needs a Spark-DAG workload and metric bridge, not a GPU co-location probe"
    )
    return blockers


def _md(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:1000]


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_decima_spark_dag_gate(run_smoke=not args.no_smoke)
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.decima_spark_dag_gate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Decima Spark-DAG compatibility gate")
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
