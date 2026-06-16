"""Decima same-domain Spark-DAG benchmark gate.

Decima is not a GPU co-location scheduler.  This gate therefore compares inside
Decima's own Spark-DAG simulator domain instead of forcing it into the GPU
full-stack gate.  The comparison is deliberately small and reproducible: run
Decima's dynamic-partition heuristic and a simple work-conserving first-frontier
surrogate under the same simulator seed and executor cap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DECIMA_REPO = REPO_ROOT / "reference" / "repos" / "decima_sim"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "decima_same_domain_benchmark_gate_20260614.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "md" / "decima_same_domain_benchmark_gate_20260614.md"


def build_decima_same_domain_benchmark_gate(
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    exec_cap: int = 4,
    num_init_dags: int = 1,
    num_stream_dags: int = 2,
    step_limit: int = 10000,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        dynamic = _run_agent(
            agent="dynamic_partition",
            seed=seed,
            exec_cap=exec_cap,
            num_init_dags=num_init_dags,
            num_stream_dags=num_stream_dags,
            step_limit=step_limit,
        )
        first = _run_agent(
            agent="first_frontier",
            seed=seed,
            exec_cap=exec_cap,
            num_init_dags=num_init_dags,
            num_stream_dags=num_stream_dags,
            step_limit=step_limit,
        )
        rows.append({
            "seed": seed,
            "dynamic_partition": dynamic,
            "first_frontier": first,
            "both_completed": bool(dynamic.get("done") and first.get("done")),
            "dynamic_wall_time": dynamic.get("sim_wall_time"),
            "first_frontier_wall_time": first.get("sim_wall_time"),
            "dynamic_better_or_equal_wall_time": _leq(dynamic.get("sim_wall_time"), first.get("sim_wall_time")),
            "dynamic_better_or_equal_mean_completion": _leq(dynamic.get("mean_completion_time"), first.get("mean_completion_time")),
        })
    completed_rows = [row for row in rows if row["both_completed"]]
    all_pairs_completed = len(completed_rows) == len(rows)
    wall_ready = bool(completed_rows) and all(row["dynamic_better_or_equal_wall_time"] for row in completed_rows)
    mean_ready = bool(completed_rows) and all(row["dynamic_better_or_equal_mean_completion"] for row in completed_rows)
    performance_noninferior_ready = bool(all_pairs_completed and wall_ready and mean_ready)
    benchmark_ready = bool(all_pairs_completed)
    return {
        "gate": "decima_same_domain_benchmark_gate",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": (
            "DECIMA_SAME_DOMAIN_BENCHMARK_EXECUTION_PASS_PERFORMANCE_NONINFERIOR"
            if performance_noninferior_ready
            else "DECIMA_SAME_DOMAIN_BENCHMARK_EXECUTION_PASS_PERFORMANCE_MIXED"
            if benchmark_ready else "DECIMA_SAME_DOMAIN_BENCHMARK_PENDING"
        ),
        "repo": str(DECIMA_REPO),
        "repo_ready": DECIMA_REPO.exists(),
        "seed_count": len(seeds),
        "completed_pair_count": len(completed_rows),
        "exec_cap": int(exec_cap),
        "num_init_dags": int(num_init_dags),
        "num_stream_dags": int(num_stream_dags),
        "step_limit": int(step_limit),
        "dynamic_partition_wall_time_noninferior_ready": wall_ready,
        "dynamic_partition_mean_completion_noninferior_ready": mean_ready,
        "dynamic_partition_all_metric_noninferior_ready": performance_noninferior_ready,
        "spark_dag_same_domain_benchmark_ready": benchmark_ready,
        "spark_dag_same_domain_execution_ready": benchmark_ready,
        "spark_dag_same_domain_performance_superiority_ready": performance_noninferior_ready,
        "scheduleurm_gpu_colocation_comparable_ready": False,
        "direct_fullstack_gpu_sota_claim_ready": False,
        "scoped_claim_ready": benchmark_ready,
        "strong_claim_ready": False,
        "rows": rows,
        "blocker": (
            "The Decima same-domain Spark-DAG benchmark execution is closed, "
            "but Decima remains an adjacent Spark-DAG simulator.  Mixed "
            "same-domain performance rows must be reported as such and do not "
            "establish GPU co-location full-stack superiority."
        ),
        "scope": (
            "Same-domain Decima Spark-DAG simulator benchmark.  The closed "
            "claim is executable same-domain benchmark coverage, not Decima "
            "performance superiority and not GPU co-location or Scheduleurm "
            "production-cluster comparability."
        ),
        "pass": benchmark_ready,
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Decima Same-Domain Benchmark Gate",
        "",
        "| Quantity | Value |",
        "|---|---:|",
        f"| `pass` | {str(bool(report.get('pass'))).lower()} |",
        f"| `status` | `{report.get('status')}` |",
        f"| `seed_count` | {report.get('seed_count')} |",
        f"| `completed_pair_count` | {report.get('completed_pair_count')} |",
        f"| `spark_dag_same_domain_benchmark_ready` | {str(bool(report.get('spark_dag_same_domain_benchmark_ready'))).lower()} |",
        f"| `scheduleurm_gpu_colocation_comparable_ready` | {str(bool(report.get('scheduleurm_gpu_colocation_comparable_ready'))).lower()} |",
        "",
        "## Rows",
        "",
        "| Seed | Dynamic wall | First-frontier wall | Dynamic mean completion | First-frontier mean completion | Completed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        dynamic = row.get("dynamic_partition") or {}
        first = row.get("first_frontier") or {}
        lines.append(
            f"| {row.get('seed')} | {dynamic.get('sim_wall_time')} | {first.get('sim_wall_time')} | "
            f"{dynamic.get('mean_completion_time')} | {first.get('mean_completion_time')} | "
            f"{str(bool(row.get('both_completed'))).lower()} |"
        )
    lines.extend(["", "## Scope", "", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def _run_agent(
    *,
    agent: str,
    seed: int,
    exec_cap: int,
    num_init_dags: int,
    num_stream_dags: int,
    step_limit: int,
) -> dict[str, Any]:
    code = _runner_code(agent=agent, seed=seed, exec_cap=exec_cap, num_init_dags=num_init_dags, num_stream_dags=num_stream_dags, step_limit=step_limit)
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            cwd=DECIMA_REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "agent": agent,
            "seed": seed,
            "returncode": None,
            "done": False,
            "error": str(exc),
        }
    parsed: dict[str, Any] = {}
    for line in reversed(proc.stdout.splitlines()):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    parsed.update({
        "agent": agent,
        "seed": seed,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    })
    if "done" not in parsed:
        parsed["done"] = False
    return parsed


def _runner_code(
    *,
    agent: str,
    seed: int,
    exec_cap: int,
    num_init_dags: int,
    num_stream_dags: int,
    step_limit: int,
) -> str:
    return textwrap.dedent(f"""
        import json, sys, time
        sys.argv = [
            'decima_same_domain',
            '--exec_cap', '{int(exec_cap)}',
            '--num_init_dags', '{int(num_init_dags)}',
            '--num_stream_dags', '{int(num_stream_dags)}',
            '--canvs_visualization', '0',
            '--num_exp', '1',
        ]
        sys.path.insert(0, '.')
        from spark_env.env import Environment
        from heuristic_agent import DynamicPartitionAgent

        class FirstFrontierAgent(object):
            def get_action(self, obs):
                job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, moving_executors, action_map = obs
                for node in frontier_nodes:
                    use_exec = min(
                        node.num_tasks - node.next_task_idx - exec_commit.node_commit[node] - moving_executors.count(node),
                        num_source_exec,
                    )
                    if use_exec > 0:
                        return node, use_exec
                return None, num_source_exec

        env = Environment()
        env.seed({int(seed)})
        env.reset()
        agent = DynamicPartitionAgent() if '{agent}' == 'dynamic_partition' else FirstFrontierAgent()
        obs = env.observe()
        total_reward = 0.0
        steps = 0
        start = time.time()
        done = False
        while not done and steps < {int(step_limit)}:
            node, use_exec = agent.get_action(obs)
            obs, reward, done = env.step(node, use_exec)
            total_reward += float(reward)
            steps += 1
        completions = [float(j.completion_time) for j in env.finished_job_dags if getattr(j, 'completion_time', None) is not None]
        out = {{
            'done': bool(done),
            'steps': int(steps),
            'total_reward': total_reward,
            'python_wall_s': round(time.time() - start, 6),
            'sim_wall_time': float(env.wall_time.curr_time),
            'finished_jobs': len(env.finished_job_dags),
            'mean_completion_time': sum(completions) / len(completions) if completions else None,
            'max_completion_time': max(completions) if completions else None,
        }}
        print(json.dumps(out, sort_keys=True))
        raise SystemExit(0 if done else 2)
    """)


def _leq(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return float(left) <= float(right) + 1e-9


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_build(args: argparse.Namespace) -> int:
    seeds = tuple(int(x) for x in str(args.seeds).split(",") if str(x).strip())
    report = build_decima_same_domain_benchmark_gate(
        seeds=seeds,
        exec_cap=args.exec_cap,
        num_init_dags=args.num_init_dags,
        num_stream_dags=args.num_stream_dags,
        step_limit=args.step_limit,
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
        prog="python -m algorithm.experiments.decima_same_domain_benchmark_gate"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="Build Decima same-domain Spark-DAG benchmark gate")
    build.add_argument("--seeds", default="1,2,3")
    build.add_argument("--exec-cap", type=int, default=4)
    build.add_argument("--num-init-dags", type=int, default=1)
    build.add_argument("--num-stream-dags", type=int, default=2)
    build.add_argument("--step-limit", type=int, default=10000)
    build.add_argument("--output", default=str(DEFAULT_OUTPUT))
    build.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    build.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
