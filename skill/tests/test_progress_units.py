from algorithm.experiments.progress_units import (
    parse_progress_line,
    parse_progress_observation,
    task_progress_observation,
)
from algorithm.experiments.progress_wrapper import build_progress_line
from algorithm.experiments.sweetspot_ab_validation import _summarize_phase


def test_progress_units_parse_benchmark_step_rate(check, sch):
    obs = parse_progress_line(
        "Step 124/2400 label=x rate=10.353019 step/s elapsed=12.0s"
    )
    check("progress parser reads benchmark current/total",
          obs is not None
          and obs.current == 124
          and obs.total == 2400
          and obs.unit == "step",
          diag=str(obs))
    check("progress parser reads benchmark step/s rate",
          obs is not None and abs(float(obs.rate_per_s) - 10.353019) < 1e-9,
          diag=str(obs))


def test_progress_units_parse_rl_seconds_per_iter(check, sch):
    obs = parse_progress_line(
        "Iter 1989 | Reward: 4739.1 | Time: 32142s | 10.6s/iter",
        cmd="python train.py --max_iters 2000",
    )
    check("progress parser reads RL iter and cmd total",
          obs is not None
          and obs.current == 1989
          and obs.total == 2000
          and obs.unit == "iter",
          diag=str(obs))
    check("progress parser converts s/iter to iter/s",
          obs is not None
          and abs(float(obs.rate_per_s) - (1.0 / 10.6)) < 1e-12
          and abs(float(obs.seconds_per_unit) - 10.6) < 1e-12,
          diag=str(obs))


def test_progress_wrapper_emits_scheduler_stable_iter_line(check, sch):
    child_obs = parse_progress_line(
        "Iter    14 | Reward: 123.0 | Time: 456s | 21.0s/iter",
        cmd="python train.py --max_iters 1500",
    )
    line = build_progress_line(child_obs, total_override=1500, unit_override="iter")
    obs = parse_progress_line(line)
    check("progress wrapper emits line with current total and rate",
          line is not None
          and "ScheduleurmProgress Iter 14/1500" in line
          and "ETA " in line
          and obs is not None
          and obs.current == 14
          and obs.total == 1500
          and abs(float(obs.rate_per_s) - (1.0 / 21.0)) < 1e-9,
          diag=f"line={line}, obs={obs}")


def test_progress_parser_ignores_units_per_iter_banner(check, sch):
    obs = parse_progress_line(
        "  Updates/iter: 250  Samples/iter: 4000",
        cmd="python train.py --max_iters 1500",
    )
    child_obs = parse_progress_line("Iter 4000 | bogus | 1.0s/iter", cmd="python train.py --max_iters 1500")
    line = build_progress_line(child_obs, total_override=1500, unit_override="iter") if child_obs else None
    check("progress parser ignores banner unit metadata and impossible current totals",
          obs is None and child_obs is None and line is None,
          diag=f"obs={obs}, child_obs={child_obs}, line={line}")


def test_progress_units_latest_line_wins_and_task_fallbacks(check, sch):
    tail = "\n".join([
        "Iter 10 | Reward: 0.0 | 50.0s/iter",
        "Iter 11 | Reward: 1.0 | 25.0s/iter",
    ])
    obs = parse_progress_observation(tail, cmd="python train.py --max_iters 100")
    check("progress parser uses latest RL progress line",
          obs is not None
          and obs.current == 11
          and obs.total == 100
          and abs(float(obs.rate_per_s) - 0.04) < 1e-12,
          diag=str(obs))

    task_obs = task_progress_observation({
        "last_progress_line": "rate=12.5 step/s",
        "runtime_current_unit": 33,
        "runtime_total_units": 100,
    })
    check("task progress parser fills scheduler runtime counters",
          task_obs is not None
          and task_obs.current == 33
          and task_obs.total == 100
          and task_obs.unit == "step",
          diag=str(task_obs))


def test_sweetspot_summary_counts_rl_iter_rate(check, sch):
    tasks = [
        {
            "id": "t1",
            "status": "running",
            "gpu_idx": 3,
            "cmd": "python train.py --max_iters 2000",
            "last_progress_line": "Iter 1105 | Reward: 123.4 | 25.6s/iter",
        },
        {
            "id": "t2",
            "status": "running",
            "gpu_idx": 3,
            "cmd": "python train.py --max_iters 2000",
            "last_progress_line": "Iter 1110 | Reward: 120.0 | 12.8s/iter",
        },
    ]
    summary = _summarize_phase("rl_2_per_gpu", tasks)
    check("sweetspot summary counts RL iter/s rates",
          summary["running_with_rate_count"] == 2
          and summary["rate_units"] == ["iter"]
          and abs(summary["aggregate_active_rate_unit_s"] - (1 / 25.6 + 1 / 12.8)) < 1e-12,
          diag=str(summary))
