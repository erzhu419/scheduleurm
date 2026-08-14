import importlib.util
import time
from pathlib import Path


def _load_eta_tracker(sch):
    path = Path(sch.__file__).resolve().parent / "eta_tracker.py"
    spec = importlib.util.spec_from_file_location("scheduleurm_eta_tracker_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load eta_tracker from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _old_rate_eta(et, tail_text, elapsed_s, fallback_ewma_s=0, cmd=None):
    progress = et.parse_progress(tail_text, cmd=cmd)
    elapsed = max(1.0, float(elapsed_s))
    if progress is not None:
        current, total = progress
        if current >= 1:
            rate = float(current) / elapsed
            if rate > 0:
                return int(max(0, (float(total) - float(current)) / rate))
    if fallback_ewma_s > 0:
        return int(max(0, float(fallback_ewma_s) - elapsed))
    return 0


def test_eta_warmup_threshold_suppresses_absurd_startup_projection(check, sch):
    et = _load_eta_tracker(sch)
    tail = "Iter 1 | loss=0.0"
    cmd = "python train.py --max_iters 2000"

    old_eta = _old_rate_eta(et, tail, elapsed_s=3600, fallback_ewma_s=0, cmd=cmd)
    new_eta = et.compute_eta_seconds(tail, elapsed_s=3600, fallback_ewma_s=0, cmd=cmd)
    new_projection = et.runtime_projection(tail, elapsed_s=3600, cmd=cmd)

    check("ETA warmup: old Iter 1/2000 projection is absurd",
          old_eta == 7196400,
          diag=f"old_eta={old_eta}")
    check("ETA warmup: new Iter 1/2000 without history stays unknown",
          new_eta == 0 and new_projection is None,
          diag=f"new_eta={new_eta}, projection={new_projection}")


def test_eta_warmup_threshold_uses_history_until_progress_is_trusted(check, sch):
    et = _load_eta_tracker(sch)
    cmd = "python train.py --max_iters 2000"

    old_eta = _old_rate_eta(
        et, "Iter 19 | loss=0.0", elapsed_s=3600,
        fallback_ewma_s=21600, cmd=cmd)
    new_eta = et.compute_eta_seconds(
        "Iter 19 | loss=0.0", elapsed_s=3600,
        fallback_ewma_s=21600, cmd=cmd)

    check("ETA warmup: old Iter 19/2000 still overprojects",
          old_eta == 375347,
          diag=f"old_eta={old_eta}")
    check("ETA warmup: new Iter 19/2000 uses history remainder",
          new_eta == 18000,
          diag=f"new_eta={new_eta}")


def test_eta_warmup_threshold_restores_rate_after_threshold(check, sch):
    et = _load_eta_tracker(sch)
    cmd = "python train.py --max_iters 2000"
    eta = et.compute_eta_seconds(
        "Iter 20 | loss=0.0", elapsed_s=3600,
        fallback_ewma_s=0, cmd=cmd)
    projection = et.runtime_projection(
        "Iter 20 | loss=0.0", elapsed_s=3600, cmd=cmd)

    check("ETA warmup: Iter 20/2000 crosses default trust threshold",
          eta == 356400
          and projection
          and projection.get("source") == "progress_rate"
          and projection.get("total_s") == 360000,
          diag=f"eta={eta}, projection={projection}")


def test_eta_warmup_keeps_tqdm_eta_priority(check, sch):
    et = _load_eta_tracker(sch)
    text = " 10%|# | 10/100 [00:42<03:21, 12.34it/s]"
    eta = et.compute_eta_seconds(text, elapsed_s=3600, fallback_ewma_s=99999)
    projection = et.runtime_projection(text, elapsed_s=3600)
    check("ETA warmup: tqdm's own ETA still wins over threshold/rate math",
          eta == 201
          and projection
          and projection.get("source") == "tqdm"
          and projection.get("total_s") == 3801,
	          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_accepts_tqdm_step_units(check, sch):
    et = _load_eta_tracker(sch)
    fast = "bench:  20%|##--------| 20/100 [00:08<00:32, 2.5step/s]"
    slow = "bench:  20%|##--------| 20/100 [00:08<02:40, 2.0s/step]"

    fast_eta = et.compute_eta_seconds(fast, elapsed_s=100, fallback_ewma_s=99999)
    fast_projection = et.runtime_projection(fast, elapsed_s=100)
    slow_eta = et.compute_eta_seconds(slow, elapsed_s=100, fallback_ewma_s=99999)
    slow_projection = et.runtime_projection(slow, elapsed_s=100)

    check("ETA tracker trusts tqdm step/s remaining time",
          fast_eta == 32
          and fast_projection
          and fast_projection.get("source") == "tqdm"
          and fast_projection.get("current") == 20
          and fast_projection.get("total_units") == 100,
          diag=f"eta={fast_eta}, projection={fast_projection}")
    check("ETA tracker trusts tqdm s/step remaining time",
          slow_eta == 160
          and slow_projection
          and slow_projection.get("source") == "tqdm"
          and slow_projection.get("current") == 20
          and slow_projection.get("total_units") == 100,
          diag=f"eta={slow_eta}, projection={slow_projection}")


def test_eta_tracker_trusts_scheduleurm_progress_inline_eta(check, sch):
    et = _load_eta_tracker(sch)
    text = "ScheduleurmProgress Iter 14/1500 rate=0.0476190476 iter/s ETA 31206.0s source=seconds_per_unit"
    eta = et.compute_eta_seconds(text, elapsed_s=3600, fallback_ewma_s=99999)
    projection = et.runtime_projection(text, elapsed_s=3600)

    check("ETA tracker uses task-native inline ETA from progress wrapper",
          eta == 31206
          and projection
          and projection.get("source") == "inline_eta"
          and projection.get("current") == 14
          and projection.get("total_units") == 1500,
          diag=f"eta={eta}, projection={projection}")


def test_h2oplus_lowercase_epoch_uses_recent_timestamp_window(check, sch):
    et = _load_eta_tracker(sch)
    tail = """\
2026-08-14 21:00:29,681 INFO epoch=170/190 reward=-1836 rollout=4.78s train=15.81s
2026-08-14 21:00:50,689 INFO epoch=171/190 reward=-2946 rollout=3.81s train=14.61s
2026-08-14 21:01:11,687 INFO epoch=172/190 reward=-3908 rollout=5.41s train=15.58s
2026-08-14 21:01:31,302 INFO epoch=173/190 reward=-3813 rollout=4.13s train=15.47s
2026-08-14 21:01:52,367 INFO epoch=174/190 reward=-3627 rollout=5.38s train=15.67s
"""
    cmd = "python train_transit_v8_fewshot.py --epochs 190"

    progress = et.parse_progress(tail, cmd=cmd)
    projection = et.runtime_projection(tail, elapsed_s=4500, cmd=cmd)
    eta = et.compute_eta_seconds(tail, elapsed_s=4500, cmd=cmd)

    check(
        "H2O+ ETA: lower-case epoch=current/total is parsed",
        progress == (174, 190),
        diag=f"progress={progress}",
    )
    check(
        "H2O+ ETA: recent timestamp window excludes one-time startup",
        projection
        and projection.get("source") == "progress_window"
        and projection.get("current") == 174
        and projection.get("total_units") == 190
        and 320 <= int(projection.get("eta_s") or 0) <= 340
        and eta == projection.get("eta_s"),
        diag=f"eta={eta}, projection={projection}",
    )
    check(
        "H2O+ ETA: watcher progress-line cache recognizes lower-case epoch",
        "epoch=174/190" in sch._last_progress_line(tail),
        diag=sch._last_progress_line(tail),
    )


def test_eta_tracker_removes_kg_inner_setup_from_future_stage_eta(check, sch):
    et = _load_eta_tracker(sch)
    tail = "\n".join([
        "Step 1100/2000 [kg-inner] kind=iteration_done "
        "stage=10/20 elapsed=516.8s step_elapsed=127.6s ETA 4651.2s eval=60.280s",
        "Step 1200/2000 [kg-inner] kind=iteration_done "
        "stage=11/20 elapsed=585.1s step_elapsed=68.3s ETA 2340.5s eval=0.000s",
        "Step 1300/2000 [kg-inner] kind=iteration_done "
        "stage=12/20 elapsed=654.8s step_elapsed=69.6s ETA 1527.8s eval=0.000s",
        "Step 1400/2000 [kg-inner] kind=iteration_done "
        "stage=13/20 elapsed=723.9s step_elapsed=69.1s ETA 1085.8s eval=0.000s",
        "Step 1500/2000 [kg-inner] kind=iteration_done "
        "stage=14/20 elapsed=792.3s step_elapsed=68.4s ETA 792.3s eval=0.000s",
    ])
    cmd = "python run_lodo_manifest_shard.py --N 20 --n0 10"
    projection = et.runtime_projection(tail, elapsed_s=900, cmd=cmd)
    eta = et.compute_eta_seconds(
        tail, elapsed_s=900, fallback_ewma_s=99999, cmd=cmd)

    check("KG inner ETA ignores one-time initialization replay",
          projection
          and projection.get("source") == "kg_inner_progress"
          and eta == projection.get("eta_s")
          and 470 <= eta <= 480
          and projection.get("future_eval_count") == 2
          and 68.0 <= projection.get("base_step_s", 0) <= 69.0
          and 120.5 <= projection.get("terminal_gap_s", 0) <= 121.0
          and projection.get("one_time_setup_s", 0) >= 380
          and et.parse_inline_eta(tail) == 792,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_kg_inner_uses_observed_eval_interval(check, sch):
    et = _load_eta_tracker(sch)
    tail = "\n".join([
        "Step 1100/2000 [kg-inner] kind=iteration_done "
        "stage=10/20 elapsed=160.0s step_elapsed=150.0s ETA 1440.0s eval=70.0s",
        "Step 1600/2000 [kg-inner] kind=iteration_done "
        "stage=15/20 elapsed=590.0s step_elapsed=150.0s ETA 393.3s eval=70.0s",
    ])
    projection = et.runtime_projection(
        tail, elapsed_s=600, cmd="python run.py --N 20 --n0 10")

    check("KG inner ETA learns periodic eval cadence from stage markers",
          projection
          and projection.get("source") == "kg_inner_progress"
          and projection.get("eval_interval") == 5
          and projection.get("future_eval_count") == 1
          and projection.get("eta_s") == 520,
          diag=f"projection={projection}")


def test_eta_tracker_keeps_terminal_eta_after_kg_inner_reaches_total(check, sch):
    et = _load_eta_tracker(sch)
    tail = "\n".join([
        "Step 1900/2000 [kg-inner] kind=iteration_done "
        "stage=18/20 elapsed=1331.0s step_elapsed=123.5s ETA 147.9s eval=0.0s",
        "Step 2000/2000 [kg-inner] kind=iteration_done "
        "stage=19/20 elapsed=1535.6s step_elapsed=226.2s ETA 0.0s eval=103.0s",
    ])
    projection = et.runtime_projection(
        tail, elapsed_s=1719, cmd="python run.py --N 20 --n0 10")

    check("KG Step N/N retains terminal recommendation/finalization ETA",
          projection
          and projection.get("source") == "kg_inner_progress"
          and projection.get("current") == 2000
          and projection.get("terminal_gap_s") == 206.0
          and 20 <= projection.get("eta_s", 0) <= 25,
          diag=f"projection={projection}")


def test_eta_tracker_overrides_naive_saasbo_inline_eta_with_growing_cost(check, sch):
    et = _load_eta_tracker(sch)
    total = 404
    exponent = 1.7
    scale = 5.0
    lines = []
    for current in range(40, 241, 5):
        elapsed = scale * current ** exponent
        naive_eta = elapsed / current * (total - current)
        lines.append(
            f"Iter {current}/{total} [botorch-canonical] "
            f"label=test:botorch_saasbo Time: {elapsed:.1f}s ETA {naive_eta:.1f}s"
        )
    tail = "\n".join(lines)
    elapsed = scale * 240 ** exponent
    cmd = (
        "python performance/benchmark_sota_fairness.py --protocol target_n404 "
        "--method botorch_saasbo --N 404"
    )

    naive_eta = et.parse_inline_eta(tail)
    projection = et.runtime_projection(tail, elapsed_s=elapsed, cmd=cmd)
    eta = et.compute_eta_seconds(
        tail, elapsed_s=elapsed, fallback_ewma_s=0, cmd=cmd)

    check("SAASBO ETA fixture contains the misleading cumulative-average ETA",
          naive_eta is not None and naive_eta > 0,
          diag=f"naive_eta={naive_eta}")
    check("SAASBO ETA uses growing per-iteration fit instead of inline average",
          projection
          and projection.get("source") == "growing_iter_cost"
          and eta == projection.get("eta_s")
          and eta > 1.5 * naive_eta
          and 1.0 <= projection.get("unit_growth_s", 0) <= 2.2,
          diag=f"naive_eta={naive_eta}, projection={projection}")


def test_eta_tracker_excludes_saas_initial_design_and_counts_terminal_refit(
    check, sch,
):
    et = _load_eta_tracker(sch)
    times = [
        0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.10,
        825.0, 1663.3, 2350.0, 2967.4,
        3588.6, 4215.6, 4840.1, 5449.6,
    ]
    lines = [
        (
            f"Iter {current}/40 [botorch-canonical] "
            f"label=test:botorch_saasbo Time: {elapsed:.1f}s "
            f"ETA 39063.1s eta_model=growing_iter_cost"
        )
        for current, elapsed in enumerate(times, start=1)
    ]
    tail = "\n".join(lines)
    cmd = (
        "python performance/benchmark_sota_fairness.py "
        "--method botorch_saasbo --target-budget 40 --n0 10"
    )

    projection = et.runtime_projection(
        tail, elapsed_s=5600.0, cmd=cmd)

    check(
        "SAAS ETA preserves the task-native current-run projection",
        projection
        and projection.get("source") == "saas_native_eta"
        and projection.get("current") == 18
        and projection.get("total_units") == 40
        and projection.get("eta_s") == 39063
        and projection.get("robust_eta_s", 0) > 0,
        diag=f"projection={projection}",
    )


def test_eta_tracker_surfaces_saas_contention_in_primary_eta(check, sch):
    et = _load_eta_tracker(sch)
    cumulative = [
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.1,
        257.6, 489.9, 748.0, 1003.2, 1253.1,
        1505.3, 1762.1, 2020.9, 2267.5, 2557.3,
        2847.4, 3193.7, 3491.8, 3801.7, 4238.1,
        16870.5, 18538.2, 20993.2, 24059.8, 26815.4,
    ]
    lines = [
        (
            f"Iter {current}/80 [botorch-canonical] "
            f"label=test:botorch_saasbo Time: {elapsed:.1f}s "
            "ETA 394931.9s eta_model=growing_iter_cost"
        )
        for current, elapsed in enumerate(cumulative, start=1)
    ]
    cmd = (
        "python performance/benchmark_sota_fairness.py "
        "--method botorch_saasbo --target-budget 80 --n0 10 "
        "--saas-refit-schedule every_iteration"
    )

    projection = et.runtime_projection(
        "\n".join(lines), elapsed_s=26815.4, cmd=cmd)

    check(
        "SAAS ETA keeps the live contention-aware estimate and audits robust alternative",
        projection
        and projection.get("source") == "saas_native_eta"
        and projection.get("eta_s") == 394931
        and 0 < projection.get("robust_eta_s", 0) < 24 * 3600,
        diag=f"projection={projection}",
    )


def test_eta_tracker_ignores_zero_saas_initial_design_eta(check, sch):
    et = _load_eta_tracker(sch)
    tail = (
        "Iter 10/40 [botorch-canonical] label=test:botorch_saasbo "
        "Time: 0.2s ETA 0.5s eta_model=current_run_average"
    )
    cmd = (
        "python performance/benchmark_sota_fairness.py "
        "--method botorch_saasbo --target-budget 40 --n0 10 "
        "--saas-refit-schedule every_iteration"
    )

    projection = et.runtime_projection(tail, elapsed_s=600, cmd=cmd)
    eta = et.compute_eta_seconds(
        tail, elapsed_s=600, fallback_ewma_s=7200, cmd=cmd)

    check(
        "SAAS initial-design clock does not replace runtime prior with zero ETA",
        projection is None and eta == 6600,
        diag=f"projection={projection}, eta={eta}",
    )


def test_eta_tracker_keeps_generic_inline_eta_priority(check, sch):
    et = _load_eta_tracker(sch)
    tail = "\n".join([
        "Iter 10/100 Time: 100.0s ETA 900.0s",
        "Iter 20/100 Time: 220.0s ETA 880.0s",
        "Iter 30/100 Time: 360.0s ETA 840.0s",
        "Iter 40/100 Time: 520.0s ETA 780.0s",
    ])
    projection = et.runtime_projection(
        tail, elapsed_s=520, cmd="python generic.py --max_iters 100")

    check("growing-cost override is scoped to explicit botorch_saasbo tasks",
          projection
          and projection.get("source") == "inline_eta"
          and projection.get("eta_s") == 780,
          diag=f"projection={projection}")


def test_eta_tracker_parses_scolhkg_structured_progress(check, sch):
    et = _load_eta_tracker(sch)
    tail = (
        'SCOLHKG_PROGRESS {"kind":"target_call_done","done":8,'
        '"total":20,"method":"hyperbo_cbo","phase_elapsed_s":160.0}'
    )
    projection = et.runtime_projection(tail, elapsed_s=220, cmd="python transfer.py")

    check("SC-OLH-KG structured target progress produces remaining ETA",
          projection
          and projection.get("source") == "scolhkg_progress"
          and projection.get("eta_s") == 240
          and projection.get("current") == 8
          and projection.get("total_units") == 20,
          diag=f"projection={projection}")


def test_eta_tracker_does_not_misread_legacy_fsbo_target_progress(check, sch):
    et = _load_eta_tracker(sch)
    tail = (
        'SCOLHKG_PROGRESS {"kind":"target_call_done","done":1,'
        '"total":20,"method":"fsbo_cbo"}'
    )
    projection = et.runtime_projection(tail, elapsed_s=7200, cmd="python transfer.py")
    eta = et.compute_eta_seconds(
        tail, elapsed_s=7200, fallback_ewma_s=9000, cmd="python transfer.py")

    check("legacy FSBO target count does not multiply hidden source-fit time",
          projection is None and eta == 1800,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_parses_bapr_run_seed_seconds_per_iter(check, sch):
    et = _load_eta_tracker(sch)
    cmd = "./run_seed.sh bapr Walker2d-v2 2 800 60 independent tag"
    line = (
        "Iter  259 | Reward: 355.7 | Time: 6151s | TaskID: 2 | "
        "Q-std: 91.38 | [ACTIVE] | 12.0s/iter"
    )
    eta = et.compute_eta_seconds(line, elapsed_s=6151, fallback_ewma_s=10800, cmd=cmd)
    projection = et.runtime_projection(line, elapsed_s=6151, cmd=cmd)

    check("ETA tracker extracts positional max_iters from run_seed.sh",
          et._extract_total_from_cmd(cmd) == 800,
          diag=f"total={et._extract_total_from_cmd(cmd)}")
    check("ETA tracker uses current task-native seconds/iter for BAPR/RE-SAC",
          eta == int((800 - 259) * 12.0)
          and projection
          and projection.get("source") == "seconds_per_unit"
          and projection.get("current") == 259
          and projection.get("total_units") == 800,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_uses_iter_time_window_over_last_bapr_line(check, sch):
    et = _load_eta_tracker(sch)
    cmd = "./run_seed.sh bapr HalfCheetah-v2 1 800 60 independent tag"
    tail = "\n".join([
        "Iter  491 | Reward: 11.0 | Time: 30624s | [ACTIVE] | 19.9s/iter",
        "Iter  495 | Reward: 11.0 | Time: 30793s | [ACTIVE] | 109.5s/iter",
        "Iter  500 | Reward: 11.0 | Time: 30981s | [ACTIVE] | 108.9s/iter",
        "Iter  569 | Reward: 11.0 | Time: 33509s | [ACTIVE] | 20.1s/iter",
    ])
    eta = et.compute_eta_seconds(tail, elapsed_s=33509, fallback_ewma_s=27012, cmd=cmd)
    projection = et.runtime_projection(tail, elapsed_s=33509, cmd=cmd)

    check("ETA tracker uses recent Iter/Time window for BAPR periodic eval cost",
          8500 <= eta <= 8600
          and projection
          and projection.get("source") == "iter_time_window"
          and projection.get("current") == 569
          and projection.get("total_units") == 800
          and 36.0 <= projection.get("unit_s", 0) <= 38.0,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_counts_future_bapr_fork_stages_from_shared_base(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.run_bapr_v3_structured_channel_headroom "
        "--family structured_channel --env HalfCheetah-v2 --seed 0 --resume"
    )
    tail = "\n".join([
        "TRAIN SUBPROCESS: python -u -m jax_experiments.train "
        "--run_name shared_base --max_iters 700",
        "Iter  450 | Reward: -18.6 | Time: 3857s | [ACTIVE] | 23.7s/iter",
        "Iter  451 | Reward: -18.6 | Time: 3870s | [ACTIVE] | 7.7s/iter",
        "Iter  452 | Reward: -18.6 | Time: 3877s | [ACTIVE] | 7.8s/iter",
        "Iter  453 | Reward: -18.6 | Time: 3885s | [ACTIVE] | 7.7s/iter",
        "Iter  457 | Reward: -18.6 | Time: 3916s | [ACTIVE] | 7.8s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=4200, cmd=cmd)
    eta = et.compute_eta_seconds(tail, elapsed_s=4200, cmd=cmd)

    check("BAPR fork ETA includes both future branches from shared base",
          projection
          and projection.get("source") == "bapr_fork_protocol"
          and projection.get("protocol_stage") == "shared_base"
          and projection.get("future_stages") == 2
          and projection.get("current") == 457
          and projection.get("total_units") == 2100
          and 12000 <= eta <= 14000,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_counts_future_oracle_branch_during_bapr_robust_stage(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.run_bapr_v3_structured_channel_headroom "
        "--family structured_channel --env Ant-v2 --seed 0 --resume"
    )
    tail = "\n".join([
        "TRAIN SUBPROCESS: python -u -m jax_experiments.train "
        "--run_name shared_base --max_iters 700",
        "Training complete! Total time: 3501s (1.0h)",
        "TRAIN SUBPROCESS: python -u -m jax_experiments.train "
        "--run_name robust_long --max_iters 1400 --min_resume_iteration 700",
        "Iter  733 | Reward: 914.1 | Time: 194s | [ACTIVE] | 4.5s/iter",
        "Iter  734 | Reward: 914.1 | Time: 199s | [ACTIVE] | 4.5s/iter",
        "Iter  735 | Reward: 914.1 | Time: 203s | [ACTIVE] | 4.5s/iter",
        "Iter  750 | Reward: 914.1 | Time: 278s | [ACTIVE] | 12.6s/iter",
        "Iter  776 | Reward: 914.1 | Time: 402s | [ACTIVE] | 4.5s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=4100, cmd=cmd)
    eta = et.compute_eta_seconds(tail, elapsed_s=4100, cmd=cmd)

    check("BAPR robust ETA includes the not-yet-started oracle branch",
          projection
          and projection.get("source") == "bapr_fork_protocol"
          and projection.get("protocol_stage") == "robust_long"
          and projection.get("future_stages") == 1
          and projection.get("current") == 776
          and projection.get("total_units") == 2100
          and 5800 <= eta <= 6500,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_projects_hidden_bapr_independent_specialist_budget(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.launch_bapr_v3_stochastic_independent_specialist "
        "--profile structured_channel --family structured_channel "
        "--env HalfCheetah-v2 --mode 0 --resume"
    )
    tail = "\n".join([
        "Iter 1000 | Reward: -22.7 | Time: 5428s | Eval: -20.3 | 51.0s/iter",
        "Iter 1001 | Reward: -22.7 | Time: 5449s | 16.5s/iter",
        "Iter 1050 | Reward: -23.1 | Time: 6318s | Eval: -20.6 | 50.9s/iter",
        "Iter 1068 | Reward: -23.1 | Time: 6629s | 17.0s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=6690, cmd=cmd)
    eta = et.compute_eta_seconds(
        tail, elapsed_s=6690, fallback_ewma_s=0, cmd=cmd)

    check("BAPR specialist ETA recovers the outer protocol's default target",
          projection
          and projection.get("source") == "bapr_independent_specialist"
          and projection.get("current") == 1069
          and projection.get("total_units") == 1400
          and projection.get("remaining_train_units") == 331
          and projection.get("rate_source") == "recent_time_window"
          and 17.5 <= projection.get("unit_s", 0) <= 18.0
          and 5900 <= eta <= 6100,
          diag=f"eta={eta}, projection={projection}")


def test_eta_tracker_projects_hidden_bapr_persistent_option_budget(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.run_bapr_v4_persistent_option "
        "--profile formal --resume"
    )
    tail = "\n".join([
        "Iter  168 | Reward: 175.5 | Time: 1781s | [ACTIVE] | 8.4s/iter",
        "Iter  169 | Reward: 175.5 | Time: 1789s | [ACTIVE] | 8.4s/iter",
        "Iter  170 | Reward: 175.5 | Time: 1798s | [ACTIVE] | 8.4s/iter",
        "Iter  171 | Reward: 175.5 | Time: 1806s | [ACTIVE] | 8.4s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=1830, cmd=cmd)
    audit_cmd = cmd.replace(
        "run_bapr_v4_persistent_option ",
        "run_bapr_v4_persistent_option_audit ",
    )

    check("BAPR persistent-option formal profile recovers its hidden target",
          et._extract_total_from_cmd(cmd) == 1400
          and projection
          and projection.get("source") == "iter_time_window"
          and projection.get("current") == 171
          and projection.get("total_units") == 1400
          and 10000 <= projection.get("eta_s", 0) <= 11000,
          diag=f"projection={projection}")
    check("BAPR persistent-option audit is not treated as formal training",
          et._extract_total_from_cmd(audit_cmd) is None,
          diag=f"total={et._extract_total_from_cmd(audit_cmd)}")


def test_eta_tracker_parses_explicit_scheduler_total_marker(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "SCHEDULEURM_ETA_TOTAL_UNITS=7200 python -u -m "
        "future_training_protocol --profile formal --max_iters 10"
    )

    check("ETA tracker accepts a protocol-independent submitted total",
          et._extract_total_from_cmd(cmd) == 7200,
          diag=f"total={et._extract_total_from_cmd(cmd)}")


def test_eta_tracker_projects_running_bapr_v7_without_new_marker(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.run_bapr_v7_data_equivalent_option "
        "--profile formal --resume"
    )
    tail = "\n".join([
        "Iter  735 | Reward: 1.0 | Time: 8130s | [ACTIVE] | 9.9s/iter",
        "Iter  736 | Reward: 1.0 | Time: 8140s | [ACTIVE] | 10.0s/iter",
        "Iter  737 | Reward: 1.0 | Time: 8150s | [ACTIVE] | 10.0s/iter",
        "Iter  738 | Reward: 1.0 | Time: 8160s | [ACTIVE] | 10.0s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=8200, cmd=cmd)

    check("running BAPR-v7 formal task recovers its frozen 5600 target",
          et._extract_total_from_cmd(cmd) == 5600
          and projection
          and projection.get("source") == "iter_time_window"
          and projection.get("current") == 738
          and projection.get("total_units") == 5600
          and projection.get("eta_s") == 48620,
          diag=f"projection={projection}")


def test_eta_tracker_projects_running_bapr_v8_wrapper_budget(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "python -u -m "
        "jax_experiments.analysis.run_bapr_v8_seed_controller "
        "--seed 3 --role specialist --mode 3 --resume"
    )
    tail = "\n".join([
        "Iter  200 | Reward: 528.1 | Time: 3584s | Eval: 496.8 | 65.9s/iter",
        "Iter  220 | Reward: 539.0 | Time: 3965s | Eval: 515.7 | 66.0s/iter",
        "Iter  240 | Reward: 591.7 | Time: 4342s | Eval: 790.6 | 65.8s/iter",
        "Iter  260 | Reward: 654.5 | Time: 4719s | Eval: 766.2 | 66.0s/iter",
        "Iter  267 | Reward: 654.5 | Time: 4833s | 16.3s/iter",
    ])
    projection = et.runtime_projection(tail, elapsed_s=4890, cmd=cmd)

    check("running BAPR-v8 wrapper recovers its frozen 1400 target",
          et._extract_total_from_cmd(cmd) == 1400
          and projection
          and projection.get("source") == "iter_time_window"
          and projection.get("current") == 267
          and projection.get("total_units") == 1400
          and 21000 <= projection.get("eta_s", 0) <= 21500,
          diag=f"projection={projection}")


def test_eta_tracker_parses_freqduet_shard_progress(check, sch):
    et = _load_eta_tracker(sch)
    cmd = (
        "PYTHONPATH=. python -u scripts/run_freqduet_external_baselines.py "
        "--episodes 100 --job-start 0 --job-end 10 --workers 4"
    )
    tail = "\n".join([
        "Shard jobs [0,10) of 100",
        "DONE cfg_a fixed_headway_600 seed=1: /tmp/cfg_a_fixed_headway_600_seed1",
        "DONE cfg_b_route_value_rf_fd_trainall_seed2: /tmp/cfg_b_route_value_rf_fd_trainall_seed2",
        "fixed_headway_600 cfg_c seed=3 ep=049 N=15 wait=5.52 cv=0.414 over=3.0 comp=1.566",
        "route_value_rf_fd_trainall cfg_d seed=4 ep=099 N=12 wait=4.20 cv=0.210 over=0.0 comp=0.900",
    ])
    eta = et.compute_eta_seconds(tail, elapsed_s=350, fallback_ewma_s=99999, cmd=cmd)
    projection = et.runtime_projection(tail, elapsed_s=350, cmd=cmd)

    check("ETA tracker parses FreqDuet shard DONE count plus active episode progress",
          eta == 650
          and projection
          and projection.get("source") == "freqduet_shard"
          and projection.get("completed_jobs") == 2
          and projection.get("active_jobs") == 2
          and projection.get("current") == 3
          and projection.get("total_units") == 10,
          diag=f"eta={eta}, projection={projection}")


def test_running_history_overrun_keeps_positive_eta_load(check, sch):
    saved_history_get = sch.history_get
    saved_load_runtime_history = sch.load_runtime_history
    sig = "eta/overrun/%d" % time.time_ns()
    task = {
        "id": "t-overrun",
        "status": "running",
        "node": "local",
        "signature": sig,
        "started_at": time.time() - 7200,
        "cmd": "python train.py --max_iters 100",
    }
    try:
        sch.history_get = lambda s: {"dur_s_ewma": 3600} if s == sig else None
        sch.load_runtime_history = lambda: {}
        sch._refresh_eta_from_logs({"tasks": [task]})
    finally:
        sch.history_get = saved_history_get
        sch.load_runtime_history = saved_load_runtime_history

    check("ETA overrun fallback keeps running task visible to eta_load",
          int(task.get("eta_seconds") or 0) > 0
          and task.get("eta_source") == "duration_ewma_overrun"
          and "overrun" in (task.get("eta_detail") or ""),
          diag=str(task))


def test_runtime_history_closest_index_reuses_tokenized_records(check, sch):
    calls = {"n": 0}
    original_tokens = sch._runtime_cmd_tokens
    history = {
        f"exact:{i}": {
            "total_s": 1000 + i,
            "cmd": f"python train.py --n_steps {1000 + i} --lr 0.{i % 10}",
            "project": "ETA",
            "cwd": "/tmp/eta",
        }
        for i in range(200)
    }
    task = {
        "cmd": "python train.py --n_steps 2000 --lr 0.1",
        "project": "ETA",
        "cwd": "/tmp/eta",
        "signature": "eta/index",
    }

    def counted(cmd):
        calls["n"] += 1
        return original_tokens(cmd)

    sch._runtime_cmd_tokens = counted
    try:
        index = sch._runtime_history_closest_index(history)
        for _ in range(10):
            got = sch._runtime_total_history_s(
                task,
                runtime_history=history,
                closest_index=index,
            )
            check("runtime closest index still returns a runtime estimate",
                  got > 0,
                  diag=f"got={got}")
    finally:
        sch._runtime_cmd_tokens = original_tokens

    check("runtime closest index tokenizes history once, not once per task lookup",
          calls["n"] <= len(history) + 15,
          diag=f"token_calls={calls['n']}, history={len(history)}")
