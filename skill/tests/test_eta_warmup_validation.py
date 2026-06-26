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
