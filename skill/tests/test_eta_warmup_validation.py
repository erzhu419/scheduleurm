import importlib.util
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

