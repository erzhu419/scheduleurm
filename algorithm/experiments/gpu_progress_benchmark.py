"""Short GPU progress benchmark for Scheduleurm placement validation.

The script is intentionally small and framework-adaptive.  It prints
`Step i/N` lines so Scheduleurm's progress parser can treat each step as a
service unit.  It prefers JAX when available, then PyTorch CUDA, and falls back
to NumPy for dry runs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time


def _progress(iterable, *, total: int, desc: str, unit: str):
    try:
        from tqdm import tqdm
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            mininterval=0.5,
            dynamic_ncols=False,
            ascii=True,
            file=sys.stdout,
        )
    except Exception:
        return iterable


def _phase(name: str, event: str) -> None:
    print(f"ScheduleurmPhase name={name} event={event}", flush=True)


def _wait_for_coordinated_start() -> None:
    ready = os.environ.get("SCHEDULEURM_READY_FILE", "").strip()
    start = os.environ.get("SCHEDULEURM_START_FILE", "").strip()
    if not ready or not start:
        return
    _phase("coordination_barrier", "start")
    Path(ready).parent.mkdir(parents=True, exist_ok=True)
    Path(ready).touch()
    deadline = time.monotonic() + float(
        os.environ.get("SCHEDULEURM_BARRIER_TIMEOUT_S", "600")
    )
    while not Path(start).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("coordinated benchmark start barrier timed out")
        time.sleep(0.05)
    _phase("coordination_barrier", "end")


def _save_checkpoint(
    *,
    checkpoint_dir: str,
    label: str,
    step: int,
    phase_name: str,
    payload_bytes: int,
    backend: str,
    matrix_size: int,
) -> None:
    root = Path(checkpoint_dir or "/tmp/scheduleurm_gpu_checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}_{phase_name}_{int(step)}.bin"
    _phase(phase_name, "start")
    header = (
        f"scheduleurm-gpu-checkpoint-v1\nbackend={backend}\n"
        f"matrix_size={int(matrix_size)}\nstep={int(step)}\n"
    ).encode("ascii")
    remaining = max(0, int(payload_bytes) - len(header))
    chunk = bytes(min(1024 * 1024, max(1, remaining)))
    with path.open("wb") as handle:
        handle.write(header)
        while remaining > 0:
            part = chunk[: min(len(chunk), remaining)]
            handle.write(part)
            remaining -= len(part)
        handle.flush()
        os.fsync(handle.fileno())
    _phase(phase_name, "end")
    print(
        f"ScheduleurmCheckpoint path={path} bytes={path.stat().st_size} "
        f"step={int(step)} phase={phase_name}",
        flush=True,
    )


def _jax_runner(size: int):
    import jax
    import jax.numpy as jnp

    x = jnp.ones((size, size), dtype=jnp.float32)

    @jax.jit
    def step(a):
        y = jnp.tanh(a @ a.T)
        return y.sum()

    step(x).block_until_ready()

    def run_once():
        step(x).block_until_ready()

    return "jax", str(jax.devices()), run_once


def _torch_runner(size: int):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda is not available")
    device = torch.device("cuda")
    x = torch.ones((size, size), dtype=torch.float32, device=device)

    def run_once():
        y = torch.tanh(x @ x.T)
        y.sum().item()
        torch.cuda.synchronize()

    run_once()
    return "torch", torch.cuda.get_device_name(device), run_once


def _numpy_runner(size: int):
    import numpy as np

    x = np.ones((min(size, 1024), min(size, 1024)), dtype=np.float32)

    def run_once():
        y = np.tanh(x @ x.T)
        float(y.sum())

    run_once()
    return "numpy", "cpu-fallback", run_once


def _load_runner(size: int):
    def fallback_reason(exc: Exception) -> str:
        text = f"{type(exc).__name__}: {str(exc)[:160]}"
        return (
            text.replace("Error", "Issue")
            .replace("error", "issue")
            .replace("Exception", "Issue")
        )

    if os.environ.get("SCHEDULEURM_BENCH_FORCE_NUMPY") != "1":
        try:
            return _jax_runner(size)
        except Exception as e:
            jax_error = fallback_reason(e)
        try:
            return _torch_runner(size)
        except Exception as e:
            torch_error = fallback_reason(e)
        print(f"GPU_BACKEND_FALLBACK jax={jax_error} torch={torch_error}", flush=True)
    return _numpy_runner(size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--label", default="scheduleurm-gpu-bench")
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--checkpoint-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--final-save", action="store_true")
    args = parser.parse_args()

    _phase("initialization", "start")
    steps = max(1, int(args.steps))
    size = max(64, int(args.size))
    backend, device, run_once = _load_runner(size)
    if args.require_gpu and backend not in {"jax", "torch"}:
        raise SystemExit("CUDA backend is required for theorem-facing GPU measurements")
    _phase("initialization", "end")
    print(
        f"BENCH_START label={args.label} backend={backend} device={device} "
        f"pid={os.getpid()} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    _wait_for_coordinated_start()
    durations = []
    start = time.time()
    _phase("outer_loop", "start")
    for i in _progress(range(1, steps + 1), total=steps, desc=str(args.label), unit="step"):
        t0 = time.time()
        run_once()
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
        dt = time.time() - t0
        durations.append(dt)
        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        remaining = max(0, steps - i)
        eta = remaining / max(rate, 1e-9)
        print(
            f"Step {i}/{steps} dt={dt:.6f}s elapsed={elapsed:.3f}s "
            f"rate={rate:.6f} step/s ETA {eta:.1f}s",
            flush=True,
        )
        if int(args.checkpoint_interval) > 0 and i % int(args.checkpoint_interval) == 0:
            _save_checkpoint(
                checkpoint_dir=args.checkpoint_dir,
                label=args.label,
                step=i,
                phase_name="checkpoint",
                payload_bytes=args.checkpoint_bytes,
                backend=backend,
                matrix_size=size,
            )
    _phase("outer_loop", "end")
    if args.final_save:
        _save_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            label=args.label,
            step=steps,
            phase_name="final_save",
            payload_bytes=args.checkpoint_bytes,
            backend=backend,
            matrix_size=size,
        )
    total = time.time() - start
    median = statistics.median(durations) if durations else 0.0
    mean = statistics.fmean(durations) if durations else 0.0
    print(
        f"BENCH_DONE label={args.label} backend={backend} steps={steps} "
        f"elapsed={total:.6f}s rate={steps / max(total, 1e-9):.6f} "
        f"mean_step_s={mean:.6f} median_step_s={median:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
