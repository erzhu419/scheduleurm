"""Short GPU progress benchmark for Scheduleurm placement validation.

The script is intentionally small and framework-adaptive.  It prints
`Step i/N` lines so Scheduleurm's progress parser can treat each step as a
service unit.  It prefers JAX when available, then PyTorch CUDA, and falls back
to NumPy for dry runs.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time


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
    if os.environ.get("SCHEDULEURM_BENCH_FORCE_NUMPY") != "1":
        try:
            return _jax_runner(size)
        except Exception as e:
            jax_error = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            return _torch_runner(size)
        except Exception as e:
            torch_error = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"GPU_BACKEND_FALLBACK jax={jax_error} torch={torch_error}", flush=True)
    return _numpy_runner(size)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--label", default="scheduleurm-gpu-bench")
    parser.add_argument("--sleep-s", type=float, default=0.0)
    args = parser.parse_args()

    steps = max(1, int(args.steps))
    size = max(64, int(args.size))
    backend, device, run_once = _load_runner(size)
    print(
        f"BENCH_START label={args.label} backend={backend} device={device} "
        f"pid={os.getpid()} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    durations = []
    start = time.time()
    for i in range(1, steps + 1):
        t0 = time.time()
        run_once()
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
        dt = time.time() - t0
        durations.append(dt)
        elapsed = time.time() - start
        rate = i / max(elapsed, 1e-9)
        print(
            f"Step {i}/{steps} dt={dt:.6f}s elapsed={elapsed:.3f}s "
            f"rate={rate:.6f} step/s",
            flush=True,
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
