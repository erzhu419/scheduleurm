"""Torch CNN synthetic progress benchmark for pure-GPU service curves.

The benchmark uses standard torchvision CNN architectures with synthetic
images, so it measures GPU compute/activation pressure without a data-loader
or storage bottleneck.  It prints Scheduleurm-compatible rate lines.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
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


def _phase(name: str, event: str, *, index: int | None = None) -> None:
    suffix = f" index={int(index)}" if index is not None else ""
    print(f"ScheduleurmPhase name={name} event={event}{suffix}", flush=True)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50", "convnext_tiny"])
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=int, default=1000)
    parser.add_argument("--mode", default="train", choices=["forward", "train"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--label", default="torch_cnn")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--final-save", action="store_true")
    args = parser.parse_args()

    _phase("initialization", "start")
    import torch
    try:
        import torchvision.models as models
    except Exception:
        models = None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for torch_cnn_progress_benchmark")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = _build_model(args.model, models, torch).to(device)
    model.train(args.mode == "train")
    image = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    target = torch.randint(0, args.classes, (args.batch_size,), device=device)
    optimizer = None
    manual_sgd = False
    if args.mode == "train":
        try:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        except Exception as exc:
            manual_sgd = True
            print(
                "BENCH_OPTIMIZER_FALLBACK "
                f"mode=manual_sgd reason={type(exc).__name__}: {str(exc)[:180]}",
                flush=True,
            )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and args.mode == "train" and not manual_sgd))
    _phase("initialization", "end")

    print(
        "BENCH_START "
        f"backend=torch device={torch.cuda.get_device_name(0)!r} "
        f"model={args.model} mode={args.mode} batch={args.batch_size} "
        f"image={args.image_size} amp={str(bool(args.amp)).lower()} label={args.label}",
        flush=True,
    )
    _phase("warmup", "start")
    for _ in range(max(0, args.warmup)):
        _step(model, image, target, criterion, optimizer, scaler, args.mode, args.amp, manual_sgd=manual_sgd)
    torch.cuda.synchronize()
    _phase("warmup", "end")
    _wait_for_coordinated_start()

    start = time.perf_counter()
    window_start = start
    window_steps = 0
    total_steps = max(1, args.steps)
    _phase("outer_loop", "start")
    for step in _progress(range(1, total_steps + 1), total=total_steps, desc=args.label, unit="step"):
        _step(model, image, target, criterion, optimizer, scaler, args.mode, args.amp, manual_sgd=manual_sgd)
        window_steps += 1
        if step == total_steps or step % max(1, args.log_interval) == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(1e-9, now - window_start)
            rate = window_steps / elapsed
            total_elapsed = max(1e-9, now - start)
            total_rate = step / total_elapsed
            remaining = max(0, total_steps - step)
            eta = remaining / max(1e-9, total_rate)
            print(
                f"BENCH_PROGRESS Step {step}/{args.steps} "
                f"rate={total_rate:.9g} step/s window_rate={rate:.9g} step/s ETA {eta:.1f}s",
                flush=True,
            )
            window_start = now
            window_steps = 0
        if int(args.checkpoint_interval) > 0 and step % int(args.checkpoint_interval) == 0:
            _save_checkpoint(
                torch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                checkpoint_dir=args.checkpoint_dir,
                label=args.label,
                step=step,
                phase_name="checkpoint",
            )
    _phase("outer_loop", "end")
    if args.final_save:
        _save_checkpoint(
            torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            checkpoint_dir=args.checkpoint_dir,
            label=args.label,
            step=total_steps,
            phase_name="final_save",
        )
    print("BENCH_DONE", flush=True)
    return 0


def _save_checkpoint(
    torch,
    *,
    model,
    optimizer,
    scaler,
    checkpoint_dir: str,
    label: str,
    step: int,
    phase_name: str,
) -> None:
    root = Path(checkpoint_dir or "/tmp/scheduleurm_cnn_checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}_{phase_name}_{int(step)}.pt"
    _phase(phase_name, "start", index=step)
    torch.cuda.synchronize()
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    with path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _phase(phase_name, "end", index=step)


def _build_model(name: str, models, torch):
    if models is None:
        return _FallbackCNN(torch)
    if name == "resnet18":
        return models.resnet18(weights=None)
    if name == "resnet50":
        return models.resnet50(weights=None)
    if name == "convnext_tiny":
        return models.convnext_tiny(weights=None)
    raise ValueError(name)


class _FallbackCNN:
    """Torch-only CNN stack used when torchvision is not installed on a node."""

    def __new__(cls, torch):
        layers = []
        in_channels = 3
        channels = (32, 64, 96, 128, 128)
        for out_channels in channels:
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        layers.extend([
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_channels, 1000),
        ])
        return torch.nn.Sequential(*layers)


def _step(model, image, target, criterion, optimizer, scaler, mode: str, amp: bool, *, manual_sgd: bool = False) -> None:
    import torch

    if mode == "forward":
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=bool(amp)):
            _ = model(image)
        return
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        for param in model.parameters():
            param.grad = None
    with torch.amp.autocast("cuda", enabled=bool(amp)):
        output = model(image)
        loss = criterion(output, target)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if optimizer is not None:
            optimizer.step()
        else:
            _manual_sgd_step(model, torch, lr=1e-4)


def _manual_sgd_step(model, torch, *, lr: float) -> None:
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.add_(param.grad, alpha=-float(lr))
                param.grad = None


if __name__ == "__main__":
    raise SystemExit(main())
