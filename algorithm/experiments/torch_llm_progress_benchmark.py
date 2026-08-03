"""Torch/Hugging Face causal-LM progress benchmark for GPU service curves.

Models are loaded on the target node from the Hugging Face cache.  Inputs are
synthetic token ids so the benchmark isolates model execution from tokenizer
and dataset I/O.
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
    parser.add_argument("--model-id", default="distilgpt2")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--mode", default="forward", choices=["forward", "train"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-transformers", action="store_true")
    parser.add_argument("--label", default="torch_llm")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--final-save", action="store_true")
    args = parser.parse_args()

    _phase("initialization", "start")
    import torch
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except Exception:
        AutoConfig = None
        AutoModelForCausalLM = None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for torch_llm_progress_benchmark")
    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = None
    vocab_size = 50257
    backend = "torch_transformer_fallback"
    if AutoConfig is not None and AutoModelForCausalLM is not None:
        try:
            config = AutoConfig.from_pretrained(
                args.model_id,
                cache_dir=args.cache_dir or None,
                local_files_only=bool(args.local_files_only),
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                cache_dir=args.cache_dir or None,
                torch_dtype=dtype,
                local_files_only=bool(args.local_files_only),
            ).to(device)
            backend = "torch_transformers"
            vocab_size = int(getattr(config, "vocab_size", 50257) or 50257)
        except Exception as exc:
            print(
                "BENCH_FALLBACK "
                f"backend=torch_transformer_fallback reason={type(exc).__name__}: {str(exc)[:180]}",
                flush=True,
            )
            model = None
    if model is None and args.require_transformers:
        raise SystemExit(
            "Hugging Face model load is required for theorem-facing LLM measurements"
        )
    if model is None:
        vocab_size = 50257
        model = _FallbackCausalLM(
            torch,
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            hidden_size=384,
            num_layers=6,
            num_heads=6,
        ).to(device=device, dtype=dtype)
        backend = "torch_transformer_fallback"
    model.train(args.mode == "train")
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) if args.mode == "train" else None
    _phase("initialization", "end")

    print(
        "BENCH_START "
        f"backend={backend} device={torch.cuda.get_device_name(0)!r} "
        f"model_id={args.model_id} mode={args.mode} batch={args.batch_size} "
        f"seq_len={args.seq_len} dtype={args.dtype} label={args.label}",
        flush=True,
    )
    _phase("warmup", "start")
    for _ in range(max(0, args.warmup)):
        _step(model, input_ids, labels, optimizer, args.mode)
    torch.cuda.synchronize()
    _phase("warmup", "end")
    _wait_for_coordinated_start()

    start = time.perf_counter()
    window_start = start
    window_steps = 0
    total_steps = max(1, args.steps)
    _phase("outer_loop", "start")
    for step in _progress(range(1, total_steps + 1), total=total_steps, desc=args.label, unit="step"):
        _step(model, input_ids, labels, optimizer, args.mode)
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
    checkpoint_dir: str,
    label: str,
    step: int,
    phase_name: str,
) -> None:
    root = Path(checkpoint_dir or "/tmp/scheduleurm_llm_checkpoints")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}_{phase_name}_{int(step)}.pt"
    _phase(phase_name, "start", index=step)
    torch.cuda.synchronize()
    state_owner = getattr(model, "module", model)
    payload = {
        "step": int(step),
        "model": state_owner.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    with path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _phase(phase_name, "end", index=step)


def _step(model, input_ids, labels, optimizer, mode: str) -> None:
    import torch

    if mode == "forward":
        with torch.inference_mode():
            _ = model(input_ids=input_ids, use_cache=False)
        return
    assert optimizer is not None
    optimizer.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    out.loss.backward()
    optimizer.step()


class _FallbackOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


class _FallbackCausalLM:
    """Small decoder-only Transformer when Hugging Face is unavailable."""

    def __new__(
        cls,
        torch,
        *,
        vocab_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
    ):
        return _TorchCausalLM(
            torch,
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )


class _TorchCausalLM:
    def __init__(
        self,
        torch,
        *,
        vocab_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
    ):
        self._torch = torch
        self.module = torch.nn.Module()
        self.module.token = torch.nn.Embedding(vocab_size, hidden_size)
        self.module.pos = torch.nn.Embedding(seq_len, hidden_size)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.module.blocks = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.module.norm = torch.nn.LayerNorm(hidden_size)
        self.module.head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def parameters(self):
        return self.module.parameters()

    def __call__(self, *, input_ids, labels=None, use_cache=False):
        torch = self._torch
        batch, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        h = self.module.token(input_ids) + self.module.pos(pos)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(length, device=input_ids.device)
        h = self.module.blocks(h, mask=mask)
        h = self.module.norm(h)
        logits = self.module.head(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
            )
        return _FallbackOutput(loss=loss, logits=logits)


if __name__ == "__main__":
    raise SystemExit(main())
