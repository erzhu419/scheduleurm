"""Finite feature maps for scheduleurm placement experiments.

The proof route treats actions through finite fabric/interference features
rather than an unstructured global action id.  This module is the code-side
counterpart: it turns scheduler tasks and GPU snapshots into stable class,
regime, and candidate-bucket keys plus bounded numeric features for scoring.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any, Dict, Iterable, Mapping

_LABEL_RE = re.compile(r"[^a-z0-9_.:+-]+")


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compact_label(value: Any, default: str = "unknown", max_len: int = 48) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return default
    text = _LABEL_RE.sub("_", text).strip("_")
    if not text:
        return default
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:8]
    return f"{text[:max_len - 9]}_{digest}"


def stable_digest(value: Any, length: int = 10) -> str:
    text = str(value or "")
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:length]


def _split_signature(signature: Any) -> list[str]:
    raw = str(signature or "").strip()
    if not raw:
        return []
    return [compact_label(x) for x in re.split(r"[/|,:]+", raw) if str(x).strip()]


def signature_prefix(task: Mapping[str, Any], depth: int = 3) -> str:
    parts = _split_signature(task.get("signature"))
    if not parts:
        cmd = task.get("cmd") or task.get("description") or ""
        return "cmd:" + stable_digest(cmd)
    return "/".join(parts[:max(1, int(depth or 1))])


def task_kind(task: Mapping[str, Any]) -> str:
    est_vram = as_int(task.get("est_vram_mb"), 0)
    cmd = str(task.get("cmd") or task.get("description") or "").lower()
    slurm = any(task.get(k) for k in ("slurm_partition", "slurm_account", "slurm_qos"))
    if est_vram <= 0:
        base = "cpu"
    elif slurm:
        base = "slurm_gpu"
    else:
        base = "gpu"
    if any(tok in cmd for tok in ("eval", "evaluate", "test.py", "rollout")):
        suffix = "eval"
    elif any(tok in cmd for tok in ("train", "fit", "learn", "pretrain")):
        suffix = "train"
    elif any(tok in cmd for tok in ("render", "video", "plot")):
        suffix = "render"
    else:
        suffix = "job"
    return f"{base}_{suffix}"


def bucket_leq(value: float, cuts: Iterable[float], prefix: str) -> str:
    cut_tuple = tuple(float(c) for c in cuts)
    val = float(value)
    for idx, cut in enumerate(cut_tuple):
        if val <= cut:
            return f"{prefix}{idx}"
    return f"{prefix}{len(cut_tuple)}p"


def _bucket_from_cuts(value: float, cuts: tuple[float, ...], prefix: str) -> str:
    val = float(value)
    for idx, cut in enumerate(cuts):
        if val <= cut:
            return f"{prefix}{idx}"
    return f"{prefix}{len(cuts)}p"


def resource_bucket(task: Mapping[str, Any]) -> str:
    vram = max(0, as_int(task.get("est_vram_mb"), 0))
    ram = max(0, as_int(task.get("ram_mb"), 0))
    cpu = max(0, as_float(task.get("cpu_cores"), 0.0))
    v = _bucket_from_cuts(vram, (0, 1024, 4096, 8192, 16384, 32768, 65536), "v")
    r = _bucket_from_cuts(ram, (1024, 4096, 8192, 16384, 32768, 65536, 131072), "r")
    c = _bucket_from_cuts(cpu, (1, 2, 4, 8, 16, 32, 64), "c")
    return f"{v}.{r}.{c}"


def class_key(task: Mapping[str, Any], prefix_depth: int = 3) -> str:
    project = compact_label(task.get("project") or task.get("cwd") or "unknown")
    sig = signature_prefix(task, prefix_depth)
    return (
        "class:v1|"
        f"project={project}|sig={sig}|kind={task_kind(task)}|res={resource_bucket(task)}"
    )


def _count_bucket(count: int) -> str:
    return _bucket_from_cuts(count, (0, 1, 2, 3, 4, 6, 8), "n")


def _fraction_bucket(value: float, prefix: str) -> str:
    return _bucket_from_cuts(value, (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0), prefix)


def _util_bucket(value: float) -> str:
    return _bucket_from_cuts(value, (0, 10, 30, 50, 70, 85, 95), "u")


def regime_key(node_state: Mapping[str, Any], gpu: Mapping[str, Any]) -> str:
    node = compact_label(node_state.get("name") or node_state.get("node") or "unknown")
    total = max(1, as_int(gpu.get("total_mb"), 1))
    used = max(0, as_int(gpu.get("used_mb"), 0))
    util = max(0.0, as_float(gpu.get("util_pct"), 0.0))
    count = max(0, as_int(gpu.get("running_task_count"), 0))
    mem = _fraction_bucket(float(used) / float(total), "m")
    return f"regime:v1|node={node}|gpu={gpu.get('idx', 'na')}|{_count_bucket(count)}|{mem}|{_util_bucket(util)}"


def priority_weight(task: Mapping[str, Any]) -> float:
    p = compact_label(task.get("priority") or "normal")
    if p in ("urgent", "critical", "p0"):
        return 4.0
    if p in ("high", "p1"):
        return 2.0
    if p in ("low", "p3", "background"):
        return 0.5
    return 1.0


def _queue_age_seconds(task: Mapping[str, Any], context: Mapping[str, Any]) -> float:
    now = as_float(context.get("now_ts"), time.time())
    submitted = as_float(task.get("submitted_at"), now)
    if submitted <= 0:
        return 0.0
    return max(0.0, now - submitted)


def _legacy_runtime_features(legacy_score: Any) -> tuple[int, float]:
    if isinstance(legacy_score, list):
        legacy_score = tuple(legacy_score)
    if not isinstance(legacy_score, tuple) or len(legacy_score) < 3:
        return 1, 0.0
    unknown = as_int(legacy_score[1], 1)
    runtime_s = max(0.0, as_float(legacy_score[2], 0.0))
    return unknown, runtime_s


def gpu_candidate_features(
    task: Mapping[str, Any],
    node_state: Mapping[str, Any],
    gpu: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    context = context or {}
    total = max(1, as_int(gpu.get("total_mb"), 1))
    used = max(0, as_int(gpu.get("used_mb"), 0))
    free = max(0, as_int(gpu.get("free_mb"), max(0, total - used)))
    need = max(0, as_int(task.get("est_vram_mb"), 0))
    util = max(0.0, as_float(gpu.get("util_pct"), 0.0))
    running = max(0, as_int(gpu.get("running_task_count"), 0))
    post_used = used + need
    post_count = running + 1
    empty_used_mb = max(0, as_int(context.get("gpu_empty_used_mb"), 100))
    sweet = max(0, as_int(context.get("sweet_spot_tasks_per_gpu"), 0))
    over_sweet = max(0, post_count - sweet) if sweet > 0 else 0
    sweet_gap = abs(post_count - sweet) if sweet > 0 else 0
    legacy_unknown, legacy_runtime_s = _legacy_runtime_features(context.get("legacy_score"))
    post_vram_frac = float(post_used) / float(total)
    used_vram_frac = float(used) / float(total)
    task_cls = class_key(task, as_int(context.get("signature_prefix_depth"), 3))
    regime = regime_key(node_state, gpu)
    finite = {
        "class_key": task_cls,
        "regime_key": regime,
        "node": compact_label(node_state.get("name") or "unknown"),
        "gpu_idx": str(gpu.get("idx", "na")),
        "post_count_bucket": _count_bucket(post_count),
        "post_vram_bucket": _fraction_bucket(post_vram_frac, "pm"),
        "util_bucket": _util_bucket(util),
        "resource_bucket": resource_bucket(task),
        "task_kind": task_kind(task),
    }
    bucket = candidate_bucket_key(finite)
    return {
        "finite": finite,
        "candidate_bucket": bucket,
        "class_key": task_cls,
        "regime_key": regime,
        "node": finite["node"],
        "gpu_idx": gpu.get("idx"),
        "need_vram_mb": need,
        "gpu_total_mb": total,
        "gpu_used_mb": used,
        "gpu_free_mb": free,
        "post_used_mb": post_used,
        "running_task_count": running,
        "post_task_count": post_count,
        "used_vram_frac": used_vram_frac,
        "post_vram_frac": post_vram_frac,
        "util_pct": util,
        "occupied": 1 if used >= empty_used_mb else 0,
        "warm": 1 if used >= empty_used_mb else 0,
        "sweet_spot_tasks_per_gpu": sweet,
        "over_sweet_tasks": over_sweet,
        "sweet_gap_tasks": sweet_gap,
        "queue_age_s": _queue_age_seconds(task, context),
        "priority_weight": priority_weight(task),
        "legacy_runtime_unknown": legacy_unknown,
        "legacy_runtime_s": legacy_runtime_s,
    }


def candidate_bucket_key(finite: Mapping[str, Any]) -> str:
    keys = (
        "class_key",
        "regime_key",
        "post_count_bucket",
        "post_vram_bucket",
        "util_bucket",
    )
    return "bucket:v1|" + "|".join(str(finite.get(k, "")) for k in keys)


def finite_feature_metric(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """A bounded fabric/interference metric over extracted feature dicts.

    This is deliberately finite-feature based.  It is the profiling-facing
    metric used by the algorithm layer; theorem constants such as rho and L
    still have to be calibrated from perturbation/profile data.
    """
    lf = left.get("finite", {})
    rf = right.get("finite", {})
    finite_keys = (
        "class_key",
        "regime_key",
        "post_count_bucket",
        "post_vram_bucket",
        "util_bucket",
        "resource_bucket",
        "task_kind",
    )
    finite_distance = sum(1.0 for k in finite_keys if lf.get(k) != rf.get(k))
    numeric_scales = {
        "post_vram_frac": 1.0,
        "used_vram_frac": 1.0,
        "util_pct": 100.0,
        "running_task_count": 8.0,
        "post_task_count": 8.0,
        "legacy_runtime_s": 3600.0,
    }
    numeric_distance = 0.0
    for key, scale in numeric_scales.items():
        delta = abs(as_float(left.get(key), 0.0) - as_float(right.get(key), 0.0))
        numeric_distance += min(1.0, delta / max(scale, 1e-9))
    out = finite_distance + numeric_distance
    if not math.isfinite(out):
        return float(len(finite_keys) + len(numeric_scales))
    return out
