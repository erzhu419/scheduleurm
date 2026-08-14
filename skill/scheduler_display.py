"""Human-facing display formatting helpers for scheduleurm."""
from __future__ import annotations


def format_mem_gb(mb, *, approx: bool = False) -> str:
    """Format a memory quantity stored in MB as a compact GB string."""
    try:
        val = float(mb or 0) / 1024.0
    except Exception:
        val = 0.0
    digits = 2 if abs(val) < 10 else 1
    prefix = "~" if approx else ""
    return f"{prefix}{val:.{digits}f}GB"


def format_node_ram_summary(node: dict) -> str:
    free = node.get("free_ram_mb")
    if free is None:
        return ""
    host_free = node.get("host_free_ram_mb")
    if host_free is not None:
        return f"ram_free={format_mem_gb(free)}(eff)"
    return f"ram_free={format_mem_gb(free)}"


def format_task_location(
    task: dict,
    *,
    legacy_bucket: str | None = None,
    slurm_bucket: str | None = None,
) -> str:
    if not task.get("node"):
        return "-"
    if task.get("slurm_job_id"):
        bucket = legacy_bucket if legacy_bucket is not None else slurm_bucket
        kind = "LEGACY-GPU" if bucket == "gpu" else "LEGACY-CPU"
        state = task.get("slurm_state")
        state_part = f":{state}" if state else ""
        return f"{task['node']}:{kind}#{task['slurm_job_id']}{state_part}"
    if task.get("gpu_idx") is None:
        return f"{task['node']}:CPU"
    return f"{task['node']}:GPU{task['gpu_idx']}"


def format_task_vram_usage(task: dict) -> str:
    if task.get("status") == "running":
        cur = int(task.get("current_vram_mb") or 0)
        if cur > 0:
            return f"cur={format_mem_gb(cur)}"
        if task.get("vram_estimation_source") == "aggregate_observed_zero":
            return "cur=0.00GB"
    if task.get("peak_vram_mb"):
        return f"peak={format_mem_gb(task['peak_vram_mb'])}"
    return format_mem_gb(task.get("est_vram_mb", 0), approx=True)


def format_task_ram_usage(task: dict) -> str:
    if task.get("status") == "running":
        cur = int(task.get("current_ram_mb") or 0)
        if cur > 0:
            return f"Rcur={format_mem_gb(cur)}"
    if task.get("peak_ram_mb"):
        return f"Rpeak={format_mem_gb(task['peak_ram_mb'])}"
    return format_mem_gb(task.get("ram_mb", 0), approx=True)
