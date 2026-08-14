"""Resource estimate and CPU worker planning for submitted tasks."""

from __future__ import annotations

from typing import Any

try:
    from .task_common import SubmitTaskDeps
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from .task_common import SubmitTaskDeps


def _estimate_submit_resources(
    state: dict,
    args: Any,
    *,
    sig: str,
    deps: SubmitTaskDeps,
    default_ram_mb: int,
) -> tuple[dict, int, int]:
    hist = deps.history_get(sig) or {}
    full_hist = None
    if args.vram is not None:
        est_vram = args.vram
    elif hist.get("vram_mb"):
        est_vram = hist["vram_mb"]
    else:
        full_hist = deps.load_history()
        probe_task = {
            "id": None,
            "signature": sig,
            "project": args.project or deps.project_from_path(args.cwd),
            "description": args.description,
            "est_vram_mb": 0,
        }
        est_vram = deps.effective_est_vram(probe_task, state, full_hist)

    if args.ram_mb is not None:
        ram_mb = args.ram_mb
    elif hist.get("ram_mb"):
        ram_mb = hist["ram_mb"]
    else:
        ram_full_hist = full_hist or deps.load_history()
        probe_task_for_ram = {
            "id": None,
            "signature": sig,
            "project": args.project or deps.project_from_path(args.cwd),
            "description": args.description,
            "ram_mb": default_ram_mb,
        }
        ram_mb = deps.effective_est_ram(probe_task_for_ram, state, ram_full_hist)
    return hist, int(est_vram), int(ram_mb)


def _submit_cpu_plan(
    args: Any,
    hist: dict,
    parallel: dict,
    *,
    deps: SubmitTaskDeps,
    default_cpu_cores: int,
) -> tuple[int, dict | None]:
    cpu_batch_plan = getattr(args, "cpu_batch_plan", None)
    items = parallel["items"]
    if args.cpu is not None:
        cpu_cores = args.cpu
        if items <= 0:
            return int(cpu_cores), None
        if isinstance(cpu_batch_plan, dict):
            cpu_auto_plan = {
                "workers": int(cpu_batch_plan.get("workers") or cpu_cores),
                "waves": int(cpu_batch_plan.get("waves") or (
                    (items + max(1, cpu_cores) - 1) // max(1, cpu_cores)
                )),
                "physical_cores": int(cpu_batch_plan.get("physical_cores") or cpu_cores),
                "total_physical_cores": int(
                    cpu_batch_plan.get("total_physical_cores")
                    or cpu_batch_plan.get("physical_cores")
                    or cpu_cores
                ),
                "last_wave_items": int(cpu_batch_plan.get("last_wave_items") or items),
            }
        else:
            cpu_auto_plan = deps.cpu_worker_plan_for_items(items, max(1, cpu_cores))
        return int(cpu_cores), cpu_auto_plan

    if items > 0:
        ref_node = args.require_node or args.preferred_node
        if ref_node:
            physical = deps.node_physical_cores(ref_node)
        else:
            physical = max(
                [deps.node_physical_cores(n) for n in deps.cpu_labor_node_names()]
                or [default_cpu_cores]
            )
        cpu_auto_plan = deps.cpu_worker_plan_for_items(items, physical)
        return int(cpu_auto_plan["workers"]), cpu_auto_plan
    return int(hist.get("cpu_cores", default_cpu_cores)), None
