"""Check that theorem dispatch consumes the live v2 service cache correctly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from algorithm.features import gpu_candidate_features
from algorithm.theorem_dispatch.global_dispatch import select_global_action
from algorithm.theorem_dispatch.service_registry import bind_service, infer_workload_key
from algorithm.theorem_dispatch.state_service import StateDependentServiceCache
from simulation.service_cache import ServiceRateCache


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"
DEFAULT_CACHE = ARTIFACT_ROOT / "service_cache_v2_live_merged_20260629.json"


def build_live_v2_dispatch_alignment_gate(
    *,
    cache_path: str | Path = DEFAULT_CACHE,
) -> dict[str, Any]:
    cache = ServiceRateCache.load(Path(cache_path))
    checks = [
        _env_key_check(cache),
        _node_lookup_check(cache),
        _rl_node_assignment_check(cache),
        _state_lookup_check(cache),
        _global_action_check(cache),
    ]
    passed = all(bool(row.get("pass")) for row in checks)
    return {
        "gate": "live_v2_dispatch_alignment_gate",
        "cache_path": str(cache_path),
        "status": "LIVE_V2_DISPATCH_ALIGNMENT_PASS" if passed else "LIVE_V2_DISPATCH_ALIGNMENT_OPEN",
        "pass": bool(passed),
        "checks": checks,
        "claim_boundary": (
            "This is an algorithm-alignment gate over the measured live v2 cache. "
            "It proves that dispatch lookup and scoring consume env/node/state keys; "
            "it is not a new live production launch trace."
        ),
    }


def _env_key_check(cache: ServiceRateCache) -> dict[str, Any]:
    task = {
        "description": "RE-SAC HalfCheetah-v2 controlled benchmark",
        "cmd": "python -m jax_experiments.train --algo resac --env HalfCheetah-v2",
        "est_vram_mb": 2048,
    }
    key = infer_workload_key(task, cache)
    return {
        "name": "env_specific_workload_key",
        "pass": key == "hybrid_rl_resac_halfcheetah",
        "inferred_key": key,
        "expected_key": "hybrid_rl_resac_halfcheetah",
    }


def _node_lookup_check(cache: ServiceRateCache) -> dict[str, Any]:
    task = {
        "description": "Torch CNN ResNet50 controlled benchmark",
        "theorem_workload_key": "gpu_cnn_torch_resnet50",
        "workload_env": "cnn",
        "est_vram_mb": 2048,
    }
    jtl110 = _bind_for_node(cache, task, "jtl110gpu", total_mb=12288, used_mb=0)
    jtl311 = _bind_for_node(cache, task, "jtl311linux", total_mb=8192, used_mb=0)
    return {
        "name": "node_specific_empty_cnn_lookup",
        "pass": bool(jtl110.certified and jtl311.certified and jtl110.lower_service > jtl311.lower_service),
        "jtl110": jtl110.snapshot(),
        "jtl311": jtl311.snapshot(),
    }


def _rl_node_assignment_check(cache: ServiceRateCache) -> dict[str, Any]:
    task = {
        "id": "resac-halfcheetah",
        "description": "RE-SAC HalfCheetah-v2 controlled benchmark",
        "cmd": "python -m jax_experiments.train --algo resac --env HalfCheetah-v2",
        "workload_env": "halfcheetah",
        "est_vram_mb": 2048,
    }
    jtl110 = _bind_for_node(cache, task, "jtl110gpu", total_mb=12288, used_mb=0)
    jtl311 = _bind_for_node(cache, task, "jtl311linux", total_mb=8192, used_mb=0)
    rows = [
        _action_row("halfcheetah-to-jtl110", "jtl110gpu:0", jtl110),
        _action_row("halfcheetah-to-jtl311", "jtl311linux:0", jtl311),
    ]
    result = select_global_action(rows, {"hybrid_rl_resac_halfcheetah": 10.0}, max_batch_size=1)
    expected = "launch:halfcheetah-to-jtl110:jtl110gpu:0"
    if jtl311.lower_service > jtl110.lower_service:
        expected = "launch:halfcheetah-to-jtl311:jtl311linux:0"
    return {
        "name": "rl_env_node_specific_assignment",
        "pass": bool(jtl110.certified and jtl311.certified and result.selected_action_ids == (expected,)),
        "expected_selected": expected,
        "selected": result.snapshot(),
        "jtl110": jtl110.snapshot(),
        "jtl311": jtl311.snapshot(),
    }


def _state_lookup_check(cache: ServiceRateCache) -> dict[str, Any]:
    state_cache = StateDependentServiceCache.from_cache(cache)
    jtl110_half = cache.get_statewise(
        "gpu_cnn_torch_resnet50",
        2,
        node_bucket="jtl110gpu:gpu_3080ti_12gb_dual",
        workload_env="cnn",
        resource_state="half_loaded",
        strict=True,
    )
    high_vram = state_cache.lookup(
        "gpu_cnn_torch_resnet50",
        1,
        "high_vram_resident",
        allow_base_fallback=False,
    )
    cpu = state_cache.lookup(
        "cpu_heavy_local_bench",
        1,
        "cpu_resident",
        allow_base_fallback=False,
    )
    return {
        "name": "resource_state_marginal_lookup",
        "pass": bool(jtl110_half and high_vram.certified and cpu.certified),
        "jtl110_half_loaded_rate": None if jtl110_half is None else float(jtl110_half.aggregate_rate),
        "high_vram": high_vram.snapshot(),
        "cpu_resident": cpu.snapshot(),
    }


def _global_action_check(cache: ServiceRateCache) -> dict[str, Any]:
    task_a = {"id": "cnn-a", "description": "Torch CNN ResNet50", "theorem_workload_key": "gpu_cnn_torch_resnet50", "workload_env": "cnn"}
    task_b = {"id": "cnn-b", "description": "Torch CNN ResNet50", "theorem_workload_key": "gpu_cnn_torch_resnet50", "workload_env": "cnn"}
    jtl110 = _bind_for_node(cache, task_a, "jtl110gpu", total_mb=12288, used_mb=0)
    jtl311 = _bind_for_node(cache, task_b, "jtl311linux", total_mb=8192, used_mb=0)
    rows = [
        _action_row(task_a["id"], "jtl110gpu:0", jtl110),
        _action_row(task_b["id"], "jtl311linux:0", jtl311),
    ]
    result = select_global_action(rows, {"gpu_cnn_torch_resnet50": 10.0}, max_batch_size=1)
    return {
        "name": "global_maxweight_selects_higher_lower_service",
        "pass": result.selected_action_ids == ("launch:cnn-a:jtl110gpu:0",),
        "selected": result.snapshot(),
        "candidate_rows": rows,
    }


def _bind_for_node(cache: ServiceRateCache, task: Mapping[str, Any], node: str, *, total_mb: int, used_mb: int):
    features = gpu_candidate_features(
        task,
        {"name": node},
        {
            "idx": 0,
            "total_mb": int(total_mb),
            "used_mb": int(used_mb),
            "free_mb": max(0, int(total_mb) - int(used_mb)),
            "running_task_count": 0,
        },
    )
    return bind_service(task, features, cache=cache)


def _action_row(task_id: str, resource_id: str, binding) -> dict[str, Any]:
    return {
        "action_id": f"launch:{task_id}:{resource_id}",
        "task_id": task_id,
        "resource_id": resource_id,
        "score_semantics": "robust_maxweight_lower_service",
        "theorem_ready": bool(binding.certified),
        "lower_service": binding.lower_service_vector(),
        "penalty_units": 0.0,
        "service_binding": binding.snapshot(),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live v2 Dispatch Alignment Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Cache: `{report.get('cache_path')}`",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    for row in report.get("checks") or []:
        lines.append(f"| `{row.get('name')}` | {str(bool(row.get('pass'))).lower()} |")
    lines.extend(["", str(report.get("claim_boundary") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "live_v2_dispatch_alignment_gate_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "live_v2_dispatch_alignment_gate_20260629.md")
    args = parser.parse_args()
    report = build_live_v2_dispatch_alignment_gate(cache_path=args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
