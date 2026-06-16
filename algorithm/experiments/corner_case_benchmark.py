"""Corner-case benchmark plan for Scheduleurm OR validation.

This module is deliberately experiment-side.  It does not mutate scheduler.py
and does not dispatch production jobs.  It builds a precise, runnable plan for
probing marginal efficiency, ETA caching, hard-rule modes, queue-aware
placement, and SOTA-style baselines under empty/half/full resource states.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path.home() / ".claude" / "scheduler" / "experiments" / "runs"


ETA_CACHE_KEY_FIELDS = [
    "workload_key",
    "command_fingerprint",
    "normalized_parameters",
    "node",
    "device_family",
    "device_model",
    "device_count",
    "gpu_indices",
    "cpu_worker_count",
    "co_location_profile",
    "resource_occupancy_state",
    "algorithm_mode",
    "hard_rule_mode",
    "python_or_conda_env",
    "cuda_visible_devices",
    "driver_or_runtime_version",
]


HARD_RULE_MODES = [
    {
        "mode": "safety_on",
        "scheduler_hook": "--hard-rule-mode safety",
        "meaning": "legacy safety gates stay active; this is the production-facing run.",
    },
    {
        "mode": "clean_bench",
        "scheduler_hook": "--hard-rule-mode clean_bench",
        "meaning": "nonessential legacy packing freezes are bypassed for theorem/algorithm comparison.",
    },
]


ALGORITHM_MODES = [
    "legacy",
    "sweetspot_v1",
    "theorem_maxweight_v1",
]


SOTA_POLICY_SEMANTICS = [
    {
        "policy": "gavel_finish_time_fairness",
        "source": "Gavel-style profiled-throughput trace scheduling",
        "comparison_level": "same measured service cache / native simulator where available",
    },
    {
        "policy": "pollux_goodput_resize",
        "source": "Pollux/AdaptDL-style goodput and resource resize semantics",
        "comparison_level": "same measured service cache policy semantics",
    },
    {
        "policy": "sia_multi_cluster_goodput",
        "source": "Sia-style heterogeneity-aware goodput placement",
        "comparison_level": "same measured service cache policy semantics",
    },
    {
        "policy": "gandiva_pack_time_slice",
        "source": "Gandiva-style GPU packing/time-slicing control",
        "comparison_level": "same measured service cache policy semantics",
    },
    {
        "policy": "salus_memory_pack",
        "source": "Salus-style GPU memory-aware packing",
        "comparison_level": "same measured service cache policy semantics",
    },
    {
        "policy": "iadeep_interference_aware",
        "source": "IADeep-style interference-aware colocation heuristic",
        "comparison_level": "same measured service cache policy semantics",
    },
]


def build_corner_case_plan(*, run_id: str | None = None) -> dict[str, Any]:
    run_id = run_id or time.strftime("corner_case_plan_%Y%m%d_%H%M%S")
    scenarios = _scenario_rows()
    return {
        "gate": "corner_case_benchmark_plan",
        "run_id": run_id,
        "created_at_unix": time.time(),
        "objective": (
            "Validate whether the theorem/sweet-spot Scheduleurm algorithm can "
            "place small tasks beside large GPU/CPU jobs and improve total work "
            "completion time without relying on stale TUI ETA."
        ),
        "resource_population": {
            "gpu_nodes": ["jtl110gpu", "jtl110gpu2", "node007-direct"],
            "node007_direct_gpu_shape": "4 x RTX 2080 Ti class, about 11GB each",
            "cpu_nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
            "cpu_node_shape": "192 logical cores per node",
        },
        "algorithm_modes": list(ALGORITHM_MODES),
        "hard_rule_modes": list(HARD_RULE_MODES),
        "sota_policy_semantics": list(SOTA_POLICY_SEMANTICS),
        "eta_policy": {
            "source_of_truth": "benchmark log progress lines, not tui-top ETA",
            "required_log_tokens": ["rate=", "ETA"],
            "stable_eta_rule": {
                "min_windows": 3,
                "max_window_cv": 0.08,
                "max_last_two_relative_delta": 0.05,
                "action_after_stable": "terminate probe window and fast-forward replay/simulation",
            },
            "cache_key_fields": list(ETA_CACHE_KEY_FIELDS),
            "cache_reuse_rule": (
                "reuse only when every cache-key field matches; otherwise probe "
                "or mark measurement_required"
            ),
        },
        "queue_state_contract": {
            "includes": [
                "queued task count by workload class",
                "running task count by workload class",
                "per-task ETA and ETA source",
                "node/gpu eta_load from existing placements",
                "candidate lower-service vector",
                "oracle gap per dispatch slot",
            ],
            "must_refresh_each_round": True,
            "scheduler_surface": [
                "algorithm/theorem_dispatch/queue_state.py",
                "algorithm/theorem_dispatch/policy.py",
                "skill/scheduler.py optional A/B hook only",
            ],
        },
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "execution_order": [
            "resource_audit",
            "empty-resource single-task ETA probes",
            "half-loaded marginal probes",
            "full-loaded marginal probes",
            "large multi-GPU memory resident plus small marginal probes",
            "large 192-core CPU resident plus small marginal probes",
            "queue backlog replay with ETA-cache reuse",
            "policy-semantics SOTA replay on the measured service rows",
            "production-safety-on rerun for claim-facing subset",
        ],
    }


def audit_resources(nodes: list[str] | None = None) -> dict[str, Any]:
    """Audit resources through scheduler's configured SSH/sudo paths."""

    nodes = nodes or [
        "jtl110gpu",
        "jtl110gpu2",
        "node007-direct",
        "node001",
        "node002",
        "node003",
        "node004",
        "node005",
        "node006",
    ]
    scheduler = _load_scheduler_module()
    shell_cmd = (
        "hostname; "
        "echo NPROC=$(nproc); "
        "awk '/MemTotal|MemAvailable/ {print $1$2}' /proc/meminfo; "
        "if command -v nvidia-smi >/dev/null 2>&1; then "
        "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits; "
        "elif [ -x /cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi ]; then "
        "/cm/local/apps/cuda-driver/libs/535.261.03/bin/nvidia-smi "
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits; fi"
    )
    rows = []
    for node in nodes:
        try:
            rc, out, err = scheduler.run_on(node, shell_cmd, timeout=18, check=False)
            rows.append({
                "node": node,
                "returncode": rc,
                "raw_stdout": out,
                "raw_stderr": err,
                "parsed": _parse_resource_stdout(out),
                "reachable": rc == 0,
            })
        except Exception as exc:
            rows.append({
                "node": node,
                "returncode": None,
                "raw_stdout": "",
                "raw_stderr": repr(exc),
                "parsed": {},
                "reachable": False,
            })
    return {
        "gate": "corner_case_resource_audit",
        "created_at_unix": time.time(),
        "nodes": rows,
        "pass": all(row.get("reachable") for row in rows),
    }


def write_plan_artifacts(plan: Mapping[str, Any], *, output_dir: Path | None = None) -> dict[str, str]:
    out_dir = output_dir or (ARTIFACT_DIR / str(plan.get("run_id") or "corner_case_plan") / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "corner_case_benchmark_plan.json"
    md_path = out_dir / "corner_case_benchmark_plan.md"
    json_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_plan_markdown(plan, json_path), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def write_resource_audit_artifacts(report: Mapping[str, Any], *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "corner_case_resource_audit.json"
    md_path = output_dir / "corner_case_resource_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_audit_markdown(report, json_path), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def _scenario_rows() -> list[dict[str, Any]]:
    py = "python3 -u"
    gpu_mem = f"{py} algorithm/experiments/gpu_multigpu_memory_progress_benchmark.py"
    gpu_cnn = f"{py} algorithm/experiments/torch_cnn_progress_benchmark.py"
    gpu_llm = f"{py} algorithm/experiments/torch_llm_progress_benchmark.py"
    cpu_single = f"{py} algorithm/experiments/cpu_progress_benchmark.py"
    cpu_parallel = f"{py} algorithm/experiments/cpu_parallel_progress_benchmark.py"
    return [
        {
            "id": "gpu_empty_single_eta",
            "quadrant": "q01_gpu_heavy",
            "resource_state": "empty",
            "question": "Baseline pure GPU/CNN/LLM ETA on an idle GPU.",
            "nodes": ["jtl110gpu", "jtl110gpu2", "node007-direct"],
            "probe_commands": [
                f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 80 --warmup 8 --log-interval 5 --label gpu_empty_cnn",
                f"CUDA_VISIBLE_DEVICES=0 {gpu_llm} --model-id distilgpt2 --steps 60 --warmup 4 --log-interval 5 --label gpu_empty_llm",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "gpu_half_loaded_add_one",
            "quadrant": "q01_gpu_heavy",
            "resource_state": "half_loaded",
            "question": "Marginal rate of adding one GPU task when the card already has two residents.",
            "nodes": ["jtl110gpu", "jtl110gpu2", "node007-direct"],
            "background": f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 240 --warmup 8 --log-interval 10 --label gpu_half_bg",
            "probe_commands": [
                f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 80 --warmup 4 --log-interval 5 --label gpu_half_add_cnn",
                f"CUDA_VISIBLE_DEVICES=0 {gpu_llm} --model-id distilgpt2 --steps 60 --warmup 4 --log-interval 5 --label gpu_half_add_llm",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "gpu_full_loaded_add_one",
            "quadrant": "q01_gpu_heavy",
            "resource_state": "full_loaded",
            "question": "Marginal rate near the measured capacity boundary before memory/OOM failure.",
            "nodes": ["jtl110gpu", "jtl110gpu2", "node007-direct"],
            "background": f"CUDA_VISIBLE_DEVICES=0 {gpu_mem} --reserve-gb-per-gpu 8.5 --steps 240 --log-interval 10 --label gpu_full_bg",
            "probe_commands": [
                f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 60 --warmup 4 --log-interval 5 --label gpu_full_add_cnn",
                f"CUDA_VISIBLE_DEVICES=0 {gpu_llm} --model-id distilgpt2 --steps 40 --warmup 3 --log-interval 5 --label gpu_full_add_llm",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "node007_30gb_multigpu_plus_small",
            "quadrant": "q11_cpu_gpu_coupled",
            "resource_state": "large_multigpu_resident",
            "question": "30GB-class resident LLM-style memory job over 4x12GB plus small marginal tasks.",
            "nodes": ["node007-direct"],
            "background": (
                f"CUDA_VISIBLE_DEVICES=0,1,2,3 {gpu_mem} --devices 0,1,2,3 "
                "--reserve-gb-per-gpu 7.0 --matrix-size 1536 --steps 240 "
                "--log-interval 10 --label node007_30gb_resident"
            ),
            "probe_commands": [
                f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 60 --warmup 4 --log-interval 5 --label node007_30gb_add_cnn",
                f"CUDA_VISIBLE_DEVICES=1 {gpu_llm} --model-id distilgpt2 --steps 40 --warmup 3 --log-interval 5 --label node007_30gb_add_llm",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "cpu_empty_single_eta",
            "quadrant": "q10_cpu_host_bound",
            "resource_state": "empty",
            "question": "Single CPU and parallel CPU ETA on an idle 192-core node.",
            "nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
            "probe_commands": [
                f"{cpu_single} --mode cpu --steps 50 --work-items 200000 --label cpu_empty_single",
                f"{cpu_parallel} --workers 96 --steps 16 --work-items 70000 --log-interval 1 --label cpu_empty_parallel96",
            ],
            "policies": _policy_matrix(cpu=True),
        },
        {
            "id": "cpu_half_loaded_add_small",
            "quadrant": "q10_cpu_host_bound",
            "resource_state": "half_loaded",
            "question": "Marginal rate when a 96-worker resident job occupies half of a 192-core node.",
            "nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
            "background": f"{cpu_parallel} --workers 96 --steps 80 --work-items 90000 --log-interval 2 --label cpu_half_bg",
            "probe_commands": [
                f"{cpu_single} --mode cpu --steps 40 --work-items 200000 --label cpu_half_add_single",
                f"{cpu_parallel} --workers 16 --steps 20 --work-items 90000 --log-interval 1 --label cpu_half_add_parallel16",
            ],
            "policies": _policy_matrix(cpu=True),
        },
        {
            "id": "cpu_full_loaded_add_small",
            "quadrant": "q10_cpu_host_bound",
            "resource_state": "full_loaded",
            "question": "Marginal rate near 180/192 workers to find whether small work should be deferred.",
            "nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
            "background": f"{cpu_parallel} --workers 180 --steps 80 --work-items 90000 --log-interval 2 --label cpu_full_bg",
            "probe_commands": [
                f"{cpu_single} --mode cpu --steps 30 --work-items 200000 --label cpu_full_add_single",
                f"{cpu_parallel} --workers 8 --steps 18 --work-items 90000 --log-interval 1 --label cpu_full_add_parallel8",
            ],
            "policies": _policy_matrix(cpu=True),
        },
        {
            "id": "mixed_gpu_cpu_resident_plus_small",
            "quadrant": "q11_cpu_gpu_coupled",
            "resource_state": "mixed_loaded",
            "question": "Hybrid resident job plus small GPU/CPU tasks; validates queue-aware MaxWeight over classes.",
            "nodes": ["node007-direct", "node001"],
            "background": (
                f"CUDA_VISIBLE_DEVICES=0 {gpu_cnn} --steps 180 --warmup 8 --log-interval 10 "
                "--label mixed_gpu_bg && "
                f"{cpu_parallel} --workers 96 --steps 60 --work-items 80000 --log-interval 2 --label mixed_cpu_bg"
            ),
            "probe_commands": [
                f"CUDA_VISIBLE_DEVICES=1 {gpu_llm} --model-id distilgpt2 --steps 40 --warmup 3 --log-interval 5 --label mixed_add_llm",
                f"{cpu_single} --mode cpu --steps 40 --work-items 200000 --label mixed_add_cpu",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "queue_backlog_eta_load_balance",
            "quadrant": "portfolio",
            "resource_state": "queued_backlog",
            "question": "Queued count, per-task ETA, eta_load, and load balancing before launch.",
            "nodes": ["jtl110gpu", "jtl110gpu2", "node007-direct", "node001", "node002"],
            "probe_commands": [
                "SCHEDULEURM_ALGORITHM=theorem_maxweight_v1 python3 skill/scheduler.py status --json --brief",
                "python3 -m algorithm.experiments.live_trace_dryrun build --algorithm theorem_maxweight_v1 --hard-rule-mode safety",
            ],
            "policies": _policy_matrix(),
        },
        {
            "id": "eta_cache_identical_signature",
            "quadrant": "portfolio",
            "resource_state": "cache_reuse",
            "question": "Identical task/env/profile reuses cached ETA; changed colocation profile forces a new probe.",
            "nodes": ["local"],
            "probe_commands": [
                "PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q skill/tests/test_corner_case_benchmark_plan.py::test_eta_cache_contract_has_colocation_key",
            ],
            "policies": _policy_matrix(),
        },
    ]


def _policy_matrix(cpu: bool = False) -> list[str]:
    if cpu:
        return ["legacy", "theorem_maxweight_v1", "gavel_finish_time_fairness", "pollux_goodput_resize", "iadeep_interference_aware"]
    return ["legacy", "sweetspot_v1", "theorem_maxweight_v1"] + [
        row["policy"] for row in SOTA_POLICY_SEMANTICS
    ]


def _parse_resource_stdout(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"gpus": []}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("NPROC="):
            try:
                out["nproc"] = int(line.split("=", 1)[1])
            except Exception:
                pass
        elif line.startswith("MemTotal:"):
            out["mem_total_kb"] = _trailing_int(line)
        elif line.startswith("MemAvailable:"):
            out["mem_available_kb"] = _trailing_int(line)
        elif re.match(r"^[0-9]+,", line):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                out["gpus"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": _safe_int(parts[2]),
                    "memory_total_mb": _safe_int(parts[3]),
                    "utilization_gpu_pct": _safe_int(parts[4]),
                })
        elif "hostname" not in out:
            out["hostname"] = line
    return out


def _trailing_int(text: str) -> int | None:
    m = re.search(r"([0-9]+)$", text.replace(" ", ""))
    return int(m.group(1)) if m else None


def _safe_int(text: str) -> int | None:
    try:
        return int(float(str(text).strip()))
    except Exception:
        return None


def _load_scheduler_module():
    path = REPO_ROOT / "skill" / "scheduler.py"
    spec = importlib.util.spec_from_file_location("scheduleurm_scheduler_for_corner_case", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan_markdown(plan: Mapping[str, Any], json_path: Path) -> str:
    lines = [
        "# Corner-Case Benchmark Plan",
        "",
        f"JSON artifact: `{json_path}`",
        "",
        "This plan is the experiment-side contract for the current empty-resource window. "
        "It uses benchmark logs with explicit `rate=` and `ETA` lines as the ETA source of truth.",
        "",
        "## Resource Population",
        "",
        f"- GPU nodes: {', '.join(plan.get('resource_population', {}).get('gpu_nodes') or [])}",
        f"- CPU nodes: {', '.join(plan.get('resource_population', {}).get('cpu_nodes') or [])}",
        "",
        "## ETA Cache Contract",
        "",
    ]
    for field in plan.get("eta_policy", {}).get("cache_key_fields") or []:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Scenarios", "", "| id | quadrant | resource state | nodes | policies |", "|---|---|---|---|---|"])
    for row in plan.get("scenarios") or []:
        lines.append(
            f"| `{row.get('id')}` | `{row.get('quadrant')}` | `{row.get('resource_state')}` | "
            f"{', '.join(row.get('nodes') or [])} | {', '.join(row.get('policies') or [])} |"
        )
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This plan can support OR claim-facing experiments only after the corresponding "
        "probe summaries show stable ETA windows and the replay/live gate consumes those "
        "measured rows.  It is not itself a performance claim.",
        "",
    ])
    return "\n".join(lines)


def _audit_markdown(report: Mapping[str, Any], json_path: Path) -> str:
    lines = [
        "# Corner-Case Resource Audit",
        "",
        f"JSON artifact: `{json_path}`",
        "",
        "| node | reachable | nproc | mem available GB | GPUs |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report.get("nodes") or []:
        parsed = row.get("parsed") or {}
        mem_gb = (parsed.get("mem_available_kb") or 0) / 1024 / 1024
        gpus = parsed.get("gpus") or []
        gpu_text = "; ".join(
            f"{g.get('index')}:{g.get('memory_used_mb')}/{g.get('memory_total_mb')}MB,{g.get('utilization_gpu_pct')}%"
            for g in gpus
        )
        lines.append(
            f"| `{row.get('node')}` | {str(bool(row.get('reachable'))).lower()} | "
            f"{parsed.get('nproc') or ''} | {mem_gb:.1f} | {gpu_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m algorithm.experiments.corner_case_benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("build-plan")
    plan.add_argument("--run-id", default="")
    plan.add_argument("--output-dir", default="")
    plan.set_defaults(func=_cmd_build_plan)
    audit = sub.add_parser("audit-resources")
    audit.add_argument("--run-id", default="")
    audit.add_argument("--nodes", default="")
    audit.add_argument("--output-dir", default="")
    audit.set_defaults(func=_cmd_audit_resources)
    return parser


def _cmd_build_plan(args: argparse.Namespace) -> int:
    plan = build_corner_case_plan(run_id=args.run_id or None)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    paths = write_plan_artifacts(plan, output_dir=output_dir)
    print(json.dumps({"pass": True, "paths": paths}, indent=2, sort_keys=True))
    return 0


def _cmd_audit_resources(args: argparse.Namespace) -> int:
    nodes = [x.strip() for x in str(args.nodes or "").split(",") if x.strip()] or None
    report = audit_resources(nodes)
    run_id = args.run_id or time.strftime("corner_case_resource_audit_%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else ARTIFACT_DIR / run_id / "reports"
    )
    paths = write_resource_audit_artifacts(report, output_dir=output_dir)
    print(json.dumps({"pass": bool(report.get("pass")), "paths": paths}, indent=2, sort_keys=True))
    return 0 if report.get("pass") else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
