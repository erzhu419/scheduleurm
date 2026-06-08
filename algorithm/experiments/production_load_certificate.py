"""Empirical production-load capacity certificate.

This module estimates arrival rates from Scheduleurm task records and checks
them against the measured service-action slice.  It separates two questions:

* measured-bucket capacity: can the mapped workload classes be supported?
* global production coverage: did every production task map to a measured
  service bucket?

The second question is intentionally strict.  A positive capacity LP on mapped
tasks is not a global theorem certificate when a large fraction of production
tasks are unmapped or only representative-mapped.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.defaults import build_default_cache
from simulation.tasksets import TaskSetMember, taskset_by_name

from .capacity_lp import solve_capacity_slack
from .empirical_slack_certificate import (
    _finite_support_second_moment_bound,
    _profile_domain,
    _product_actions,
)


DEFAULT_TASKSETS = (
    "q00_light_control",
    "q01_gpu_bound_compute",
    "q10_cpu_host_bound",
    "production_freqduet_cpu_c17_32",
    "production_freqduet_cpu_ablation_c9_16",
    "production_sumo_eval_simple_sac_c_le2",
    "production_freqduet_runner_v3_allfreq_alllayers_c9_16",
    "q11_cpu_gpu_coupled",
)


def build_production_load_certificate(
    *,
    records: Iterable[Mapping[str, Any]],
    window_days: float = 30.0,
    include_representative: bool = False,
    taskset_names: Iterable[str] = DEFAULT_TASKSETS,
    now_ts: float | None = None,
) -> dict[str, Any]:
    materialized = _dedupe_records(records)
    end_ts = _window_end(materialized, now_ts=now_ts)
    window_s = max(1.0, float(window_days) * 86400.0)
    start_ts = end_ts - window_s
    window_records = [
        row for row in materialized
        if start_ts <= _submitted_at(row) <= end_ts
    ]
    members = _members_for_tasksets(taskset_names)
    member_by_key = {member.workload_key: member for member in members}
    classification_rows = []
    mapped_counts: Counter[str] = Counter()
    mapped_units: Counter[str] = Counter()
    representative_counts: Counter[str] = Counter()
    unmapped_reasons: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()

    for row in window_records:
        status_counts[str(row.get("status") or "")] += 1
        project_counts[str(row.get("project") or "")] += 1
        cls = classify_record(row, include_representative=include_representative)
        out = {
            "task_id": row.get("id") or row.get("task_id"),
            "status": row.get("status"),
            "project": row.get("project"),
            "signature": row.get("signature"),
            "submitted_at": _submitted_at(row),
            "classification": cls,
        }
        classification_rows.append(out)
        key = cls.get("workload_key")
        if key in member_by_key:
            units = _classified_units(cls, fallback=member_by_key[key].total_units)
            mapped_counts[key] += 1
            mapped_units[key] += units
            if cls.get("mapping_mode") != "strict_measured":
                representative_counts[key] += 1
        else:
            unmapped_reasons[str(cls.get("reason") or "unmapped")] += 1

    lam = {
        key: float(units) / window_s
        for key, units in sorted(mapped_units.items())
        if units > 0
    }
    cache = build_default_cache()
    domains = {member.workload_key: _profile_domain(cache, member) for member in members}
    actions = _product_actions(cache, tuple(members), domains)
    capacity = solve_capacity_slack(actions, lam)
    moment = {
        "B": _finite_support_second_moment_bound(actions, lam),
        "status": "finite_support_bound_from_measured_action_slice_and_empirical_load",
        "usable_for_theorem": bool(actions),
    }
    mapped_task_count = sum(mapped_counts.values())
    representative_task_count = sum(representative_counts.values())
    unmapped_task_count = len(window_records) - mapped_task_count
    mapped_capacity_usable = bool(capacity.get("usable_for_theorem"))
    global_coverage_usable = unmapped_task_count == 0 and representative_task_count == 0
    return {
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "window_days": float(window_days),
            "window_s": window_s,
        },
        "taskset_names": list(taskset_names),
        "include_representative": bool(include_representative),
        "record_count_total": len(materialized),
        "record_count_window": len(window_records),
        "mapped_task_count": mapped_task_count,
        "representative_mapped_task_count": representative_task_count,
        "unmapped_task_count": unmapped_task_count,
        "mapped_fraction": mapped_task_count / len(window_records) if window_records else 0.0,
        "strict_mapped_fraction": (
            (mapped_task_count - representative_task_count) / len(window_records)
            if window_records else 0.0
        ),
        "status_counts": dict(status_counts),
        "top_projects": dict(project_counts.most_common(20)),
        "mapped_counts": dict(mapped_counts),
        "mapped_units": dict(mapped_units),
        "lambda": lam,
        "capacity": capacity,
        "moment": moment,
        "action_profile_domains": domains,
        "full_action_count": len(actions),
        "mapped_capacity_usable_for_theorem": mapped_capacity_usable,
        "global_coverage_usable_for_theorem": global_coverage_usable,
        "usable_for_global_theorem": mapped_capacity_usable and global_coverage_usable,
        "unmapped_reasons": dict(unmapped_reasons),
        "classification_sample": classification_rows[:200],
        "interpretation": (
            "Positive capacity on mapped measured buckets is a load certificate only "
            "for those buckets. Global production stability remains open unless "
            "unmapped and representative-mapped tasks are eliminated or separately "
            "certified by service measurements."
        ),
    }


def classify_record(
    row: Mapping[str, Any],
    *,
    include_representative: bool = False,
) -> dict[str, Any]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("project", "signature", "description", "cmd", "cwd")
    ).lower()
    est_vram = _as_float(row.get("est_vram_mb", row.get("vram_mb", 0.0)))
    cpu = _as_float(row.get("cpu_cores", 0.0))

    if "scheduleurmbench" in text:
        if any(token in text for token in ("q00", "light_control", "light-control")):
            return _mapped("light_control_local", "strict_measured", "scheduleurmbench_q00")
        if any(token in text for token in ("q01", "jax_matmul", "matmul", "gpu_heavy")):
            return _mapped("gpu_heavy_jax_matmul", "strict_measured", "scheduleurmbench_q01")
        if any(token in text for token in ("q10", "cpu_heavy", "cpu-heavy")):
            return _mapped("cpu_heavy_local_bench", "strict_measured", "scheduleurmbench_q10")
        if any(token in text for token in (
            "q11", "resac_ant", "resac_walker", "bapr_ant", "hybrid_rl",
        )):
            return _mapped("hybrid_rl_resac_ant", "strict_measured", "scheduleurmbench_q11")
        return {"workload_key": None, "mapping_mode": "unmapped", "reason": "scheduleurmbench_unknown"}

    if _is_freqduet_cpu_ablation_c17_32(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "freqduet_cpu_ablation_c17_32",
            "strict_measured",
            "module56_freqduet_cpu_ablation_c17_32",
        )

    c9_16_units = _freqduet_ablation_c9_16_units(row=row, est_vram=est_vram, cpu=cpu)
    if c9_16_units is not None:
        return _mapped(
            "freqduet_cpu_ablation_c9_16",
            "strict_measured",
            "module58_freqduet_cpu_ablation_c9_16",
            units=c9_16_units,
        )

    if _is_simple_sac_sumo_eval_c_le2(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "sumo_eval_simple_sac_c_le2",
            "strict_measured",
            "module59_simple_sac_sumo_eval_c_le2_completed_history",
            units=1.0,
        )

    if _is_freqduet_runner_v3_allfreq_alllayers_c9_16(row=row, est_vram=est_vram, cpu=cpu):
        return _mapped(
            "freqduet_runner_v3_allfreq_alllayers_c9_16",
            "strict_measured",
            "module57_freqduet_runner_v3_allfreq_alllayers_c9_16",
        )

    if include_representative:
        if est_vram > 0 and any(
            token in text for token in (
                "re-sac", "resac", "bapr", "jax_experiments.train",
                "sac_", "halfcheetah", "walker2d", "hopper", "ant-v2", "humanoid",
            )
        ):
            return _mapped("hybrid_rl_resac_ant", "representative", "gpu_rl_representative")
        if est_vram <= 0 and cpu >= 4 and any(
            token in text for token in (
                "analysis", "audit", "cpu_eval", "stage", "sweep", "preprocess",
            )
        ):
            return _mapped("cpu_heavy_local_bench", "representative", "cpu_heavy_representative")

    if est_vram > 0:
        return {"workload_key": None, "mapping_mode": "unmapped", "reason": "unmapped_gpu"}
    return {"workload_key": None, "mapping_mode": "unmapped", "reason": "unmapped_cpu"}


def load_scheduler_records(
    *,
    queue_path: str | Path | None = None,
    archive_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    state_dir = Path.home() / ".claude" / "scheduler"
    queue = Path(queue_path).expanduser() if queue_path else state_dir / "queue.json"
    archive = Path(archive_path).expanduser() if archive_path else state_dir / "queue_archive.jsonl"
    rows: list[dict[str, Any]] = []
    if archive.exists():
        with archive.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if queue.exists():
        payload = json.loads(queue.read_text(encoding="utf-8"))
        tasks = payload.get("tasks", payload if isinstance(payload, list) else [])
        rows.extend(dict(row) for row in tasks)
    return _dedupe_records(rows)


def _members_for_tasksets(taskset_names: Iterable[str]) -> tuple[TaskSetMember, ...]:
    seen: set[str] = set()
    members: list[TaskSetMember] = []
    for name in taskset_names:
        for member in taskset_by_name(str(name)).members:
            if member.workload_key in seen:
                continue
            seen.add(member.workload_key)
            members.append(member)
    return tuple(members)


def _mapped(workload_key: str, mode: str, reason: str, *, units: float | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"workload_key": workload_key, "mapping_mode": mode, "reason": reason}
    if units is not None and math.isfinite(float(units)) and float(units) > 0:
        out["units"] = float(units)
    return out


def _classified_units(cls: Mapping[str, Any], *, fallback: float) -> float:
    raw = cls.get("units")
    try:
        units = float(raw)
        if math.isfinite(units) and units > 0:
            return units
    except (TypeError, ValueError):
        pass
    return float(fallback)


def _is_freqduet_cpu_ablation_c17_32(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> bool:
    if est_vram > 0:
        return False
    if not (16.0 < float(cpu) <= 32.0):
        return False
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "").lower()
    if (
        project == "bamor"
        or "/bamor" in cwd
        or "run_bamor_diagnostic_shard.py" in cmd
        or "train_compare_baselines.py" in cmd
    ):
        return False
    return "run_freqduet_ablation.py" in cmd


def _freqduet_ablation_c9_16_units(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> float | None:
    if est_vram > 0:
        return None
    if not (8.0 < float(cpu) <= 16.0):
        return None
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project == "bamor" or "/bamor" in cwd:
        return None
    if "run_freqduet_ablation.py" not in cmd_lower:
        return None
    if "--worker-threads" in cmd_lower and "--worker-threads 1" not in cmd_lower:
        return None
    units = _parse_freqduet_ablation_units(cmd)
    return units if units is not None and units > 0 else None


def _parse_freqduet_ablation_units(cmd: str) -> float | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()

    def opt(name: str) -> str | None:
        if name not in tokens:
            return None
        idx = tokens.index(name)
        return tokens[idx + 1] if idx + 1 < len(tokens) else None

    episodes_raw = opt("--episodes")
    try:
        episodes = int(episodes_raw) if episodes_raw is not None else 0
    except ValueError:
        episodes = 0
    if episodes <= 0:
        return None

    job_start_raw = opt("--job-start")
    job_end_raw = opt("--job-end")
    if job_start_raw is not None or job_end_raw is not None:
        try:
            job_start = int(job_start_raw) if job_start_raw is not None else 0
            job_end = int(job_end_raw) if job_end_raw is not None else 0
        except ValueError:
            return None
        jobs = max(0, job_end - job_start)
        return float(jobs * episodes) if jobs > 0 else None

    configs_raw = opt("--configs") or ""
    seeds_raw = opt("--seeds") or ""
    if "$" in configs_raw or "$" in seeds_raw:
        return None
    configs = [part for part in configs_raw.split(",") if part.strip()]
    seeds = [part for part in seeds_raw.split(",") if part.strip()]
    jobs = len(configs) * len(seeds)
    return float(jobs * episodes) if jobs > 0 else None


def _is_simple_sac_sumo_eval_c_le2(*, row: Mapping[str, Any], est_vram: float, cpu: float) -> bool:
    if est_vram > 0:
        return False
    if float(cpu) > 2.0:
        return False
    project = str(row.get("project") or "").lower()
    cmd = str(row.get("cmd") or "")
    cmd_lower = cmd.lower()
    if project != "simplesac":
        return False
    if "run_multiseed_eval.sh" not in cmd_lower:
        return False
    if "bash -lc" in cmd_lower or "$" in cmd:
        return False
    return _parse_simple_sac_multiseed_eval_identity(cmd) is not None


def _parse_simple_sac_multiseed_eval_identity(cmd: str) -> tuple[str, int, float] | None:
    import shlex

    try:
        tokens = shlex.split(str(cmd))
    except ValueError:
        tokens = str(cmd).split()
    idx = next((i for i, token in enumerate(tokens) if token.endswith("run_multiseed_eval.sh")), None)
    if idx is None or idx + 3 >= len(tokens):
        return None
    method = str(tokens[idx + 1])
    if not method or method.startswith("-"):
        return None
    try:
        seed = int(tokens[idx + 2])
        od_scale = float(tokens[idx + 3])
    except ValueError:
        return None
    return (method, seed, od_scale)


def _is_freqduet_runner_v3_allfreq_alllayers_c9_16(
    *,
    row: Mapping[str, Any],
    est_vram: float,
    cpu: float,
) -> bool:
    if est_vram > 0:
        return False
    if not (8.0 < float(cpu) <= 16.0):
        return False
    project = str(row.get("project") or "").lower()
    cwd = str(row.get("cwd") or "").lower()
    cmd = str(row.get("cmd") or "").lower()
    text = " ".join(str(row.get(key) or "") for key in ("project", "signature", "description", "cmd", "cwd")).lower()
    if project == "bamor" or "/bamor" in cwd:
        return False
    if "runner_v3.py" not in cmd:
        return False
    if "configs_freqduet/f_allfreq_alllayers_hiro.yaml" not in cmd:
        return False
    return "freqduet" in text or "/transitduet/freqduet/" in cwd


def _dedupe_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    no_id = []
    for row in records:
        rec = dict(row)
        key = str(rec.get("id") or rec.get("task_id") or "")
        if not key:
            no_id.append(rec)
            continue
        old = by_id.get(key)
        if old is None or _record_timestamp(rec) >= _record_timestamp(old):
            by_id[key] = rec
    return list(by_id.values()) + no_id


def _window_end(records: list[Mapping[str, Any]], *, now_ts: float | None) -> float:
    if now_ts is not None and math.isfinite(float(now_ts)):
        return float(now_ts)
    timestamps = [_record_timestamp(row) for row in records if _record_timestamp(row) > 0]
    return max(timestamps, default=time.time())


def _record_timestamp(row: Mapping[str, Any]) -> float:
    return max(
        _as_float(row.get("finished_at")),
        _as_float(row.get("started_at")),
        _as_float(row.get("submitted_at")),
        _as_float(row.get("created_at")),
    )


def _submitted_at(row: Mapping[str, Any]) -> float:
    return _as_float(row.get("submitted_at"), _record_timestamp(row))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    cap = report.get("capacity") or {}
    rows = [
        "# Production Load Capacity Certificate",
        "",
        "```text",
        f"window_days = {report.get('window', {}).get('window_days')}",
        f"include_representative = {report.get('include_representative')}",
        f"record_count_window = {report.get('record_count_window')}",
        f"mapped_task_count = {report.get('mapped_task_count')}",
        f"representative_mapped_task_count = {report.get('representative_mapped_task_count')}",
        f"unmapped_task_count = {report.get('unmapped_task_count')}",
        f"mapped_fraction = {report.get('mapped_fraction')}",
        "```",
        "",
        "| Workload | Count | Lambda |",
        "|---|---:|---:|",
    ]
    counts = report.get("mapped_counts") or {}
    lam = report.get("lambda") or {}
    for key in sorted(set(counts) | set(lam)):
        rows.append(f"| `{key}` | {int(counts.get(key, 0))} | {float(lam.get(key, 0.0)):.9f} |")
    rows.extend(
        [
            "",
            "| Quantity | Value |",
            "|---|---:|",
            f"| `delta` | {float(cap.get('delta') or 0.0):.9f} |",
            f"| `mapped_capacity_usable_for_theorem` | {str(bool(report.get('mapped_capacity_usable_for_theorem'))).lower()} |",
            f"| `global_coverage_usable_for_theorem` | {str(bool(report.get('global_coverage_usable_for_theorem'))).lower()} |",
            f"| `usable_for_global_theorem` | {str(bool(report.get('usable_for_global_theorem'))).lower()} |",
            "",
            "## Unmapped Reasons",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted((report.get("unmapped_reasons") or {}).items()):
        rows.append(f"| `{reason}` | {int(count)} |")
    rows.extend(["", str(report.get("interpretation") or ""), ""])
    return "\n".join(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m algorithm.experiments.production_load_certificate")
    p.add_argument("--queue-path", default="")
    p.add_argument("--archive-path", default="")
    p.add_argument("--window-days", type=float, default=30.0)
    p.add_argument("--include-representative", action="store_true")
    p.add_argument("--tasksets", default=",".join(DEFAULT_TASKSETS))
    p.add_argument("--output", required=True)
    p.add_argument("--markdown-output", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_production_load_certificate(
        records=load_scheduler_records(
            queue_path=args.queue_path or None,
            archive_path=args.archive_path or None,
        ),
        window_days=args.window_days,
        include_representative=args.include_representative,
        taskset_names=[name.strip() for name in args.tasksets.split(",") if name.strip()],
    )
    _write_json(args.output, report)
    if args.markdown_output:
        p = Path(args.markdown_output).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_markdown(report), encoding="utf-8")
    print(args.output)
    return 0 if report.get("mapped_capacity_usable_for_theorem") else 2


if __name__ == "__main__":
    raise SystemExit(main())
