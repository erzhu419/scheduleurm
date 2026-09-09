"""Physical checkpoint/sync/resume migration-cost gate.

This gate measures the cost decomposition used by migration actions without
touching production jobs.  It creates controlled checkpoint payloads, copies
them between explicitly named nodes, resumes by reading/verifying the payload,
and records the measured terms that become ``K_migration``.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from algorithm.theorem_dispatch.migration import MigrationCost, build_migration_action_row


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "md" / "experiment_artifacts"


@dataclass(frozen=True)
class MigrationSpec:
    spec_id: str
    workload_key: str
    family: str
    source_node: str
    dest_node: str
    checkpoint_mib: int
    total_work_units: float
    current_rate: float
    target_rate: float
    checkpoint_policy: str
    sync_policy: str = "tar_ssh"
    resume_policy: str = "hash_verify_resume"


DEFAULT_SPECS = (
    MigrationSpec(
        spec_id="cnn_gpu_checkpoint",
        workload_key="gpu_cnn_torch_resnet50",
        family="pure_gpu",
        source_node="jtl110gpu",
        dest_node="jtl311linux",
        checkpoint_mib=96,
        total_work_units=1000.0,
        current_rate=20.0,
        target_rate=28.0,
        checkpoint_policy="controlled_cnn_state_payload",
    ),
    MigrationSpec(
        spec_id="resac_halfcheetah_checkpoint",
        workload_key="hybrid_rl_resac_halfcheetah",
        family="hybrid_rl",
        source_node="jtl110gpu",
        dest_node="jtl311linux",
        checkpoint_mib=48,
        total_work_units=80.0,
        current_rate=0.10110832175,
        target_rate=0.1440999673,
        checkpoint_policy="controlled_resac_cycle_state_payload",
    ),
    MigrationSpec(
        spec_id="cpu_freqduet_surrogate_checkpoint",
        workload_key="freqduet_cpu_surrogate",
        family="pure_cpu",
        source_node="node003",
        dest_node="node005",
        checkpoint_mib=32,
        total_work_units=200.0,
        current_rate=12.530672,
        target_rate=16.392399,
        checkpoint_policy="controlled_cpu_state_payload",
    ),
)


def build_live_checkpoint_migration_cost_gate(
    *,
    allow_launch: bool = False,
    specs: Iterable[str] | None = None,
    extra_cpu_pairs: Iterable[str] | None = None,
    run_id: str = "live_checkpoint_migration_cost_20260629",
) -> dict[str, Any]:
    selected_ids = {str(x) for x in specs or []}
    universe = [*DEFAULT_SPECS, *_extra_cpu_specs(extra_cpu_pairs or ())]
    selected = [spec for spec in universe if not selected_ids or spec.spec_id in selected_ids]
    points = (0.25, 0.50, 0.75)
    rows: list[dict[str, Any]] = []
    for spec in selected:
        for point in points:
            rows.append(
                _measure_one(spec, point=point, run_id=run_id)
                if allow_launch
                else _manifest_one(spec, point=point, run_id=run_id)
            )
    measured = [row for row in rows if row.get("measurement_valid")]
    blocked = [row for row in rows if row.get("launched") and not row.get("measurement_valid")]
    pending = [row for row in rows if not row.get("launched")]
    return {
        "gate": "live_checkpoint_migration_cost_gate",
        "run_id": run_id,
        "allow_launch": bool(allow_launch),
        "pass": bool(measured) and not pending,
        "status": (
            "LIVE_CHECKPOINT_MIGRATION_COST_READY"
            if measured and not pending and not blocked
            else ("LIVE_CHECKPOINT_MIGRATION_COST_PARTIAL" if measured else ("DRY_RUN_READY" if not allow_launch else "LIVE_CHECKPOINT_MIGRATION_COST_OPEN"))
        ),
        "row_count": len(rows),
        "measured_count": len(measured),
        "blocked_count": len(blocked),
        "pending_count": len(pending),
        "migration_points": list(points),
        "rows": rows,
        "measured_rows": measured,
        "blocked_rows": blocked,
        "pending_rows": pending,
        "scope": (
            "Physical checkpoint/sync/resume timings for controlled benchmark "
            "payloads only.  No production task is migrated."
        ),
    }


def _extra_cpu_specs(pairs: Iterable[str]) -> tuple[MigrationSpec, ...]:
    out: list[MigrationSpec] = []
    for pair in pairs:
        text = str(pair or "").strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(f"extra CPU pair must be source:dest, got {text!r}")
        src, dst = [part.strip() for part in text.split(":", 1)]
        if not src or not dst:
            raise ValueError(f"extra CPU pair must be source:dest, got {text!r}")
        safe_src = src.replace("-", "_")
        safe_dst = dst.replace("-", "_")
        out.append(
            MigrationSpec(
                spec_id=f"cpu_freqduet_surrogate_checkpoint_{safe_src}_to_{safe_dst}",
                workload_key="freqduet_cpu_surrogate",
                family="pure_cpu",
                source_node=src,
                dest_node=dst,
                checkpoint_mib=32,
                total_work_units=200.0,
                current_rate=12.530672,
                target_rate=16.392399,
                checkpoint_policy="controlled_cpu_state_payload",
            )
        )
    return tuple(out)


def _manifest_one(spec: MigrationSpec, *, point: float, run_id: str) -> dict[str, Any]:
    return {
        "spec_id": spec.spec_id,
        "run_id": run_id,
        "progress_fraction": float(point),
        "source_node": spec.source_node,
        "dest_node": spec.dest_node,
        "workload_key": spec.workload_key,
        "family": spec.family,
        "checkpoint_mib": int(spec.checkpoint_mib),
        "checkpoint_policy": spec.checkpoint_policy,
        "sync_policy": spec.sync_policy,
        "resume_policy": spec.resume_policy,
        "launched": False,
        "measurement_valid": False,
        "status": "PENDING_ALLOW_LAUNCH",
    }


def _measure_one(spec: MigrationSpec, *, point: float, run_id: str) -> dict[str, Any]:
    row = _manifest_one(spec, point=point, run_id=run_id)
    remote_root = f"/tmp/scheduleurm_live_migration/{run_id}/{spec.spec_id}/p{int(point * 100):02d}"
    source_dir = f"{remote_root}/source_payload"
    dest_parent = f"{remote_root}/dest_parent"
    dest_dir = f"{dest_parent}/source_payload"
    try:
        source_ok = _ssh_ok(spec.source_node)
        dest_ok = _ssh_ok(spec.dest_node)
        if not source_ok or not dest_ok:
            return {
                **row,
                "launched": True,
                "status": "WAIT_NODE_ACCESS",
                "measurement_valid": False,
                "block_reason": f"source_ssh={source_ok},dest_ssh={dest_ok}",
            }

        checkpoint = _time_call(
            _remote_cmd(spec.source_node, _remote_python(_checkpoint_script(spec, point, source_dir))),
            timeout_s=180,
        )
        if checkpoint["returncode"] != 0:
            return {**row, "launched": True, "status": "CHECKPOINT_FAILED", "measurement_valid": False, "checkpoint_result": checkpoint}
        meta = json.loads(str(checkpoint["stdout"]).strip().splitlines()[-1])

        sync_cmd = (
            f"{_remote_pipe_endpoint(spec.source_node, 'tar -C ' + shlex.quote(str(Path(source_dir).parent)) + ' -cf - source_payload')} "
            f"| {_remote_pipe_endpoint(spec.dest_node, 'rm -rf ' + shlex.quote(dest_parent) + ' && mkdir -p ' + shlex.quote(dest_parent) + ' && tar -C ' + shlex.quote(dest_parent) + ' -xf -')}"
        )
        sync = _time_shell_retry(sync_cmd, timeout_s=240, attempts=3)
        if sync["returncode"] != 0:
            return {**row, "launched": True, "status": "SYNC_FAILED", "measurement_valid": False, "checkpoint_meta": meta, "sync_result": sync}

        staging = _time_call(
            _remote_cmd(
                spec.dest_node,
                "python3 -V >/dev/null && mkdir -p " + shlex.quote(f"{remote_root}/resume_logs"),
            ),
            timeout_s=60,
        )
        resume = _time_call(
            _remote_cmd(spec.dest_node, _remote_python(_resume_script(dest_dir))),
            timeout_s=180,
        )
        if resume["returncode"] != 0:
            return {
                **row,
                "launched": True,
                "status": "RESUME_FAILED",
                "measurement_valid": False,
                "checkpoint_meta": meta,
                "sync_result": sync,
                "staging_result": staging,
                "resume_result": resume,
            }
        resume_meta = json.loads(str(resume["stdout"]).strip().splitlines()[-1])
        hash_ok = bool(resume_meta.get("hash_ok"))
        risk = 0.0 if hash_ok else 50.0
        cost = MigrationCost(
            checkpoint_flush_s=float(meta.get("checkpoint_flush_s") or checkpoint["elapsed_s"]),
            sync_s=float(sync["elapsed_s"]),
            environment_staging_s=float(staging["elapsed_s"]),
            resume_warmup_s=float(resume_meta.get("resume_warmup_s") or resume["elapsed_s"]),
            lost_work_s=0.0,
            risk_penalty_units=risk,
        )
        remaining = float(spec.total_work_units) * (1.0 - float(point))
        action_row = build_migration_action_row(
            task={
                "id": spec.spec_id,
                "controlled_benchmark": True,
                "checkpoint_verified": hash_ok,
                "resume_verified": hash_ok,
            },
            workload_key=spec.workload_key,
            from_node=spec.source_node,
            to_node=spec.dest_node,
            current_rate=spec.current_rate,
            target_lower_service=spec.target_rate,
            remaining_work=remaining,
            progress_fraction=float(point),
            cost=cost,
        )
        return {
            **row,
            "launched": True,
            "measurement_valid": bool(hash_ok),
            "status": "MEASURED" if hash_ok else "HASH_MISMATCH",
            "checkpoint_meta": meta,
            "resume_meta": resume_meta,
            "checkpoint_flush_s": cost.checkpoint_flush_s,
            "sync_s": cost.sync_s,
            "environment_staging_s": cost.environment_staging_s,
            "resume_warmup_s": cost.resume_warmup_s,
            "lost_work_s": cost.lost_work_s,
            "risk_penalty_units": cost.risk_penalty_units,
            "total_time_s": cost.total_time_s,
            "total_penalty_units": cost.total_penalty_units,
            "remaining_work": remaining,
            "current_rate": float(spec.current_rate),
            "target_rate": float(spec.target_rate),
            "beneficial_by_threshold": bool(action_row.get("beneficial_by_threshold")),
            "theorem_ready": bool(action_row.get("theorem_ready")),
            "migration_action_row": action_row,
        }
    except Exception as exc:
        return {
            **row,
            "launched": True,
            "measurement_valid": False,
            "status": "EXCEPTION",
            "block_reason": repr(exc),
        }


def _ssh_ok(node: str) -> bool:
    result = subprocess.run(
        _remote_cmd(node, "true", connect_timeout=8),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    return result.returncode == 0


def _checkpoint_script(spec: MigrationSpec, point: float, out_dir: str) -> str:
    return f"""
import hashlib, json, os, time
out = {out_dir!r}
if not os.path.isdir(out):
    os.makedirs(out)
payload = os.path.join(out, 'checkpoint.bin')
meta_path = os.path.join(out, 'checkpoint.json')
size = int({int(spec.checkpoint_mib)}) * 1024 * 1024
chunk = (({spec.spec_id!r} + '|' + {spec.family!r}).encode('utf-8') * 4096)[:1024 * 1024]
h = hashlib.sha256()
start = time.time()
with open(payload, 'wb') as fh:
    remaining = size
    while remaining > 0:
        data = chunk[:min(len(chunk), remaining)]
        fh.write(data)
        h.update(data)
        remaining -= len(data)
    fh.flush()
    os.fsync(fh.fileno())
elapsed = time.time() - start
meta = {{
    'spec_id': {spec.spec_id!r},
    'family': {spec.family!r},
    'workload_key': {spec.workload_key!r},
    'progress_fraction': float({float(point)}),
    'checkpoint_mib': int({int(spec.checkpoint_mib)}),
    'payload_bytes': size,
    'sha256': h.hexdigest(),
    'checkpoint_flush_s': elapsed,
    'created_at': time.time(),
}}
with open(meta_path, 'w') as fh:
    fh.write(json.dumps(meta, sort_keys=True))
print(json.dumps(meta, sort_keys=True))
"""


def _resume_script(checkpoint_dir: str) -> str:
    return f"""
import hashlib, json, time
import os
root = {checkpoint_dir!r}
with open(os.path.join(root, 'checkpoint.json'), 'r') as fh:
    meta = json.loads(fh.read())
payload = os.path.join(root, 'checkpoint.bin')
h = hashlib.sha256()
start = time.time()
with open(payload, 'rb') as fh:
    for chunk in iter(lambda: fh.read(1024 * 1024), b''):
        h.update(chunk)
elapsed = time.time() - start
out = {{
    'spec_id': meta.get('spec_id'),
    'payload_bytes': meta.get('payload_bytes'),
    'sha256': h.hexdigest(),
    'expected_sha256': meta.get('sha256'),
    'hash_ok': h.hexdigest() == meta.get('sha256'),
    'resume_warmup_s': elapsed,
    'resumed_at': time.time(),
}}
with open(os.path.join(root, 'resume.json'), 'w') as fh:
    fh.write(json.dumps(out, sort_keys=True))
print(json.dumps(out, sort_keys=True))
"""


def _remote_python(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    payload = shlex.quote(f"import base64; exec(base64.b64decode({encoded!r}))")
    return f"PY=$(command -v python3 2>/dev/null || command -v python); \"$PY\" -c {payload}"


def _remote_cmd(node: str, inner_command: str, *, connect_timeout: int = 10) -> list[str]:
    node_text = str(node)
    base = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(connect_timeout)}"]
    if _is_hpc_compute_node(node_text):
        login = "zhengliang01@202.197.46.16"
        relay = f"sudo -n ssh {shlex.quote(node_text)} {shlex.quote(inner_command)}"
        return [*base, "-J", "jtl110gpu2,jtl110gpu", login, relay]
    return [*base, node_text, inner_command]


def _remote_pipe_endpoint(node: str, inner_command: str, *, connect_timeout: int = 10) -> str:
    return " ".join(shlex.quote(part) for part in _remote_cmd(node, inner_command, connect_timeout=connect_timeout))


def _is_hpc_compute_node(node: str) -> bool:
    return str(node) in {"node001", "node002", "node003", "node004", "node005", "node006"}


def _time_call(cmd: list[str], *, timeout_s: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return {
        "returncode": int(proc.returncode),
        "elapsed_s": time.perf_counter() - start,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
    }


def _time_shell(cmd: str, *, timeout_s: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    return {
        "returncode": int(proc.returncode),
        "elapsed_s": time.perf_counter() - start,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
    }


def _time_shell_retry(cmd: str, *, timeout_s: int, attempts: int = 3) -> dict[str, Any]:
    tries = []
    for attempt in range(1, max(1, int(attempts)) + 1):
        result = _time_shell(cmd, timeout_s=timeout_s)
        result["attempt"] = attempt
        tries.append(result)
        if int(result.get("returncode") or 0) == 0:
            return {**result, "attempts": tries}
        time.sleep(min(2.0 * attempt, 5.0))
    last = dict(tries[-1])
    last["attempts"] = tries
    return last


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Live Checkpoint Migration Cost Gate",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Pass: `{str(bool(report.get('pass'))).lower()}`",
        f"- Measured rows: `{report.get('measured_count')}`",
        f"- Blocked rows: `{report.get('blocked_count')}`",
        f"- Pending rows: `{report.get('pending_count')}`",
        "",
        "| Spec | Family | Point | Nodes | Status | K total | ckpt | sync | stage | resume | Beneficial |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("rows") or []:
        lines.append(
            f"| `{row.get('spec_id')}` | `{row.get('family')}` | "
            f"{float(row.get('progress_fraction') or 0.0):.2f} | "
            f"`{row.get('source_node')}->{row.get('dest_node')}` | `{row.get('status')}` | "
            f"{float(row.get('total_penalty_units') or 0.0):.3f} | "
            f"{float(row.get('checkpoint_flush_s') or 0.0):.3f} | "
            f"{float(row.get('sync_s') or 0.0):.3f} | "
            f"{float(row.get('environment_staging_s') or 0.0):.3f} | "
            f"{float(row.get('resume_warmup_s') or 0.0):.3f} | "
            f"{str(bool(row.get('beneficial_by_threshold'))).lower()} |"
        )
    lines.extend(["", str(report.get("scope") or ""), ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-launch", action="store_true")
    parser.add_argument("--specs", default="")
    parser.add_argument("--extra-cpu-pairs", default="")
    parser.add_argument("--run-id", default="live_checkpoint_migration_cost_20260629")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "live_checkpoint_migration_cost_gate_20260629.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "md" / "live_checkpoint_migration_cost_gate_20260629.md")
    args = parser.parse_args()
    specs = [part.strip() for part in str(args.specs or "").split(",") if part.strip()]
    extra_cpu_pairs = [part.strip() for part in str(args.extra_cpu_pairs or "").split(",") if part.strip()]
    report = build_live_checkpoint_migration_cost_gate(
        allow_launch=bool(args.allow_launch),
        specs=specs or None,
        extra_cpu_pairs=extra_cpu_pairs or None,
        run_id=str(args.run_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown_report(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
