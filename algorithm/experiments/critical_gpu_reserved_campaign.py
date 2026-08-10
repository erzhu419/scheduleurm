"""Run frozen GPU completion protocols behind a TTL-bound scheduler reservation.

This wrapper does not alter any workload, progress parser, or stochastic gate.
It only keeps the local Scheduleurm watcher from filling a node between the
idle preflight and a long natural-completion wave.  Existing tasks drain
naturally; the wrapper never cancels, migrates, or signals them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from algorithm.experiments.critical_gpu_completion_campaign import (
    CALIBRATION_WAVES as EMPTY_CALIBRATION_WAVES,
    HOLDOUT_WAVE as EMPTY_HOLDOUT_WAVE,
    NODE_SPECS,
    TRAINING_WAVES as EMPTY_TRAINING_WAVES,
    _gpu_idle_snapshot,
)
from algorithm.experiments.critical_gpu_loaded_completion_campaign import (
    ALL_WAVES as LOADED_ALL_WAVES,
    ARTIFACT_ROOT,
    CAMPAIGN_DATE,
)
from algorithm.experiments.theorem_measurement_reservation import (
    MeasurementReservation,
)


QUEUE_FILE = Path.home() / ".claude" / "scheduler" / "queue.json"
NODE_PROBE_CACHE = Path.home() / ".claude" / "scheduler" / "node_probe_cache.json"
ACTIVE_TASK_STATES = frozenset({"running", "launching"})
EMPTY_ALL_WAVES = (
    *EMPTY_TRAINING_WAVES,
    *EMPTY_CALIBRATION_WAVES,
    EMPTY_HOLDOUT_WAVE,
)


def _parse_waves(raw: str, allowed: Sequence[int]) -> tuple[int, ...]:
    text = str(raw or "").strip()
    if not text:
        return ()
    values: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"invalid descending wave range {token!r}")
            values.extend(range(start, stop + 1))
        else:
            values.append(int(token))
    allowed_set = set(int(value) for value in allowed)
    invalid = sorted(set(values) - allowed_set)
    if invalid:
        raise ValueError(f"unsupported waves {invalid}; allowed={sorted(allowed_set)}")
    return tuple(dict.fromkeys(values))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _active_scheduler_tasks(node: str, *, queue_file: Path = QUEUE_FILE) -> list[dict[str, Any]]:
    payload = _read_json(queue_file)
    return [
        {
            "id": task.get("id"),
            "status": task.get("status"),
            "project": task.get("project"),
            "description": task.get("description"),
        }
        for task in payload.get("tasks") or []
        if isinstance(task, Mapping)
        and str(task.get("node") or task.get("assigned_node") or "") == str(node)
        and str(task.get("status") or "") in ACTIVE_TASK_STATES
    ]


def _reservation_acknowledged(
    node: str,
    reservation_id: str,
    created_at: float,
    *,
    cache_file: Path = NODE_PROBE_CACHE,
) -> bool:
    payload = _read_json(cache_file)
    try:
        if float(payload.get("ts") or 0.0) < float(created_at):
            return False
    except (TypeError, ValueError):
        return False
    state = (payload.get("states") or {}).get(node) or {}
    reservation = state.get("measurement_reservation") or {}
    return bool(
        state.get("measurement_reservation_active") is True
        and str(reservation.get("reservation_id") or "") == str(reservation_id)
    )


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _wait_for_scheduler_ack(
    *,
    node: str,
    reservation: Mapping[str, Any],
    lease: MeasurementReservation,
    poll_s: float,
) -> None:
    reservation_id = str(reservation["reservation_id"])
    created_at = float(reservation["created_at"])
    while True:
        lease.ensure_healthy()
        if _reservation_acknowledged(node, reservation_id, created_at):
            print(
                f"[{node}] scheduler acknowledged reservation {reservation_id}",
                flush=True,
            )
            return
        print(
            f"[{node}] waiting for scheduler reservation acknowledgement",
            flush=True,
        )
        time.sleep(max(1.0, float(poll_s)))


def _wait_for_clean_node(
    *,
    node: str,
    lease: MeasurementReservation,
    poll_s: float,
    consecutive_samples: int,
) -> dict[str, Any]:
    clean = 0
    latest: dict[str, Any] = {}
    while clean < max(1, int(consecutive_samples)):
        lease.ensure_healthy()
        scheduler_tasks = _active_scheduler_tasks(node)
        try:
            idle = _gpu_idle_snapshot(NODE_SPECS[node])
        except Exception as exc:
            idle = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        latest = {
            "sampled_at": time.time(),
            "gpu_idle": idle,
            "active_scheduler_tasks": scheduler_tasks,
        }
        if idle.get("ready") is True and not scheduler_tasks:
            clean += 1
        else:
            clean = 0
        print(
            f"[{node}] clean_guard={clean}/{max(1, int(consecutive_samples))} "
            f"gpu_ready={idle.get('ready') is True} active_tasks={len(scheduler_tasks)}",
            flush=True,
        )
        if clean < max(1, int(consecutive_samples)):
            time.sleep(max(1.0, float(poll_s)))
    return latest


def build_commands(
    *,
    node: str,
    empty_waves: Sequence[int],
    loaded_waves: Sequence[int],
    run_empty_gate: bool,
    run_loaded_gate: bool,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for wave in empty_waves:
        commands.append({
            "phase": "empty",
            "wave": int(wave),
            "argv": [
                sys.executable,
                "-m",
                "algorithm.experiments.critical_gpu_completion_campaign",
                "--node",
                node,
                "--wave",
                str(int(wave)),
                "--allow-launch",
            ],
        })
    if run_empty_gate:
        commands.append({
            "phase": "empty_gate",
            "wave": None,
            "argv": [
                sys.executable,
                "-m",
                "algorithm.experiments.critical_gpu_stochastic_lcb_gate",
                "--node",
                node,
            ],
        })
    for wave in loaded_waves:
        commands.append({
            "phase": "loaded",
            "wave": int(wave),
            "argv": [
                sys.executable,
                "-m",
                "algorithm.experiments.critical_gpu_loaded_wave_runner",
                "--node",
                node,
                "--wave",
                str(int(wave)),
                "--allow-launch",
                "--keep-remote-output",
            ],
        })
    if run_loaded_gate:
        commands.append({
            "phase": "loaded_gate",
            "wave": None,
            "argv": [
                sys.executable,
                "-m",
                "algorithm.experiments.critical_gpu_loaded_stochastic_lcb_gate",
                "--node",
                node,
            ],
        })
    return commands


def _max_command_attempts(phase: str, wave_retries: int) -> int:
    if int(wave_retries) < 0:
        raise ValueError("wave_retries must be nonnegative")
    return 1 + int(wave_retries) if str(phase) in {"empty", "loaded"} else 1


def run_reserved_campaign(
    *,
    node: str,
    empty_waves: Sequence[int] = (),
    loaded_waves: Sequence[int] = (),
    run_empty_gate: bool = False,
    run_loaded_gate: bool = False,
    poll_s: float = 60.0,
    clean_samples: int = 5,
    ttl_s: float = 900.0,
    wave_retries: int = 2,
    output: Path,
) -> dict[str, Any]:
    if node not in NODE_SPECS:
        raise ValueError(f"unsupported node {node!r}")
    commands = build_commands(
        node=node,
        empty_waves=empty_waves,
        loaded_waves=loaded_waves,
        run_empty_gate=run_empty_gate,
        run_loaded_gate=run_loaded_gate,
    )
    if not commands:
        raise ValueError("at least one wave or gate is required")
    report: dict[str, Any] = {
        "gate": "critical_gpu_reserved_campaign",
        "schema_version": 1,
        "status": "STARTING",
        "pass": False,
        "node": node,
        "empty_waves": list(empty_waves),
        "loaded_waves": list(loaded_waves),
        "run_empty_gate": bool(run_empty_gate),
        "run_loaded_gate": bool(run_loaded_gate),
        "started_at": time.time(),
        "ordinary_running_tasks_touched": False,
        "existing_tasks_drain_naturally": True,
        "legacy_scheduler_limits_bypassed_by_frozen_campaign": True,
        "external_manual_launches_still_rejected_by_process_audit": True,
        "wave_retries_after_initial_attempt": int(wave_retries),
        "gates_retried": False,
        "commands": [],
    }
    if int(wave_retries) < 0:
        raise ValueError("wave_retries must be nonnegative")
    _write_report(output, report)
    lease = MeasurementReservation(
        node=node,
        purpose="theorem-facing natural-completion GPU calibration",
        ttl_s=ttl_s,
    )
    try:
        with lease as reservation:
            report["reservation"] = dict(reservation)
            report["status"] = "WAIT_SCHEDULER_ACK"
            _write_report(output, report)
            _wait_for_scheduler_ack(
                node=node,
                reservation=reservation,
                lease=lease,
                poll_s=poll_s,
            )
            report["status"] = "WAIT_NODE_DRAIN"
            _write_report(output, report)
            report["clean_preflight"] = _wait_for_clean_node(
                node=node,
                lease=lease,
                poll_s=poll_s,
                consecutive_samples=clean_samples,
            )
            for index, command in enumerate(commands):
                max_attempts = _max_command_attempts(
                    str(command["phase"]), int(wave_retries)
                )
                command_passed = False
                for attempt in range(1, max_attempts + 1):
                    lease.ensure_healthy()
                    if (
                        command["phase"] in {"empty", "loaded"}
                        and (index > 0 or attempt > 1)
                    ):
                        _wait_for_clean_node(
                            node=node,
                            lease=lease,
                            poll_s=poll_s,
                            consecutive_samples=clean_samples,
                        )
                    wave_suffix = (
                        "" if command["wave"] is None
                        else f"_r{int(command['wave']):02d}"
                    )
                    attempt_suffix = (
                        "" if max_attempts == 1 else f"_try{attempt:02d}"
                    )
                    log_path = output.with_name(
                        f"{output.stem}_{index:02d}_{command['phase']}"
                        f"{wave_suffix}{attempt_suffix}.log"
                    )
                    row = {
                        **command,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "started_at": time.time(),
                        "log_path": str(log_path),
                    }
                    report["commands"].append(row)
                    report["status"] = f"RUNNING_{str(command['phase']).upper()}"
                    _write_report(output, report)
                    print(
                        f"[{node}] start {command['phase']} wave={command['wave']} "
                        f"attempt={attempt}/{max_attempts} log={log_path}",
                        flush=True,
                    )
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        completed = subprocess.run(
                            command["argv"],
                            cwd=str(Path(__file__).resolve().parents[2]),
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    row["finished_at"] = time.time()
                    row["returncode"] = int(completed.returncode)
                    row["pass"] = completed.returncode == 0
                    _write_report(output, report)
                    print(
                        f"[{node}] finish {command['phase']} wave={command['wave']} "
                        f"attempt={attempt}/{max_attempts} rc={completed.returncode}",
                        flush=True,
                    )
                    if completed.returncode == 0:
                        command_passed = True
                        break
                    if attempt < max_attempts:
                        print(
                            f"[{node}] retry {command['phase']} wave={command['wave']} "
                            "after fail-closed nonzero return",
                            flush=True,
                        )
                if not command_passed:
                    report["status"] = "COMMAND_FAILED"
                    report["failed_command_index"] = index
                    report["failed_after_attempts"] = max_attempts
                    report["finished_at"] = time.time()
                    _write_report(output, report)
                    return report
            lease.ensure_healthy()
            report["status"] = "PASS"
            report["pass"] = True
            report["finished_at"] = time.time()
            _write_report(output, report)
            return report
    except BaseException as exc:
        report["status"] = "ABORTED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at"] = time.time()
        _write_report(output, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, choices=tuple(NODE_SPECS))
    parser.add_argument("--empty-waves", default="")
    parser.add_argument("--loaded-waves", default="")
    parser.add_argument("--run-empty-gate", action="store_true")
    parser.add_argument("--run-loaded-gate", action="store_true")
    parser.add_argument("--poll-s", type=float, default=60.0)
    parser.add_argument("--clean-samples", type=int, default=5)
    parser.add_argument("--ttl-s", type=float, default=900.0)
    parser.add_argument("--wave-retries", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    empty_waves = _parse_waves(args.empty_waves, EMPTY_ALL_WAVES)
    loaded_waves = _parse_waves(args.loaded_waves, LOADED_ALL_WAVES)
    output = args.output or (
        ARTIFACT_ROOT
        / f"critical_gpu_reserved_campaign_{args.node}_{CAMPAIGN_DATE}.json"
    )
    report = run_reserved_campaign(
        node=args.node,
        empty_waves=empty_waves,
        loaded_waves=loaded_waves,
        run_empty_gate=args.run_empty_gate,
        run_loaded_gate=args.run_loaded_gate,
        poll_s=args.poll_s,
        clean_samples=args.clean_samples,
        ttl_s=args.ttl_s,
        wave_retries=args.wave_retries,
        output=output,
    )
    print(json.dumps({"status": report["status"], "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
