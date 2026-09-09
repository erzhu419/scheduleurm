"""Durable lease orchestration for full-factorial parallel probe waves.

Preparing or recovering a wave never launches work.  A single controlled
probe may be launched only through ``launch_leased_probe`` with explicit
authorization, and only when its argv targets a whitelisted experiment module.
Ordinary scheduler/user-task launch commands are rejected.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .full_factorial_parallel_probe_plan import (
    REPO_ROOT,
    T0_CPU_RUNNER,
    T0_GPU_RUNNER,
    T2_MARGINAL_RUNNER,
)


SCHEMA_VERSION = 1
CONTROLLED_PROBE_MODULES = frozenset({T0_CPU_RUNNER, T0_GPU_RUNNER, T2_MARGINAL_RUNNER})
ACTIVE_LEASE_STATES = frozenset({"leased", "launching", "running", "recovery_required"})
TERMINAL_LEASE_STATES = frozenset({"succeeded", "failed", "released", "expired"})


class LeaseConflict(RuntimeError):
    """Raised when a row, node, owner, or wave already has an active lease."""


class UnsafeLaunchCommand(ValueError):
    """Raised when a plan command is outside the controlled probe boundary."""


def prepare_or_resume_wave(
    plan: Mapping[str, Any],
    *,
    state_path: Path,
    owner: str,
    lease_ttl_s: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Atomically create leases, or return the existing same-owner wave.

    The operation is idempotent for the same plan and owner.  Expired leases
    are not silently reclaimed, and an expired running launch is moved to
    ``recovery_required`` so a restart cannot duplicate remote work.
    """

    owner = str(owner or "").strip()
    if not owner:
        raise ValueError("owner is required")
    ttl = _positive_ttl(lease_ttl_s)
    timestamp = time.time() if now is None else float(now)
    planned = _flatten_plan(plan)
    wave_id = str(plan.get("wave_id") or _plan_fingerprint(planned))
    fingerprint = _plan_fingerprint(planned)

    with _state_lock(state_path):
        existing = _read_state_unlocked(state_path)
        if existing:
            _apply_expiry_recovery(existing, now=timestamp)
            if str(existing.get("wave_id") or "") != wave_id:
                if _active_leases(existing):
                    raise LeaseConflict("another wave still owns active row/node leases")
                existing = {}
            elif str(existing.get("plan_fingerprint") or "") != fingerprint:
                raise LeaseConflict("the persisted wave id has a different plan fingerprint")

        if not existing:
            existing = {
                "schema_version": SCHEMA_VERSION,
                "gate": "full_factorial_parallel_orchestrator",
                "wave_id": wave_id,
                "plan_fingerprint": fingerprint,
                "design_csv": str(plan.get("design_csv") or ""),
                "automatic_launch": False,
                "created_at": timestamp,
                "updated_at": timestamp,
                "leases": [],
                "lease_history": [],
            }

        by_row_lock = {
            str(lease.get("row_lock_key") or ""): lease
            for lease in existing.get("leases") or []
        }
        for item in planned:
            prior = by_row_lock.get(item["row_lock_key"])
            if prior is not None:
                if str(prior.get("owner") or "") != owner and str(prior.get("status") or "") in ACTIVE_LEASE_STATES:
                    raise LeaseConflict(f"row {item['row_id']} is leased by {prior.get('owner')}")
                continue
            _assert_no_lock_conflict(existing.get("leases") or [], item)
            lease = _new_lease(item, owner=owner, ttl=ttl, now=timestamp)
            existing["leases"].append(lease)
            by_row_lock[item["row_lock_key"]] = lease

        existing["updated_at"] = timestamp
        _refresh_state_summary(existing)
        _atomic_write_json(state_path, existing)
        return _json_copy(existing)


def recover_wave(*, state_path: Path, now: float | None = None) -> dict[str, Any]:
    """Apply conservative expiry recovery without reclaiming or launching."""

    timestamp = time.time() if now is None else float(now)
    with _state_lock(state_path):
        state = _require_state(state_path)
        _apply_expiry_recovery(state, now=timestamp)
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)
        return _json_copy(state)


def renew_wave_leases(
    *,
    state_path: Path,
    owner: str,
    lease_ttl_s: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Renew active, non-recovery leases owned by ``owner``."""

    owner = str(owner or "").strip()
    if not owner:
        raise ValueError("owner is required")
    ttl = _positive_ttl(lease_ttl_s)
    timestamp = time.time() if now is None else float(now)
    with _state_lock(state_path):
        state = _require_state(state_path)
        _apply_expiry_recovery(state, now=timestamp)
        renewed = 0
        for lease in state.get("leases") or []:
            if str(lease.get("owner") or "") != owner:
                continue
            if str(lease.get("status") or "") not in {"leased", "launching", "running"}:
                continue
            lease["expires_at"] = timestamp + ttl
            lease["renewed_at"] = timestamp
            renewed += 1
        state["last_renewed_count"] = renewed
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)
        return _json_copy(state)


def reclaim_expired_lease(
    *,
    state_path: Path,
    lease_id: str,
    owner: str,
    lease_ttl_s: float = 3600.0,
    now: float | None = None,
) -> dict[str, Any]:
    """Explicitly reclaim a never-launched expired lease with a new lease id."""

    owner = str(owner or "").strip()
    if not owner:
        raise ValueError("owner is required")
    ttl = _positive_ttl(lease_ttl_s)
    timestamp = time.time() if now is None else float(now)
    with _state_lock(state_path):
        state = _require_state(state_path)
        _apply_expiry_recovery(state, now=timestamp)
        lease = _find_lease(state, lease_id)
        if str(lease.get("status") or "") != "expired":
            raise LeaseConflict("only an expired, never-running lease can be reclaimed")
        state.setdefault("lease_history", []).append(_json_copy(lease))
        lease["previous_lease_id"] = str(lease.get("lease_id") or "")
        lease["lease_id"] = f"lease-{uuid.uuid4().hex}"
        lease["owner"] = owner
        lease["status"] = "leased"
        lease["attempt"] = int(lease.get("attempt") or 1) + 1
        lease["leased_at"] = timestamp
        lease["expires_at"] = timestamp + ttl
        lease.pop("expired_at", None)
        lease.pop("recovery_reason", None)
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)
        return _json_copy(lease)


def record_lease_status(
    *,
    state_path: Path,
    lease_id: str,
    owner: str,
    status: str,
    result_path: str = "",
    error: str = "",
    now: float | None = None,
) -> dict[str, Any]:
    """Record an explicit terminal or recovery disposition for one lease."""

    target = str(status or "")
    if target not in {"succeeded", "failed", "released", "recovery_required"}:
        raise ValueError("status must be succeeded, failed, released, or recovery_required")
    timestamp = time.time() if now is None else float(now)
    with _state_lock(state_path):
        state = _require_state(state_path)
        _apply_expiry_recovery(state, now=timestamp)
        lease = _find_lease(state, lease_id)
        _assert_owner(lease, owner)
        current = str(lease.get("status") or "")
        if current in TERMINAL_LEASE_STATES and current != target:
            raise LeaseConflict(f"terminal lease is already {current}")
        lease["status"] = target
        lease["updated_at"] = timestamp
        if target in TERMINAL_LEASE_STATES:
            lease["finished_at"] = timestamp
        if result_path:
            lease["result_path"] = str(result_path)
        if error:
            lease["error"] = str(error)
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)
        return _json_copy(lease)


def validate_controlled_probe_command(command: str | Sequence[str]) -> list[str]:
    """Return safe argv or reject anything outside controlled probe modules."""

    argv = shlex.split(command) if isinstance(command, str) else [str(value) for value in command]
    if len(argv) < 4 or Path(argv[0]).name not in {"python", "python3"} or argv[1] != "-m":
        raise UnsafeLaunchCommand("controlled probes must use `python3 -m <module>`")
    module = argv[2]
    if module not in CONTROLLED_PROBE_MODULES:
        raise UnsafeLaunchCommand(f"module {module!r} is not a controlled full-factorial runner")
    if "--allow-launch" not in argv:
        raise UnsafeLaunchCommand("controlled launch command is missing --allow-launch")
    if any(token in {";", "&&", "||", "|", "`"} for token in argv):
        raise UnsafeLaunchCommand("shell control operators are not allowed")
    return argv


def launch_leased_probe(
    *,
    state_path: Path,
    lease_id: str,
    owner: str,
    allow_controlled_launch: bool = False,
    now: float | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Launch exactly one leased controlled probe; never a general user task."""

    if not allow_controlled_launch:
        raise PermissionError("controlled launch requires explicit allow_controlled_launch=True")
    timestamp = time.time() if now is None else float(now)
    with _state_lock(state_path):
        state = _require_state(state_path)
        _apply_expiry_recovery(state, now=timestamp)
        lease = _find_lease(state, lease_id)
        _assert_owner(lease, owner)
        if str(lease.get("status") or "") != "leased":
            raise LeaseConflict(f"lease is {lease.get('status')}, not launchable")
        argv = validate_controlled_probe_command(str(lease.get("command") or ""))
        lease["status"] = "launching"
        lease["launching_at"] = timestamp
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)

    try:
        process = popen_factory(argv, cwd=str(REPO_ROOT), start_new_session=True)
    except Exception as exc:
        with _state_lock(state_path):
            state = _require_state(state_path)
            lease = _find_lease(state, lease_id)
            if str(lease.get("status") or "") == "launching":
                lease["status"] = "failed"
                lease["finished_at"] = timestamp
                lease["error"] = f"launch_failed:{type(exc).__name__}:{exc}"
            state["updated_at"] = timestamp
            _refresh_state_summary(state)
            _atomic_write_json(state_path, state)
        raise

    with _state_lock(state_path):
        state = _require_state(state_path)
        lease = _find_lease(state, lease_id)
        if str(lease.get("status") or "") != "launching":
            raise LeaseConflict("lease changed while the controlled process was starting")
        lease["status"] = "running"
        lease["pid"] = int(process.pid)
        lease["started_at"] = timestamp
        state["updated_at"] = timestamp
        _refresh_state_summary(state)
        _atomic_write_json(state_path, state)
        return _json_copy(lease)


def load_wave_state(*, state_path: Path) -> dict[str, Any]:
    with _state_lock(state_path, shared=True):
        return _json_copy(_require_state(state_path))


def _flatten_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if str(plan.get("gate") or "") != "full_factorial_parallel_probe_plan":
        raise ValueError("not a full-factorial parallel probe plan")
    out: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    seen_factors: set[str] = set()
    seen_nodes: set[str] = set()
    for lane in plan.get("lanes") or []:
        rows = list(lane.get("rows") or [])
        commands = list(lane.get("commands") or [])
        assignments = list(lane.get("assignment_csvs") or [])
        if len(rows) > 1:
            raise ValueError(f"lane {lane.get('lane_id')} contains more than one wave row")
        if len(rows) != len(commands):
            raise ValueError(f"lane {lane.get('lane_id')} has mismatched rows and commands")
        for index, (row, command) in enumerate(zip(rows, commands)):
            validate_controlled_probe_command(str(command))
            row_id = str(row.get("row_id") or "")
            row_lock = str(row.get("row_lock_key") or "")
            factor_lock = str(row.get("factor_lock_key") or "")
            node_lock = str(lane.get("node_lock") or row.get("node_lock") or "")
            if not all((row_id, row_lock, factor_lock, node_lock)):
                raise ValueError("planned rows require row, factor, and node lock keys")
            if row_lock in seen_rows or factor_lock in seen_factors or node_lock in seen_nodes:
                raise ValueError("plan contains a duplicate row, factor, or physical node lock")
            seen_rows.add(row_lock)
            seen_factors.add(factor_lock)
            seen_nodes.add(node_lock)
            factor_values = row.get("factor_values") or []
            if not isinstance(factor_values, list):
                raise ValueError("factor_values must be a list")
            out.append(
                {
                    "row_id": row_id,
                    "row_lock_key": row_lock,
                    "factor_lock_key": factor_lock,
                    "factor_values": [str(value) for value in factor_values],
                    "node_lock": node_lock,
                    "lane_id": str(lane.get("lane_id") or ""),
                    "lane_mode": str(lane.get("lane_mode") or ""),
                    "observed_resident": bool(lane.get("observed_resident")),
                    "runner_route": str(row.get("runner_route") or ""),
                    "command": str(command),
                    "assignment_csv": str(assignments[index]) if index < len(assignments) else "",
                }
            )
    if len(out) != int(plan.get("selected_row_count") or 0):
        raise ValueError("selected_row_count does not match lane assignments")
    return out


def _new_lease(item: Mapping[str, Any], *, owner: str, ttl: float, now: float) -> dict[str, Any]:
    return {
        **dict(item),
        "lease_id": f"lease-{uuid.uuid4().hex}",
        "owner": owner,
        "status": "leased",
        "attempt": 1,
        "leased_at": now,
        "expires_at": now + ttl,
        "automatic_launch": False,
    }


def _assert_no_lock_conflict(leases: Iterable[Mapping[str, Any]], item: Mapping[str, Any]) -> None:
    for lease in leases:
        if str(lease.get("status") or "") not in ACTIVE_LEASE_STATES:
            continue
        for key in ("row_lock_key", "factor_lock_key", "node_lock"):
            if str(lease.get(key) or "") == str(item.get(key) or ""):
                raise LeaseConflict(f"active {key} conflict for {item.get('row_id')}")


def _apply_expiry_recovery(state: dict[str, Any], *, now: float) -> None:
    for lease in state.get("leases") or []:
        status = str(lease.get("status") or "")
        if status not in {"leased", "launching", "running"}:
            continue
        if float(lease.get("expires_at") or 0.0) > now:
            continue
        if status == "leased":
            lease["status"] = "expired"
            lease["expired_at"] = now
            lease["recovery_reason"] = "unlaunched_lease_expired"
        else:
            lease["status"] = "recovery_required"
            lease["recovery_required_at"] = now
            lease["recovery_reason"] = f"{status}_lease_expired_no_automatic_relaunch"


def _refresh_state_summary(state: dict[str, Any]) -> None:
    counts: dict[str, int] = {}
    for lease in state.get("leases") or []:
        status = str(lease.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    state["lease_counts"] = counts
    state["lease_count"] = sum(counts.values())
    state["active_lease_count"] = sum(counts.get(status, 0) for status in ACTIVE_LEASE_STATES)
    if counts and counts.get("succeeded", 0) == sum(counts.values()):
        state["status"] = "COMPLETE"
    elif counts.get("recovery_required", 0):
        state["status"] = "RECOVERY_REQUIRED"
    elif counts.get("running", 0) or counts.get("launching", 0):
        state["status"] = "RUNNING"
    elif counts.get("leased", 0):
        state["status"] = "LEASED"
    elif counts:
        state["status"] = "INCOMPLETE"
    else:
        state["status"] = "EMPTY"


def _active_leases(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        lease
        for lease in state.get("leases") or []
        if str(lease.get("status") or "") in ACTIVE_LEASE_STATES
    ]


def _find_lease(state: Mapping[str, Any], lease_id: str) -> dict[str, Any]:
    for lease in state.get("leases") or []:
        if str(lease.get("lease_id") or "") == str(lease_id):
            return lease
    raise KeyError(f"unknown lease_id {lease_id}")


def _assert_owner(lease: Mapping[str, Any], owner: str) -> None:
    if str(lease.get("owner") or "") != str(owner or ""):
        raise LeaseConflict(f"lease is owned by {lease.get('owner')}")


def _positive_ttl(value: float) -> float:
    ttl = float(value)
    if ttl <= 0:
        raise ValueError("lease_ttl_s must be positive")
    return ttl


def _plan_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"plan-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _require_state(path: Path) -> dict[str, Any]:
    state = _read_state_unlocked(path)
    if not state:
        raise FileNotFoundError(path)
    return state


def _read_state_unlocked(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _state_lock(path: Path, *, shared: bool = False):
    lock_path = Path(f"{Path(path)}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, required=True)
    prepare.add_argument("--state", type=Path, required=True)
    prepare.add_argument("--owner", required=True)
    prepare.add_argument("--lease-ttl-s", type=float, default=3600.0)

    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--state", type=Path, required=True)

    renew = subparsers.add_parser("renew")
    renew.add_argument("--state", type=Path, required=True)
    renew.add_argument("--owner", required=True)
    renew.add_argument("--lease-ttl-s", type=float, default=3600.0)

    reclaim = subparsers.add_parser("reclaim")
    reclaim.add_argument("--state", type=Path, required=True)
    reclaim.add_argument("--lease-id", required=True)
    reclaim.add_argument("--owner", required=True)
    reclaim.add_argument("--lease-ttl-s", type=float, default=3600.0)

    mark = subparsers.add_parser("mark")
    mark.add_argument("--state", type=Path, required=True)
    mark.add_argument("--lease-id", required=True)
    mark.add_argument("--owner", required=True)
    mark.add_argument("--status", choices=("succeeded", "failed", "released", "recovery_required"), required=True)
    mark.add_argument("--result-path", default="")
    mark.add_argument("--error", default="")

    launch = subparsers.add_parser("launch")
    launch.add_argument("--state", type=Path, required=True)
    launch.add_argument("--lease-id", required=True)
    launch.add_argument("--owner", required=True)
    launch.add_argument("--allow-controlled-launch", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        result = prepare_or_resume_wave(
            plan,
            state_path=args.state,
            owner=args.owner,
            lease_ttl_s=args.lease_ttl_s,
        )
    elif args.command == "status":
        result = load_wave_state(state_path=args.state)
    elif args.command == "recover":
        result = recover_wave(state_path=args.state)
    elif args.command == "renew":
        result = renew_wave_leases(
            state_path=args.state,
            owner=args.owner,
            lease_ttl_s=args.lease_ttl_s,
        )
    elif args.command == "reclaim":
        result = reclaim_expired_lease(
            state_path=args.state,
            lease_id=args.lease_id,
            owner=args.owner,
            lease_ttl_s=args.lease_ttl_s,
        )
    elif args.command == "mark":
        result = record_lease_status(
            state_path=args.state,
            lease_id=args.lease_id,
            owner=args.owner,
            status=args.status,
            result_path=args.result_path,
            error=args.error,
        )
    else:
        result = launch_leased_probe(
            state_path=args.state,
            lease_id=args.lease_id,
            owner=args.owner,
            allow_controlled_launch=bool(args.allow_controlled_launch),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
