"""Crash-safe local node reservations for theorem-facing measurements.

The live scheduler and the measurement harness run on the same control host.
This module gives the harness a small, explicit coordination surface without
changing the default placement algorithm or touching already-running jobs.
An active reservation only prevents *new* Scheduleurm launches on its node.

Reservations have both a TTL and a local holder-process identity.  A crashed
harness therefore releases itself on the next read, while a healthy long run
renews in the background.  External/manual launches remain outside this local
coordination mechanism and are still rejected by the campaign's process audit.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
import threading
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SCHEMA_VERSION = 1
DEFAULT_TTL_S = 900.0
DEFAULT_STATE_PATH = Path.home() / ".claude" / "scheduler" / "measurement_reservations.json"


class ReservationConflict(RuntimeError):
    """Raised when another live measurement already owns a node."""


def reservation_state_path() -> Path:
    raw = os.environ.get("SCHEDULEURM_MEASUREMENT_RESERVATION_FILE")
    return Path(raw).expanduser() if raw else DEFAULT_STATE_PATH


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


def _empty_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "reservations": []}


def _load_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid measurement reservation state {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("reservations"), list):
        raise RuntimeError(f"invalid measurement reservation schema in {path}")
    return payload


def _write_unlocked(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _pid_start_ticks(pid: int) -> int | None:
    try:
        tail = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
        fields = tail.split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _record_active(record: Mapping[str, Any], *, now: float) -> bool:
    try:
        if float(record.get("expires_at") or 0.0) <= now:
            return False
    except (TypeError, ValueError):
        return False
    holder_host = str(record.get("holder_host") or "")
    if holder_host and holder_host != socket.gethostname():
        return True
    try:
        holder_pid = int(record.get("holder_pid") or 0)
    except (TypeError, ValueError):
        return False
    if holder_pid <= 0:
        return False
    observed_start = _pid_start_ticks(holder_pid)
    if observed_start is None:
        return False
    try:
        expected_start = int(record.get("holder_start_ticks"))
    except (TypeError, ValueError):
        return False
    return observed_start == expected_start


def _active_rows(payload: Mapping[str, Any], *, now: float) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in payload.get("reservations") or []
        if isinstance(row, Mapping) and _record_active(row, now=now)
    ]


def _with_lock(
    path: Path,
    *,
    exclusive: bool,
    operation: Callable[[dict[str, Any]], Any],
) -> Any:
    lock = _lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            payload = _load_unlocked(path)
            return operation(payload)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def list_active_reservations(
    *,
    path: Path | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    target = path or reservation_state_path()
    timestamp = time.time() if now is None else float(now)
    return _with_lock(
        target,
        exclusive=False,
        operation=lambda payload: _active_rows(payload, now=timestamp),
    )


def acquire_reservation(
    *,
    node: str,
    purpose: str,
    ttl_s: float = DEFAULT_TTL_S,
    path: Path | None = None,
    now: float | None = None,
    holder_pid: int | None = None,
) -> dict[str, Any]:
    if not str(node).strip():
        raise ValueError("node is required")
    if float(ttl_s) <= 0.0:
        raise ValueError("ttl_s must be positive")
    target = path or reservation_state_path()
    timestamp = time.time() if now is None else float(now)
    pid = os.getpid() if holder_pid is None else int(holder_pid)
    start_ticks = _pid_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError(f"cannot verify reservation holder pid {pid}")

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        active = _active_rows(payload, now=timestamp)
        conflicts = [row for row in active if str(row.get("node")) == str(node)]
        if conflicts:
            row = conflicts[0]
            raise ReservationConflict(
                f"node {node} is reserved by {row.get('reservation_id')} "
                f"for {row.get('purpose')}"
            )
        record = {
            "reservation_id": f"measure-{uuid.uuid4().hex}",
            "node": str(node),
            "purpose": str(purpose),
            "holder_host": socket.gethostname(),
            "holder_pid": pid,
            "holder_start_ticks": start_ticks,
            "created_at": timestamp,
            "renewed_at": timestamp,
            "expires_at": timestamp + float(ttl_s),
            "ordinary_running_tasks_touched": False,
            "blocks_new_scheduleurm_launches_only": True,
        }
        payload["schema_version"] = SCHEMA_VERSION
        payload["reservations"] = active + [record]
        _write_unlocked(target, payload)
        return record

    return _with_lock(target, exclusive=True, operation=mutate)


def renew_reservation(
    reservation_id: str,
    *,
    ttl_s: float = DEFAULT_TTL_S,
    path: Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    target = path or reservation_state_path()
    timestamp = time.time() if now is None else float(now)

    def mutate(payload: dict[str, Any]) -> dict[str, Any]:
        active = _active_rows(payload, now=timestamp)
        found = None
        for row in active:
            if str(row.get("reservation_id")) != str(reservation_id):
                continue
            if int(row.get("holder_pid") or 0) != os.getpid():
                raise ReservationConflict("reservation holder pid changed")
            row["renewed_at"] = timestamp
            row["expires_at"] = timestamp + float(ttl_s)
            found = row
            break
        if found is None:
            raise ReservationConflict(f"reservation {reservation_id} is no longer active")
        payload["schema_version"] = SCHEMA_VERSION
        payload["reservations"] = active
        _write_unlocked(target, payload)
        return dict(found)

    return _with_lock(target, exclusive=True, operation=mutate)


def release_reservation(
    reservation_id: str,
    *,
    path: Path | None = None,
    now: float | None = None,
) -> bool:
    target = path or reservation_state_path()
    timestamp = time.time() if now is None else float(now)

    def mutate(payload: dict[str, Any]) -> bool:
        active = _active_rows(payload, now=timestamp)
        kept = [
            row for row in active
            if str(row.get("reservation_id")) != str(reservation_id)
        ]
        removed = len(kept) != len(active)
        payload["schema_version"] = SCHEMA_VERSION
        payload["reservations"] = kept
        _write_unlocked(target, payload)
        return removed

    return bool(_with_lock(target, exclusive=True, operation=mutate))


def fold_measurement_reservations_into_probe(
    nodes: Iterable[dict[str, Any]],
    *,
    path: Path | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Annotate live probes with active measurement reservations.

    Reservation is an admission-control fact, not observed resource pressure.
    Keep the live CPU, RAM, and GPU telemetry intact so preemption, eviction,
    migration, and monitoring code cannot misinterpret a calibration hold as a
    saturated node. Placement code must consume
    ``measurement_reservation_active`` explicitly.
    """
    rows = list(nodes)
    active = {
        str(row.get("node")): row
        for row in list_active_reservations(path=path, now=now)
        if row.get("node")
    }
    for node in rows:
        reservation = active.get(str(node.get("name")))
        if reservation is None or node.get("alive") is not True:
            continue
        node["measurement_reservation_active"] = True
        node["measurement_reservation"] = {
            key: reservation.get(key)
            for key in (
                "reservation_id",
                "purpose",
                "holder_host",
                "holder_pid",
                "created_at",
                "renewed_at",
                "expires_at",
            )
        }
        for gpu in node.get("gpus") or []:
            gpu["measurement_reservation_active"] = True
    return rows


class MeasurementReservation(AbstractContextManager[dict[str, Any]]):
    """Acquire, renew, and release one node reservation."""

    def __init__(
        self,
        *,
        node: str,
        purpose: str,
        ttl_s: float = DEFAULT_TTL_S,
        path: Path | None = None,
    ) -> None:
        self.node = node
        self.purpose = purpose
        self.ttl_s = float(ttl_s)
        self.path = path
        self.record: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.renewal_error: str = ""

    def __enter__(self) -> dict[str, Any]:
        self.record = acquire_reservation(
            node=self.node,
            purpose=self.purpose,
            ttl_s=self.ttl_s,
            path=self.path,
        )
        interval = max(1.0, min(self.ttl_s / 3.0, 60.0))

        def renew_loop() -> None:
            while not self._stop.wait(interval):
                try:
                    renew_reservation(
                        str(self.record["reservation_id"]),
                        ttl_s=self.ttl_s,
                        path=self.path,
                    )
                except Exception as exc:  # The caller checks this before launch steps.
                    self.renewal_error = f"{type(exc).__name__}: {exc}"
                    self._stop.set()

        self._thread = threading.Thread(
            target=renew_loop,
            name=f"measurement-reservation-{self.node}",
            daemon=True,
        )
        self._thread.start()
        return self.record

    def ensure_healthy(self) -> None:
        if self.renewal_error:
            raise RuntimeError(f"measurement reservation renewal failed: {self.renewal_error}")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.record is not None:
            release_reservation(str(self.record["reservation_id"]), path=self.path)


__all__ = [
    "DEFAULT_STATE_PATH",
    "DEFAULT_TTL_S",
    "MeasurementReservation",
    "ReservationConflict",
    "acquire_reservation",
    "fold_measurement_reservations_into_probe",
    "list_active_reservations",
    "release_reservation",
    "renew_reservation",
    "reservation_state_path",
]
