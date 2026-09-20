"""Campaign-owned node reservations in the scheduler queue state."""

from __future__ import annotations

from pathlib import Path


def _released(reservation: dict) -> bool:
    marker = Path(str(reservation.get("until_file") or ""))
    return marker.is_file() and marker.stat().st_size > 0


def active_reservations(state: dict) -> list[dict]:
    return [
        reservation
        for reservation in (state.get("node_reservations") or [])
        if not _released(reservation)
    ]


def prune_released_reservations(state: dict) -> int:
    before = len(state.get("node_reservations") or [])
    state["node_reservations"] = active_reservations(state)
    return before - len(state["node_reservations"])


def blocked_reserved_nodes(state: dict, task: dict) -> set[str]:
    signature = str(task.get("signature") or "")
    task_id = str(task.get("id") or "")
    return {
        node
        for reservation in active_reservations(state)
        if task_id != str(reservation.get("owner_task_id") or "")
        and not signature.startswith(str(reservation.get("signature_prefix") or ""))
        for node in reservation.get("nodes") or []
    }


def make_reservation(
    state: dict,
    owner_task: dict,
    *,
    nodes: list[str],
    signature_prefix: str,
    until_file: str,
    known_nodes: set[str],
) -> dict:
    requested = list(dict.fromkeys(str(node) for node in nodes if str(node)))
    prefix = str(signature_prefix or "").strip()
    marker = Path(str(until_file or ""))
    if not requested or not prefix or not marker.is_absolute():
        raise ValueError(
            "node reservation requires --reserve-node, --reserve-signature-prefix, "
            "and an absolute --reserve-until-file"
        )
    if any(node not in known_nodes for node in requested):
        raise ValueError("node reservation names an unknown node")
    if marker.exists():
        raise ValueError(f"reservation completion marker already exists: {marker}")
    active = active_reservations(state)
    taken = {node for reservation in active for node in reservation.get("nodes") or []}
    conflict = taken.intersection(requested)
    if conflict:
        raise ValueError(f"node already reserved: {', '.join(sorted(conflict))}")
    occupied = [
        task.get("id")
        for task in state.get("tasks") or []
        if task.get("status") in ("running", "launching")
        and task.get("node") in requested
    ]
    if occupied:
        raise ValueError(f"node has active scheduler tasks: {', '.join(map(str, occupied))}")
    return {
        "owner_task_id": owner_task["id"],
        "nodes": requested,
        "signature_prefix": prefix,
        "until_file": str(marker),
        "created_at": owner_task.get("submitted_at"),
    }
