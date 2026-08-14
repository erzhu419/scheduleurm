#!/usr/bin/env python3
"""Audit and apply a local queue migration for permanently lost nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
if str(SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_DIR))

import scheduler as sch  # noqa: E402
from scheduler_node.retirement import (  # noqa: E402
    preview_retired_node_reconciliation,
    reconcile_retired_nodes,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes", nargs="*", help="Retired scheduler node names")
    parser.add_argument(
        "--reason",
        default="cluster node permanently retired and no longer reachable",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Apply the migration; without this flag only print a dry-run report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configured_retired = sorted(
        name for name, info in sch.NODES.items() if (info or {}).get("retired")
    )
    nodes = sorted(set(args.nodes or configured_retired))
    invalid = [
        node for node in nodes
        if node not in sch.NODES or not (sch.NODES[node] or {}).get("retired")
    ]
    if invalid:
        raise SystemExit(
            "refusing nodes not marked retired in inventory: " + ", ".join(invalid)
        )
    now = time.time()
    with sch.state_lock(purpose="retire-nodes"):
        state = sch.load_state()
        if not args.confirm:
            _, report = preview_retired_node_reconciliation(
                state,
                nodes,
                now=now,
                reason=args.reason,
            )
            report["mode"] = "dry-run"
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
        backup = sch.STATE_DIR / f"queue.json.pre-retire-{timestamp}"
        shutil.copy2(sch.QUEUE_FILE, backup)
        before_sha256 = _sha256(backup)
        report = reconcile_retired_nodes(
            state,
            nodes,
            now=now,
            reason=args.reason,
        )
        sch.save_state(state)
        after_sha256 = _sha256(sch.QUEUE_FILE)

        audit_dir = sch.STATE_DIR / "retirement_audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"retire_nodes_{timestamp}.json"
        report.update({
            "mode": "applied",
            "applied_at": now,
            "backup": str(backup),
            "queue_sha256_before": before_sha256,
            "queue_sha256_after": after_sha256,
        })
        tmp = audit_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        tmp.replace(audit_path)
        report["audit_path"] = str(audit_path)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
