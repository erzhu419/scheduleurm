from __future__ import annotations

from types import SimpleNamespace

from skill.scheduler_backend.local_backend import LocalBackend
from skill.scheduler_node.inventory import NODES
from skill.scheduler_node.retirement import reconcile_retired_nodes
from skill.scheduler_probe.all import ProbeAllDeps, probe_all_nodes


HPC_NODES = {f"node{index:03d}" for index in range(1, 8)}


def test_temporarily_unreachable_cluster_nodes_remain_in_live_inventory():
    assert HPC_NODES.issubset(NODES)
    assert all(not NODES[name].get("retired", False) for name in HPC_NODES)


def test_probe_all_skips_retired_nodes():
    calls = []
    nodes = {
        "active": {},
        "retired": {"retired": True},
        "monitor": {"monitor_only": True},
    }
    result = probe_all_nodes(
        deps=ProbeAllDeps(
            node_configs=nodes,
            max_workers=4,
            route_max_workers=1,
            outer_ssh_route_key=lambda _name: None,
            probe_all_route_failures=lambda: {},
            probe_node=lambda name: calls.append(name) or {"name": name, "alive": True},
            fold_claims_into_probe=lambda records: records,
        )
    )

    assert calls == ["active"]
    assert [record["name"] for record in result] == ["active"]


def test_retirement_reconciliation_preserves_history_and_safe_routes():
    state = {
        "tasks": [
            {
                "id": "running-old",
                "status": "running",
                "node": "node002",
                "started_at": 50.0,
                "remote_pids": [123],
            },
            {
                "id": "launching-old",
                "status": "launching",
                "assigned_node": "node007",
            },
            {
                "id": "mixed",
                "status": "queued",
                "allowed_nodes": ["node001", "jtl110gpu"],
                "allowed_nodes_submitted": ["node001", "jtl110gpu"],
                "allowed_nodes_user_explicit": True,
                "preferred_node": "node001",
            },
            {
                "id": "old-only",
                "status": "queued",
                "allowed_nodes": ["node003", "node004"],
                "allowed_nodes_submitted": ["node003", "node004"],
                "allowed_nodes_user_explicit": True,
            },
            {
                "id": "active",
                "status": "running",
                "node": "jtl110gpu2",
                "remote_pids": [456],
            },
        ]
    }

    report = reconcile_retired_nodes(
        state,
        HPC_NODES,
        now=100.0,
        reason="node retired",
    )
    tasks = {task["id"]: task for task in state["tasks"]}

    assert report["counts"] == {
        "terminalized": 2,
        "rerouted_queued": 1,
        "blocked_queued": 1,
        "cleared_preferences": 1,
    }
    assert tasks["running-old"]["status"] == "failed"
    assert tasks["running-old"]["remote_pids"] == [123]
    assert tasks["running-old"]["node_retirement"]["remote_termination_attempted"] is False
    assert tasks["launching-old"]["status"] == "failed"
    assert tasks["mixed"]["allowed_nodes"] == ["jtl110gpu"]
    assert tasks["mixed"]["allowed_nodes_submitted"] == ["jtl110gpu"]
    assert "preferred_node" not in tasks["mixed"]
    assert tasks["old-only"]["status"] == "queued"
    assert tasks["old-only"]["allowed_nodes"] == ["node003", "node004"]
    assert "every explicitly allowed node is retired" in tasks["old-only"]["last_block_reason"]
    assert tasks["active"]["status"] == "running"


def test_scheduler_submission_surfaces_keep_hpc_nodes_available(sch):
    cli_deps = sch._scheduler_cli_deps()
    submit_deps = sch._submit_task_deps()
    bulk_deps = sch._bulk_submit_deps()

    assert HPC_NODES.issubset(cli_deps.node_names)
    assert HPC_NODES.issubset(submit_deps.known_nodes)
    assert HPC_NODES.issubset(bulk_deps.known_nodes)
    assert {"jtl110gpu", "jtl110gpu2"}.issubset(cli_deps.node_names)


def test_local_backend_does_not_contact_retired_node():
    calls = []

    def run_on(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("retired node must not be contacted")

    backend = LocalBackend(
        SimpleNamespace(
            node_configs={"node001": {"retired": True}},
            run_on=run_on,
            task_pids=lambda task: list(task.get("remote_pids") or []),
            descendants_of=lambda roots, _parents: set(roots),
            running_probe_max_workers=2,
            running_probe_timeout_s=1,
        )
    )
    result = backend.batch_probe(
        {
            "tasks": [
                {
                    "id": "old",
                    "status": "running",
                    "node": "node001",
                    "remote_pids": [123],
                }
            ]
        }
    )

    assert calls == []
    assert result["old"]["state"] == "unknown"
    assert "retired" in result["old"]["error"]
