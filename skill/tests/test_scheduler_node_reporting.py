from __future__ import annotations

from skill.scheduler_node.reporting import (
    NodeReportingDeps,
    build_heartbeat_payload,
    format_claim_intent_hint_for_task,
    format_claim_record,
    format_node_claim_summary,
    iter_node_summary_nodes,
    print_node_summary,
)


def _deps(*, now=1000.0, printed=None):
    return NodeReportingDeps(
        node_configs={"node001": {"reserved_cpu_cores": 4}},
        node_display_name=lambda name: name,
        format_mem_gb=lambda mb: f"{int(mb)}MB",
        format_node_ram_summary=lambda node: (
            f"ram={int(node.get('free_ram_mb', 0))}MB/{int(node.get('total_ram_mb', 0))}MB"
        ),
        scheduler_id=lambda: "sid",
        cpu_ownership_snapshot=lambda state, nodes: [
            {"node": node["name"]} for node in nodes
        ],
        now=lambda: now,
        print_fn=(printed.append if printed is not None else print),
    )


def test_iter_node_summary_nodes_hides_jump_host_and_orders_gpu_before_cpu():
    nodes = [
        {"name": "node006", "alive": True, "gpus": []},
        {"name": "zhengliang-hpc", "alive": True, "gpus": []},
        {"name": "local", "alive": True, "gpus": [{"idx": 0}]},
        {"name": "node001", "alive": True, "gpus": []},
        {"name": "nodeX", "alive": True, "gpus": []},
    ]

    ordered = iter_node_summary_nodes(nodes, deps=_deps())

    assert [node["name"] for node in ordered] == ["local", "node001", "node006", "nodeX"]


def test_print_node_summary_includes_cpu_only_host_load_ram_and_claims():
    printed = []
    nodes = [
        {
            "name": "node001",
            "alive": True,
            "gpus": [],
            "loadavg": 1.2,
            "wsl_loadavg": 2.5,
            "host_cpu_load_pct": 33,
            "probe_fallback": "cached",
            "free_cpu": 6,
            "total_cpu": 64,
            "free_ram_mb": 100,
            "total_ram_mb": 200,
            "claim_intents": [
                {
                    "owner": "alice",
                    "scheduler_id": "sid",
                    "task_id": "t2",
                    "intent_at": 990,
                    "vram_mb": 0,
                    "cpu_cores": 8,
                    "ram_mb": 50,
                }
            ],
        },
        {"name": "nodeX", "alive": False, "error": "ssh"},
    ]

    print_node_summary(nodes, deps=_deps(printed=printed))

    text = "\n".join(printed)
    assert "=== nodes ===" in text
    assert "node001" in text
    assert "CPU-only" in text
    assert "wsl_load 2.5" in text
    assert "host 33%" in text
    assert "cached" in text
    assert "reserve 4" in text
    assert "ram=100MB/200MB" in text
    assert "claims(intents=1 head=t2@10s)" in text
    assert "nodeX" in text
    assert "DOWN (ssh)" in text


def test_claim_record_and_summaries_mark_other_scheduler_owners():
    node = {
        "active_claims": [
            {
                "owner": "other-user",
                "scheduler_id": "other",
                "task_id": "t1",
                "claimed_at": 900,
                "gpu_idx": 1,
                "vram_mb": 123,
                "cpu_cores": 2,
                "ram_mb": 456,
            }
        ],
        "pending_claims": [],
        "claim_intents": [],
    }

    summary = format_node_claim_summary(node, deps=_deps())
    record = format_claim_record(node["active_claims"][0], deps=_deps())

    assert "active_claims=1" in summary
    assert "other_schedulers=1 owner=other-user" in summary
    assert record == "other-user/other/t1:GPU1 vram=123MB cpu=2 ram=456MB age=100s"


def test_claim_intent_hint_reports_position_or_queue_head():
    snap = {
        "intents": [
            {"owner": "bob", "scheduler_id": "other", "task_id": "t0", "intent_at": 800},
            {"owner": "alice", "scheduler_id": "sid", "task_id": "t1", "intent_at": 900},
        ]
    }

    own = format_claim_intent_hint_for_task({"id": "t1"}, "node001", snap, deps=_deps())
    missing = format_claim_intent_hint_for_task({"id": "t2"}, "node001", snap, deps=_deps())

    assert own.startswith("claim-intent: position 2/2; head=bob/other/t0:CPU")
    assert missing.startswith("claim-intents: 2 queued; head=bob/other/t0:CPU")


def test_build_heartbeat_payload_counts_tasks_and_formats_node_briefs():
    state = {
        "tasks": [
            {"id": "r", "status": "running"},
            {"id": "l", "status": "launching"},
            {"id": "q", "status": "queued"},
            {"id": "d", "status": "done"},
        ]
    }
    nodes = [
        {"name": "nodeA", "alive": True, "gpus": [{"used_mb": 12}, {"used_mb": 34}]},
        {"name": "nodeB", "alive": False},
        {"name": "nodeC", "alive": True, "gpus": []},
    ]

    payload = build_heartbeat_payload(state, nodes, deps=_deps())

    assert payload["running"] == 1
    assert payload["launching"] == 1
    assert payload["queued"] == 1
    assert payload["nodes"] == ["nodeA:12MB/34MB", "nodeB:DOWN", "nodeC:"]
    assert payload["cpu_accounting"] == [
        {"node": "nodeA"},
        {"node": "nodeB"},
        {"node": "nodeC"},
    ]
