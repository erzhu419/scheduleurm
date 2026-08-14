from __future__ import annotations

import pytest

from skill.scheduler_algorithm.bridge import AlgorithmBridge


class _Policy:
    name = "unit_policy"

    def snapshot(self):
        return {"name": self.name, "score_weights": {"free": 1}}

    def gpu_fit_block_reason(self, task, gpu, node_info, context):
        if gpu.get("running_task_count", 0) >= context["max_tasks"]:
            return "full"
        return ""

    def gpu_score(self, task, node_state, gpu, legacy_score, context):
        return ("unit", context["queue_file"], *legacy_score)

    def selected_gpu_audit(self, task, node_state, gpu, context):
        return {"algorithm": self.name, "queue_file": context["queue_file"]}

    def global_batch_select(self, tasks, nodes, context):
        return {
            "scheduler_hook_ready": True,
            "placements": {
                tasks[0]["id"]: {"node": nodes[0]["name"], "gpu_idx": 1},
                "bad": "not-a-dict",
                "missing-node": {"gpu_idx": 2},
            },
            "action": {"oracle_gap_alpha0": 0.1, "oracle_gap_alpha1": 0.2},
        }


def _bridge(*, environ=None, node_configs=None, loader=None, bypass=None, snapshot=None):
    env = {} if environ is None else environ
    nodes = {} if node_configs is None else node_configs
    return AlgorithmBridge(
        runtime_context_factory=lambda: {
            "queue_file": env.get("QUEUE_FILE", "/tmp/queue.json"),
            "max_tasks": int(env.get("MAX_TASKS", "2")),
        },
        node_configs_factory=lambda: nodes,
        environ=env,
        load_placement_policy=loader,
        should_bypass_hard_rule=bypass,
        hard_rule_snapshot=snapshot,
    )


def test_configure_uses_env_and_falls_back_to_legacy_on_implicit_error():
    calls = []

    def loader(name):
        calls.append(name)
        if name == "broken":
            raise ValueError("boom")
        policy = _Policy()
        policy.name = name
        return policy

    bridge = _bridge(environ={"SCHEDULEURM_ALGORITHM": "broken"}, loader=loader)

    policy = bridge.configure()

    assert policy.name == "legacy"
    assert calls == ["broken", "legacy"]
    assert bridge.config_snapshot()["load_error"] == "boom"


def test_configure_explicit_error_raises_system_exit():
    bridge = _bridge(loader=lambda name: (_ for _ in ()).throw(ValueError("bad policy")))

    with pytest.raises(SystemExit, match="invalid --algorithm 'broken'"):
        bridge.configure("broken")


def test_policy_scoring_audit_and_fit_use_dynamic_runtime_context():
    bridge = _bridge(environ={"QUEUE_FILE": "/tmp/q1.json", "MAX_TASKS": "1"}, loader=lambda name: _Policy())
    bridge.configure("unit")
    legacy = (1, 2)

    assert bridge.gpu_score({}, {}, {"idx": 0}, legacy) == ("unit", "/tmp/q1.json", 1, 2)
    assert bridge.selected_gpu_audit({}, {}, {"idx": 0}) == {
        "algorithm": "unit_policy",
        "queue_file": "/tmp/q1.json",
    }
    assert bridge.gpu_fit_block_reason({}, {"running_task_count": 1}, {}) == "full"


def test_global_batch_plan_uses_dynamic_node_configs_and_env_hook():
    nodes = {"nodeA": {"max_tasks_per_gpu": 3}}
    env = {"SCHEDULEURM_GLOBAL_BATCH_HOOK": "1"}
    bridge = _bridge(environ=env, node_configs=nodes, loader=lambda name: _Policy())
    bridge.configure("unit")

    plan, event = bridge.global_batch_plan(
        [{"id": "t1"}, {"id": "t2"}],
        [{"name": "nodeA", "gpus": [{"idx": 0}]}],
    )

    assert plan == {
        "t1": {
            "node": "nodeA",
            "gpu_idx": 1,
            "algorithm": "unit_policy",
            "execution_contract": "enforced_exact_placement_v1",
        }
    }
    assert event["type"] == "algorithm_global_batch_plan"
    assert event["scheduler_hook_ready"] is True
    assert event["execution_contract"] == "enforced_exact_placement_v1"
    assert event["fail_closed"] is True
    assert event["oracle_gap_applies_to_execution"] is False
    assert event["planned_task_ids"] == ["t1"]
    assert event["candidate_tasks"] == 2
    assert event["planned_tasks"] == 1
    assert event["oracle_gap_alpha0"] == 0.1
    assert event["oracle_gap_alpha1"] == 0.2


def test_global_batch_plan_rejects_uncertified_placements_fail_closed():
    class _UnreadyPolicy(_Policy):
        def global_batch_select(self, tasks, nodes, context):
            return {
                "scheduler_hook_ready": False,
                "placements": {
                    tasks[0]["id"]: {"node": nodes[0]["name"], "gpu_idx": 0},
                },
            }

    bridge = _bridge(
        environ={"SCHEDULEURM_GLOBAL_BATCH_HOOK": "1"},
        node_configs={"nodeA": {}},
        loader=lambda name: _UnreadyPolicy(),
    )
    bridge.configure("unit")

    plan, event = bridge.global_batch_plan(
        [{"id": "t1"}],
        [{"name": "nodeA", "gpus": [{"idx": 0}]}],
    )

    assert plan == {}
    assert event["type"] == "algorithm_global_batch_plan_rejected"
    assert event["execution_contract"] == "enforced_exact_placement_v1"
    assert event["fail_closed"] is True


def test_hard_rule_helpers_delegate_and_tolerate_errors():
    calls = []

    def bypass(rule, **kwargs):
        calls.append((rule, kwargs["context"]["algorithm"], kwargs["task"]["id"]))
        return True

    def snapshot(task=None):
        return {"mode": "unit", "task": (task or {}).get("id"), "active_rules": ["x"]}

    bridge = _bridge(loader=lambda name: _Policy(), bypass=bypass, snapshot=snapshot)
    bridge.configure("unit")

    assert bridge.hard_rule_bypassed("x", task={"id": "t1"}) is True
    assert calls == [("x", "unit_policy", "t1")]
    assert bridge.hard_rule_override_snapshot({"id": "t2"}) == {
        "mode": "unit",
        "task": "t2",
        "active_rules": ["x"],
    }

    error_bridge = _bridge(
        loader=lambda name: _Policy(),
        bypass=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nope")),
        snapshot=lambda task=None: (_ for _ in ()).throw(RuntimeError("snap")),
    )
    error_bridge.configure("unit")

    assert error_bridge.hard_rule_bypassed("x") is False
    assert error_bridge.hard_rule_override_snapshot()["mode"] == "error"
