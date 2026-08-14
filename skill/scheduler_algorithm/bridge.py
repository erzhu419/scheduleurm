from __future__ import annotations

import os
from typing import Callable, Optional

GLOBAL_BATCH_EXECUTION_CONTRACT = "enforced_exact_placement_v1"

try:
    from algorithm import load_placement_policy as _default_load_placement_policy
except Exception:
    _default_load_placement_policy = None

try:
    from algorithm.hard_rules import (
        should_bypass_hard_rule as _default_should_bypass_hard_rule,
        snapshot as _default_hard_rule_snapshot,
    )
except Exception:
    _default_should_bypass_hard_rule = None
    _default_hard_rule_snapshot = None


class AlgorithmBridge:
    """Adapter between scheduler runtime state and optional /algorithm policies."""

    def __init__(
        self,
        *,
        runtime_context_factory: Callable[[], dict],
        node_configs_factory: Callable[[], dict],
        environ=None,
        load_placement_policy: Optional[Callable] = None,
        should_bypass_hard_rule: Optional[Callable] = None,
        hard_rule_snapshot: Optional[Callable] = None,
    ):
        self._runtime_context_factory = runtime_context_factory
        self._node_configs_factory = node_configs_factory
        self._environ = environ if environ is not None else os.environ
        self._load_placement_policy = (
            _default_load_placement_policy if load_placement_policy is None else load_placement_policy
        )
        self._should_bypass_hard_rule = (
            _default_should_bypass_hard_rule
            if should_bypass_hard_rule is None else should_bypass_hard_rule
        )
        self._hard_rule_snapshot = (
            _default_hard_rule_snapshot if hard_rule_snapshot is None else hard_rule_snapshot
        )
        self._placement_policy = None
        self._placement_policy_name = ""
        self._placement_policy_load_error = ""

    def configure(self, name: Optional[str] = None):
        explicit = bool(name)
        selected = (
            name
            or self._environ.get("SCHEDULEURM_ALGORITHM")
            or self._environ.get("SCHEDULEURM_PLACEMENT_POLICY")
            or "legacy"
        )
        if self._load_placement_policy is None:
            selected = "legacy"
            policy = None
        else:
            try:
                policy = self._load_placement_policy(selected)
                self._placement_policy_load_error = ""
            except Exception as exc:
                if explicit:
                    raise SystemExit(f"invalid --algorithm {selected!r}: {exc}") from exc
                self._placement_policy_load_error = str(exc)
                policy = self._load_placement_policy("legacy")
        self._placement_policy = policy
        self._placement_policy_name = getattr(policy, "name", selected) if policy is not None else "legacy"
        return policy

    def runtime_context(self) -> dict:
        return dict(self._runtime_context_factory())

    def name(self) -> str:
        if self._placement_policy is None:
            return "legacy"
        return getattr(self._placement_policy, "name", "legacy")

    def config_snapshot(self) -> dict:
        policy = self._placement_policy
        if policy is None or not hasattr(policy, "snapshot"):
            return {"name": "legacy", "hard_rule_override": self.hard_rule_override_snapshot()}
        try:
            out = policy.snapshot()
            if self._placement_policy_load_error:
                out["load_error"] = self._placement_policy_load_error
            out["hard_rule_override"] = self.hard_rule_override_snapshot()
            return out
        except Exception:
            return {
                "name": self.name(),
                "snapshot_error": True,
                "hard_rule_override": self.hard_rule_override_snapshot(),
            }

    def gpu_fit_block_reason(self, task: dict, gpu: dict, node_info: dict) -> str:
        policy = self._placement_policy
        if policy is None:
            return ""
        try:
            return str(policy.gpu_fit_block_reason(task, gpu, node_info, self.runtime_context()) or "")
        except Exception as exc:
            return f"algorithm:{self.name()}: error {str(exc)[:120]}"

    def gpu_score(self, task: dict, node_state: dict, gpu: dict, legacy_score):
        policy = self._placement_policy
        if policy is None:
            return legacy_score
        try:
            return policy.gpu_score(task, node_state, gpu, legacy_score, self.runtime_context())
        except Exception:
            return legacy_score

    def selected_gpu_audit(self, task: dict, node_state: dict, gpu: dict) -> dict:
        policy = self._placement_policy
        if policy is None or not hasattr(policy, "selected_gpu_audit"):
            return {}
        try:
            return policy.selected_gpu_audit(task, node_state, gpu, self.runtime_context()) or {}
        except Exception as exc:
            return {"error": str(exc)[:120], "algorithm": self.name()}

    def global_batch_hook_enabled(self) -> bool:
        raw = str(self._environ.get("SCHEDULEURM_GLOBAL_BATCH_HOOK") or "").strip().lower()
        if raw in ("0", "false", "no", "off"):
            return False
        if raw in ("1", "true", "yes", "on"):
            return True
        return str(self.name()).startswith("global_theorem")

    def global_batch_plan(self, tasks: list, nodes: list) -> tuple[dict, dict]:
        policy = self._placement_policy
        if (
            policy is None
            or not self.global_batch_hook_enabled()
            or not hasattr(policy, "global_batch_select")
        ):
            return {}, {}
        limit = max(1, int(self._environ.get("SCHEDULEURM_GLOBAL_BATCH_TASK_LIMIT", "32")))
        ctx = self.runtime_context()
        ctx["global_batch_size"] = max(1, int(self._environ.get("SCHEDULEURM_GLOBAL_BATCH_SIZE", "4")))
        ctx["global_batch_max_configurations"] = max(
            1,
            int(self._environ.get("SCHEDULEURM_GLOBAL_BATCH_MAX_CONFIGURATIONS", "10000")),
        )
        node_configs = self._node_configs_factory()
        node_rows = []
        for node in nodes:
            row = dict(node)
            row["node_info"] = dict(node_configs.get(node.get("name"), {}) or {})
            node_rows.append(row)
        try:
            report = policy.global_batch_select(list(tasks[:limit]), node_rows, ctx) or {}
        except Exception as exc:
            return {}, {
                "type": "algorithm_global_batch_plan_error",
                "algorithm": self.name(),
                "reason": str(exc)[:220],
                "execution_contract": GLOBAL_BATCH_EXECUTION_CONTRACT,
                "fail_closed": True,
                "planned_tasks": 0,
            }
        placements = report.get("placements") if isinstance(report, dict) else {}
        if not isinstance(placements, dict):
            placements = {}
        scheduler_hook_ready = (
            bool(report.get("scheduler_hook_ready"))
            if isinstance(report, dict) else False
        )
        plan = {}
        for task_id, placement in placements.items():
            if not isinstance(placement, dict):
                continue
            node = str(placement.get("node") or "")
            if not node:
                continue
            plan[str(task_id)] = {
                "node": node,
                "gpu_idx": placement.get("gpu_idx"),
                "algorithm": self.name(),
                "execution_contract": GLOBAL_BATCH_EXECUTION_CONTRACT,
            }
        if plan and not scheduler_hook_ready:
            return {}, {
                "type": "algorithm_global_batch_plan_rejected",
                "algorithm": self.name(),
                "reason": "policy returned placements without a scheduler-ready certificate",
                "execution_contract": GLOBAL_BATCH_EXECUTION_CONTRACT,
                "fail_closed": True,
                "candidate_tasks": min(len(tasks), limit),
                "planned_tasks": 0,
            }
        event = {
            "type": "algorithm_global_batch_plan",
            "algorithm": self.name(),
            "candidate_tasks": min(len(tasks), limit),
            "planned_tasks": len(plan),
            "planned_task_ids": sorted(plan),
            "scheduler_hook_ready": scheduler_hook_ready,
            "execution_contract": GLOBAL_BATCH_EXECUTION_CONTRACT,
            "fail_closed": True,
            "oracle_gap_applies_to_execution": False,
            "oracle_gap_alpha0": (
                (report.get("action") or {}).get("oracle_gap_alpha0") if isinstance(report, dict) else None
            ),
            "oracle_gap_alpha1": (
                (report.get("action") or {}).get("oracle_gap_alpha1") if isinstance(report, dict) else None
            ),
        }
        return plan, event

    def hard_rule_bypassed(
        self,
        rule: str,
        *,
        task: Optional[dict] = None,
        node_info: Optional[dict] = None,
        node_state: Optional[dict] = None,
        gpu: Optional[dict] = None,
    ) -> bool:
        if self._should_bypass_hard_rule is None:
            return False
        try:
            return bool(self._should_bypass_hard_rule(
                rule,
                task=task or {},
                node_info=node_info or {},
                node_state=node_state or {},
                gpu=gpu or {},
                context={"algorithm": self.name()},
            ))
        except Exception:
            return False

    def hard_rule_override_snapshot(self, task: Optional[dict] = None) -> dict:
        if self._hard_rule_snapshot is None:
            return {"mode": "none", "active_rules": []}
        try:
            return self._hard_rule_snapshot(task)
        except Exception as exc:
            return {"mode": "error", "error": str(exc)[:120], "active_rules": []}

    def set_hard_rule_mode(self, mode: Optional[str]) -> None:
        raw = str(mode or "").strip()
        if raw:
            self._environ["SCHEDULEURM_AB_HARD_RULE_MODE"] = raw
