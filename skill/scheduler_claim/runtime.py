from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ClaimManagerDeps:
    state_dir: Any
    node_configs: dict
    claim_ttl_s: int
    claim_intent_ttl_s: int
    claim_fifo_strict_after_s: int
    claim_live_check: bool
    vram_margin_mb: int
    one_third_pack_rule: bool
    one_third_pack_grace_mb: int
    gpu_empty_used_mb: int
    default_cpu_cores: int
    default_ram_mb: int
    ignore_cpu_for_server_gpu_task: Callable[..., bool]
    task_ignores_one_third_pack_rule: Callable[[dict, Optional[dict]], bool]
    claims_remote_op: Callable[..., dict]


class ClaimManager:
    """Cross-scheduler resource claims via remote flock + JSON."""

    _deps_source: ClaimManagerDeps | Callable[[], ClaimManagerDeps] | None = None

    @classmethod
    def configure(cls, deps: ClaimManagerDeps | Callable[[], ClaimManagerDeps]) -> None:
        cls._deps_source = deps

    @classmethod
    def deps(cls) -> ClaimManagerDeps:
        if cls._deps_source is None:
            raise RuntimeError("ClaimManager dependencies are not configured")
        if callable(cls._deps_source):
            return cls._deps_source()
        return cls._deps_source

    @classmethod
    def enabled_for(cls, node: str) -> bool:
        return bool(cls.deps().node_configs.get(node, {}).get("enable_claims"))

    @classmethod
    def scheduler_id(cls) -> str:
        cached = getattr(cls, "_cached_owner_id", None)
        if cached:
            return cached
        owner_file = Path(cls.deps().state_dir) / "claim_owner_id"
        try:
            if owner_file.exists():
                value = owner_file.read_text().strip()
                if value:
                    cls._cached_owner_id = value
                    return value
        except Exception:
            pass
        try:
            host = os.uname().nodename
        except Exception:
            host = "unknown"
        try:
            import uuid as _uuid
            new_id = f"{host}:{_uuid.uuid4().hex[:12]}"
        except Exception:
            new_id = f"{host}:{os.getpid()}-{int(time.time())}"
        try:
            owner_file.parent.mkdir(parents=True, exist_ok=True)
            owner_file.write_text(new_id)
        except Exception:
            pass
        cls._cached_owner_id = new_id
        return new_id

    @classmethod
    def _ttl_for(cls, node: str) -> int:
        deps = cls.deps()
        return int(deps.node_configs.get(node, {}).get("claim_ttl_s", deps.claim_ttl_s))

    @classmethod
    def _intent_ttl_for(cls, node: str) -> int:
        deps = cls.deps()
        return int(deps.node_configs.get(node, {}).get("claim_intent_ttl_s", deps.claim_intent_ttl_s))

    @classmethod
    def _fifo_strict_after_for(cls, node: str) -> int:
        deps = cls.deps()
        return int(deps.node_configs.get(node, {}).get(
            "claim_fifo_strict_after_s",
            deps.claim_fifo_strict_after_s,
        ))

    @classmethod
    def _live_check_for(cls, node: str) -> bool:
        deps = cls.deps()
        return bool(deps.node_configs.get(node, {}).get("claim_live_check", deps.claim_live_check))

    @classmethod
    def _build_capacity(cls, node: str, node_state: Optional[dict] = None) -> dict:
        deps = cls.deps()
        info = deps.node_configs.get(node, {})
        declared_ram = int(info.get("ram_mb") or 0)
        probed_ram = int((node_state or {}).get("total_ram_mb") or 0)
        if declared_ram and probed_ram:
            ram_cap = min(declared_ram, probed_ram)
        else:
            ram_cap = probed_ram or declared_ram
        cap = {
            "cpu_cores": int(info.get("cpu_cores", 0)),
            "ram_mb": ram_cap,
            "gpu_vram_mb": {},
            "max_vram_per_task": info.get("max_vram_per_task"),
            "vram_margin_mb": int(deps.vram_margin_mb),
            "third_pack_rule": bool(deps.one_third_pack_rule),
            "one_third_grace_mb": int(deps.one_third_pack_grace_mb),
            "gpu_empty_used_mb": int(deps.gpu_empty_used_mb),
            "fifo_strict_after_s": cls._fifo_strict_after_for(node),
            "live_check": cls._live_check_for(node),
            "live_check_timeout_s": int(info.get("claim_live_check_timeout_s", 3)),
        }
        if node_state and node_state.get("gpus"):
            for gpu in node_state["gpus"]:
                cap["gpu_vram_mb"][str(gpu["idx"])] = int(gpu["total_mb"])
        return cap

    @classmethod
    def claim(cls, node: str, task: dict, gpu_idx: Optional[int],
              node_state: Optional[dict] = None) -> tuple:
        deps = cls.deps()
        if not cls.enabled_for(node):
            return (True, {"reason": "claims disabled for this node"}, "ok")
        now = time.time()
        ttl = cls._ttl_for(node)
        intent_ttl = cls._intent_ttl_for(node)
        try:
            owner = os.environ.get("USER") or os.getlogin() or "?"
        except Exception:
            owner = "?"
        node_info = deps.node_configs.get(node, {})
        ignore_cpu = deps.ignore_cpu_for_server_gpu_task(
            task,
            node_state=node_state,
            node_info=node_info,
            node_name=node,
            gpu_idx=gpu_idx,
        )
        ignore_one_third = deps.task_ignores_one_third_pack_rule(task, node_info)
        record = {
            "owner": owner,
            "scheduler_id": cls.scheduler_id(),
            "task_id": task["id"],
            "gpu_idx": gpu_idx,
            "vram_mb": int(task.get("est_vram_mb") or 0),
            "cpu_cores": 0 if ignore_cpu else int(task.get("cpu_cores") or deps.default_cpu_cores),
            "ram_mb": int(task.get("ram_mb") or deps.default_ram_mb),
            "ignore_cpu_capacity": bool(ignore_cpu),
            "ignore_one_third_pack_rule": bool(ignore_one_third),
            "claimed_at": now,
            "expires_at": now + ttl,
            "intent_expires_at": now + intent_ttl,
            "pid": None,
        }
        result = deps.claims_remote_op(node, "claim", record, cls._build_capacity(node, node_state))
        if result.get("ok"):
            return (True, record, "ok")
        if result.get("conflict"):
            return (False, result["conflict"], "conflict")
        return (False, result.get("error") or "claim transport failed", "error")

    @classmethod
    def release(cls, node: str, task_id: str) -> bool:
        if not cls.enabled_for(node):
            return True
        result = cls.deps().claims_remote_op(node, "release", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
        })
        return bool(result.get("ok"))

    @classmethod
    def update_pid(cls, node: str, task_id: str, pid: Optional[int]) -> bool:
        if not cls.enabled_for(node):
            return True
        result = cls.deps().claims_remote_op(node, "update_pid", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
            "pid": int(pid) if pid else None,
        })
        return bool(result.get("ok"))

    @classmethod
    def renew(cls, node: str, task_id: str) -> bool:
        if not cls.enabled_for(node):
            return True
        new_exp = time.time() + cls._ttl_for(node)
        result = cls.deps().claims_remote_op(node, "renew", {
            "scheduler_id": cls.scheduler_id(),
            "task_id": task_id,
            "expires_at": new_exp,
        })
        return bool(result.get("ok"))

    @classmethod
    def renew_many(cls, node: str, task_ids: list, records: Optional[dict] = None) -> int:
        if not cls.enabled_for(node):
            return 0
        if not task_ids:
            return 0
        new_exp = time.time() + cls._ttl_for(node)
        result = cls.deps().claims_remote_op(node, "renew_many", {
            "scheduler_id": cls.scheduler_id(),
            "task_ids": list(task_ids),
            "expires_at": new_exp,
            "records": records or {},
        })
        if result.get("ok"):
            return int(result.get("renewed", 0))
        return -1

    @classmethod
    def gc_stale(cls, node: str) -> int:
        if not cls.enabled_for(node):
            return 0
        result = cls.deps().claims_remote_op(node, "gc", {})
        if result.get("ok"):
            return int(result.get("removed", 0))
        return -1

    @classmethod
    def enumerate(cls, node: str) -> list:
        return list(cls.snapshot(node).get("claims") or [])

    @classmethod
    def enumerate_intents(cls, node: str) -> list:
        return list(cls.snapshot(node).get("intents") or [])

    @classmethod
    def snapshot(cls, node: str) -> dict:
        if not cls.enabled_for(node):
            return {"ok": True, "claims": [], "intents": [], "disabled": True}
        try:
            result = cls.deps().claims_remote_op(node, "list", {})
        except Exception as exc:
            return {"ok": False, "claims": [], "intents": [], "error": str(exc)[:200]}
        if result.get("ok"):
            result["claims"] = list(result.get("claims", []))
            result["intents"] = list(result.get("intents", []))
            return result
        return {
            "ok": False,
            "claims": [],
            "intents": [],
            "error": result.get("error") or result.get("conflict") or "claims list failed",
        }
