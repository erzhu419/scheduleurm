"""Runtime-bound CPU batch submit command wrappers for scheduler.py."""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_cpu_submit_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _parse_cpu_node_list(text: Optional[str]) -> list:
        if not text:
            return _ns(namespace, "_cpu_labor_node_names")()
        names = [x.strip() for x in str(text).split(",") if x.strip()]
        bad = [
            n for n in names
            if n not in _ns(namespace, "NODES")
            or (_ns(namespace, "NODES").get(n, {}) or {}).get("retired")
        ]
        if bad:
            sys.exit(f"unknown or retired node(s): {', '.join(bad)}")
        return names

    def _cpu_plan_live_node_states(node_names: list) -> dict:
        want = set(node_names or [])
        try:
            nodes = _ns(namespace, "probe_all")()
        except Exception:
            return {}
        return {n.get("name"): n for n in nodes if n.get("name") in want}

    def _print_cpu_plan(plan: list, total_items: int) -> None:
        total_free = sum(int(p.get("physical_cores") or 0) for p in plan)
        total_capacity = sum(
            int(p.get("total_physical_cores") or p.get("physical_cores") or 0)
            for p in plan
        )
        global_waves = max(
            [int(p.get("global_waves") or p.get("waves") or 0) for p in plan] or [0]
        )
        print(
            f"cpu-plan: total_items={total_items} free_physical={total_free}/"
            f"{total_capacity} global_waves={global_waves}"
        )
        for p in plan:
            print(
                f"  {p['node']:11s} range=[{p['start']},{p['end']}) "
                f"items={p['items']} free_physical={p['physical_cores']}/"
                f"{p.get('total_physical_cores', p['physical_cores'])} "
                f"workers={p['workers']} waves={p['waves']} "
                f"last_wave={p['last_wave_items']}"
            )

    def _cpu_batch_submit_spec(
        args,
        p: dict,
        total_items: int,
        logical_items: int,
        item_multiplier: int,
        names: list[str],
    ) -> tuple[dict, dict]:
        return _ns(namespace, "_build_cpu_batch_submit_spec_impl")(
            args,
            p,
            total_items=total_items,
            logical_items=logical_items,
            item_multiplier=item_multiplier,
            node_names=names,
            deps=_ns(namespace, "_CpuBatchSubmitDeps")(
                cpu_parallel_template_values=_ns(namespace, "_cpu_parallel_template_values"),
                format_cpu_parallel_template=_ns(namespace, "_format_cpu_parallel_template"),
                rewrite_cpu_parallel_cmd=_ns(namespace, "_rewrite_cpu_parallel_cmd"),
            ),
        )

    def _validate_cpu_batch_submit_spec(spec: dict, args) -> None:
        return _ns(namespace, "_validate_cpu_batch_submit_spec_impl")(
            spec,
            args,
            deps=_ns(namespace, "_CpuBatchSubmitValidationDeps")(
                submit_policy_text=_ns(namespace, "_submit_policy_text"),
                simple_sac_large_data_reason=_ns(namespace, "_simple_sac_large_data_reason"),
                cpu_training_policy_reason=_ns(namespace, "_cpu_training_policy_reason"),
                task_looks_like_training=_ns(namespace, "_task_looks_like_training"),
                cmd_looks_like_training=_ns(namespace, "_cmd_looks_like_training"),
                resume_capability_reason=_ns(namespace, "_resume_capability_reason"),
                checkpoint_contract_reason=_ns(namespace, "_checkpoint_contract_reason"),
                same_declared_path=_ns(namespace, "_same_declared_path"),
            ),
        )

    def _validate_bulk_submit_result_conflicts(
        state: dict,
        specs: list[dict],
        *,
        allow_shared_result_dir: bool,
    ) -> None:
        msg = _ns(namespace, "_bulk_submit_result_conflict_message")(
            state,
            specs,
            allow_shared_result_dir=allow_shared_result_dir,
            active_or_unsynced_result_status=_ns(namespace, "_active_or_unsynced_result_status"),
            conflict_path_key=_ns(namespace, "_conflict_path_key"),
            same_declared_path=_ns(namespace, "_same_declared_path"),
        )
        if msg:
            sys.exit(msg)

    def _submit_cpu_batch_command_deps():
        return _ns(namespace, "_build_submit_cpu_batch_command_deps")(namespace)

    def cmd_cpu_plan(args):
        logical_items, item_multiplier, total_items = _ns(namespace, "_cpu_batch_item_counts")(args)
        names = _ns(namespace, "_parse_cpu_node_list")(getattr(args, "nodes", None))
        node_states = (
            None
            if getattr(args, "use_total_cores", False)
            else _ns(namespace, "_cpu_plan_live_node_states")(names)
        )
        plan = _ns(namespace, "_cpu_batch_plan")(total_items, names, node_states=node_states)
        if args.json:
            print(json.dumps({
                "logical_items": logical_items,
                "item_multiplier": item_multiplier,
                "total_items": total_items,
                "plan": plan,
            }, indent=2))
            return
        if item_multiplier != 1:
            print(
                f"cpu-plan input: logical_items={logical_items} "
                f"item_multiplier={item_multiplier} total_items={total_items}"
            )
        _ns(namespace, "_print_cpu_plan")(plan, total_items)

    def cmd_submit_cpu_batch(args):
        return _ns(namespace, "_cmd_submit_cpu_batch_impl")(
            args,
            deps=_ns(namespace, "_submit_cpu_batch_command_deps")(),
        )

    return {
        "_parse_cpu_node_list": _parse_cpu_node_list,
        "_cpu_plan_live_node_states": _cpu_plan_live_node_states,
        "_print_cpu_plan": _print_cpu_plan,
        "_cpu_batch_submit_spec": _cpu_batch_submit_spec,
        "_validate_cpu_batch_submit_spec": _validate_cpu_batch_submit_spec,
        "_validate_bulk_submit_result_conflicts": _validate_bulk_submit_result_conflicts,
        "_submit_cpu_batch_command_deps": _submit_cpu_batch_command_deps,
        "cmd_cpu_plan": cmd_cpu_plan,
        "cmd_submit_cpu_batch": cmd_submit_cpu_batch,
    }
