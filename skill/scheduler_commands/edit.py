from __future__ import annotations

import fnmatch
import sys
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class EditCommandDeps:
    node_configs: dict
    canonicalize_node_list: Callable[[list], list]
    release_task_claims_and_intents: Callable[..., Any]
    state_lock: Callable[..., Any]
    load_state: Callable[[], dict]
    save_state: Callable[[dict], Any]


EDIT_FIELDS = (
    "vram_mb", "ram_mb", "cpu", "description", "preferred_node",
    "require_node", "require_gpu_idx", "allow_gpu_over_one_third",
    "allowed_nodes", "clear_allowed_nodes", "resource_family",
    "vram_resource_family", "ram_resource_family",
)


def _validate_requested_edit(args) -> None:
    if all(getattr(args, key, None) is None for key in EDIT_FIELDS):
        sys.exit("specify at least one of --vram-mb / --ram-mb / --cpu / "
                 "--description / --preferred-node / --require-node / --require-gpu / "
                 "--allow-gpu-over-one-third / --allowed-node / --clear-allowed-nodes / "
                 "--resource-family / --vram-resource-family / --ram-resource-family")
    if getattr(args, "allowed_nodes", None) and getattr(args, "clear_allowed_nodes", False):
        sys.exit("use either --allowed-node or --clear-allowed-nodes, not both")


def _parse_required_gpu(value) -> int | None:
    raw = str(value).strip()
    if raw == "":
        return None
    try:
        new_gpu = int(raw)
    except ValueError:
        sys.exit(f"--require-gpu expects a non-negative integer or empty string, got {raw!r}")
    if new_gpu < 0:
        sys.exit(f"--require-gpu expects a non-negative integer or empty string, got {raw!r}")
    return new_gpu


def _known_node_or_exit(node: str, *, field: str, deps: EditCommandDeps) -> None:
    if node not in deps.node_configs:
        sys.exit(f"--{field} {node!r} not in NODES ({list(deps.node_configs.keys())})")


def apply_edit_to_task(task: dict, args, *, deps: EditCommandDeps) -> tuple[list, bool]:
    changes = []
    placement_changed = False

    if args.vram_mb is not None:
        changes.append(("est_vram_mb", task.get("est_vram_mb"), int(args.vram_mb)))
        task["est_vram_mb"] = int(args.vram_mb)
        task["est_vram_mb_explicit"] = True
    if args.ram_mb is not None:
        changes.append(("ram_mb", task.get("ram_mb"), int(args.ram_mb)))
        task["ram_mb"] = int(args.ram_mb)
        task["ram_mb_explicit"] = True
    if args.cpu is not None:
        changes.append(("cpu_cores", task.get("cpu_cores"), int(args.cpu)))
        task["cpu_cores"] = int(args.cpu)
        task["cpu_cores_explicit"] = True
    if args.description is not None:
        changes.append(("description", task.get("description"), args.description))
        task["description"] = args.description
    for arg_name, field_name in (
        ("resource_family", "resource_family"),
        ("vram_resource_family", "vram_resource_family"),
        ("ram_resource_family", "ram_resource_family"),
    ):
        value = getattr(args, arg_name, None)
        if value is None:
            continue
        normalized = str(value).strip() or None
        changes.append((field_name, task.get(field_name), normalized))
        if normalized is None:
            task.pop(field_name, None)
        else:
            task[field_name] = normalized
        placement_changed = True
    if args.preferred_node is not None:
        _known_node_or_exit(args.preferred_node, field="preferred-node", deps=deps)
        changes.append(("preferred_node", task.get("preferred_node"), args.preferred_node))
        task["preferred_node"] = args.preferred_node
        placement_changed = True
    if args.require_node is not None:
        if args.require_node:
            _known_node_or_exit(args.require_node, field="require-node", deps=deps)
        changes.append(("require_node", task.get("require_node"), args.require_node or None))
        task["require_node"] = args.require_node or None
        placement_changed = True

    require_gpu_idx_arg = getattr(args, "require_gpu_idx", None)
    if require_gpu_idx_arg is not None:
        new_gpu = _parse_required_gpu(require_gpu_idx_arg)
        changes.append(("require_gpu_idx", task.get("require_gpu_idx"), new_gpu))
        if new_gpu is None:
            task.pop("require_gpu_idx", None)
        else:
            task["require_gpu_idx"] = new_gpu
        placement_changed = True

    allowed_nodes_arg = getattr(args, "allowed_nodes", None)
    if allowed_nodes_arg is not None:
        allowed = deps.canonicalize_node_list(allowed_nodes_arg or [])
        unknown_allowed = [node for node in allowed if node not in deps.node_configs]
        if unknown_allowed:
            sys.exit(f"--allowed-node contains unknown node(s): {unknown_allowed} "
                     f"({list(deps.node_configs.keys())})")
        changes.append(("allowed_nodes", task.get("allowed_nodes"), allowed or None))
        if allowed:
            task["allowed_nodes"] = allowed
            task["allowed_nodes_user_explicit"] = True
            task["allowed_nodes_submitted"] = list(allowed)
        else:
            task.pop("allowed_nodes", None)
            task.pop("allowed_nodes_user_explicit", None)
            task.pop("allowed_nodes_submitted", None)
        task.pop("allowed_nodes_expanded_reason", None)
        placement_changed = True

    if getattr(args, "clear_allowed_nodes", False):
        changes.append(("allowed_nodes", task.get("allowed_nodes"), None))
        task.pop("allowed_nodes", None)
        task.pop("allowed_nodes_user_explicit", None)
        task.pop("allowed_nodes_submitted", None)
        task.pop("allowed_nodes_expanded_reason", None)
        placement_changed = True

    allow_over_one_third_arg = getattr(args, "allow_gpu_over_one_third", None)
    if allow_over_one_third_arg is not None:
        new_allow = bool(allow_over_one_third_arg)
        changes.append(("allow_gpu_over_one_third", task.get("allow_gpu_over_one_third"), new_allow))
        if new_allow:
            task["allow_gpu_over_one_third"] = True
        else:
            task.pop("allow_gpu_over_one_third", None)
        placement_changed = True

    return changes, placement_changed


def _select_queued_tasks(state: dict, args) -> tuple[list[dict], bool]:
    task_id = str(getattr(args, "id", "") or "").strip()
    project = str(getattr(args, "project", "") or "").strip()
    signature = str(getattr(args, "signature", "") or "").strip()
    selector_mode = bool(project or signature)

    if task_id and selector_mode:
        sys.exit("use either a task id or --project/--signature selectors, not both")
    if not task_id and not selector_mode:
        sys.exit("specify a task id or at least one of --project/--signature")
    if selector_mode and not bool(getattr(args, "confirm", False)):
        sys.exit("selector-based edit requires --confirm")

    if task_id:
        for task in state.get("tasks", []):
            if task.get("id") != task_id:
                continue
            if task.get("status") != "queued":
                sys.exit(
                    f"task {task_id} is {task.get('status')!r}, not queued; "
                    "cannot edit resources of in-flight tasks. Cancel + "
                    "resubmit if you really need to change them."
                )
            return [task], False
        sys.exit(f"task {task_id} not found")

    selected = []
    matched_nonqueued = 0
    for task in state.get("tasks", []):
        if project and not fnmatch.fnmatch(str(task.get("project") or ""), project):
            continue
        if signature and not fnmatch.fnmatch(
            str(task.get("signature") or ""), signature
        ):
            continue
        if task.get("status") != "queued":
            matched_nonqueued += 1
            continue
        selected.append(task)
    if not selected:
        suffix = (
            f"; {matched_nonqueued} matching task(s) were not queued"
            if matched_nonqueued else ""
        )
        sys.exit(f"no queued tasks matched the requested selectors{suffix}")
    return selected, True


def cmd_edit(args, *, deps: EditCommandDeps) -> None:
    _validate_requested_edit(args)
    with deps.state_lock():
        state = deps.load_state()
        selected, selector_mode = _select_queued_tasks(state, args)
        all_changes = []
        for task in selected:
            changes, placement_changed = apply_edit_to_task(task, args, deps=deps)
            all_changes.append((task["id"], changes))
            if placement_changed:
                keep = task.get("require_node") or task.get("preferred_node")
                if keep:
                    deps.release_task_claims_and_intents(
                        task,
                        exclude_nodes={keep},
                        clear_markers=False,
                    )
                else:
                    deps.release_task_claims_and_intents(task)

        deps.save_state(state)
        if selector_mode:
            print(f"updated {len(selected)} queued task(s)")
            return

        for task_id, changes in all_changes:
            for field, old, new in changes:
                print(f"  {field}: {old!r} → {new!r}")
            print(f"updated {task_id}")
