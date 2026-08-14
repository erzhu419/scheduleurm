from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_external_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    exports: dict[str, Any] = {
        "_DESCENDANTS_CAP": 500,
    }

    def _adopt_probe_deps():
        return _ns(namespace, "_build_adopt_probe_deps")(namespace)

    def _node_processes(name):
        return _ns(namespace, "_node_processes_impl")(
            name,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _node_ppid_map(name):
        return _ns(namespace, "_node_ppid_map_impl")(
            name,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _descendants_of(roots, ppid_of):
        return _ns(namespace, "_descendants_of_impl")(
            roots,
            ppid_of,
            cap=_ns(namespace, "_DESCENDANTS_CAP"),
        )

    def _node_expected_process_owner(name: str) -> str:
        return _ns(namespace, "_node_expected_process_owner_impl")(
            name,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _node_auto_adopt_owners(name: str) -> set[str]:
        return _ns(namespace, "_node_auto_adopt_owners_impl")(
            name,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _node_auto_adopt_roots(name: str, owner: str) -> list[str]:
        return _ns(namespace, "_node_auto_adopt_roots_impl")(
            name,
            owner,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _path_under_roots(path: str, roots: list[str]) -> bool:
        return _ns(namespace, "_path_under_roots_impl")(path, roots)

    def _infer_adopt_cwd_from_cmdline(cmdline: str, roots: list[str]) -> str:
        return _ns(namespace, "_infer_adopt_cwd_from_cmdline_impl")(cmdline, roots)

    def _is_scheduler_control_cmdline(cmdline: str) -> bool:
        return _ns(namespace, "_is_scheduler_control_cmdline_impl")(
            cmdline,
            scheduler_file=namespace.get("__file__") or __file__,
            control_subcommands=_ns(namespace, "SCHEDULER_CONTROL_SUBCOMMANDS"),
            path_markers=_ns(namespace, "SCHEDULER_CONTROL_PATH_MARKERS"),
        )

    def _is_scheduleurm_repo_path(path: str) -> bool:
        return _ns(namespace, "_is_repo_path_impl")(
            path,
            repo_root=_ns(namespace, "_SCHEDULEURM_ROOT"),
        )

    def _is_scheduler_control_process(cwd: str, cmdline: str) -> bool:
        return _ns(namespace, "_is_scheduler_control_process_impl")(
            cwd,
            cmdline,
            repo_root=_ns(namespace, "_SCHEDULEURM_ROOT"),
            scheduler_file=namespace.get("__file__") or __file__,
            control_subcommands=_ns(namespace, "SCHEDULER_CONTROL_SUBCOMMANDS"),
            path_markers=_ns(namespace, "SCHEDULER_CONTROL_PATH_MARKERS"),
        )

    def _node_cpu_processes(name):
        return _ns(namespace, "_node_cpu_processes_impl")(
            name,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _refresh_adopted_resources(state, gpu_proc_lists, cpu_proc_lists):
        return _ns(namespace, "_refresh_adopted_resources_impl")(
            state,
            gpu_proc_lists,
            cpu_proc_lists,
            deps=_ns(namespace, "_adopt_probe_deps")(),
        )

    def _collect_external_task_probe_data(
        state: Optional[dict] = None,
        *,
        eligible_node_names: Optional[set[str]] = None,
    ) -> dict:
        return _ns(namespace, "_collect_external_task_probe_data_impl")(
            state,
            deps=_ns(namespace, "_adopt_probe_deps")(),
            eligible_node_names=eligible_node_names,
            node_processes_fn=_ns(namespace, "_node_processes"),
            node_cpu_processes_fn=_ns(namespace, "_node_cpu_processes"),
            node_ppid_map_fn=_ns(namespace, "_node_ppid_map"),
        )

    def _reconcile_external_tasks(state, probe_data: Optional[dict] = None):
        return _ns(namespace, "_reconcile_external_tasks_impl")(
            state,
            probe_data=probe_data,
            deps=_ns(namespace, "_external_reconcile_deps")(),
        )

    def _external_reconcile_deps():
        return _ns(namespace, "_build_external_reconcile_deps")(namespace)

    def _resource_accounting_deps():
        return _ns(namespace, "_build_resource_accounting_deps")(namespace)

    def _cpu_ownership_snapshot(state, nodes) -> list:
        return _ns(namespace, "_cpu_ownership_snapshot_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_resource_accounting_deps")(),
        )

    def _resource_accounting_payload(state, nodes) -> dict:
        return _ns(namespace, "_resource_accounting_payload_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_resource_accounting_deps")(),
        )

    def _dispatch_cycle_log_payload(state, nodes, events, queued_count: int) -> dict:
        return _ns(namespace, "_dispatch_cycle_log_payload_impl")(
            state,
            nodes,
            events,
            queued_count,
            deps=_ns(namespace, "_resource_accounting_deps")(),
        )

    def _build_heartbeat_payload(state, nodes):
        return _ns(namespace, "_build_heartbeat_payload_impl")(
            state,
            nodes,
            deps=_ns(namespace, "_node_reporting_deps")(),
        )

    exports.update({
        "_adopt_probe_deps": _adopt_probe_deps,
        "_node_processes": _node_processes,
        "_node_ppid_map": _node_ppid_map,
        "_descendants_of": _descendants_of,
        "_node_expected_process_owner": _node_expected_process_owner,
        "_node_auto_adopt_owners": _node_auto_adopt_owners,
        "_node_auto_adopt_roots": _node_auto_adopt_roots,
        "_path_under_roots": _path_under_roots,
        "_infer_adopt_cwd_from_cmdline": _infer_adopt_cwd_from_cmdline,
        "_is_scheduler_control_cmdline": _is_scheduler_control_cmdline,
        "_is_scheduleurm_repo_path": _is_scheduleurm_repo_path,
        "_is_scheduler_control_process": _is_scheduler_control_process,
        "_node_cpu_processes": _node_cpu_processes,
        "_refresh_adopted_resources": _refresh_adopted_resources,
        "_collect_external_task_probe_data": _collect_external_task_probe_data,
        "_reconcile_external_tasks": _reconcile_external_tasks,
        "_external_reconcile_deps": _external_reconcile_deps,
        "_resource_accounting_deps": _resource_accounting_deps,
        "_cpu_ownership_snapshot": _cpu_ownership_snapshot,
        "_resource_accounting_payload": _resource_accounting_payload,
        "_dispatch_cycle_log_payload": _dispatch_cycle_log_payload,
        "_build_heartbeat_payload": _build_heartbeat_payload,
    })
    return exports
