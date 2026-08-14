from __future__ import annotations

from typing import Any, Mapping, Optional


def _ns(namespace: Mapping[str, Any], name: str) -> Any:
    return namespace[name]


def build_doctor_runtime_exports(namespace: dict[str, Any]) -> dict[str, Any]:
    def _doctor_deps():
        return _ns(namespace, "_build_doctor_deps")(namespace)

    def _flag_values_any(tokens: list, flags: set) -> list:
        return _ns(namespace, "_flag_values_any_impl")(
            tokens,
            flags,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def _simple_sac_large_data_reason(cmd: str, cwd: str) -> Optional[str]:
        return _ns(namespace, "_simple_sac_large_data_reason_impl")(
            cmd,
            cwd,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def _simple_sac_ckpt_for_method(script_path: str, method: str) -> Optional[str]:
        return _ns(namespace, "_simple_sac_ckpt_for_method_impl")(
            script_path,
            method,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def _simple_sac_eval_prereq_files(cmd: str, cwd: str) -> list:
        return _ns(namespace, "_simple_sac_eval_prereq_files_impl")(
            cmd,
            cwd,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def _simple_sac_train_best_ckpt_from_cmd(cmd: str, cwd: str) -> Optional[str]:
        return _ns(namespace, "_simple_sac_train_best_ckpt_from_cmd_impl")(
            cmd,
            cwd,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def _doctor_issue(
        task: Optional[dict],
        code: str,
        severity: str,
        message: str,
        fix: str = "",
        fixed: bool = False,
        path: str = "",
    ) -> dict:
        return _ns(namespace, "_doctor_issue_impl")(
            task,
            code,
            severity,
            message,
            fix=fix,
            fixed=fixed,
            path=path,
        )

    def _doctor_scan_state(state: dict, fix: bool = False, project: str = ""):
        return _ns(namespace, "_doctor_scan_state_impl")(
            state,
            fix=fix,
            project=project,
            deps=_ns(namespace, "_doctor_deps")(),
        )

    def cmd_doctor(args):
        return _ns(namespace, "_cmd_doctor_impl")(args, deps=_ns(namespace, "_doctor_deps")())

    def _profile_local_deps():
        return _ns(namespace, "_build_profile_local_deps")(namespace)

    def cmd_profile_local(args):
        return _ns(namespace, "_cmd_profile_local_impl")(
            args,
            deps=_ns(namespace, "_profile_local_deps")(),
        )

    return {
        "_doctor_deps": _doctor_deps,
        "_flag_values_any": _flag_values_any,
        "_simple_sac_large_data_reason": _simple_sac_large_data_reason,
        "_simple_sac_ckpt_for_method": _simple_sac_ckpt_for_method,
        "_simple_sac_eval_prereq_files": _simple_sac_eval_prereq_files,
        "_simple_sac_train_best_ckpt_from_cmd": _simple_sac_train_best_ckpt_from_cmd,
        "_doctor_issue": _doctor_issue,
        "_doctor_scan_state": _doctor_scan_state,
        "cmd_doctor": cmd_doctor,
        "_profile_local_deps": _profile_local_deps,
        "cmd_profile_local": cmd_profile_local,
    }
