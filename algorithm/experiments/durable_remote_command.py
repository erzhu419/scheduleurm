"""Idempotent remote execution for long controlled measurement commands.

The scheduler transport retries short SSH commands. Replaying a long benchmark
after a transient disconnect can start a duplicate workload, so long probes use
an idempotent remote supervisor instead: one deterministic control directory,
one detached command, and reconnect-safe status polling.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shlex
import time
from typing import Any

from algorithm.experiments.remote_workload_selected_profile_probe import (
    _run_remote_capture,
    _transient_ssh_error,
)


CONTROL_ROOT = "/tmp/scheduleurm_durable_control"
POLL_INTERVAL_S = 15.0
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9_.-]+")


def run_durable_remote_capture(
    node: str,
    shell_cmd: str,
    prefix: Path,
    *,
    run_id: str,
    timeout_s: int,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> tuple[int, str, str, dict[str, Any]]:
    """Run one long command exactly once and resume observation after SSH loss."""

    safe_run_id = str(run_id).strip()
    if not _SAFE_RUN_ID.fullmatch(safe_run_id):
        raise ValueError(f"unsafe durable run id: {run_id!r}")
    if int(timeout_s) <= 0:
        raise ValueError("timeout_s must be positive")
    if float(poll_interval_s) < 0.0:
        raise ValueError("poll_interval_s must be nonnegative")

    control_dir = f"{CONTROL_ROOT}/{safe_run_id}"
    launch_cmd = _launch_command(control_dir=control_dir, shell_cmd=shell_cmd)
    poll_cmd = _poll_command(control_dir)
    started = time.monotonic()
    deadline = started + float(timeout_s)
    launch_attempt_count = 0
    poll_count = 0
    transient_transport_error_count = 0
    resumed_existing = False
    supervisor_pid: int | None = None
    last_transport_error = ""
    state = "ABSENT"

    def launch() -> tuple[int, str, str]:
        nonlocal launch_attempt_count, transient_transport_error_count
        nonlocal resumed_existing, supervisor_pid, last_transport_error
        launch_attempt_count += 1
        rc, out, err = _run_remote_capture(
            node,
            launch_cmd,
            _derived_prefix(prefix, "durable_launch"),
            timeout_s=min(300, max(90, int(timeout_s))),
        )
        markers = _markers(out)
        launch_state = markers.get("DURABLE_STATE")
        resumed_existing = resumed_existing or launch_state == "RESUMED"
        supervisor_pid = _positive_int(markers.get("DURABLE_PID")) or supervisor_pid
        if rc != 0:
            last_transport_error = (err or out)[-2000:]
            if _transport_transient(out, err):
                transient_transport_error_count += 1
        return int(rc), out, err

    launch_rc, launch_out, launch_err = launch()
    if launch_rc != 0 and not _transport_transient(launch_out, launch_err):
        return _finish(
            prefix=prefix,
            rc=launch_rc,
            stdout=launch_out,
            stderr=launch_err,
            audit=_audit(
                run_id=safe_run_id,
                control_dir=control_dir,
                state="LAUNCH_FAILED",
                started=started,
                launch_attempt_count=launch_attempt_count,
                poll_count=poll_count,
                transient_transport_error_count=transient_transport_error_count,
                resumed_existing=resumed_existing,
                supervisor_pid=supervisor_pid,
                last_transport_error=last_transport_error,
            ),
        )

    lost_observations = 0
    while time.monotonic() < deadline:
        poll_count += 1
        rc, out, err = _run_remote_capture(
            node,
            poll_cmd,
            _derived_prefix(prefix, "durable_poll"),
            timeout_s=180,
        )
        if rc != 0:
            last_transport_error = (err or out)[-2000:]
            if _transport_transient(out, err):
                transient_transport_error_count += 1
                _sleep_until_next_poll(deadline, poll_interval_s)
                continue
            state = "POLL_FAILED"
            return _finish(
                prefix=prefix,
                rc=rc,
                stdout=out,
                stderr=err,
                audit=_audit(
                    run_id=safe_run_id,
                    control_dir=control_dir,
                    state=state,
                    started=started,
                    launch_attempt_count=launch_attempt_count,
                    poll_count=poll_count,
                    transient_transport_error_count=transient_transport_error_count,
                    resumed_existing=resumed_existing,
                    supervisor_pid=supervisor_pid,
                    last_transport_error=last_transport_error,
                ),
            )
        markers = _markers(out)
        state = str(markers.get("DURABLE_STATE") or "UNKNOWN")
        supervisor_pid = _positive_int(markers.get("DURABLE_PID")) or supervisor_pid
        if state == "ABSENT":
            launch_rc, launch_out, launch_err = launch()
            if launch_rc != 0 and not _transport_transient(launch_out, launch_err):
                state = "LAUNCH_FAILED"
                break
        elif state == "DONE":
            remote_rc = _nonnegative_int(markers.get("DURABLE_RC"))
            if remote_rc is None:
                state = "RETURN_CODE_MISSING"
                break
            fetched = _fetch_result(
                node=node,
                control_dir=control_dir,
                prefix=prefix,
                deadline=deadline,
            )
            if fetched is None:
                state = "RESULT_FETCH_FAILED"
                break
            stdout, stderr = fetched
            cleanup_rc, _, cleanup_err = _run_remote_capture(
                node,
                f"rm -rf -- {shlex.quote(control_dir)}",
                _derived_prefix(prefix, "durable_cleanup"),
                timeout_s=180,
            )
            audit = _audit(
                run_id=safe_run_id,
                control_dir=control_dir,
                state="DONE",
                started=started,
                launch_attempt_count=launch_attempt_count,
                poll_count=poll_count,
                transient_transport_error_count=transient_transport_error_count,
                resumed_existing=resumed_existing,
                supervisor_pid=supervisor_pid,
                last_transport_error=last_transport_error,
            )
            audit["remote_returncode"] = remote_rc
            audit["control_cleanup_ready"] = cleanup_rc == 0
            audit["control_cleanup_error"] = cleanup_err[-1000:]
            return _finish(
                prefix=prefix,
                rc=remote_rc,
                stdout=stdout,
                stderr=stderr,
                audit=audit,
            )
        elif state == "LOST":
            lost_observations += 1
            if lost_observations >= 2:
                break
        else:
            lost_observations = 0
        _sleep_until_next_poll(deadline, poll_interval_s)

    timed_out = time.monotonic() >= deadline
    rc = 124 if timed_out else 255
    stderr = (
        f"durable remote command {state.lower()} for {safe_run_id}; "
        f"last transport error: {last_transport_error}"
    )
    return _finish(
        prefix=prefix,
        rc=rc,
        stdout="",
        stderr=stderr,
        audit=_audit(
            run_id=safe_run_id,
            control_dir=control_dir,
            state="TIMEOUT" if timed_out else state,
            started=started,
            launch_attempt_count=launch_attempt_count,
            poll_count=poll_count,
            transient_transport_error_count=transient_transport_error_count,
            resumed_existing=resumed_existing,
            supervisor_pid=supervisor_pid,
            last_transport_error=last_transport_error,
        ),
    )


def _launch_command(*, control_dir: str, shell_cmd: str) -> str:
    command_payload = base64.b64encode(shell_cmd.encode("utf-8")).decode("ascii")
    supervisor = "\n".join(
        (
            "#!/usr/bin/env bash",
            "set +e",
            f"bash {shlex.quote(control_dir + '/command.sh')} "
            f">{shlex.quote(control_dir + '/stdout')} "
            f"2>{shlex.quote(control_dir + '/stderr')}",
            "rc=$?",
            f"printf '%s\\n' \"$rc\" > {shlex.quote(control_dir + '/returncode.tmp')}",
            f"mv {shlex.quote(control_dir + '/returncode.tmp')} "
            f"{shlex.quote(control_dir + '/returncode')}",
            f"touch {shlex.quote(control_dir + '/done')}",
            "exit 0",
        )
    )
    supervisor_payload = base64.b64encode(supervisor.encode("utf-8")).decode("ascii")
    parent = str(Path(control_dir).parent)
    q_control = shlex.quote(control_dir)
    return "\n".join(
        (
            "set -u",
            f"mkdir -p {shlex.quote(parent)}",
            f"if mkdir {q_control} 2>/dev/null; then",
            f"  printf '%s' {shlex.quote(command_payload)} | base64 -d > "
            f"{shlex.quote(control_dir + '/command.sh')}",
            f"  printf '%s' {shlex.quote(supervisor_payload)} | base64 -d > "
            f"{shlex.quote(control_dir + '/supervisor.sh')}",
            f"  chmod 700 {shlex.quote(control_dir + '/command.sh')} "
            f"{shlex.quote(control_dir + '/supervisor.sh')}",
            f"  setsid bash {shlex.quote(control_dir + '/supervisor.sh')} "
            "</dev/null >/dev/null 2>&1 &",
            "  supervisor_pid=$!",
            f"  printf '%s\\n' \"$supervisor_pid\" > {shlex.quote(control_dir + '/pid.tmp')}",
            f"  mv {shlex.quote(control_dir + '/pid.tmp')} {shlex.quote(control_dir + '/pid')}",
            f"  touch {shlex.quote(control_dir + '/launched')}",
            "  printf '__DURABLE_STATE__ LAUNCHED\\n'",
            "else",
            f"  if [ -f {shlex.quote(control_dir + '/launched')} ] || "
            f"[ -f {shlex.quote(control_dir + '/done')} ]; then",
            "    printf '__DURABLE_STATE__ RESUMED\\n'",
            "  else",
            "    printf '__DURABLE_STATE__ PENDING\\n'",
            "  fi",
            "fi",
            f"if [ -f {shlex.quote(control_dir + '/pid')} ]; then "
            f"printf '__DURABLE_PID__ '; cat {shlex.quote(control_dir + '/pid')}; fi",
        )
    )


def _poll_command(control_dir: str) -> str:
    q_done = shlex.quote(control_dir + "/done")
    q_launched = shlex.quote(control_dir + "/launched")
    q_pid = shlex.quote(control_dir + "/pid")
    q_rc = shlex.quote(control_dir + "/returncode")
    q_control = shlex.quote(control_dir)
    return "\n".join(
        (
            f"if [ ! -d {q_control} ]; then",
            "  printf '__DURABLE_STATE__ ABSENT\\n'",
            f"elif [ -f {q_done} ]; then",
            "  printf '__DURABLE_STATE__ DONE\\n'",
            f"  printf '__DURABLE_RC__ '; cat {q_rc}",
            f"  printf '__DURABLE_PID__ '; cat {q_pid}",
            f"elif [ -f {q_launched} ] && [ -f {q_pid} ]; then",
            f"  supervisor_pid=$(cat {q_pid})",
            "  if kill -0 \"$supervisor_pid\" 2>/dev/null; then",
            "    printf '__DURABLE_STATE__ RUNNING\\n'",
            "  else",
            "    printf '__DURABLE_STATE__ LOST\\n'",
            "  fi",
            "  printf '__DURABLE_PID__ %s\\n' \"$supervisor_pid\"",
            "else",
            "  printf '__DURABLE_STATE__ PENDING\\n'",
            "fi",
        )
    )


def _fetch_result(
    *, node: str, control_dir: str, prefix: Path, deadline: float
) -> tuple[str, str] | None:
    outputs = []
    for name in ("stdout", "stderr"):
        while time.monotonic() < deadline:
            rc, out, err = _run_remote_capture(
                node,
                f"cat -- {shlex.quote(control_dir + '/' + name)}",
                _derived_prefix(prefix, f"durable_fetch_{name}"),
                timeout_s=180,
            )
            if rc == 0:
                outputs.append(out)
                break
            if not _transport_transient(out, err):
                return None
            _sleep_until_next_poll(deadline, POLL_INTERVAL_S)
        else:
            return None
    return outputs[0], outputs[1]


def _markers(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in str(text).splitlines():
        match = re.fullmatch(r"__([A-Z0-9_]+)__\s*(.*)", line.strip())
        if match:
            result[match.group(1)] = match.group(2).strip()
    return result


def _transport_transient(out: str | None, err: str | None) -> bool:
    text = f"{out or ''}\n{err or ''}".lower()
    return _transient_ssh_error(out, err) or bool(
        re.search(r"server\s+\S+\s+not responding", text)
    )


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_int(value: object) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _sleep_until_next_poll(deadline: float, interval_s: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0.0 and interval_s > 0.0:
        time.sleep(min(float(interval_s), remaining))


def _audit(
    *,
    run_id: str,
    control_dir: str,
    state: str,
    started: float,
    launch_attempt_count: int,
    poll_count: int,
    transient_transport_error_count: int,
    resumed_existing: bool,
    supervisor_pid: int | None,
    last_transport_error: str,
) -> dict[str, Any]:
    return {
        "protocol": "idempotent_detached_remote_supervisor_v1",
        "run_id": run_id,
        "control_dir": control_dir,
        "state": state,
        "elapsed_wall_s": time.monotonic() - started,
        "launch_attempt_count": launch_attempt_count,
        "poll_count": poll_count,
        "transient_transport_error_count": transient_transport_error_count,
        "resumed_existing": resumed_existing,
        "supervisor_pid": supervisor_pid,
        "long_command_replayed": False,
        "last_transport_error": last_transport_error,
    }


def _finish(
    *, prefix: Path, rc: int, stdout: str, stderr: str, audit: dict[str, Any]
) -> tuple[int, str, str, dict[str, Any]]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".stdout").write_text(stdout or "", encoding="utf-8")
    prefix.with_suffix(".stderr").write_text(stderr or "", encoding="utf-8")
    prefix.with_suffix(".meta.json").write_text(
        json.dumps(
            {"returncode": int(rc), "durable_transport": audit},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(rc), stdout or "", stderr or "", audit


def _derived_prefix(prefix: Path, suffix: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{suffix}")


__all__ = ["run_durable_remote_capture"]
