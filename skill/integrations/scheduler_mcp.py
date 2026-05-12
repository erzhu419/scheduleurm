#!/usr/bin/env python3
"""MCP server wrapping ~/.claude/skills/scheduler/scheduler.py for cross-AI use.

Same scheduler — different transport. Claude Code uses the SKILL.md auto-route; this MCP
server exposes the same operations as standard MCP tools so any MCP-aware client (ChatGPT
Desktop, Claude Desktop, Cursor, Cline, Continue, custom agents) can drive the queue.

Tool descriptions are written so the host LLM can auto-route user intent to the right call
(e.g. "跑这个 python script" → submit_task; "GPU 还空吗" → node_status; "看看 t0007" → show_task).

Usage (one-time, per client):
    pip install mcp
    Add to client's MCP config:
      {"command": "/usr/bin/python3",
       "args": ["/home/erzhu419/.claude/skills/scheduler/integrations/scheduler_mcp.py"]}

Then restart the client. Tools auto-appear with `submit_task`, `dispatch`, etc.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.exit("MCP SDK not found. Run: pip install mcp")

SCHED = os.environ.get(
    "SCHEDULER_PY",
    str(Path.home() / ".claude" / "skills" / "scheduler" / "scheduler.py"),
)
QUEUE = Path.home() / ".claude" / "scheduler" / "queue.json"

mcp = FastMCP("scheduler")


def _run(args: list[str], timeout: int = 60) -> dict:
    """Invoke scheduler.py with args, return {ok, stdout, stderr, exit_code}."""
    try:
        r = subprocess.run(
            [sys.executable, SCHED, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "ok": r.returncode == 0,
            "exit_code": r.returncode,
            "stdout": r.stdout,
            "stderr": r.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "exit_code": -1, "stdout": "", "stderr": str(e)}


@mcp.tool()
def submit_task(
    description: str,
    cmd: str,
    cwd: str,
    signature: str,
    vram_mb: Optional[int] = None,
    ram_mb: Optional[int] = None,
    cpu_cores: Optional[int] = None,
    require_node: Optional[str] = None,
    preferred_node: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
    resume_flag: Optional[str] = None,
    result_dir: Optional[str] = None,
    local_result_dir: Optional[str] = None,
    wait_for_files: Optional[list[str]] = None,
    test_log: Optional[str] = None,
    test_peak_vram_mb: Optional[int] = None,
    test_peak_ram_mb: Optional[int] = None,
    test_cpu: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    priority: str = "normal",
    allow_cpu_training: bool = False,
    cpu_training_justification: Optional[str] = None,
    allow_no_resume: bool = False,
    allow_no_ckpt: bool = False,
    allow_shared_ckpt_dir: bool = False,
    allow_remote_large_data: bool = False,
    env_spec: str = "none",
    image: Optional[str] = None,
) -> dict:
    """Submit a new task to the scheduler queue.

    Use this when the user wants to launch ANY long-running computation: training, eval,
    data prep, sweep. Pick `cwd` as the absolute path on the target node where the cmd
    runs. `signature` is a stable id like "ProjectName/config-name" — same signature for
    re-runs so peak vram/ram history accumulates and auto-fills future submissions.

    Resource hints (vram_mb/ram_mb/cpu_cores): pass 0 to let the scheduler use sig history
    or DEFAULT (512MB vram, 4096MB ram, 1 cpu). Pass an explicit number when you have a
    specific reason. CPU-only tasks must use vram_mb=0.

    Resume flow: pass `ckpt_dir` (absolute path on target node) + `resume_flag` (e.g.
    '--resume_from') and the scheduler will inject the latest ckpt on every relaunch.

    Dependency flow: pass `wait_for_files` for eval jobs that require a local checkpoint
    file to exist before dispatch. This is how train-before-eval is enforced without
    burning CPU on missing-model eval loops.

    Local preflight profiling: pass `test_log` if the command has already been run locally
    with tqdm/progress output. The scheduler records its runtime projection before launch
    so ETA/walltime do not start from a blind guess. Pass test_peak_* values when the local
    preflight measured RAM/VRAM/CPU peaks.

    Submit-time guards (refuse with explanatory error):
      - training cmd + vram_mb=0 → REFUSED unless allow_cpu_training=True AND
        cpu_training_justification (>=30 chars) is provided.
      - training cmd without ckpt_dir → REFUSED unless allow_no_ckpt=True.
      - training cmd with ckpt_dir but no `--resume*` flag in cmd nor resume_flag set →
        REFUSED unless allow_no_resume=True.
      - ckpt_dir already in use by an active task with a DIFFERENT signature →
        REFUSED unless allow_shared_ckpt_dir=True.

    Env delivery (env_spec / image):
      - 'none' (default): cmd assumes env (e.g. conda) exists on target. Use when target
        already has the right conda env at the absolute python path in cmd.
      - 'docker:IMAGE[:TAG]' (with image=IMAGE param OR baked into spec): launch wraps cmd
        in a named `docker run --rm` container with `--gpus device=N`,
        `CUDA_VISIBLE_DEVICES=0`, `--memory <ram_mb>m`, `--cpus <cpu_cores>`, and
        `-v cwd:cwd -w cwd`. First time or digest drift on a target, scheduler pushes the
        image via `docker save | ssh docker load`. Requires target docker daemon access.
      - 'conda:/abs/path/to/env': before dispatch, scheduler incrementally rsyncs the
        local conda env to remote targets at the same absolute path. The cmd should use
        that env's absolute python path.
      - 'auto': probes target docker, falls back to 'none' if unavailable. Provide `image`
        param to enable the docker fallback path.

    Returns: {ok, exit_code, stdout, stderr}. On success stdout has "submitted tXXXX".
    """
    args = [
        "submit",
        "--description", description,
        "--cmd", cmd,
        "--cwd", cwd,
        "--signature", signature,
        "--priority", priority,
    ]
    # Explicit-zero check: vram_mb=0 means "CPU-only task" (legitimate); only skip when None.
    # Was `if vram_mb:` (treating 0 as missing) → CPU-only tasks silently became default-VRAM
    # GPU tasks. Codex review caught this.
    if vram_mb is not None:    args += ["--vram", str(vram_mb)]
    if ram_mb is not None:     args += ["--ram-mb", str(ram_mb)]
    if cpu_cores is not None:  args += ["--cpu", str(cpu_cores)]
    if require_node:   args += ["--require-node", require_node]
    if preferred_node: args += ["--preferred-node", preferred_node]
    if ckpt_dir:       args += ["--ckpt-dir", ckpt_dir]
    if resume_flag:    args += [f"--resume-flag={resume_flag}"]
    if result_dir:     args += ["--result-dir", result_dir]
    if local_result_dir: args += ["--local-result-dir", local_result_dir]
    if wait_for_files:
        for path in wait_for_files:
            args += ["--wait-for-file", str(path)]
    if test_log:      args += ["--test-log", test_log]
    if test_peak_vram_mb is not None: args += ["--test-peak-vram-mb", str(test_peak_vram_mb)]
    if test_peak_ram_mb is not None:  args += ["--test-peak-ram-mb", str(test_peak_ram_mb)]
    if test_cpu is not None:          args += ["--test-cpu", str(test_cpu)]
    if extra_env:
        args += ["--env"] + [f"{k}={v}" for k, v in extra_env.items()]
    if allow_cpu_training:
        args += ["--allow-cpu-training"]
        if cpu_training_justification:
            args += ["--cpu-training-justification", cpu_training_justification]
    if allow_no_resume:        args += ["--allow-no-resume"]
    if allow_no_ckpt:          args += ["--allow-no-ckpt"]
    if allow_shared_ckpt_dir:  args += ["--allow-shared-ckpt-dir"]
    if allow_remote_large_data: args += ["--allow-remote-large-data"]
    if env_spec and env_spec != "none":
        args += ["--env-spec", env_spec]
    if image:
        args += ["--image", image]
    return _run(args)


@mcp.tool()
def dispatch() -> dict:
    """Trigger one dispatch cycle: probe nodes, evict at thresholds, launch what fits.

    Use after `submit_task` to wake the scheduler immediately (otherwise the watcher polls
    every 60s). Also useful when user says "rebalance" or "派活" — replays placement.
    """
    return _run(["dispatch"], timeout=120)


@mcp.tool()
def status() -> dict:
    """Get scheduler status: per-node resources (CPU/RAM/GPU mem+util) + per-task state.

    Use when user asks "GPU/CPU/RAM 还空吗", "现在跑啥呢", "节点状态", "任务到哪了".
    Returns text-formatted overview; parse stdout for structured info.
    """
    return _run(["status"])


@mcp.tool()
def doctor(fix: bool = False, project: Optional[str] = None) -> dict:
    """Audit active queue invariants and optionally apply safe queued-task repairs.

    Use before or after scheduler writes when user asks why tasks launched out of order,
    eval ran before train, SimpleSAC data was sent remote, or completed results seem
    invisible. With fix=False this is read-only. With fix=True it only edits queued
    records: add wait_for_files to evals, force known large SimpleSAC data local,
    promote producer trains to high priority when queued evals depend on their
    ckpt, and clear stale live ETA/progress inherited by queued retries.
    Running tasks are never modified; they are reported for manual decision.
    """
    args = ["doctor", "--json"]
    if fix:
        args.append("--fix")
    if project:
        args += ["--project", project]
    return _run(args)


@mcp.tool()
def show_task(task_id: str) -> dict:
    """Show full details of a single task (cmd, signature, state, peak vram/ram, log path).

    Use when user asks "看看 tXXXX" or wants to debug a specific task.
    """
    return _run(["show", task_id])


@mcp.tool()
def cancel_task(task_id: str, force: bool = False) -> dict:
    """Cancel a queued task or kill a running task.

    For a queued task: instant cancel, no force needed.
    For a running task: pass force=True to send SIGKILL. **Confirm with user first** —
    SIGKILL discards in-memory progress; only ckpt-saved state survives.

    Resumable-task safety rule: never force-kill a task with usable ckpt without explicit
    user approval. The user has memorialized this as a hard constraint.
    """
    args = ["cancel", task_id]
    if force: args.append("--force")
    return _run(args)


@mcp.tool()
def history(signature: Optional[str] = None) -> dict:
    """Get resource peak history per signature (peak_vram_mb, peak_ram_mb, n_runs).

    Pass `signature` to filter to one. Useful before submitting a new task with a known
    signature: peek history to size vram/ram correctly without manual estimation.
    """
    args = ["history"]
    if signature: args += ["--signature", signature]
    return _run(args)


@mcp.tool()
def queue_dump(filter_status: Optional[str] = None) -> dict:
    """Dump the raw queue.json. Optional filter_status: 'queued','running','done','failed','cancelled'.

    Returns parsed JSON of matching task records. Use for programmatic analysis when the
    `status` formatted output isn't enough (e.g. "list all failed offline-sumo tasks in
    last hour", "find tasks with >50h lifetime").
    """
    try:
        data = json.loads(QUEUE.read_text())
    except Exception as e:
        return {"ok": False, "stderr": f"could not read queue.json: {e}", "tasks": []}
    tasks = data.get("tasks", [])
    if filter_status:
        tasks = [t for t in tasks if t.get("status") == filter_status]
    return {"ok": True, "tasks": tasks, "count": len(tasks)}


@mcp.tool()
def task_log(task_id: str, tail_lines: int = 50) -> dict:
    """Read the last N lines of a task's stdout/stderr log.

    Use to debug failures or check progress. The log path differs by node (local: under
    ~/.claude/scheduler/logs/, remote: /tmp/sched_<id>.log on the remote host).
    """
    # Get log_path from queue.json
    try:
        data = json.loads(QUEUE.read_text())
    except Exception as e:
        return {"ok": False, "stderr": f"queue read failed: {e}"}
    task = next((t for t in data.get("tasks", []) if t.get("id") == task_id), None)
    if not task:
        return {"ok": False, "stderr": f"task {task_id} not found in queue"}
    log_path = task.get("log_path")
    node = task.get("node")
    if not log_path:
        return {"ok": False, "stderr": f"task {task_id} has no log_path (likely not yet launched)"}
    # Local node: read directly
    if node == "local" or not node:
        try:
            with open(log_path) as f:
                lines = f.readlines()
            return {"ok": True, "log_path": log_path, "tail": "".join(lines[-tail_lines:])}
        except Exception as e:
            return {"ok": False, "stderr": f"could not read {log_path}: {e}"}
    # Remote node: ssh tail
    try:
        r = subprocess.run(
            ["ssh", node, f"tail -n {tail_lines} {log_path}"],
            capture_output=True, text=True, timeout=15,
        )
        return {
            "ok": r.returncode == 0,
            "log_path": f"{node}:{log_path}",
            "tail": r.stdout,
            "stderr": r.stderr,
        }
    except Exception as e:
        return {"ok": False, "stderr": f"ssh tail failed: {e}"}


if __name__ == "__main__":
    mcp.run()
