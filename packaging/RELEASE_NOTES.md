# Scheduleurm 0.1.1

Scheduler runtime update for Linux and WSL, requiring Python 3.10+.

- Account for projected CPU and GPU load across a dispatch batch and keep explicit
  node restrictions during checkpoint migration.
- Avoid preempting running jobs for queued work whose prerequisite files are missing.
- Improve task completion tracking and distinguish measured RAM from estimates.
  On Linux, use proportional memory where available to avoid counting shared pages twice.
- Keep large queue reads off the TUI event loop and show observed external workloads
  without adding them to the scheduler queue.
- Keep the CLI, optional TUI and stdio MCP entry points. Node configuration stays
  user-owned; installation does not start services or alter existing task state.

Install the wheel from this release:

```bash
python3 -m venv ~/.venvs/scheduleurm
~/.venvs/scheduleurm/bin/python -m pip install \
  'scheduleurm[mcp,tui] @ https://github.com/erzhu419/scheduleurm/releases/download/v0.1.1/scheduleurm-0.1.1-py3-none-any.whl'
~/.venvs/scheduleurm/bin/scheduleurm --help
```

Configure `~/.claude/scheduler/nodes.json` before using remote nodes. Start
`scheduleurm watch --interval 60` for automatic dispatch. Installing with pip
does not automatically register Claude skills or start systemd services; the
repository installer remains available for that workflow.
