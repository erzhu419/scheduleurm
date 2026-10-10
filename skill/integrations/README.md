# Scheduler MCP Integration

This local stdio MCP server exposes the existing scheduler CLI. It does not
create another queue or change placement algorithms. A client must support
launching a local stdio MCP process.

## Install

From the cloned repository on Linux or WSL:

```bash
./install.sh
python3 -m venv ~/.claude/scheduler/mcp-venv
~/.claude/scheduler/mcp-venv/bin/python -m pip install -r skill/integrations/requirements.txt
```

The CLI itself uses Python's standard library. MCP is optional. This wrapper uses
the official Python SDK's FastMCP v1 API, so the requirements select SDK v1.

Configure your own nodes in `~/.claude/scheduler/nodes.json`; see
[configuration](../../docs/CONFIGURATION.md). Keep identities in `~/.ssh/`.
For Windows clients, run the server inside WSL; the installation script and local
backend are Linux-based.

## Codex

Resolve paths on the machine running Codex:

```bash
codex mcp add scheduler -- \
  "$HOME/.claude/scheduler/mcp-venv/bin/python" \
  "$HOME/.claude/skills/scheduler/integrations/scheduler_mcp.py"
codex mcp list
```

Alternatively, use absolute paths in `~/.codex/config.toml`:

```toml
[mcp_servers.scheduler]
command = "/ABS/HOME/.claude/scheduler/mcp-venv/bin/python"
args = ["/ABS/HOME/.claude/skills/scheduler/integrations/scheduler_mcp.py"]
startup_timeout_sec = 20
tool_timeout_sec = 180
```

See the [official Codex MCP configuration](https://developers.openai.com/codex/mcp).

## Other Local MCP Clients

In a client that accepts an `mcpServers` configuration, replace the absolute
paths below with your installation paths:

```json
{
  "mcpServers": {
    "scheduler": {
      "command": "/ABS/HOME/.claude/scheduler/mcp-venv/bin/python",
      "args": ["/ABS/HOME/.claude/skills/scheduler/integrations/scheduler_mcp.py"]
    }
  }
}
```

For a native Windows client driving WSL, the equivalent process command is
`wsl.exe`, with arguments `["--", "/ABS/WSL/HOME/.claude/scheduler/mcp-venv/bin/python",
"/ABS/WSL/HOME/.claude/skills/scheduler/integrations/scheduler_mcp.py"]`.
Both absolute paths refer to files inside WSL.

The wrapper locates its adjacent `scheduler.py`, so it also runs directly from
the clone. `SCHEDULER_PY` can select a different CLI explicitly.
An optional `SCHEDULEURM_NODES_FILE` must match the watcher configuration.
Queue and history always live in the home directory of the user running this
process. Two different users do not share task state automatically.

## Tools

| Tool | Scheduler operation |
|---|---|
| `submit_task` | Submit one task |
| `cpu_plan` | Plan CPU placement |
| `submit_cpu_batch` | Submit a CPU batch |
| `dispatch` | Request dispatch |
| `status` | Inspect nodes and tasks |
| `doctor` | Diagnose queue invariants |
| `show_task` | Inspect one task |
| `cancel_task` | Cancel one task |
| `cancel_tasks` | Cancel IDs, ranges or a selected batch |
| `history` | Inspect resource history |
| `queue_dump` | Read structured queue state |
| `task_log` | Read a bounded task log |

The watcher handles automatic dispatch after installation. Calling `dispatch`
manually is optional. Cancellation and repair tools can modify or terminate the
operator's tasks; use the client's normal tool confirmation controls.
