# scheduler MCP integration

Wraps `~/.claude/skills/scheduler/scheduler.py` as a Model Context Protocol (MCP) server so any MCP-aware AI client (ChatGPT Desktop, Claude Desktop, Cursor, Cline, custom agents) can dispatch / inspect / cancel scheduler tasks the same way Claude Code does — but through a transport every modern client speaks.

The Python tool is **untouched**: this is a thin sub-process wrapper that translates MCP tool calls into `python scheduler.py <subcommand> ...` invocations.

## What you get

| Tool | Maps to | Use case |
|---|---|---|
| `submit_task` | `scheduler.py submit ...` | "跑这个 python script", "launch 5 seeds of WSRL" |
| `dispatch` | `scheduler.py dispatch` | "rebalance", wake the scheduler after a manual submit |
| `status` | `scheduler.py status` | "GPU 还空吗", "现在跑啥呢" |
| `show_task` | `scheduler.py show tXXXX` | "看看 t0007", debug a specific task |
| `cancel_task` | `scheduler.py cancel tXXXX [--force]` | "取消 tXXXX" — running needs confirm + force |
| `history` | `scheduler.py history [--signature]` | "看看 RE-SAC 的资源画像" |
| `queue_dump` | reads `queue.json` directly | structured filtering: "list all failed offline-sumo tasks in last hour" |
| `task_log` | reads task `log_path` | debug failures, check progress |

The tool descriptions in `scheduler_mcp.py` are written so the host LLM can auto-route user intent to the right call without explicit "use scheduler" instruction — same auto-detection Claude Code's SKILL.md provides, just via MCP standard.

## One-time setup

```bash
pip install mcp
```

(Or `uv pip install mcp` if using uv. The `mcp` package is the official Python SDK.)

## Per-client config

### ChatGPT Desktop (macOS / Windows)

`~/Library/Application Support/ChatGPT/mcp.json` (mac) or `%APPDATA%\ChatGPT\mcp.json` (Win):

```json
{
  "mcpServers": {
    "scheduler": {
      "command": "/usr/bin/python3",
      "args": ["/home/erzhu419/.claude/skills/scheduler/integrations/scheduler_mcp.py"]
    }
  }
}
```

Restart ChatGPT Desktop. The tools appear in the tool picker.

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (mac) or equivalent on Win:

```json
{
  "mcpServers": {
    "scheduler": {
      "command": "/usr/bin/python3",
      "args": ["/home/erzhu419/.claude/skills/scheduler/integrations/scheduler_mcp.py"]
    }
  }
}
```

### Cursor / Cline / Continue

Each has an MCP config UI; point it at the same `command` + `args`.

### Custom agent (Python OpenAI SDK / anthropic SDK / etc.)

Use the `mcp` SDK's client (`from mcp import ClientSession` + `from mcp.client.stdio import stdio_client`) to spawn the server and discover tools. ~10 lines.

## Auto-detection / "GPT 自己识别该派啥"

The MCP host LLM (GPT-4o, Claude, etc.) reads each tool's docstring + parameter schema as part of its tool-calling context. The docstrings in `scheduler_mcp.py` include the same routing hints from SKILL.md ("use this when user says X / Y / Z"). Result: the user can say "跑这个 python script for me" and the LLM routes to `submit_task` with the right args, just like Claude Code.

If the LLM you're using has weak tool-routing, paste the relevant section of `~/.claude/skills/scheduler/SKILL.md` into its system prompt — that's the same convention guide Claude Code's skill loader injects.

## Environment

The wrapper invokes `python scheduler.py ...` so it inherits whatever python is on `PATH` when the MCP client launches it. If your client runs in a different env than the one with scheduler deps, set `SCHEDULER_PY` env var to override the script path:

```json
{
  "mcpServers": {
    "scheduler": {
      "command": "/home/erzhu419/anaconda3/bin/python",
      "args": ["/home/erzhu419/.claude/skills/scheduler/integrations/scheduler_mcp.py"],
      "env": {"SCHEDULER_PY": "/home/erzhu419/.claude/skills/scheduler/scheduler.py"}
    }
  }
}
```

## Caveats

- **No shell access via this MCP**: tools wrap *scheduler operations*, not arbitrary commands. If the host LLM needs to read code / edit files / run other commands, pair this with a generic Bash-MCP server.
- **Local file access**: `queue_dump` / `task_log` read directly from `~/.claude/scheduler/queue.json` and the local log paths. For remote-node logs, the wrapper uses `ssh <node> tail ...` — same SSH config as scheduler.py.
- **MCP server runs as the user that launched it**. Don't expose this over network without auth — anyone connected can submit/cancel tasks.

## Maintenance

When `scheduler.py` adds a new subcommand or flag, the MCP wrapper needs a matching `@mcp.tool()` (or new parameter on an existing tool). Regression tests for scheduler.py do NOT cover this wrapper — add tests separately if you start depending on programmatic correctness vs just live-driving.
