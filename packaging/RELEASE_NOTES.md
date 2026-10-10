# Scheduleurm 0.1.0

First installable public release for Linux and WSL, requiring Python 3.10+.

- `scheduleurm` CLI and `scheduleurm-mcp` local stdio MCP entry points.
- Optional `mcp` and `tui` dependencies.
- Existing modular scheduler, placement algorithms and portable measured caches.
- User-owned node and SSH configuration; no cluster credentials bundled.
- Installation leaves watcher services and existing task state unchanged.

Install the wheel from this release:

```bash
python3 -m venv ~/.venvs/scheduleurm
~/.venvs/scheduleurm/bin/python -m pip install \
  'scheduleurm[mcp,tui] @ https://github.com/erzhu419/scheduleurm/releases/download/v0.1.0/scheduleurm-0.1.0-py3-none-any.whl'
~/.venvs/scheduleurm/bin/scheduleurm --help
```

Configure `~/.claude/scheduler/nodes.json` before using remote nodes. Start
`scheduleurm watch --interval 60` for automatic dispatch. Installing with pip
does not automatically register Claude skills or start systemd services; the
repository installer remains available for that workflow.
