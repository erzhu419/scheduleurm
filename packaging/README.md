# Scheduleurm

Schedule CPU, RAM and GPU jobs across heterogeneous Linux or WSL nodes.
The CLI and optional local stdio MCP server share the same queue and algorithms.

## Install

Use Python 3.10 or newer on Linux or WSL:

```bash
python3 -m venv ~/.venvs/scheduleurm
~/.venvs/scheduleurm/bin/python -m pip install \
  'scheduleurm[mcp,tui] @ https://github.com/erzhu419/scheduleurm/releases/download/v0.1.0/scheduleurm-0.1.0-py3-none-any.whl'
~/.venvs/scheduleurm/bin/scheduleurm --help
```

Omit `[mcp,tui]` for the standard-library-only CLI. The wheel and source
distribution are available from [GitHub Releases](https://github.com/erzhu419/scheduleurm/releases).

## Configure and Run

Configure your own nodes in `~/.claude/scheduler/nodes.json`, or select a file
with `SCHEDULEURM_NODES_FILE`. SSH aliases and identities belong in your own
`~/.ssh/config`. See the
[configuration guide](https://github.com/erzhu419/scheduleurm/blob/master/docs/CONFIGURATION.md)
and [node example](https://github.com/erzhu419/scheduleurm/blob/master/docs/nodes.example.json).
Without node configuration, the scheduler uses only the local machine with a
conservative one-core limit.

```bash
~/.venvs/scheduleurm/bin/scheduleurm status
~/.venvs/scheduleurm/bin/scheduleurm watch --interval 60
# With the tui extra:
~/.venvs/scheduleurm/bin/scheduleurm tui-top
```

Installing the package does not start a watcher or modify services, credentials
or existing task state. Queue, history and logs use `~/.claude/scheduler/`.
Run `watch` to enable automatic dispatch; retain your process manager for
continuous operation.

## MCP

With the `mcp` extra installed, configure a local stdio MCP client to launch
`/ABS/HOME/.venvs/scheduleurm/bin/scheduleurm-mcp` without arguments.
It exposes 12 tools for submission, inspection, dispatch and cancellation.
The [MCP guide](https://github.com/erzhu419/scheduleurm/blob/master/skill/integrations/README.md)
contains client-specific configuration examples.

## Source and License

[Source repository](https://github.com/erzhu419/scheduleurm). MIT license.
The package includes scheduler modules, placement algorithms and portable
service-rate summaries, but not research result directories or cluster login
configuration. The complete research sources remain in the repository.
