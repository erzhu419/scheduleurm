# Configuration

## Nodes

The node inventory is loaded from `~/.claude/scheduler/nodes.json`. Set
`SCHEDULEURM_NODES_FILE` to use another absolute path. An explicitly selected
missing or invalid file is an error; it never silently falls back to another cluster.
The JSON object replaces the entire inventory, so omit `local` to prohibit local
dispatch. Without a nodes file, only the local machine is enabled, with a one-core
budget and live RAM/GPU probing.

Start from [nodes.example.json](nodes.example.json), replace the paths and SSH
aliases, and set each node's physical CPU budget before enabling the watcher:

```bash
mkdir -p ~/.claude/scheduler
cp docs/nodes.example.json ~/.claude/scheduler/nodes.json
chmod 600 ~/.claude/scheduler/nodes.json
```

Remote hosts use your existing `~/.ssh/config`. Keep login names, identities,
jump-host addresses and ports there, not in the repository. Check each alias with
`ssh -o BatchMode=yes compute-a true`. No scheduler-specific password or API key
is required.

| Field | Meaning |
|---|---|
| `host` | SSH alias, or `null` for local execution |
| `cpu_cores` | Physical CPU budget, before subtracting reserved cores |
| `reserved_cpu_cores` | Cores left for the OS and other work |
| `ram_mb` | RAM capacity in MB; `0` uses the live probe |
| `ram_headroom_mb` / `ram_headroom_frac` | Fixed or fractional RAM headroom |
| `capabilities` | Only advertise installed runtimes: e.g. `cpu`, `cuda`, `torch_cuda`, `jax_cuda` |
| `max_vram_per_task` | Optional per-task VRAM cap |
| `max_tasks_per_gpu` | Concurrent tasks admitted on each GPU |
| `max_concurrent_running` | Optional per-node task cap |
| `enable_claims` | Cross-scheduler resource claims on shared nodes |
| `remote_workspace_root` | Remote staging destination |
| `remote_path_prefixes` | Absolute local project roots mapped into that destination |
| `ssh_proxy_jump` / `ssh_proxy_jumps` | Optional jump aliases; SSH config can also supply these |
| `monitor_only` | Probe a login node without dispatching tasks there |

Restart `systemctl --user restart scheduler` after changing nodes. When using a
custom file, also supply `SCHEDULEURM_NODES_FILE` to the watcher service and MCP
client; using the default location avoids per-process environment configuration.

## Runtime Settings

Scheduler-wide defaults are in `skill/scheduler_config.py`; supported
`SCHEDULEURM_*` environment settings are read there. The source is modular:
the CLI and MCP both load the same modules and algorithm layer.

`./install.sh` defaults to symlinking the repository. `./install.sh --copy`
installs a snapshot of the skill, algorithms and simulation helpers.
`--no-systemd` installs without starting the watcher. Neither mode overwrites
your nodes file or task state.

Training admission, dependency files, resume flags, priorities and resource
hints are task submission parameters. See the CLI `--help` and
[CLI examples](../README.md#directly-from-the-cli). Measured RAM/VRAM history belongs to the user
state directory and is not distributed as cluster configuration.

## State And Publication

Queue, resource/runtime history, logs, launch records and transport budgets live
under `~/.claude/scheduler/`. Do not edit task state while the watcher runs;
use `submit`, `cancel-batch`, `doctor` and the inspection commands.

The public repository includes code, tests, research source and compact evidence,
with anonymized host labels and paths. Raw per-task transport logs, queue snapshots,
large generated replay outputs and private operational configuration are excluded.
Those research outputs can be regenerated with the experiment scripts, but live
hardware experiments require your own configured nodes and environments.

## Known Issues

Archived hardware experiment registrations retain their original provenance
digests. Anonymizing source paths changes the corresponding source bytes, and
some raw outputs are intentionally not distributed. Tests that require those
frozen live-hardware bundles are not a clean-install test suite. Register new
hardware measurements for your own cluster rather than reusing those launch
authorizations. Core scheduler tests are `skill/tests/test_scheduler_*.py`;
the public node configuration tests are `skill/tests/test_public_node_config.py`.
