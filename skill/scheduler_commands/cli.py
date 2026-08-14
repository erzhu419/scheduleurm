from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class SchedulerCliDeps:
    node_names: list[str]
    default_vram_mb: int
    default_ram_mb: int
    default_cpu_cores: int
    dispatch_intent_ttl_s: float
    resource_log_interval_s: int
    archive_age_days: float
    archive_max_hot_terminal: int
    node_down_requeue_s: int
    cmd_submit: Callable
    cmd_submit_jsonl: Callable
    cmd_cpu_plan: Callable
    cmd_submit_cpu_batch: Callable
    cmd_dispatch: Callable
    cmd_wait_for: Callable
    cmd_watch: Callable
    cmd_status: Callable
    cmd_compact: Callable
    cmd_doctor: Callable
    cmd_profile_local: Callable
    cmd_claims: Callable
    cmd_show: Callable
    cmd_task_log: Callable
    cmd_results: Callable
    cmd_cancel: Callable
    cmd_forget: Callable
    cmd_clear_queue: Callable
    cmd_adopt: Callable
    cmd_record_vram: Callable
    cmd_history: Callable
    cmd_priority: Callable
    cmd_edit: Callable
    cmd_why: Callable
    cmd_tui: Callable


def _add_submit_parser(sub, deps: SchedulerCliDeps) -> None:
    nodes = deps.node_names
    parser = sub.add_parser("submit", help="Add a task to the queue")
    parser.add_argument("--description", required=True)
    parser.add_argument("--cmd", required=True, help="Shell command to run (will be wrapped with cd cwd && env)")
    parser.add_argument("--cwd", required=True, help="Working directory on target node")
    parser.add_argument("--signature", required=True, help="Stable id like 'RE-SAC/b1' for VRAM history lookup")
    parser.add_argument(
        "--resource-family",
        dest="resource_family",
        help=(
            "Optional stable resource class shared by smoke/formal/eval tasks. "
            "Running siblings in the same family calibrate queued RAM/VRAM estimates."
        ),
    )
    parser.add_argument(
        "--vram-resource-family",
        dest="vram_resource_family",
        help="Resource family used only for queued VRAM calibration.",
    )
    parser.add_argument(
        "--ram-resource-family",
        dest="ram_resource_family",
        help="Resource family used only for queued host-RAM calibration.",
    )
    parser.add_argument(
        "--vram",
        type=int,
        help="Override est VRAM in MB (else from history; else %d). Use 0 for CPU-only tasks."
        % deps.default_vram_mb,
    )
    parser.add_argument(
        "--ram-mb",
        type=int,
        dest="ram_mb",
        help="Override est RAM in MB (else from history; else %d)" % deps.default_ram_mb,
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="CPU cores needed (else from history; else %d)" % deps.default_cpu_cores,
    )
    parser.add_argument(
        "--cpu-parallel-items",
        dest="cpu_parallel_items",
        type=int,
        default=0,
        help=(
            "For CPU-only multi-worker jobs with M independent items, auto-size workers as "
            "e=ceil(M/physical_cores), workers=ceil(M/e). Commands may use "
            "{workers}/{n_workers}/{num_workers} placeholders or pass --workers auto."
        ),
    )
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--project", help="Project name (else derived from cwd basename)")
    parser.add_argument("--preferred-node", choices=nodes, help="SOFT preference: try this node first, fall back if full")
    parser.add_argument(
        "--require-node",
        dest="require_node",
        choices=nodes,
        help="HARD pin: only place on this node, never fall back. Use when the cmd has node-specific paths/env that won't work elsewhere.",
    )
    parser.add_argument(
        "--allowed-node",
        dest="allowed_nodes",
        choices=nodes,
        action="append",
        help="Restrict candidate placement to this node. Repeatable; unlike --require-node, scheduler still chooses among the allowed nodes.",
    )
    parser.add_argument("--git-repo", help="Local + remote path of git repo to sync-check before launch")
    parser.add_argument("--ckpt-dir", help="Checkpoint directory on TARGET node, for resume detection. Must be a dedicated directory, not equal to --cwd.")
    parser.add_argument(
        "--result-dir",
        dest="result_dir",
        help=(
            "Phase 3.5: directory on TARGET node containing the experiment results "
            "(logs / final models / metrics). On task completion, scheduleurm rsyncs "
            "this dir back to local (delta sync; no ckpts unless they live here). "
            "Set this to opt in. Must be a dedicated directory, not equal to --cwd; "
            "intermediate ckpts should stay in --ckpt-dir which is NOT synced automatically."
        ),
    )
    parser.add_argument(
        "--local-result-dir",
        dest="local_result_dir",
        help=(
            "Phase 3.5: where on local to land the rsync'd results. Defaults to "
            "mirroring the remote path (same absolute path on local as on the target). "
            "Use a per-task subdirectory; exact sharing is refused unless "
            "--allow-shared-result-dir is passed."
        ),
    )
    parser.add_argument(
        "--wait-for-file",
        dest="wait_for_files",
        action="append",
        help="Defer dispatch until this local prerequisite file exists and is non-empty. Repeat for multiple prerequisites; useful for eval tasks waiting on train ckpts.",
    )
    parser.add_argument(
        "--test-log",
        dest="test_log",
        help="Local preflight/test log containing tqdm/progress output. Parsed at submit time and recorded into runtime history so ETA/walltime use the local test profile before the real experiment launches.",
    )
    parser.add_argument("--test-peak-vram-mb", dest="test_peak_vram_mb", type=int, help="Peak VRAM observed during local preflight. Recorded into signature history before sizing this submit.")
    parser.add_argument("--test-peak-ram-mb", dest="test_peak_ram_mb", type=int, help="Peak RAM observed during local preflight. Recorded into signature history before sizing this submit.")
    parser.add_argument("--test-cpu", dest="test_cpu", type=int, help="CPU cores observed during local preflight. Recorded into signature history before sizing this submit.")
    parser.add_argument("--ckpt-glob", default="*", help="Glob within ckpt-dir (default '*')")
    parser.add_argument(
        "--resume-flag",
        dest="resume_flag",
        default="",
        help="If set (e.g. '--resume_from'), launcher appends '<flag> <ckpt_path>' to cmd when find_resume() locates a checkpoint. Empty (default) = no injection. Pair with --ckpt-dir; the script must accept this flag.",
    )
    parser.add_argument(
        "--allow-initial-resume-scan-error",
        dest="allow_initial_resume_scan_error",
        action="store_true",
        help="For brand-new unique runs, allow first launch even if an optional remote checkpoint scan has transient SSH errors and no checkpoint is found. Retry clones still re-scan and will not use this bypass.",
    )
    parser.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    parser.add_argument("--allow-cpu-training", dest="allow_cpu_training", action="store_true", help="Override the 'training cmd + --vram 0' refusal. Use when you really do want to train on CPU; required even if the cmd contains --device cpu. MUST be paired with --cpu-training-justification.")
    parser.add_argument("--cpu-training-justification", dest="cpu_training_justification", default="", help="Required when --allow-cpu-training is set: >=30 chars explaining why this training task should run on CPU rather than GPU. Stored on the task record for audit.")
    parser.add_argument("--allow-no-ckpt", dest="allow_no_ckpt", action="store_true", help="Override the 'training cmd without --ckpt-dir' refusal. Use for short debug runs or one-shot evals where losing progress on relaunch is fine.")
    parser.add_argument("--allow-no-resume", dest="allow_no_resume", action="store_true", help="Override the 'training cmd without resume capability' refusal. Required when the script genuinely cannot resume. Acknowledges that crash/reboot/eviction will lose progress.")
    parser.add_argument(
        "--env-spec",
        dest="env_spec",
        default="none",
        help="Environment delivery strategy: 'none', 'docker:IMAGE[:TAG]', 'conda:/abs/path/to/env', or 'auto'.",
    )
    parser.add_argument("--image", dest="image", default="", help="Docker image to use when --env-spec includes docker.")
    parser.add_argument("--allow-shared-ckpt-dir", dest="allow_shared_ckpt_dir", action="store_true", help="Override the active-task ckpt-dir conflict refusal.")
    parser.add_argument("--allow-shared-result-dir", dest="allow_shared_result_dir", action="store_true", help="Override the local result destination conflict refusal.")
    parser.add_argument("--allow-remote-large-data", dest="allow_remote_large_data", action="store_true", help="Override SimpleSAC large external-data local pin.")
    parser.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true", help="Allow submission even when run identity matches an existing queued/launching/running task")
    parser.add_argument("--allow-seed-batch", dest="allow_seed_batch", action="store_true", help="Do not auto-split simple BAPR `for seed in ...` shell loops at submit time.")
    parser.add_argument(
        "--skip-launch-staging",
        dest="skip_launch_staging",
        action="store_true",
        help="Use only after the exact cwd has been provisioned on every allowed remote node. Skips scheduler-managed source staging for this task.",
    )
    parser.add_argument("--stage-exclude", dest="stage_excludes", action="append", help="Extra cwd staging exclude path/glob, relative to --cwd. Repeatable.")
    parser.add_argument("--stage-input-path", dest="stage_input_paths", action="append", help="Local directory that must be staged to the selected remote node before launch. Repeatable.")
    parser.add_argument("--reroute-on-node-down", dest="reroute_on_node_down", action="store_true", help="Opt into auto-requeue/reroute if a running task's node probe stays unknown.")
    parser.add_argument(
        "--node-down-requeue-s",
        dest="node_down_requeue_s",
        type=int,
        default=0,
        help=f"Seconds of unknown node probe before reroute when opted in (default {deps.node_down_requeue_s}).",
    )
    parser.set_defaults(func=deps.cmd_submit)


def _add_bulk_submit_parsers(sub, deps: SchedulerCliDeps) -> None:
    parser = sub.add_parser("submit-jsonl", help="Trusted fast path: append many scheduler task specs with one state lock")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true", help="Read JSONL or a JSON list from stdin")
    src.add_argument("--file", help="Read JSONL or a JSON list from this file")
    parser.add_argument("--trusted", action="store_true", help="Required acknowledgement: specs are already scheduler-compatible")
    parser.add_argument("--json", action="store_true", help="Print one JSON summary")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing")
    parser.add_argument("--intent-ttl", type=float, default=deps.dispatch_intent_ttl_s, help=f"Seconds for the bulk-submit intent that asks watch to skip (default {deps.dispatch_intent_ttl_s:g})")
    parser.add_argument("--intent-label", default="", help="Optional label written to the bulk-submit intent file")
    parser.set_defaults(func=deps.cmd_submit_jsonl)

    parser = sub.add_parser("cpu-plan", help="Plan M independent CPU items across physical-core CPU nodes")
    parser.add_argument("--items", type=int, required=True, help="Logical CPU items/checkpoints to process")
    parser.add_argument("--item-multiplier", type=int, default=1, help="Independent work items per logical item. Example: 39 ckpts * 10 eval episodes => --items 39 --item-multiplier 10.")
    parser.add_argument("--nodes", default="", help="Comma-separated CPU nodes (default: all cpu_labor_node nodes)")
    parser.add_argument("--use-total-cores", action="store_true", help="Ignore live free_cpu and plan from full physical-core capacity")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_cpu_plan)

    parser = sub.add_parser("submit-cpu-batch", help="Split a CPU-heavy batch across CPU nodes and submit one shard per node")
    parser.add_argument("--items", type=int, required=True, help="Logical CPU items/checkpoints to process")
    parser.add_argument("--item-multiplier", type=int, default=1, help="Independent work items per logical item. Example: ckpt eval with 10 episodes per checkpoint should use --item-multiplier 10 so worker planning sees ckpt_count*10 work items.")
    parser.add_argument("--cmd-template", required=True, help="Command template. Known placeholders: {start} {end} {items} {total_items} {logical_items} {item_multiplier} {workers} {node} {shard_index} {num_shards}. Use {workers} or '--workers auto' to get the computed worker count.")
    parser.add_argument("--cwd", required=True, help="Working directory template/path on target node")
    parser.add_argument("--signature", required=True, help="Signature template, e.g. Project/eval/{node}")
    parser.add_argument("--description", required=True, help="Description template")
    parser.add_argument("--nodes", default="", help="Comma-separated CPU nodes (default: all cpu_labor_node nodes)")
    parser.add_argument("--use-total-cores", action="store_true", help="Ignore live free_cpu and plan from full physical-core capacity")
    parser.add_argument("--ram-mb", dest="ram_mb", type=int, help="RAM estimate per shard")
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    parser.add_argument("--project", help="Project name")
    parser.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    parser.add_argument("--result-dir-template", dest="result_dir_template", help="Remote result dir template for each shard")
    parser.add_argument("--local-result-dir-template", dest="local_result_dir_template", help="Local result dir template for each shard")
    parser.add_argument("--wait-for-file-template", dest="wait_for_file_template", action="append", help="Prerequisite file template; repeatable")
    parser.add_argument("--allow-env-only-shard", action="store_true", help="Allow multi-node split when templates lack {start}/{end}/{node}; use only if the script reads SCHEDULEURM_CPU_* env vars.")
    parser.add_argument("--allow-cpu-training", dest="allow_cpu_training", action="store_true")
    parser.add_argument("--cpu-training-justification", dest="cpu_training_justification", default="")
    parser.add_argument("--allow-no-ckpt", dest="allow_no_ckpt", action="store_true")
    parser.add_argument("--allow-no-resume", dest="allow_no_resume", action="store_true")
    parser.add_argument("--allow-shared-result-dir", dest="allow_shared_result_dir", action="store_true")
    parser.add_argument("--allow-remote-large-data", dest="allow_remote_large_data", action="store_true")
    parser.add_argument("--allow-duplicate", dest="allow_duplicate", action="store_true")
    parser.add_argument(
        "--skip-launch-staging",
        dest="skip_launch_staging",
        action="store_true",
        help="Use only after the exact cwd has been provisioned on every CPU node.",
    )
    parser.add_argument("--stage-exclude", dest="stage_exclude", action="append", help="Extra cwd staging exclude path/glob, relative to --cwd. Repeatable.")
    parser.add_argument("--node-down-requeue-s", dest="node_down_requeue_s", type=int, default=0, help=f"Seconds of unknown node probe before CPU batch shard reroute (default {deps.node_down_requeue_s}).")
    parser.add_argument("--env-spec", dest="env_spec", default="none")
    parser.add_argument("--image", dest="image", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_submit_cpu_batch)


def _add_dispatch_watch_parsers(sub, deps: SchedulerCliDeps) -> None:
    parser = sub.add_parser("dispatch", help="Probe nodes & launch what fits (also rebalances queue)")
    parser.add_argument("--algorithm", default="", help="Placement algorithm policy (default/env: legacy). Examples: legacy, sweetspot_v1")
    parser.add_argument("--hard-rule-mode", default="", help="Experiment hard-rule override mode from algorithm.hard_rules, e.g. clean_bench")
    parser.add_argument("--task-id", dest="dispatch_task_ids", action="append", help="Experimental one-shot dispatch filter: only place this queued task id. Repeatable.")
    parser.add_argument(
        "--bulk-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Advertise a short-lived dispatch-priority window so watch/cancel let placement run (default: true).",
    )
    parser.add_argument("--intent-ttl", type=float, default=deps.dispatch_intent_ttl_s, help=f"Seconds before a bulk dispatch intent expires (default {deps.dispatch_intent_ttl_s:g}).")
    parser.add_argument("--intent-label", default="", help="Optional label written to the bulk dispatch intent file.")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing (default: wait indefinitely).")
    parser.set_defaults(func=deps.cmd_dispatch)

    parser = sub.add_parser("wait-for", help="Block until matching tasks reach terminal state; exit fires a task-notification when wrapped in Bash run_in_background. Match by --signature glob or --task-id list (or both).")
    parser.add_argument("--signature", help="fnmatch glob over task signatures (e.g. 'H2Oplus/multiseed_*')")
    parser.add_argument("--task-id", dest="task_ids", nargs="*", default=[], help="One or more explicit task IDs (e.g. t0099 t0100)")
    parser.add_argument("--poll", type=int, default=30, help="Poll interval in seconds (default 30)")
    parser.add_argument("--timeout", type=int, default=14400, help="Max seconds to wait (default 14400 = 4h; 0 = no timeout)")
    parser.add_argument("--verbose", action="store_true", help="Print a progress line every 5 minutes")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Use running-state refresh as a failover only while no watcher owns "
            "the control lease; otherwise wait-for remains readonly."
        ),
    )
    parser.set_defaults(func=deps.cmd_wait_for)

    parser = sub.add_parser("watch", help="Background daemon: probe + dispatch every --interval s; notify on done/launch/heartbeat")
    parser.add_argument("--interval", type=int, default=60, help="Probe + dispatch every N seconds (default 60)")
    parser.add_argument("--heartbeat", type=int, default=3600, help="Push a state-snapshot heartbeat every N seconds (default 3600 = 1h)")
    parser.add_argument("--resource-log-interval", dest="resource_log_interval", type=int, default=deps.resource_log_interval_s, help=f"Write detailed node CPU/RAM attribution to watcher.log every N seconds (default {deps.resource_log_interval_s})")
    parser.add_argument("--algorithm", default="", help="Placement algorithm policy (default/env: legacy). Examples: legacy, sweetspot_v1")
    parser.add_argument("--hard-rule-mode", default="", help="Experiment hard-rule override mode from algorithm.hard_rules, e.g. clean_bench")
    parser.add_argument("--ignore-dispatch-intent", action="store_true", help="Do not skip watch iterations during an active bulk dispatch intent.")
    parser.set_defaults(func=deps.cmd_watch)


def _add_status_maintenance_parsers(sub, deps: SchedulerCliDeps) -> None:
    parser = sub.add_parser("status", help="Show node + task state")
    parser.add_argument("--all", action="store_true", help="Include done/failed/cancelled tasks")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--brief", action="store_true", help="With --json, emit compact task records")
    parser.add_argument("--readonly", action="store_true", help="Read queue state with a shared lock; skip running-task updates and node probes.")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing.")
    parser.add_argument("--ids", nargs="*", default=[], help="Limit task output to specific IDs. Accepts space-separated IDs or comma-separated groups.")
    parser.set_defaults(func=deps.cmd_status)

    parser = sub.add_parser("compact", help="Move old terminal tasks from hot queue.json into queue_archive.jsonl")
    parser.add_argument("--age-days", type=float, default=None, help=f"Archive terminal tasks older than this many days (default {deps.archive_age_days:g})")
    parser.add_argument("--age-hours", type=float, default=None, help="Archive terminal tasks older than this many hours; overrides --age-days")
    parser.add_argument("--all-terminal", action="store_true", help="Archive every terminal done/failed/cancelled/forgotten task")
    parser.add_argument("--keep-terminal", type=int, default=deps.archive_max_hot_terminal, help=f"Keep at most this many newest terminal tasks hot (default {deps.archive_max_hot_terminal})")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be archived")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing.")
    parser.set_defaults(func=deps.cmd_compact)

    parser = sub.add_parser("doctor", help="Audit active queue invariants; --fix applies safe queued-task repairs")
    parser.add_argument("--fix", action="store_true", help="Apply safe repairs to queued tasks: add wait-for-file gates, force SimpleSAC large-data local, promote dependent train priority")
    parser.add_argument("--project", help="fnmatch glob over project (e.g. SimpleSAC)")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_doctor)

    parser = sub.add_parser("profile-local", help="Run a local preflight directly, monitor peak RAM/VRAM/CPU, and record tqdm/runtime history")
    parser.add_argument("--description", default="local preflight profile")
    parser.add_argument("--cmd", required=True, help="Shell command to run locally (not through scheduleurm queue)")
    parser.add_argument("--cwd", required=True, help="Working directory for the local preflight")
    parser.add_argument("--signature", required=True, help="Signature whose resource/runtime history should be updated")
    parser.add_argument("--project", help="Project name (else derived from cwd/signature)")
    parser.add_argument("--env", nargs="*", help="Extra env vars KEY=VALUE")
    parser.add_argument("--log-path", dest="log_path", help="Where to write the local preflight log")
    parser.add_argument("--sample-interval", dest="sample_interval", type=float, default=5.0, help="Seconds between local resource samples (default 5)")
    parser.add_argument("--timeout", type=int, default=0, help="Kill the local preflight after N seconds (0 = no timeout)")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_profile_local)

    parser = sub.add_parser("claims", help="Show remote shared claims/intents for claims-enabled nodes")
    parser.add_argument("--node", choices=deps.node_names, help="Limit to one node")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_claims)


def _add_task_inspection_parsers(sub, deps: SchedulerCliDeps) -> None:
    parser = sub.add_parser("show", help="Show one task's full record + how to tail logs")
    parser.add_argument("id")
    parser.set_defaults(func=deps.cmd_show)

    parser = sub.add_parser("task-log", aliases=["log"], help="Tail a task log through scheduler node routing")
    parser.add_argument("id", help="Task id, e.g. t20846")
    parser.add_argument("-n", "--lines", "--tail", type=int, default=80, help="Lines to show (default 80)")
    parser.add_argument("--no-archive", action="store_true", help="Only search hot queue.json")
    parser.add_argument("--no-header", action="store_true", help="Do not print the log path header")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_task_log)

    parser = sub.add_parser("results", help="Find inferred result artifacts in queue + archive")
    parser.add_argument("task_ids", nargs="*", help="Optional task IDs to inspect")
    parser.add_argument("--project", help="fnmatch glob over project, e.g. 'SimpleSAC' or 'bapr*'")
    parser.add_argument("--signature", help="fnmatch glob over signature, e.g. 'H2Oplus/r3_eval_*'")
    parser.add_argument("--status", nargs="*", default=["done"], help="Statuses to include (default: done). Pass e.g. --status done failed")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print (default 50; 0 = no limit)")
    parser.add_argument("--no-archive", action="store_true", help="Only search hot queue.json")
    parser.add_argument("--scan-logs", action="store_true", help="For batch queries, scan task logs for saved output paths. Task-id queries scan logs by default.")
    parser.add_argument("--no-log-scan", action="store_true", help="Do not read task logs; use stored fields and command flags only")
    parser.add_argument("--include-empty", action="store_true", help="Print matching tasks even when no result artifact can be inferred")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=deps.cmd_results)

    parser = sub.add_parser("why", help="Diagnose why a queued task isn't being dispatched")
    parser.add_argument("id", help="Task id (e.g. t0042)")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing.")
    parser.set_defaults(func=deps.cmd_why)

    sub.add_parser("tui", help="Interactive TUI: sortable + filterable + auto-refresh task table").set_defaults(func=deps.cmd_tui)


def _add_mutation_parsers(sub, deps: SchedulerCliDeps) -> None:
    parser = sub.add_parser(
        "cancel",
        aliases=["cancel-batch"],
        help="Cancel one or many tasks in one transaction (running tasks need --force)",
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help=(
            "Task IDs, comma groups, or inclusive ranges, e.g. "
            "t21001 t21002, t21010-t21030"
        ),
    )
    parser.add_argument("--project", help="fnmatch glob over project; selector-only calls preview unless --confirm")
    parser.add_argument("--signature", help="fnmatch glob over signature; selector-only calls preview unless --confirm")
    parser.add_argument(
        "--status",
        dest="statuses",
        action="append",
        choices=["queued", "launching", "running"],
        help="Restrict selected active states; repeatable (default: all active states)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preview the exact selected tasks without changing state")
    parser.add_argument("--confirm", action="store_true", help="Apply a project/signature selector-based batch cancellation")
    parser.add_argument("--ignore-missing", action="store_true", help="Ignore absent explicit IDs for idempotent cleanup")
    parser.add_argument("--lock-timeout", type=float, default=None, help="Seconds to wait for the scheduler state lock before failing.")
    parser.add_argument("--ignore-dispatch-intent", action="store_true", help="Cancel even when a bulk dispatch intent is active.")
    parser.add_argument("--force-lock-wait", action="store_true", help="Urgent cancel: ignore bulk-dispatch deferral and wait for the lock.")
    parser.set_defaults(func=deps.cmd_cancel)

    parser = sub.add_parser("forget", help="Drop a task record from tracking (NEVER touches processes - for fixing wrong adopts)")
    parser.add_argument("id")
    parser.set_defaults(func=deps.cmd_forget)

    parser = sub.add_parser("clear-queue", help="Cancel ALL queued tasks (running tasks untouched)")
    parser.add_argument("--confirm", action="store_true")
    parser.set_defaults(func=deps.cmd_clear_queue)

    parser = sub.add_parser("adopt", help="Register externally-launched PIDs as a tracked task")
    parser.add_argument("--node", required=True, choices=deps.node_names)
    parser.add_argument("--gpu", required=True, type=int, help="GPU index on the node")
    parser.add_argument("--pid", dest="pids", required=True, nargs="+", type=int, help="One or more PIDs (multi-worker tasks)")
    parser.add_argument("--description", required=True)
    parser.add_argument("--signature", required=True, help="For VRAM history; reuse same id across runs of same config")
    parser.add_argument("--project", help="Project name (else read /proc/<pid>/cwd basename on the node)")
    parser.add_argument("--cwd", help="(optional) working dir on the node, for documentation only")
    parser.add_argument("--ckpt-dir", help="(optional) abs path to ckpt dir on the node")
    parser.add_argument("--log-path", help="(optional) abs path to existing log file on the node")
    parser.add_argument("--est-vram", type=int, help="Override est VRAM (else uses current sum)")
    parser.add_argument("--allow-multi-project", action="store_true", help="Override the safety check that refuses adopting PIDs spanning different projects")
    parser.set_defaults(func=deps.cmd_adopt)

    parser = sub.add_parser("record-vram", help="Manually record peak VRAM for a signature")
    parser.add_argument("signature")
    parser.add_argument("peak_vram_mb", type=int)
    parser.set_defaults(func=deps.cmd_record_vram)

    parser = sub.add_parser("history", help="Show / edit VRAM history per signature")
    parser.add_argument("--drop", metavar="SIG", help="Remove the history entry for SIG (next runs will start fresh)")
    parser.add_argument("--set", metavar="SIG", help="Set / overwrite the history entry for SIG (use with --vram-mb / --ram-mb / --cpu)")
    parser.add_argument("--vram-mb", dest="vram_mb", type=int, help="With --set: peak VRAM in MB to record")
    parser.add_argument("--ram-mb", dest="ram_mb", type=int, help="With --set: peak RAM in MB to record")
    parser.add_argument("--cpu", type=int, help="With --set: cpu_cores to record")
    parser.set_defaults(func=deps.cmd_history)

    parser = sub.add_parser("priority", help="Change priority of a queued task (queue ordering)")
    parser.add_argument("id", help="Task id (e.g. t0042)")
    parser.add_argument("level", choices=["low", "normal", "high"])
    parser.set_defaults(func=deps.cmd_priority)

    parser = sub.add_parser("edit", help="Override resource estimates / pin on a queued task")
    parser.add_argument("id", nargs="?", help="Task id (e.g. t0042)")
    parser.add_argument("--project", help="fnmatch selector over queued task projects")
    parser.add_argument("--signature", help="fnmatch selector over queued task signatures")
    parser.add_argument("--confirm", action="store_true",
                        help="Required when using --project/--signature selectors")
    parser.add_argument("--vram-mb", dest="vram_mb", type=int, help="Override estimated peak VRAM in MB")
    parser.add_argument("--ram-mb", dest="ram_mb", type=int, help="Override estimated peak RAM in MB")
    parser.add_argument("--cpu", type=int, help="Override CPU cores")
    parser.add_argument("--description", help="Override description")
    parser.add_argument("--resource-family", dest="resource_family", help="Set generic RAM/VRAM family (use empty string to clear)")
    parser.add_argument("--vram-resource-family", dest="vram_resource_family", help="Set VRAM-only family (use empty string to clear)")
    parser.add_argument("--ram-resource-family", dest="ram_resource_family", help="Set RAM-only family (use empty string to clear)")
    parser.add_argument("--preferred-node", dest="preferred_node", help="Set / change soft preferred node (must be a known node)")
    parser.add_argument("--require-node", dest="require_node", help="Set / change hard pin (use empty string to clear)")
    parser.add_argument("--require-gpu", dest="require_gpu_idx", help="Set / change hard GPU index pin on the selected node (use empty string to clear)")
    parser.add_argument("--allowed-node", dest="allowed_nodes", choices=deps.node_names, action="append", help="Replace placement candidate whitelist with this node. Repeatable.")
    parser.add_argument("--clear-allowed-nodes", dest="clear_allowed_nodes", action="store_true", help="Clear the placement candidate whitelist")
    parser.add_argument("--allow-gpu-over-one-third", dest="allow_gpu_over_one_third", action="store_true", default=None, help="Allow this queued GPU task to exceed the 1/3 GPU packing guard")
    parser.add_argument("--clear-gpu-over-one-third", dest="allow_gpu_over_one_third", action="store_false", default=None, help="Clear the per-task 1/3 GPU packing override")
    parser.set_defaults(func=deps.cmd_edit)


def build_parser(deps: SchedulerCliDeps) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scheduler")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_submit_parser(sub, deps)
    _add_bulk_submit_parsers(sub, deps)
    _add_dispatch_watch_parsers(sub, deps)
    _add_status_maintenance_parsers(sub, deps)
    _add_task_inspection_parsers(sub, deps)
    _add_mutation_parsers(sub, deps)
    return parser


def run_scheduler_cli(deps: SchedulerCliDeps, argv: Sequence[str] | None = None) -> None:
    parser = build_parser(deps)
    args = parser.parse_args(argv)
    args.func(args)
