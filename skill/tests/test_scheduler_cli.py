from __future__ import annotations

from skill.scheduler_commands.cli import SchedulerCliDeps, build_parser, run_scheduler_cli


def _deps(calls=None):
    calls = calls if calls is not None else []

    def cb(name):
        return lambda args: calls.append((name, args))

    return SchedulerCliDeps(
        node_names=["node001", "local"],
        default_vram_mb=1000,
        default_ram_mb=2000,
        default_cpu_cores=3,
        dispatch_intent_ttl_s=12.0,
        resource_log_interval_s=34,
        archive_age_days=5.0,
        archive_max_hot_terminal=6,
        node_down_requeue_s=78,
        cmd_submit=cb("submit"),
        cmd_submit_jsonl=cb("submit-jsonl"),
        cmd_cpu_plan=cb("cpu-plan"),
        cmd_submit_cpu_batch=cb("submit-cpu-batch"),
        cmd_dispatch=cb("dispatch"),
        cmd_wait_for=cb("wait-for"),
        cmd_watch=cb("watch"),
        cmd_status=cb("status"),
        cmd_compact=cb("compact"),
        cmd_doctor=cb("doctor"),
        cmd_profile_local=cb("profile-local"),
        cmd_claims=cb("claims"),
        cmd_show=cb("show"),
        cmd_task_log=cb("task-log"),
        cmd_results=cb("results"),
        cmd_cancel=cb("cancel"),
        cmd_forget=cb("forget"),
        cmd_clear_queue=cb("clear-queue"),
        cmd_adopt=cb("adopt"),
        cmd_record_vram=cb("record-vram"),
        cmd_history=cb("history"),
        cmd_priority=cb("priority"),
        cmd_edit=cb("edit"),
        cmd_why=cb("why"),
        cmd_tui=cb("tui"),
    )


def test_run_scheduler_cli_invokes_injected_callback():
    calls = []

    run_scheduler_cli(
        _deps(calls),
        [
            "status",
            "--readonly",
            "--ids",
            "t1,t2",
        ],
    )

    assert len(calls) == 1
    name, args = calls[0]
    assert name == "status"
    assert args.readonly is True
    assert args.ids == ["t1,t2"]


def test_task_log_alias_routes_to_same_callback():
    calls = []

    run_scheduler_cli(_deps(calls), ["log", "t20846", "-n", "3", "--json"])

    assert len(calls) == 1
    name, args = calls[0]
    assert name == "task-log"
    assert args.id == "t20846"
    assert args.lines == 3
    assert args.json is True


def test_cancel_parser_accepts_multiple_ids_ranges_and_batch_alias():
    calls = []

    run_scheduler_cli(
        _deps(calls),
        ["cancel", "t1,t2", "t10-t12", "--force"],
    )
    run_scheduler_cli(
        _deps(calls),
        ["cancel-batch", "--project", "Laplace-*", "--confirm"],
    )

    first_name, first_args = calls[0]
    assert first_name == "cancel"
    assert first_args.ids == ["t1,t2", "t10-t12"]
    assert first_args.force is True
    second_name, second_args = calls[1]
    assert second_name == "cancel"
    assert second_args.cmd == "cancel-batch"
    assert second_args.project == "Laplace-*"
    assert second_args.confirm is True


def test_submit_parser_uses_dynamic_node_choices_and_defaults():
    parser = build_parser(_deps())

    args = parser.parse_args([
        "submit",
        "--description", "d",
        "--cmd", "python train.py",
        "--cwd", "/work",
        "--signature", "sig",
        "--preferred-node", "node001",
        "--node-down-requeue-s", "9",
    ])

    assert args.preferred_node == "node001"
    assert args.node_down_requeue_s == 9
    assert args.priority == "normal"
    assert args.env_spec == "none"


def test_submit_parser_rejects_removed_slurm_flags():
    parser = build_parser(_deps())

    try:
        parser.parse_args([
            "submit",
            "--description", "d",
            "--cmd", "python train.py",
            "--cwd", "/work",
            "--signature", "sig",
            "--slurm-partition", "gpu",
        ])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("removed --slurm-partition flag should be rejected")


def test_bulk_submit_intent_default_is_injected():
    parser = build_parser(_deps())

    args = parser.parse_args(["submit-jsonl", "--stdin", "--trusted"])

    assert args.intent_ttl == 12.0


def test_dispatch_bulk_window_defaults_on_and_can_be_disabled():
    parser = build_parser(_deps())

    assert parser.parse_args(["dispatch"]).bulk_window is True
    assert parser.parse_args(["dispatch", "--no-bulk-window"]).bulk_window is False
