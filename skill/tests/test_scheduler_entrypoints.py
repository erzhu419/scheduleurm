import subprocess
import sys
from pathlib import Path

from skill import tui


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER = REPO_ROOT / "skill" / "scheduler.py"


CLI_SUBCOMMANDS = [
    "submit",
    "submit-jsonl",
    "cpu-plan",
    "submit-cpu-batch",
    "dispatch",
    "wait-for",
    "watch",
    "status",
    "compact",
    "doctor",
    "profile-local",
    "claims",
    "show",
    "task-log",
    "results",
    "cancel",
    "cancel-batch",
    "forget",
    "clear-queue",
    "adopt",
    "record-vram",
    "history",
    "priority",
    "edit",
    "why",
    "tui",
]


def _scheduler_help(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(SCHEDULER), *args, "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_scheduler_top_level_help_lists_operational_subcommands():
    help_text = _scheduler_help()

    for command in CLI_SUBCOMMANDS:
        assert command in help_text


def test_scheduler_subcommand_help_smoke():
    for command in CLI_SUBCOMMANDS + ["log"]:
        help_text = _scheduler_help(command)
        assert "usage:" in help_text


def test_task_log_alias_help_routes_to_same_subcommand():
    task_log_help = _scheduler_help("task-log")
    alias_help = _scheduler_help("log")

    assert "usage: scheduler task-log" in task_log_help
    assert "usage: scheduler task-log" in alias_help
    assert "--no-archive" in task_log_help
    assert "--tail" in task_log_help
    assert "--json" in task_log_help
    assert "--no-archive" in alias_help
    assert "--tail" in alias_help
    assert "--json" in alias_help


def test_tui_exposes_task_log_keybinding():
    bindings = {(binding.key, binding.action) for binding in tui.SchedulerTUI.BINDINGS}

    assert ("l", "view_log") in bindings
