"""Verify the installed wheel without accessing an operator's cluster or queue."""

from __future__ import annotations

import asyncio
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import scheduleurm


async def check_mcp(env, home):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    def payload(result):
        assert not result.isError, result
        return json.loads(next(item.text for item in result.content if item.type == "text"))

    server = Path(sys.executable).parent / "scheduleurm-mcp"
    parameters = StdioServerParameters(command=str(server), env=env)
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert len(names) == 12, names
            result = payload(await client.call_tool("submit_task", {
                "description": "Release smoke evaluation", "cmd": "printf release-ok",
                "cwd": str(home), "signature": "release/smoke", "vram_mb": 0,
                "ram_mb": 128, "cpu_cores": 1,
                "wait_for_files": [str(home / "not-ready")],
            }))
            assert result["ok"], result
            task_id = re.search(r"\bt\d+\b", result["stdout"]).group(0)
            queue = payload(await client.call_tool("queue_dump"))
            assert queue["ok"] and queue["count"] == 1, queue
            shown = payload(await client.call_tool("show_task", {"task_id": task_id}))
            assert shown["ok"], shown
            cancelled = payload(await client.call_tool("cancel_task", {"task_id": task_id}))
            assert cancelled["ok"], cancelled
            queue = payload(await client.call_tool("queue_dump", {"filter_status": "cancelled"}))
            assert queue["count"] == 1, queue


def main():
    package = Path(scheduleurm.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    for name in ("default_service_cache.json", "corner_case_service_cache_20260613.json",
                 "service_cache_v2_live_merged_pre_native_20260727.json"):
        assert (package / "md" / "experiment_artifacts" / name).is_file(), name
    with tempfile.TemporaryDirectory(prefix="scheduleurm-release-") as directory:
        home = Path(directory)
        env = {key: value for key, value in os.environ.items()
               if key != "PYTHONPATH" and key != "SCHEDULER_PY" and not key.startswith("SCHEDULEURM_")}
        env["HOME"] = str(home)
        executable = Path(sys.executable).parent / "scheduleurm"
        for args in (["--version"], ["--help"]):
            result = subprocess.run([str(executable), *args], cwd=home, env=env,
                                    text=True, capture_output=True, check=True, timeout=60)
            if args == ["--version"]:
                assert result.stdout.strip() == version("scheduleurm"), result.stdout
            else:
                assert "submit" in result.stdout and "watch" in result.stdout, result.stdout
        code = """
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'skill'))
import scheduler
assert set(scheduler.NODES) == {'local'}, scheduler.NODES
from scheduler_algorithm.bridge import _default_load_placement_policy
assert _default_load_placement_policy is not None
from algorithm import available_policies, load_placement_policy
for name in available_policies():
    assert load_placement_policy(name) is not None, name
from simulation.defaults import build_default_cache
assert build_default_cache().snapshot()['records']
import tui
assert callable(tui.main)
"""
        subprocess.run([sys.executable, "-c", code, str(package)], cwd=home, env=env,
                       check=True, timeout=60)
        asyncio.run(check_mcp(env, home))
    print("Installed CLI, all placement policies, caches, TUI import and 12-tool MCP submit/show/cancel: passed")


if __name__ == "__main__":
    main()
