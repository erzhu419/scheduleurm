from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_PATH = REPO_ROOT / "skill" / "integrations" / "scheduler_mcp.py"


class _FakeFastMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self):
        def _decorate(fn):
            return fn

        return _decorate


def _load_mcp_module(monkeypatch, module_name: str = "scheduleurm_mcp_task_log_test"):
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")
    fake_fastmcp.FastMCP = _FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)

    spec = importlib.util.spec_from_file_location(module_name, MCP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scheduler_mcp from {MCP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_mcp_task_log_uses_scheduler_cli(monkeypatch):
    mcp = _load_mcp_module(monkeypatch)
    calls: list[tuple[list[str], int]] = []

    def fake_run(args, timeout=60):
        calls.append((args, timeout))
        return {
            "ok": True,
            "exit_code": 0,
            "stdout": json.dumps(
                {
                    "ok": True,
                    "id": "t20846",
                    "status": "done",
                    "source": "archive",
                    "tail": "DONE\n",
                    "error": "",
                }
            ),
            "stderr": "",
        }

    monkeypatch.setattr(mcp, "_run", fake_run)

    result = mcp.task_log("t20846", 3)

    assert calls == [(["task-log", "t20846", "-n", "3", "--json"], 45)]
    assert result["ok"] is True
    assert result["source"] == "archive"
    assert result["tail"] == "DONE\n"


def test_mcp_task_log_reports_non_json_cli_output(monkeypatch):
    mcp = _load_mcp_module(monkeypatch, "scheduleurm_mcp_task_log_bad_json_test")

    monkeypatch.setattr(
        mcp,
        "_run",
        lambda args, timeout=60: {
            "ok": True,
            "exit_code": 0,
            "stdout": "not-json",
            "stderr": "plain text failure",
        },
    )

    result = mcp.task_log("t99999", 0)

    assert result["ok"] is False
    assert "non-json output" in result["stderr"]
