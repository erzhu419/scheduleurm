from __future__ import annotations

import json
import subprocess
import sys

from skill.scheduler_claim.remote_script import CLAIMS_REMOTE_SCRIPT_TEXT


def _run_claim(tmp_path, *, live_used_mb: int) -> dict:
    claims_path = tmp_path / f"claims-{live_used_mb}.json"
    claims_path.write_text(
        json.dumps({"version": 1, "claims": [], "intents": []}),
        encoding="utf-8",
    )
    script_path = tmp_path / f"claim-{live_used_mb}.py"
    script_path.write_text(
        CLAIMS_REMOTE_SCRIPT_TEXT.replace(
            'CLAIMS_FILE = "/tmp/scheduleurm/claims.json"',
            f"CLAIMS_FILE = {str(claims_path)!r}",
        ),
        encoding="utf-8",
    )
    payload = {
        "owner": "test",
        "scheduler_id": "test-scheduler",
        "task_id": "large-first-task",
        "gpu_idx": 0,
        "vram_mb": 10_500,
        "cpu_cores": 0,
        "ram_mb": 1_461,
        "ignore_cpu_capacity": True,
        "ignore_one_third_pack_rule": False,
        "claimed_at": 1,
        "expires_at": 10**20,
        "intent_expires_at": 10**20,
        "pid": None,
    }
    capacity = {
        "cpu_cores": 12,
        "ram_mb": 500_000,
        "gpu_vram_mb": {"0": 12_288},
        "max_vram_per_task": None,
        "vram_margin_mb": 256,
        "third_pack_rule": True,
        "one_third_grace_mb": 512,
        "gpu_empty_used_mb": 200,
        "fifo_strict_after_s": 1_800,
        "live_check": True,
        "live_snapshot": {
            "loadavg": 0,
            "mem_available_mb": 500_000,
            "gpu_used_mb": {"0": live_used_mb},
        },
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "claim",
            json.dumps(payload),
            json.dumps(capacity),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_remote_claim_treats_driver_vram_below_empty_threshold_as_empty(tmp_path):
    result = _run_claim(tmp_path, live_used_mb=169)

    assert result["ok"] is True


def test_remote_claim_still_blocks_real_occupied_vram(tmp_path):
    result = _run_claim(tmp_path, live_used_mb=201)

    assert result["ok"] is False
    assert "occupied claim would cross" in result["conflict"]
