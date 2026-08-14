from __future__ import annotations

import json
import subprocess
import sys

from skill.scheduler_claim.remote_script import CLAIMS_REMOTE_SCRIPT_TEXT


def _claim(tmp_path, *, loadavg: float) -> dict:
    claims_path = tmp_path / f"claims-{loadavg}.json"
    claims_path.write_text(
        json.dumps({"version": 1, "claims": [], "intents": []}),
        encoding="utf-8",
    )
    script_path = tmp_path / f"claim-{loadavg}.py"
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
        "task_id": "full-cpu-batch",
        "gpu_idx": None,
        "vram_mb": 0,
        "cpu_cores": 12,
        "ram_mb": 32_768,
        "ignore_cpu_capacity": False,
        "ignore_one_third_pack_rule": True,
        "claimed_at": 1,
        "expires_at": 10**20,
        "intent_expires_at": 10**20,
        "pid": None,
    }
    capacity = {
        "cpu_cores": 12,
        "ram_mb": 500_000,
        "gpu_vram_mb": {},
        "live_check": True,
        "live_snapshot": {
            "loadavg": loadavg,
            "mem_available_mb": 500_000,
            "gpu_used_mb": {},
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


def test_sub_core_load_noise_does_not_block_full_cpu_batch(tmp_path):
    result = _claim(tmp_path, loadavg=0.49)

    assert result["ok"] is True


def test_whole_external_core_still_blocks_full_cpu_batch(tmp_path):
    result = _claim(tmp_path, loadavg=1.2)

    assert result["ok"] is False
    assert "claimed 1" in result["conflict"]
