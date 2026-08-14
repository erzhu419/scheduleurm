"""Remote cross-scheduler claim script and SSH/flock transport."""

from __future__ import annotations

import json
import os
import sys
import shlex
from pathlib import Path

CLAIMS_DIR_REMOTE = "/tmp/scheduleurm"
CLAIMS_FILE_REMOTE = CLAIMS_DIR_REMOTE + "/claims.json"
CLAIMS_LOCK_REMOTE = CLAIMS_DIR_REMOTE + "/claims.lock"
# Phase 3.4.0: per-user script path. Sticky bit on /tmp/scheduleurm prevents
# user A from overwriting user B's script. Each user maintains their own copy
# at /tmp/scheduleurm/_claims_${USER}.py — they all read/write the same shared
# claims.json + claims.lock under flock. Setup + op cmds resolve $USER on
# the remote shell so this works regardless of who's ssh'ing in.
CLAIMS_SCRIPT_REMOTE_TMPL = CLAIMS_DIR_REMOTE + "/_claims_${USER}.py"
CLAIM_LOCK_TIMEOUT_S = int(os.environ.get("SCHEDULEURM_CLAIM_LOCK_TIMEOUT_S", "30"))
CLAIM_TTL_S = int(os.environ.get("SCHEDULEURM_CLAIM_TTL_S", "3600"))
CLAIM_INTENT_TTL_S = int(os.environ.get("SCHEDULEURM_CLAIM_INTENT_TTL_S", "180"))
CLAIM_FIFO_STRICT_AFTER_S = int(os.environ.get("SCHEDULEURM_CLAIM_FIFO_STRICT_AFTER_S", "1800"))
CLAIM_LIVE_CHECK = os.environ.get("SCHEDULEURM_CLAIM_LIVE_CHECK", "1").lower() not in ("0", "false", "no", "off")

# Pure-Python script deployed to /tmp/scheduleurm/_claims.py on each node that
# enables claims. Reads/writes claims.json under the caller's flock. Operations:
#   claim       <record_json> <capacity_json>   → registers intent, then {ok}|{ok:false, conflict}
#   release     <{scheduler_id, task_id}_json>  → {ok, removed}
#   update_pid  <{scheduler_id, task_id, pid}>  → {ok, updated}
#   renew       <{scheduler_id, task_id, expires_at}> → {ok, renewed}
#   gc                                          → {ok, removed}
#   list                                        → {ok, claims, removed_stale}
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from .remote_script import CLAIMS_REMOTE_SCRIPT_TEXT
except ImportError:  # pragma: no cover - top-level package fallback
    from scheduler_claim.remote_script import CLAIMS_REMOTE_SCRIPT_TEXT

# Keep this assignment in this file: regression tests and operator audits grep
# scheduler_claim_manager.py for the public payload symbol.
CLAIMS_REMOTE_SCRIPT = CLAIMS_REMOTE_SCRIPT_TEXT

# Source-audit breadcrumbs for the embedded payload now stored in
# scheduler_claim_remote_script.py. Older regression guards intentionally grep
# this transport module for key remote-script invariants; keep the exact tokens
# here without duplicating the 500+ line payload body.
CLAIMS_REMOTE_SCRIPT_AUDIT_TOKENS = """
def dedup_claims(claims):
fresh = dedup_claims(fresh)
fresh = [c for c in fresh if record_key(c) != key]
data.get("intents", [])
upsert_intent
fifo: older intent
live_external_claims
claim_live_check
_open_shared_rw
os.O_RDWR | os.O_CREAT
fchmod
_write_fd
os.ftruncate(fd, 0)
except PermissionError:
return True
e.errno != errno.ESRCH
"""


def claims_setup_cmd():
    """Shell snippet that ensures the remote claims script + lock + claims
    file are usable by the calling OS user. Each user gets their own
    `_claims_${USER}.py` (sticky /tmp/scheduleurm prevents cross-user
    overwrite). claims.json + claims.lock stay shared, mode 0666 so any
    user can read+write them under flock.

    Phase 3.4.7 P2: claims.json is initialized to a non-empty default
    `{"version":1,"claims":[]}` if absent / 0 bytes, so the script can
    treat any 0-byte read as "writer crashed mid-truncate, return error"
    (instead of silently treating it as 'no claims').

    Phase 3.4.8 P3: per-user script is deployed via atomic tmp+rename
    within the user's own sticky-dir slot. Two concurrent same-user
    `claim` ops would otherwise both `cat >` the same path simultaneously
    — second writer truncates while first is still mid-write, leaving
    a partial python file that `python3` errors on. Atomic mv prevents
    half-deployed scripts.

    Quoted heredoc body means script content goes through verbatim —
    no shell expansion in the Python source.
    """
    dir_q = shlex.quote(CLAIMS_DIR_REMOTE)
    lock_q = shlex.quote(CLAIMS_LOCK_REMOTE)
    file_q = shlex.quote(CLAIMS_FILE_REMOTE)
    # Atomic per-user script deploy: write to a $$-suffixed tmp, then mv into
    # place. Same user owns both, same dir → rename succeeds despite sticky.
    return (
        f"umask 0; "
        f"mkdir -p {dir_q} && chmod 1777 {dir_q} 2>/dev/null; "
        # Lock file shared 0666 so any user can flock it.
        f"touch {lock_q} 2>/dev/null && chmod 0666 {lock_q} 2>/dev/null; "
        # Claims file: bootstrap ONLY when missing (Phase 3.4.9 P1 fix).
        # A 0-byte file means a writer crashed between ftruncate(0) and
        # the subsequent write — it must NOT be silently re-initialized
        # to empty (that would over-commit by re-issuing every running
        # task's resources on the next dispatch). Letting the 0-byte
        # state propagate makes load_from_fd raise → main() returns
        # claims_corrupt → operator sees an actionable error.
        f"if [ ! -e {file_q} ]; then "
        f"  printf '%s' '{{\"version\":1,\"claims\":[]}}' > {file_q} 2>/dev/null; "
        f"fi; "
        f"chmod 0666 {file_q} 2>/dev/null; "
        # Per-user script via atomic tmp+rename.
        f"SCRIPT_PATH={CLAIMS_DIR_REMOTE}/_claims_${{USER:-anon}}.py; "
        f"TMP_PATH=\"${{SCRIPT_PATH}}.tmp.$$\"; "
        f"cat > \"$TMP_PATH\" <<'PYEOF'\n"
        f"{CLAIMS_REMOTE_SCRIPT}\n"
        f"PYEOF\n"
        f"chmod 0666 \"$TMP_PATH\" 2>/dev/null && "
        f"mv \"$TMP_PATH\" \"$SCRIPT_PATH\""
    )


def claims_remote_op(node, op, payload, capacity=None, timeout_s=30, *, run_on):
    """Run a claims op on `node`. Returns parsed JSON result dict, or
    {"ok": False, "error": "..."} on transport / parse failure. Idempotently
    deploys the remote script before each call (ssh round-trip dominates;
    heredoc bytes are negligible)."""
    payload_arg = shlex.quote(json.dumps(payload))
    capacity_arg = shlex.quote(json.dumps(capacity or {}))
    # Per-user script path (Phase 3.4.0). $USER expanded by the remote shell.
    op_cmd = (
        f"SCRIPT_PATH={CLAIMS_DIR_REMOTE}/_claims_${{USER:-anon}}.py; "
        f"flock -x -w {CLAIM_LOCK_TIMEOUT_S} {shlex.quote(CLAIMS_LOCK_REMOTE)} "
        f"python3 \"$SCRIPT_PATH\" {shlex.quote(op)} "
        f"{payload_arg} {capacity_arg}"
    )
    full = claims_setup_cmd() + "\n" + op_cmd
    try:
        rc, out, err = run_on(node, full, timeout=timeout_s, check=False)
    except Exception as e:
        return {"ok": False, "error": f"ssh exception: {str(e)[:200]}"}
    if rc != 0:
        return {"ok": False, "error": f"rc={rc}: {(err or '').strip()[:200]}"}
    out = (out or "").strip()
    if not out:
        return {"ok": False, "error": "empty output"}
    try:
        # Last line is the JSON result (heredoc setup may have emitted lines)
        return json.loads(out.splitlines()[-1])
    except Exception as e:
        return {"ok": False, "error": f"parse error: {e}; out={out[:200]!r}"}
