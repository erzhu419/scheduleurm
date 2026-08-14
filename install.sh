#!/usr/bin/env bash
# scheduleurm install — installs the skill into your Claude Code skills dir and
# (optionally) installs the systemd user unit that runs the watcher every 60s.
#
# Modes:
#   ./install.sh                # LINK mode: symlink ~/.claude/skills/scheduler -> <clone>/skill
#                               #            Edits to the clone are picked up immediately.
#                               #            Don't move/delete the clone afterwards.
#   ./install.sh --copy         # COPY mode: cp skill files to ~/.claude/skills/scheduler/
#   ./install.sh --no-systemd   # skip the systemd user unit (combinable with --link)
#   ./install.sh --copy --cache-artifact /path/to/cache.json
#                               # include a validated cache artifact (repeatable)
#   SCHEDULEURM_CACHE_ARTIFACTS=/a.json:/b.json ./install.sh --copy
#                               # include a colon-separated cache selection
#
# Idempotent: re-running upgrades the skill in place. Existing state in
# ~/.claude/scheduler/ (queue.json, vram_history.json, logs/) is NEVER touched.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$REPO_DIR/skill"
ALGORITHM_SRC="$REPO_DIR/algorithm"
SIMULATION_SRC="$REPO_DIR/simulation"
SKILL_DST="${SCHEDULEURM_SKILL_DIR:-$HOME/.claude/skills/scheduler}"
INSTALL_ROOT="$(dirname "$SKILL_DST")"
ALGORITHM_DST="$INSTALL_ROOT/algorithm"
SIMULATION_DST="$INSTALL_ROOT/simulation"
CACHE_ARTIFACT_DST="$INSTALL_ROOT/md/experiment_artifacts"
UNIT_SRC="$REPO_DIR/systemd/scheduler.service"
UNIT_DST="$HOME/.config/systemd/user/scheduler.service"
UNIT_OVERRIDE_DST="$HOME/.config/systemd/user/scheduler.service.d/override.conf"
KEEPALIVE_SERVICE_SRC="$REPO_DIR/systemd/scheduler-keepalive.service"
KEEPALIVE_TIMER_SRC="$REPO_DIR/systemd/scheduler-keepalive.timer"
KEEPALIVE_SERVICE_DST="$HOME/.config/systemd/user/scheduler-keepalive.service"
KEEPALIVE_TIMER_DST="$HOME/.config/systemd/user/scheduler-keepalive.timer"

LINK_MODE=1
NO_SYSTEMD=0
CACHE_ARTIFACTS=()
if [[ -n "${SCHEDULEURM_CACHE_ARTIFACTS:-}" ]]; then
    while IFS= read -r cache_artifact; do
        [[ -n "$cache_artifact" ]] && CACHE_ARTIFACTS+=("$cache_artifact")
    done < <(printf '%s\n' "$SCHEDULEURM_CACHE_ARTIFACTS" | tr ':' '\n')
fi

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --link)
            LINK_MODE=1
            shift ;;
        --copy)
            LINK_MODE=0
            shift ;;
        --no-systemd)
            NO_SYSTEMD=1
            shift ;;
        --cache-artifact)
            if [[ "$#" -lt 2 || -z "$2" ]]; then
                echo "--cache-artifact requires a JSON file path" >&2
                exit 2
            fi
            CACHE_ARTIFACTS+=("$2")
            shift 2 ;;
        --cache-artifact=*)
            cache_artifact="${1#*=}"
            if [[ -z "$cache_artifact" ]]; then
                echo "--cache-artifact requires a JSON file path" >&2
                exit 2
            fi
            CACHE_ARTIFACTS+=("$cache_artifact")
            shift ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown arg: $1 (try --help)" >&2
            exit 2 ;;
    esac
done

PYTHON_BIN="$(command -v python3 || true)"
if [[ "$LINK_MODE" -eq 0 && -z "$PYTHON_BIN" ]]; then
    echo "COPY mode requires python3 for cache and staged-runtime validation" >&2
    exit 1
fi
if [[ "$LINK_MODE" -eq 1 && "${#CACHE_ARTIFACTS[@]}" -gt 0 ]]; then
    echo "--cache-artifact and SCHEDULEURM_CACHE_ARTIFACTS require --copy" >&2
    exit 2
fi

CACHE_ARTIFACT_DEST_NAMES=()
if [[ "$LINK_MODE" -eq 0 ]]; then
    for index in "${!CACHE_ARTIFACTS[@]}"; do
        cache_artifact="${CACHE_ARTIFACTS[$index]}"
        if [[ "$cache_artifact" == "~/"* ]]; then
            cache_artifact="$HOME/${cache_artifact#\~/}"
        elif [[ "$cache_artifact" != /* ]]; then
            cache_artifact="$PWD/$cache_artifact"
        fi
        CACHE_ARTIFACTS[$index]="$cache_artifact"

        if [[ ! -f "$cache_artifact" || ! -r "$cache_artifact" ]]; then
            echo "cache artifact is not a readable file: $cache_artifact" >&2
            exit 1
        fi
        cache_name="$(basename "$cache_artifact")"
        for existing_name in "${CACHE_ARTIFACT_DEST_NAMES[@]}"; do
            if [[ "$existing_name" == "$cache_name" ]]; then
                echo "cache artifacts collide at md/experiment_artifacts/$cache_name" >&2
                exit 1
            fi
        done
        CACHE_ARTIFACT_DEST_NAMES+=("$cache_name")

        "$PYTHON_BIN" -B - "$cache_artifact" <<'PY'
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        json.load(handle)
except Exception as exc:
    print(f"invalid JSON cache artifact {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    done
fi

path_exists() {
    [[ -e "$1" || -L "$1" ]]
}

remove_path() {
    local path="$1"
    if [[ -L "$path" || -f "$path" ]]; then
        rm -f -- "$path"
    elif [[ -d "$path" ]]; then
        rm -rf -- "$path"
    elif [[ -e "$path" ]]; then
        rm -f -- "$path"
    fi
}

copy_source_tree() {
    local source_dir="$1"
    local destination_dir="$2"
    mkdir -p "$destination_dir"
    tar -C "$source_dir" \
        --exclude='__pycache__' \
        --exclude='*/__pycache__' \
        --exclude='.pytest_cache' \
        --exclude='*/.pytest_cache' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        --exclude='*.orig' \
        -cf - . | tar -C "$destination_dir" -xf -
}

atomic_exchange_paths() {
    local left="$1"
    local right="$2"
    "$PYTHON_BIN" -B - "$left" "$right" <<'PY'
import ctypes
import errno
import os
import sys

left, right = sys.argv[1:3]
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = getattr(libc, "renameat2", None)
if renameat2 is None:
    raise SystemExit(75)

renameat2.argtypes = [
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_uint,
]
renameat2.restype = ctypes.c_int
at_fdcwd = -100
rename_exchange = 2
result = renameat2(
    at_fdcwd,
    os.fsencode(left),
    at_fdcwd,
    os.fsencode(right),
    rename_exchange,
)
if result != 0:
    error = ctypes.get_errno()
    if error in {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
        errno.EXDEV,
    }:
        raise SystemExit(75)
    raise OSError(error, os.strerror(error), f"{left} <-> {right}")
PY
}

COPY_STAGE_ROOT=""
COPY_TRANSACTION_ACTIVE=0
PUBLISHED_STAGE_PATHS=()
PUBLISHED_DEST_PATHS=()
PUBLISHED_METHODS=()
PUBLISHED_PREVIOUS_PATHS=()

publish_staged_path() {
    local staged_path="$1"
    local destination_path="$2"
    local exchange_status=0
    local method="created"
    local previous_path=""
    local record_index=0

    mkdir -p "$(dirname "$destination_path")"
    if path_exists "$destination_path"; then
        if atomic_exchange_paths "$staged_path" "$destination_path"; then
            method="exchange"
        else
            exchange_status="$?"
            if [[ "$exchange_status" -ne 75 ]]; then
                return "$exchange_status"
            fi
            echo "    filesystem lacks atomic exchange; using transactional rename for $destination_path"
            record_index="${#PUBLISHED_METHODS[@]}"
            previous_path="$COPY_STAGE_ROOT/.install-previous/$record_index"
            mkdir -p "$(dirname "$previous_path")"
            mv -- "$destination_path" "$previous_path"
            PUBLISHED_STAGE_PATHS+=("$staged_path")
            PUBLISHED_DEST_PATHS+=("$destination_path")
            PUBLISHED_METHODS+=("transactional-prepared")
            PUBLISHED_PREVIOUS_PATHS+=("$previous_path")
            if mv -- "$staged_path" "$destination_path"; then
                PUBLISHED_METHODS[$record_index]="transactional"
                return 0
            else
                exchange_status="$?"
                return "$exchange_status"
            fi
        fi
    else
        mv -- "$staged_path" "$destination_path"
    fi
    PUBLISHED_STAGE_PATHS+=("$staged_path")
    PUBLISHED_DEST_PATHS+=("$destination_path")
    PUBLISHED_METHODS+=("$method")
    PUBLISHED_PREVIOUS_PATHS+=("")
}

rollback_copy_transaction() {
    local index
    local failed=0
    set +e
    for ((index=${#PUBLISHED_DEST_PATHS[@]} - 1; index >= 0; index--)); do
        case "${PUBLISHED_METHODS[$index]}" in
            exchange)
                atomic_exchange_paths \
                    "${PUBLISHED_STAGE_PATHS[$index]}" \
                    "${PUBLISHED_DEST_PATHS[$index]}" || failed=1
                ;;
            transactional)
                if mv -- \
                    "${PUBLISHED_DEST_PATHS[$index]}" \
                    "${PUBLISHED_STAGE_PATHS[$index]}"; then
                    if ! mv -- \
                        "${PUBLISHED_PREVIOUS_PATHS[$index]}" \
                        "${PUBLISHED_DEST_PATHS[$index]}"; then
                        mv -- \
                            "${PUBLISHED_STAGE_PATHS[$index]}" \
                            "${PUBLISHED_DEST_PATHS[$index]}" || true
                        failed=1
                    fi
                else
                    failed=1
                fi
                ;;
            transactional-prepared)
                if path_exists "${PUBLISHED_DEST_PATHS[$index]}"; then
                    failed=1
                else
                    mv -- \
                        "${PUBLISHED_PREVIOUS_PATHS[$index]}" \
                        "${PUBLISHED_DEST_PATHS[$index]}" || failed=1
                fi
                ;;
            created)
                remove_path "${PUBLISHED_DEST_PATHS[$index]}" || failed=1
                ;;
            *)
                failed=1
                ;;
        esac
    done
    set -e
    return "$failed"
}

cleanup_copy_transaction() {
    local status="$?"
    local rollback_failed=0
    trap - EXIT
    if [[ "$COPY_TRANSACTION_ACTIVE" -eq 1 && "${#PUBLISHED_DEST_PATHS[@]}" -gt 0 ]]; then
        echo "==> install failed; restoring the previous COPY-mode runtime" >&2
        rollback_copy_transaction || rollback_failed=1
    fi
    if [[ "$rollback_failed" -eq 0 && -n "$COPY_STAGE_ROOT" ]] && path_exists "$COPY_STAGE_ROOT"; then
        remove_path "$COPY_STAGE_ROOT" || rollback_failed=1
    fi
    if [[ "$rollback_failed" -ne 0 ]]; then
        echo "ERROR: automatic installer rollback did not complete" >&2
        echo "       rollback data was retained at $COPY_STAGE_ROOT" >&2
        status=1
    fi
    exit "$status"
}

# If the watcher is running, keep active experiments alive and restart only the
# watcher after files/unit are updated. The unit uses KillMode=process for this.
WATCHER_WAS_RUNNING=0
if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active scheduler.service >/dev/null 2>&1; then
    WATCHER_WAS_RUNNING=1
    echo "==> watcher is running; will restart it after files/unit are updated"
fi

# Ensure parent dir exists
mkdir -p "$INSTALL_ROOT"

if [[ "$LINK_MODE" -eq 1 ]]; then
    # Remove existing destination — handles both the symlink case and the directory case.
    # CRITICAL: a bare `rm -rf "$SKILL_DST"` removes a symlink (good) but
    # `rm -rf "$SKILL_DST/"` follows it and can wipe the cloned repo.
    if [[ -L "$SKILL_DST" ]]; then
        echo "==> removing old symlink at $SKILL_DST"
        rm -f "$SKILL_DST"
    elif [[ -d "$SKILL_DST" ]]; then
        # Switching from copy → link mode. Back up first.
        bak="${SKILL_DST}.bak-$(date +%s)"
        echo "==> existing copy detected; backing up to $bak before symlinking"
        # Move OUTSIDE skills/ so Claude Code doesn't auto-discover the backup as a phantom skill
        mv "$SKILL_DST" "$HOME/.claude/scheduler.bak-$(date +%s)"
    fi

    echo "==> LINK mode: symlinking $SKILL_DST -> $SKILL_SRC"
    ln -sfn "$SKILL_SRC" "$SKILL_DST"
    echo "    edits in $SKILL_SRC are now live without re-running install.sh"
    if [[ -d "$ALGORITHM_SRC" ]]; then
        if [[ -L "$ALGORITHM_DST" ]]; then
            echo "==> removing old algorithm symlink at $ALGORITHM_DST"
            rm -f "$ALGORITHM_DST"
        elif [[ -d "$ALGORITHM_DST" ]]; then
            echo "==> existing algorithm copy detected; backing up before symlinking"
            mv "$ALGORITHM_DST" "$HOME/.claude/scheduler.algorithm.bak-$(date +%s)"
        fi
        echo "==> LINK mode: symlinking $ALGORITHM_DST -> $ALGORITHM_SRC"
        ln -sfn "$ALGORITHM_SRC" "$ALGORITHM_DST"
    fi

    echo
    echo "==> verifying skill imports cleanly"
    python3 -c "import sys; sys.path.insert(0, '$SKILL_DST'); import scheduler; print('    scheduler module loads OK (', len(scheduler.NODES), 'nodes configured)')"
else
    for required_source in "$SKILL_SRC" "$ALGORITHM_SRC" "$SIMULATION_SRC"; do
        if [[ ! -d "$required_source" ]]; then
            echo "required COPY-mode source directory is missing: $required_source" >&2
            exit 1
        fi
    done
    if [[ "$SKILL_DST" == "$SKILL_SRC" ]]; then
        echo "refusing to replace the source skill directory in COPY mode: $SKILL_DST" >&2
        exit 1
    fi

    stage_prefix=".$(basename "$SKILL_DST").stage.XXXXXXXX"
    COPY_STAGE_ROOT="$(mktemp -d "$INSTALL_ROOT/$stage_prefix")"
    COPY_TRANSACTION_ACTIVE=1
    trap cleanup_copy_transaction EXIT
    chmod 755 "$COPY_STAGE_ROOT"

    echo "==> COPY mode: staging complete runtime at $COPY_STAGE_ROOT"
    copy_source_tree "$SKILL_SRC" "$COPY_STAGE_ROOT/skill"
    copy_source_tree "$ALGORITHM_SRC" "$COPY_STAGE_ROOT/algorithm"
    copy_source_tree "$SIMULATION_SRC" "$COPY_STAGE_ROOT/simulation"
    mkdir -p "$COPY_STAGE_ROOT/md/experiment_artifacts"
    for index in "${!CACHE_ARTIFACTS[@]}"; do
        cp -- \
            "${CACHE_ARTIFACTS[$index]}" \
            "$COPY_STAGE_ROOT/md/experiment_artifacts/${CACHE_ARTIFACT_DEST_NAMES[$index]}"
    done
    chmod +x "$COPY_STAGE_ROOT/skill/scheduler.py" "$COPY_STAGE_ROOT/skill/test_hook.sh"

    verify_dir="$COPY_STAGE_ROOT/.install-verify"
    mkdir -p "$verify_dir/home"
    echo
    echo "==> verifying staged runtime in a clean process"
    (
        cd "$verify_dir"
        env -i \
            HOME="$verify_dir/home" \
            PATH="$PATH" \
            "$PYTHON_BIN" -B -I -c '
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
skill_root = root / "skill"
sys.path.insert(0, str(skill_root))

import scheduler
import algorithm
import simulation
from simulation.service_cache import ServiceRateCache

expected_roots = {
    "scheduler": skill_root,
    "algorithm": root / "algorithm",
    "simulation": root / "simulation",
}
for name, module in (
    ("scheduler", scheduler),
    ("algorithm", algorithm),
    ("simulation", simulation),
):
    module_path = pathlib.Path(module.__file__).resolve()
    expected = expected_roots[name].resolve()
    if os.path.commonpath((str(module_path), str(expected))) != str(expected):
        raise RuntimeError(
            f"{name} escaped staged runtime: {module_path} (expected under {expected})"
        )

policy = algorithm.load_placement_policy("legacy")
cache = ServiceRateCache()
artifact_root = root / "md" / "experiment_artifacts"
artifact_count = 0
for artifact in artifact_root.iterdir():
    if artifact.is_file():
        with artifact.open("r", encoding="utf-8") as handle:
            json.load(handle)
        artifact_count += 1

print(
    "    staged scheduler runtime loads OK "
    f"({len(scheduler.NODES)} nodes, "
    f"{artifact_count} selected cache artifacts, "
    f"{policy.name} policy)"
)
' "$COPY_STAGE_ROOT"
    )
    remove_path "$verify_dir"

    echo
    echo "==> publishing staged COPY-mode runtime"
    publish_staged_path "$COPY_STAGE_ROOT/algorithm" "$ALGORITHM_DST"
    publish_staged_path "$COPY_STAGE_ROOT/simulation" "$SIMULATION_DST"
    for index in "${!CACHE_ARTIFACTS[@]}"; do
        publish_staged_path \
            "$COPY_STAGE_ROOT/md/experiment_artifacts/${CACHE_ARTIFACT_DEST_NAMES[$index]}" \
            "$CACHE_ARTIFACT_DST/${CACHE_ARTIFACT_DEST_NAMES[$index]}"
    done
    # Publish the entrypoint last, after every dependency it can import is live.
    publish_staged_path "$COPY_STAGE_ROOT/skill" "$SKILL_DST"
    echo "    runtime published at $SKILL_DST"
fi

if [[ "$NO_SYSTEMD" -eq 1 ]]; then
    echo
    echo "==> --no-systemd: skipping watcher unit install"
    echo "    Run the watcher manually:  python3 $SKILL_DST/scheduler.py watch"
elif command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    echo
    echo "==> installing systemd user unit"
    mkdir -p "$(dirname "$UNIT_DST")"
    # The unit references %h/.claude/skills/scheduler/... which resolves correctly whether
    # SKILL_DST is a real dir or a symlink, so the unit body is identical for both modes.
    # Only rewrite the path if user has overridden SCHEDULEURM_SKILL_DIR away from the default.
    if [[ "$SKILL_DST" == "$HOME/.claude/skills/scheduler" ]]; then
        cp "$UNIT_SRC" "$UNIT_DST"
        cp "$KEEPALIVE_SERVICE_SRC" "$KEEPALIVE_SERVICE_DST"
    else
        sed "s|%h/.claude/skills/scheduler|$SKILL_DST|g" "$UNIT_SRC" > "$UNIT_DST"
        sed "s|%h/.claude/skills/scheduler|$SKILL_DST|g" "$KEEPALIVE_SERVICE_SRC" > "$KEEPALIVE_SERVICE_DST"
    fi
    if [[ -f "$UNIT_OVERRIDE_DST" ]] && grep -q -- "--interval 5" "$UNIT_OVERRIDE_DST"; then
        echo "==> updating stale scheduler.service override interval 5s -> 60s"
        sed -i 's/--interval 5/--interval 60/g' "$UNIT_OVERRIDE_DST"
    fi
    cp "$KEEPALIVE_TIMER_SRC" "$KEEPALIVE_TIMER_DST"
    systemctl --user daemon-reload
    systemctl --user enable scheduler.service >/dev/null
    systemctl --user enable --now scheduler-keepalive.timer >/dev/null
    if [[ "$WATCHER_WAS_RUNNING" -eq 1 ]]; then
        systemctl --user restart scheduler.service
    else
        systemctl --user start scheduler.service
    fi
    sleep 1
    if systemctl --user is-active scheduler.service >/dev/null; then
        echo "    watcher: active (running)"
    else
        echo "    watcher: NOT running — check 'journalctl --user -u scheduler -n 50'"
        exit 1
    fi
    if systemctl --user is-active scheduler-keepalive.timer >/dev/null; then
        echo "    keepalive: active (timer)"
    else
        echo "    keepalive: NOT running — check 'journalctl --user -u scheduler-keepalive -n 50'"
        exit 1
    fi
else
    echo
    echo "==> systemd not detected — skipping watcher unit"
    echo "    Run the watcher manually:  python3 $SKILL_DST/scheduler.py watch"
fi

if [[ "$LINK_MODE" -eq 0 ]]; then
    COPY_TRANSACTION_ACTIVE=0
    trap - EXIT
    if path_exists "$COPY_STAGE_ROOT" && ! remove_path "$COPY_STAGE_ROOT"; then
        echo "WARNING: installed runtime is active, but old staging data remains at $COPY_STAGE_ROOT" >&2
    fi
fi

echo
echo "==> done"
echo
echo "Next steps:"
if [[ "$LINK_MODE" -eq 1 ]]; then
    echo "  1. Edit cluster's NODES dict in $SKILL_SRC/scheduler.py (top of file)"
    echo "     Edits go live immediately — no need to re-run install.sh."
    echo "     Run 'systemctl --user restart scheduler' to pick up scheduler.py changes in the watcher."
else
    echo "  1. Edit cluster's NODES dict in $SKILL_DST/scheduler.py (top of file)"
    echo "     OR re-run ./install.sh after editing $SKILL_SRC/scheduler.py to push edits."
fi
echo "  2. Confirm SSH passwordless to remote nodes (ssh -o BatchMode=yes <node> true)"
echo "  3. Try:    python3 $SKILL_DST/scheduler.py status"
echo "  4. Submit: python3 $SKILL_DST/scheduler.py submit --help"
echo
echo "Skill is auto-discovered by Claude Code on next session start."
echo "From inside Claude Code, just say things like '跑这个脚本' or 'GPU 还空吗'."
