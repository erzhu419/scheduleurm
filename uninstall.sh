#!/usr/bin/env bash
# scheduleurm uninstall — removes the skill installation and the systemd unit.
# Handles both COPY mode (real directory) and LINK mode (symlink to clone).
# Does NOT touch ~/.claude/scheduler/ (queue.json, vram_history.json, logs).
# Pass --purge-state to also wipe state.
set -euo pipefail

SKILL_DST="${SCHEDULEURM_SKILL_DIR:-$HOME/.claude/skills/scheduler}"
UNIT_DST="$HOME/.config/systemd/user/scheduler.service"
STATE_DIR="$HOME/.claude/scheduler"

if command -v systemctl >/dev/null 2>&1 && systemctl --user is-enabled scheduler.service >/dev/null 2>&1; then
    echo "==> stopping + disabling watcher"
    systemctl --user stop scheduler.service || true
    systemctl --user disable scheduler.service || true
fi

if [[ -f "$UNIT_DST" ]]; then
    echo "==> removing $UNIT_DST"
    rm -f "$UNIT_DST"
    systemctl --user daemon-reload || true
fi

# Symlink case: rm -f (NOT rm -rf, NEVER with trailing slash) just removes the link.
# Directory case: rm -rf the actual files.
if [[ -L "$SKILL_DST" ]]; then
    target=$(readlink "$SKILL_DST")
    echo "==> removing symlink $SKILL_DST  (was -> $target)"
    rm -f "$SKILL_DST"
elif [[ -d "$SKILL_DST" ]]; then
    echo "==> removing skill directory $SKILL_DST"
    rm -rf "$SKILL_DST"
fi

if [[ "${1:-}" == "--purge-state" ]]; then
    if [[ -d "$STATE_DIR" ]]; then
        echo "==> --purge-state: removing $STATE_DIR (queue + history + logs)"
        rm -rf "$STATE_DIR"
    fi
else
    echo
    echo "State preserved at $STATE_DIR"
    echo "Re-run with --purge-state to wipe queue/history/logs as well."
fi

echo "==> done"
