#!/usr/bin/env bash
# scheduleurm install — copies the skill into your Claude Code skills dir and
# (optionally) installs the systemd user unit that runs the watcher every 60s.
#
# Idempotent: re-running upgrades the skill in place. Existing state in
# ~/.claude/scheduler/ (queue.json, vram_history.json, logs/) is NEVER touched.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$REPO_DIR/skill"
SKILL_DST="${SCHEDULEURM_SKILL_DIR:-$HOME/.claude/skills/scheduler}"
UNIT_SRC="$REPO_DIR/systemd/scheduler.service"
UNIT_DST="$HOME/.config/systemd/user/scheduler.service"

echo "==> installing skill files"
mkdir -p "$SKILL_DST/integrations"
cp "$SKILL_SRC/SKILL.md"            "$SKILL_DST/SKILL.md"
cp "$SKILL_SRC/scheduler.py"        "$SKILL_DST/scheduler.py"
cp "$SKILL_SRC/env_deploy.py"       "$SKILL_DST/env_deploy.py"
cp "$SKILL_SRC/tui.py"              "$SKILL_DST/tui.py"
cp "$SKILL_SRC/test_regression.py"  "$SKILL_DST/test_regression.py"
cp "$SKILL_SRC/test_hook.sh"        "$SKILL_DST/test_hook.sh"
cp "$SKILL_SRC/integrations/scheduler_mcp.py" "$SKILL_DST/integrations/scheduler_mcp.py"
cp "$SKILL_SRC/integrations/README.md"        "$SKILL_DST/integrations/README.md"
chmod +x "$SKILL_DST/scheduler.py" "$SKILL_DST/test_hook.sh"
echo "    skill installed at: $SKILL_DST"

echo
echo "==> verifying skill imports cleanly"
python3 -c "import sys; sys.path.insert(0, '$SKILL_DST'); import scheduler; print('    scheduler module loads OK (', len(scheduler.NODES), 'nodes configured)')"

if [[ "${1:-}" == "--no-systemd" ]]; then
    echo
    echo "==> --no-systemd: skipping watcher unit install"
    echo "    Run the watcher manually:  python3 $SKILL_DST/scheduler.py watch"
elif command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
    echo
    echo "==> installing systemd user unit"
    mkdir -p "$(dirname "$UNIT_DST")"
    # Rewrite the unit's ExecStart to point at the user's chosen skill dir
    sed "s|%h/.claude/skills/scheduler|$SKILL_DST|g" "$UNIT_SRC" > "$UNIT_DST"
    systemctl --user daemon-reload
    systemctl --user enable scheduler.service
    systemctl --user restart scheduler.service
    sleep 1
    if systemctl --user is-active scheduler.service >/dev/null; then
        echo "    watcher: active (running)"
    else
        echo "    watcher: NOT running — check 'journalctl --user -u scheduler -n 50'"
        exit 1
    fi
else
    echo
    echo "==> systemd not detected — skipping watcher unit"
    echo "    Run the watcher manually:  python3 $SKILL_DST/scheduler.py watch"
fi

echo
echo "==> done"
echo
echo "Next steps:"
echo "  1. Edit your cluster's NODES dict in $SKILL_DST/scheduler.py (top of file)"
echo "     OR set SCHEDULEURM_NODES_FILE=/path/to/nodes.json (if you wired that up)"
echo "  2. Confirm SSH passwordless to remote nodes (ssh -o BatchMode=yes <node> true)"
echo "  3. Try:    python3 $SKILL_DST/scheduler.py status"
echo "  4. Submit: python3 $SKILL_DST/scheduler.py submit --help"
echo
echo "Skill is auto-discovered by Claude Code on next session start."
echo "From inside Claude Code, just say things like '跑这个脚本' or 'GPU 还空吗'."
