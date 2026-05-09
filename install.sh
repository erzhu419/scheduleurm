#!/usr/bin/env bash
# scheduleurm install — installs the skill into your Claude Code skills dir and
# (optionally) installs the systemd user unit that runs the watcher every 60s.
#
# Modes:
#   ./install.sh                # COPY mode: cp skill files to ~/.claude/skills/scheduler/
#   ./install.sh --link         # LINK mode: symlink ~/.claude/skills/scheduler -> <clone>/skill
#                               #            Edits to the clone are picked up immediately.
#                               #            Don't move/delete the clone afterwards.
#   ./install.sh --no-systemd   # skip the systemd user unit (combinable with --link)
#
# Idempotent: re-running upgrades the skill in place. Existing state in
# ~/.claude/scheduler/ (queue.json, vram_history.json, logs/) is NEVER touched.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_SRC="$REPO_DIR/skill"
SKILL_DST="${SCHEDULEURM_SKILL_DIR:-$HOME/.claude/skills/scheduler}"
UNIT_SRC="$REPO_DIR/systemd/scheduler.service"
UNIT_DST="$HOME/.config/systemd/user/scheduler.service"

LINK_MODE=0
NO_SYSTEMD=0
for arg in "$@"; do
    case "$arg" in
        --link)        LINK_MODE=1 ;;
        --no-systemd)  NO_SYSTEMD=1 ;;
        -h|--help)
            sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# If the watcher is running, keep active experiments alive and restart only the
# watcher after files/unit are updated. The unit uses KillMode=process for this.
WATCHER_WAS_RUNNING=0
if command -v systemctl >/dev/null 2>&1 && systemctl --user is-active scheduler.service >/dev/null 2>&1; then
    WATCHER_WAS_RUNNING=1
    echo "==> watcher is running; will restart it after files/unit are updated"
fi

# Ensure parent dir exists
mkdir -p "$(dirname "$SKILL_DST")"

# Remove existing destination — handles both the symlink case and the directory case.
# CRITICAL: a bare `rm -rf "$SKILL_DST"` removes a symlink (good) but `rm -rf "$SKILL_DST/"`
# (trailing slash) follows the symlink and wipes the cloned repo. We use no trailing slash
# and also `-f` not `-rf` for the symlink case to be doubly safe.
if [[ -L "$SKILL_DST" ]]; then
    echo "==> removing old symlink at $SKILL_DST"
    rm -f "$SKILL_DST"
elif [[ -d "$SKILL_DST" ]]; then
    if [[ "$LINK_MODE" -eq 1 ]]; then
        # Switching from copy → link mode. Back up first.
        bak="${SKILL_DST}.bak-$(date +%s)"
        echo "==> existing copy detected; backing up to $bak before symlinking"
        # Move OUTSIDE skills/ so Claude Code doesn't auto-discover the backup as a phantom skill
        mv "$SKILL_DST" "$HOME/.claude/scheduler.bak-$(date +%s)"
    else
        echo "==> removing existing copy at $SKILL_DST"
        rm -rf "$SKILL_DST"
    fi
fi

if [[ "$LINK_MODE" -eq 1 ]]; then
    echo "==> LINK mode: symlinking $SKILL_DST -> $SKILL_SRC"
    ln -sfn "$SKILL_SRC" "$SKILL_DST"
    echo "    edits in $SKILL_SRC are now live without re-running install.sh"
else
    echo "==> COPY mode: copying skill files to $SKILL_DST"
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
fi

echo
echo "==> verifying skill imports cleanly"
python3 -c "import sys; sys.path.insert(0, '$SKILL_DST'); import scheduler; print('    scheduler module loads OK (', len(scheduler.NODES), 'nodes configured)')"

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
    else
        sed "s|%h/.claude/skills/scheduler|$SKILL_DST|g" "$UNIT_SRC" > "$UNIT_DST"
    fi
    systemctl --user daemon-reload
    systemctl --user enable scheduler.service >/dev/null
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
else
    echo
    echo "==> systemd not detected — skipping watcher unit"
    echo "    Run the watcher manually:  python3 $SKILL_DST/scheduler.py watch"
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
