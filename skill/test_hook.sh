#!/bin/bash
# PostToolUse hook: when scheduler.py is modified, run the regression suite.
# Reads hook JSON from stdin, writes hook JSON to stdout per Claude Code spec.
# Tests pass → emit a quiet systemMessage.
# Tests fail → emit decision:"block" + the test output as additionalContext
#              so the assistant sees exactly which checks failed.

set -uo pipefail

PAYLOAD=$(cat)
FILE=$(printf '%s' "$PAYLOAD" | jq -r '.tool_input.file_path // ""')
TARGET="/home/erzhu419/.claude/skills/scheduler/scheduler.py"

# Match against the resolved path so a symlink or relative-style input still triggers.
RESOLVED=$(realpath -m "$FILE" 2>/dev/null || printf '%s' "$FILE")
if [ "$RESOLVED" != "$TARGET" ]; then
    exit 0
fi

TEST_PY="/home/erzhu419/.claude/skills/scheduler/test_regression.py"
OUT=$(python "$TEST_PY" 2>&1)
RC=$?

if [ "$RC" -eq 0 ]; then
    # All checks passed — keep the systemMessage one-line so it doesn't spam transcript.
    SUMMARY=$(printf '%s\n' "$OUT" | grep -E '^[0-9]+/[0-9]+ checks passed' | tail -1)
    jq -nc --arg msg "scheduler regression: ${SUMMARY:-passed}" \
        '{systemMessage: $msg, suppressOutput: true}'
else
    # Block the turn so the assistant cannot ignore a regression.
    jq -nc --arg out "$OUT" \
        '{decision: "block",
          reason: "scheduler regression test FAILED — fix before continuing",
          hookSpecificOutput: {hookEventName: "PostToolUse", additionalContext: $out}}'
fi
