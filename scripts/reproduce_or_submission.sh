#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

printf '%s\n' 'Recomputing the registered replay, current summaries, cross-domain aggregation, and result figure from local frozen inputs.'
make reproduce-current-results
printf '%s\n' 'Current result generation completed. Building the main paper and electronic companion.'
make -C paper
printf '%s\n' 'Current results and paper build completed. Lean and historical evidence have separate verification commands in paper/README.md.'
