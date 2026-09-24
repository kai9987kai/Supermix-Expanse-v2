#!/usr/bin/env bash
# Unattended Connectome Lab sweep: every pre-registered run (3 seeds), verdicts, then the exploratory arms.
#   bash expanse/cns_lab/run_sweep.sh [threads]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
THREADS="${1:-8}"
python "$HERE/run_all.py" --hypotheses H-CNS-001,H-CNS-002,H-CNS-003 --threads "$THREADS"
python "$HERE/run_all.py" --hypotheses X-CNS-001 --exploratory --threads "$THREADS"
echo "[sweep] done"
