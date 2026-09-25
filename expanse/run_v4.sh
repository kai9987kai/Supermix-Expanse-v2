#!/usr/bin/env bash
# v4 = v3 + input-dependent gate on the male-CNS core (H-CNS-004 SUPPORTED), its matched static-gate
# control, then the standard v1/v3/v4 comparison on v1's dev rows.
#   bash expanse/run_v4.sh [threads]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
T="${1:-8}"
python "$HERE/build_v4.py" --gate dynamic --steps 600 --eval_every 200 --threads "$T" --tag v4 || exit 1
python "$HERE/build_v4.py" --gate static --steps 600 --eval_every 200 --threads "$T" --no_save --tag v4_control || exit 1
python "$HERE/compare_models.py" --model v1="$HERE/checkpoints/supermix_expanse.pt" \
    --model v3="$HERE/checkpoints/supermix_expanse_v3.pt" --model v4="$HERE/checkpoints/supermix_expanse_v4.pt" \
    --gen_limit 25 --threads "$T" --out "$HERE/checkpoints/compare_v1_v3_v4.json" || exit 1
echo "[v4] all done"
