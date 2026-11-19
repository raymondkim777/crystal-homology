#!/usr/bin/env bash
set -euo pipefail

# This file belongs in code/cgcnn/scripts/.
cd "$(dirname "$0")/.."   # Now in code/cgcnn/

# Shared backbone hyperparameters
# Shared backbone hyperparameters
HP_CONFIG=(
    --atom-fea-len 64
    --n-conv 4
    --h-fea-len 256
    --o-fea-len 64
    --n-h 1
    --n-o 1
    --vec-fea-len 256
    --cat-fea-len 64
    --n-vec 0
    --vec-source both
    --vector image
)

pct="${1:?Usage: bash scripts/train_atom_seeds.sh 10}"
if ! [[ "$pct" =~ ^(0|[1-9][0-9]?|100)$ ]]; then
    echo "Invalid percentage: $pct (expected 0-100)" >&2
    exit 1
fi

for seed in 42 43 44 45 46; do
    printf -v id 'bi_atom%02d_s%d' "$pct" "$seed"
    if [[ -e "out_$id" ]]; then
        echo "Existing output out_$id; refusing to overwrite." >&2
        exit 1
    fi

    echo "========== Training ${pct}%, seed ${seed} =========="
    printf 'ATOM_ENV_PERC=%s\nSEED=%s\n' "$pct" "$seed" \
        > data/pretrain/current_experiment.txt
    # Your active train_atom.sh command, with variable seed and unique ID.
    # No --delete: this script never intentionally erases a training result.
    python main.py \
        --clear-cache --seed --seed-val "$seed" \
        --dims 2 --optim Adam --scheduler adapt --attr \
        --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
        --epochs 200 --lr 0.001 --batch-size 32 \
        --id "$id" \
        --loss-reg mae --val-metric error \
        "${HP_CONFIG[@]}" \
        --train pretrain data/pretrain

    [[ -s "out_$id/test_stats_$id.csv" ]] || {
        echo "Missing test statistics for $id; stopping." >&2
        exit 1
    }
done

echo "All five seeds finished for ${pct}%."
