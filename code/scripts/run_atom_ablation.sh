#!/usr/bin/env bash
set -euo pipefail

# This file belongs in code/scripts/. Run with percentages, e.g. 10 15 20.
cd "$(dirname "$0")/.."   # Now in code/

if (( $# == 0 )); then
    echo "Usage: bash scripts/run_atom_ablation.sh 10 15 20" >&2
    exit 1
fi

# Check the project folder before we delete any generated data.
[[ -f compute_diagrams.py && -d data/pretrain/cif && -d data/pretrain/graphs ]] || {
    echo "Expected code/ directory or existing CIF/graph data not found." >&2
    exit 1
}

# Check all requested run IDs before doing any preprocessing.
for pct in "$@"; do
    if ! [[ "$pct" =~ ^(0|[1-9][0-9]?|100)$ ]]; then
        echo "Invalid percentage: $pct (expected 0-100)" >&2
        exit 1
    fi
    # for seed in 42 43 44 45 46; do
    for seed in 42; do
        printf -v id 'bl_atom%02d_s%d' "$pct" "$seed"
        if [[ -e "cgcnn/out_$id" ]]; then
            echo "Existing output cgcnn/out_$id; refusing to overwrite." >&2
            exit 1
        fi
    done
done

for pct in "$@"; do
    echo "========== ATOM ENVIRONMENT: ${pct}% =========="

    # Delete only the PH products that persistent_homology.sh regenerates.
    rm -rf -- \
        data/pretrain/diagrams_c \
        data/pretrain/diagrams_p \
        data/pretrain/diagrams_g \
        data/pretrain/diagrams_atom_p \
        data/pretrain/images_c \
        data/pretrain/images_p \
        data/pretrain/images_g \
        data/pretrain/landscapes_c \
        data/pretrain/landscapes_p \
        data/pretrain/landscapes_g \
        data/pretrain/vectors_atom_p

    # Re-run your existing scripts, unmodified except for argument forwarding.
    bash -e scripts/persistent_homology.sh --atom-env-perc "$pct"
    [[ -s data/pretrain/diagrams_atom_p/cubic.pkl && \
       -s data/pretrain/vectors_atom_p/cubic.pkl ]] || {
        echo "Atom-specific preprocessing output missing; stopping." >&2
        exit 1
    }

    # Clear the old CGCNN pretraining data; cgcnn_prep.sh recreates it.
    rm -rf -- cgcnn/data/pretrain

    bash -e scripts/cgcnn_prep.sh

    # Record the current percentage
    printf 'ATOM_ENV_PERC=%s\n' "$pct" \
        > cgcnn/data/pretrain/current_experiment.txt
        
    [[ -s cgcnn/data/pretrain/vecs/vectors_atom_p.pkl ]] || {
        echo "Prepared CGCNN atom vectors missing; stopping." >&2
        exit 1
    }

    bash -e cgcnn/scripts/train_atom_seeds.sh "$pct"
done

echo "All requested percentages completed."
