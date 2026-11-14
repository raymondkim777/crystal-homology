
#!/usr/bin/env bash
set -euo pipefail

# Run from code/cgcnn. The ST and MT scripts are independent.
if [[ ! -f main.py || ! -d data/pretrain ]]; then
    echo "ERROR: Run this script from code/cgcnn (with data/pretrain present)." >&2
    exit 1
fi

SEEDS=(42 43 44 45 46)
# Use explicit ID labels and numeric fractions; don't derive one from the other.
KEEP_LABELS=(25 50 75 100)
KEEP_FRACTIONS=(0.25 0.5 0.75 1.0)

LOG_DIR="logs/direct_gap_availability"
mkdir -p "$LOG_DIR"

# Edit these common options if hyperparameter tuning selects a different model.
COMMON_ARGS=(
    --clear-cache --seed --delete --dims 2
    --optim Adam --scheduler adapt
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
    --epochs 200 --lr 0.001 --batch-size 32
    --loss-reg mae --val-metric error
    --vec-source both --vector landscape
    --atom-fea-len 64
    --n-conv 4
    --h-fea-len 256
    --o-fea-len 64
    --n-h 1
    --n-o 1
    --vec-fea-len 256
    --cat-fea-len 64
    --n-vec 0
)

successful_runs=()
failed_runs=()

for i in "${!KEEP_LABELS[@]}"; do
    keep_label="${KEEP_LABELS[$i]}"
    keep_fraction="${KEEP_FRACTIONS[$i]}"

    for seed in "${SEEDS[@]}"; do
        run_id="st_dg${keep_label}_s${seed}"
        log_file="$LOG_DIR/${run_id}.log"

        echo "========================================"
        echo "STARTING: $run_id"
        echo "Direct-gap labels retained: $keep_fraction"
        echo "Seed: $seed"
        echo "Log: $log_file"
        echo "========================================"

        if python main.py \
            "${COMMON_ARGS[@]}" \
            --seed-val "$seed" \
            --id "$run_id" \
            --direct-gap-keep "$keep_fraction" \
            --only-direct-gap \
            --train pretrain data/pretrain \
            2>&1 | tee "$log_file"; then
            echo "SUCCESS: $run_id"
            successful_runs+=("$run_id")
        else
            exit_code=$?
            echo "FAILED: $run_id (exit code $exit_code). See $log_file" >&2
            failed_runs+=("$run_id")
        fi
    done
done

echo
echo "========== DIRECT-GAP SWEEP SUMMARY (ST) =========="
echo "Successful: ${#successful_runs[@]}"
echo "Failed:     ${#failed_runs[@]}"
if (( ${#failed_runs[@]} > 0 )); then
    printf 'FAILED RUN: %s\n' "${failed_runs[@]}"
    exit 1
fi
echo "All runs completed successfully."

