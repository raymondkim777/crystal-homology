
#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# ==========================================
# Hyperparameter configurations
# ==========================================

# Each entry is one complete hyperparameter configuration.
# Add or remove entries as needed.

HP_CONFIGS=(
    "--atom-fea-len 64 \
     --n-conv 4 \
     --h-fea-len 256 \
     --o-fea-len 64 \
     --n-h 1 \
     --n-o 1 \
     --vec-fea-len 256 \
     --cat-fea-len 64 \
     --n-vec 1"

    # "--atom-fea-len 128 \
    #  --n-conv 4 \
    #  --h-fea-len 256 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 5 \
    #  --h-fea-len 256 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 128 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 2 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 2 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 512 \
    #  --cat-fea-len 64 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 128 \
    #  --n-vec 1"

    # "--atom-fea-len 64 \
    #  --n-conv 4 \
    #  --h-fea-len 128 \
    #  --o-fea-len 64 \
    #  --n-h 1 \
    #  --n-o 1 \
    #  --vec-fea-len 256 \
    #  --cat-fea-len 64 \
    #  --n-vec 2"
)

# Seeds to evaluate
SEEDS=(42)

failed_runs=()
successful_runs=()

run_config() {
    local hp_index="$1"
    local seed="$2"

    local run_id="mt_b32_mae_loss_bl_hp${hp_index}_s${seed}"
    local log_file="logs/${run_id}.log"

    # Convert hyperparameter string to Bash arguments.
    local -a hp_args
    read -r -a hp_args <<< "${HP_CONFIGS[$hp_index]}"

    echo "======================================"
    echo "STARTING: ${run_id}"
    echo "Hyperparameters: ${HP_CONFIGS[$hp_index]}"
    echo "======================================"

    if python main.py \
        --clear-cache --seed --seed-val "$seed" --delete --dims 2 \
        --optim Adam --scheduler adapt \
        --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
        --epochs 200 --lr 0.001 --batch-size 32 \
        --id "$run_id" \
        --loss-reg mae --val-metric error \
        --vec-source both --vector landscape \
        "${hp_args[@]}" \
        --train pretrain data/pretrain \
        2>&1 | tee "$log_file"
    then
        echo "SUCCESS: ${run_id}"
        successful_runs+=("$run_id")
    else
        exit_code=$?
        echo "FAILED: ${run_id} (exit code ${exit_code})"
        echo "See: ${log_file}"
        failed_runs+=("$run_id")
    fi
}

echo "======================================"
echo "Number of HP configurations: ${#HP_CONFIGS[@]}"
echo "Number of seeds: ${#SEEDS[@]}"
echo "Total training runs: $((${#HP_CONFIGS[@]} * ${#SEEDS[@]}))"
echo "======================================"

# ==========================================
# Run hyperparameter sweep
# ==========================================

for seed in "${SEEDS[@]}"; do
    for hp_index in "${!HP_CONFIGS[@]}"; do
        run_config "$hp_index" "$seed"
    done
done

# ==========================================
# Final summary
# ==========================================

echo
echo "========== SWEEP SUMMARY =========="
echo "Successful: ${#successful_runs[@]}"
echo "Failed:     ${#failed_runs[@]}"

if (( ${#failed_runs[@]} > 0 )); then
    printf 'FAILED: %s\n' "${failed_runs[@]}"
fi

echo "==================================="

if (( ${#failed_runs[@]} > 0 )); then
    exit 1
fi
