
#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Shared backbone hyperparameters
COMMON_HP=(
    --atom-fea-len 164
    --n-conv 4
    --h-fea-len 128
    --o-fea-len 64
    --n-h 1
    --n-o 1
)

# Additional hyperparameters for PH configurations
VECTOR_HP=(
    --vec-fea-len 256
    --cat-fea-len 64
    --n-vec 1
)

failed_runs=()
skipped_runs=()
successful_runs=()

run_config() {
    local source="$1"
    local vector="$2"
    local seed="$3"

    local key short_id
    local -a vec_args

    if [[ "$vector" == "none" ]]; then
        key="none"
        short_id="none"
        vec_args=(--vector none)
    else
        key="${source}_${vector}"
        short_id="${source:0:1}${vector:0:1}"

        vec_args=(
            "${VECTOR_HP[@]}"
            --vec-source "$source"
            --vector "$vector"
        )
    fi

    local run_id="mt_b32_mae_loss_${short_id}_s${seed}"
    local log_file="logs/${run_id}.log"

    echo "======================================"
    echo "STARTING: ${run_id}"
    echo "======================================"

    if python main.py \
        --clear-cache --seed --seed-val "$seed" --delete --dims 2 \
        --optim Adam --scheduler adapt \
        --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
        --epochs 200 --lr 0.001 --batch-size 32 \
        --id "$run_id" \
        --loss-reg mae --val-metric error \
        "${COMMON_HP[@]}" \
        "${vec_args[@]}" \
        --train pretrain data/pretrain \
        2>&1 | tee "$log_file"
    then
        echo "SUCCESS: $run_id"
        successful_runs+=("$run_id")
    else
        exit_code=$?
        echo "FAILED: $run_id (exit code $exit_code)"
        failed_runs+=("$run_id")
    fi
}

for seed in 42; do
    run_config none none "$seed"

    for source in graph point both; do
        for vector in image landscape perslay; do
            run_config "$source" "$vector" "$seed"
        done
    done
done

echo
echo "========== SWEEP SUMMARY =========="
echo "Successful: ${#successful_runs[@]}"
echo "Failed:     ${#failed_runs[@]}"
echo "Skipped:    ${#skipped_runs[@]}"

if (( ${#failed_runs[@]} > 0 )); then
    printf 'FAILED: %s\n' "${failed_runs[@]}"
fi

if (( ${#skipped_runs[@]} > 0 )); then
    printf 'SKIPPED: %s\n' "${skipped_runs[@]}"
fi

echo "==================================="
