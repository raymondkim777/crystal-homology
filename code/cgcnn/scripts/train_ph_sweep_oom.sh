
#!/usr/bin/env bash
set -euo pipefail

#! CHECK FOR OOM ERROR

run_with_oom_guard() {
    local log_file="$1"
    shift

    local tmp_dir fifo pid line lower
    local status=0
    local oom=0

    tmp_dir=$(mktemp -d)
    fifo="$tmp_dir/output"
    mkfifo "$fifo"

    : > "$log_file"

    # Start training in its own process group.
    # This lets us terminate DataLoader workers too.
    setsid "$@" > "$fifo" 2>&1 &
    pid=$!

    # Read and log output while monitoring for OOM.
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "$line"
        printf '%s\n' "$line" >> "$log_file"

        lower=${line,,}

        if [[ "$lower" == *"out of memory"* ||
              "$lower" == *"outofmemoryerror"* ||
              "$lower" == *"cublas_status_alloc_failed"* ||
              "$lower" == *"cudnn_status_alloc_failed"* ]]; then

            oom=1
            echo "OOM DETECTED: terminating run (PID $pid)" \
                | tee -a "$log_file"

            kill -TERM -- "-$pid" 2>/dev/null || true
            break
        fi
    done < "$fifo"

    if wait "$pid"; then
        status=0
    else
        status=$?
    fi

    rm -r "$tmp_dir"

    if (( oom )); then
        return 42
    fi

    return "$status"
}

mkdir -p logs

# PH Configuration-specific hyperparameters.
declare -A CONFIG_HP=(
    [none]="--atom-fea-len 128 \
        --n-conv 4 \
        --h-fea-len 128 \
        --o-fea-len 64 \
        --n-h 1 \
        --n-o 1"

    [graph_image]="--atom-fea-len 64 \
        --n-conv 4 \
        --h-fea-len 128 \
        --o-fea-len 64 \
        --n-h 1 \
        --n-o 1 \
        --vec-fea-len 256 \
        --cat-fea-len 128 \
        --n-vec 1"
)

failed_runs=()
skipped_runs=()
successful_runs=()

run_config() {
    local source="$1"
    local vector="$2"
    local seed="$3"
    local key

    if [[ "$vector" == "none" ]]; then
        key="none"
        short_id="none"
    else
        key="${source}_${vector}"
        short_id="${source:0:1}${vector:0:1}"
    fi

    local run_id="mt_b32_mae_loss_${short_id}_s${seed}"
    local log_file="logs/${run_id}.log"

    # Do not silently use the wrong hyperparameters.
    if [[ -z "${CONFIG_HP[$key]+defined}" ]]; then
        echo "SKIPPED: ${run_id} (hyperparameters not configured)"
        skipped_runs+=("$run_id")
        return 0
    fi

    # Convert configuration string into Bash arguments.
    local -a hp_args
    read -r -a hp_args <<< "${CONFIG_HP[$key]}"

    # PH-related arguments
    local -a vec_args

    if [[ "$vector" == "none" ]]; then
        vec_args=(--vector none)
    else
        vec_args=(--vec-source "$source" --vector "$vector")
    fi

    echo "======================================"
    echo "STARTING: ${run_id}"
    echo "======================================"

    # if python main.py \
    #     --clear-cache --seed --seed-val "$seed" --delete --dims 2 \
    #     --optim Adam --scheduler adapt \
    #     --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
    #     --epochs 200 --lr 0.001 --batch-size 32 \
    #     --id "$run_id" \
    #     --loss-reg mae --val-metric error \
    #     "${hp_args[@]}" \
    #     "${vec_args[@]}" \
    #     --train pretrain data/pretrain \
    #     2>&1 | tee "$log_file"
    # then
    #     echo "SUCCESS: ${run_id}"
    #     successful_runs+=("$run_id")
    # else
    #     exit_code=$?
    #     echo "FAILED: ${run_id} (exit code ${exit_code})"
    #     echo "See: ${log_file}"
    #     failed_runs+=("$run_id")
    # fi


    if run_with_oom_guard "$log_file" \
        python -u main.py \
            --clear-cache --seed --seed-val "$seed" --delete --dims 2 \
            --optim Adam --scheduler adapt \
            --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
            --epochs 200 --lr 0.001 --batch-size 32 \
            --id "$run_id" \
            --loss-reg mae --val-metric error \
            "${hp_args[@]}" \
            "${vec_args[@]}" \
            --train pretrain data/pretrain
    then
        echo "SUCCESS: ${run_id}"
        successful_runs+=("$run_id")
    else
        exit_code=$?

        if (( exit_code == 42 )); then
            echo "FAILED (OOM): ${run_id}"
        else
            echo "FAILED: ${run_id} (exit code ${exit_code})"
        fi

        echo "See: ${log_file}"
        failed_runs+=("$run_id")
    fi

}

# 1. Run no-PH baseline first
for seed in 42; do
    run_config none none "$seed"

    # 2. Run all PH combinations
    for source in graph point both; do
        for vector in image landscape perslay; do
            run_config "$source" "$vector" "$seed"
        done
    done
done

# 3. Final summary
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
