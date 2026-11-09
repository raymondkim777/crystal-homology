for seed in 42 43 44 45 46; do

    echo "========== Norm bounds for seed ${seed} =========="

    printf -v id 'norm_s%d' "$seed"

    python main.py --print-norm --clear-cache --seed --seed-val "$seed" --delete --dims 2 --optim Adam --scheduler adapt --attr --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --norm-sample 0 --id "$id" --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

done