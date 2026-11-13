# SINGLE TASK

python main.py \
    --clear-cache --seed --seed-val 42 --delete --dims 2 \
    --optim Adam --scheduler adapt \
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
    --epochs 2 --lr 0.001 --batch-size 32 \
    --id st_dg10_s42 \
    --loss-reg mae --val-metric error \
    --vec-source both --vector landscape \
    --atom-fea-len 64 \
    --n-conv 4 \
    --h-fea-len 128 \
    --o-fea-len 64 \
    --n-h 1 \
    --n-o 1 \
    --vec-fea-len 256 \
    --cat-fea-len 64 \
    --n-vec 0 \
    --direct-gap-keep 1.0 \
    --only-direct-gap \
    --train pretrain data/pretrain


# MULTI TASK

python main.py \
    --clear-cache --seed --seed-val 42 --delete --dims 2 \
    --optim Adam --scheduler adapt \
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 \
    --epochs 2 --lr 0.001 --batch-size 32 \
    --id mt_dg10_s42 \
    --loss-reg mae --val-metric error \
    --vec-source both --vector landscape \
    --atom-fea-len 64 \
    --n-conv 4 \
    --h-fea-len 128 \
    --o-fea-len 64 \
    --n-h 1 \
    --n-o 1 \
    --vec-fea-len 256 \
    --cat-fea-len 64 \
    --n-vec 0 \
    --direct-gap-keep 1.0 \
    --train pretrain data/pretrain
