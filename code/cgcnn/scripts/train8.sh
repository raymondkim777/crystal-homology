for seed in 42; do

    # python main.py --clear-cache --seed --seed-val "$seed" --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 32 --id "test_s${seed}" --val-metric loss --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source both --vector landscape --train pretrain data/pretrain

    python main.py --clear-cache --seed --seed-val "$seed" --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 32 --id "test_s${seed}" --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source both --vector landscape --train pretrain data/pretrain

done