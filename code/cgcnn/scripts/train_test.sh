for seed in 42; do

    python main.py --clear-cache --seed --seed-val "$seed" --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --batch-size 64 --id "mt_b64_mae_loss_s${seed}" --loss-reg mae --val-metric error --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source both --vector landscape --train pretrain data/pretrain

    python main.py --clear-cache --seed --seed-val "$seed" --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --batch-size 16 --id "mt_b16_mae_loss_s${seed}" --loss-reg mae --val-metric error --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source both --vector landscape --train pretrain data/pretrain

done