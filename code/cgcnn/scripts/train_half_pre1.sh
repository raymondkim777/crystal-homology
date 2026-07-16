# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id pre_0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id pre_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id pre_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id pre_3 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id pre_4 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --train pretrain data/pretrain
