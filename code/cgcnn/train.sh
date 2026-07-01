python main.py --debug --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --id 1 --train pretrain --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 3 --n-o 2 data/pretrain


# python main.py --debug --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --n-h 1 --n-o 2 --vec-fea-len 64 --n-vec 1 --vector image data/pretrain