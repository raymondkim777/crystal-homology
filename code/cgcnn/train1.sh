python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 2 --n-o 2 --vec-fea-len 64 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain

python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 2 --atom-fea-len 64 --n-conv 5 --h-fea-len 256 --n-h 1 --n-o 2 --vec-fea-len 64 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain

python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 4 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 1 --n-o 3 --vec-fea-len 64 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain

python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 6 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 1 --n-o 2 --vec-fea-len 128 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain

python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 8 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 1 --n-o 2 --vec-fea-len 64 --n-vec 2 --vector image --vec-source graph --train pretrain data/pretrain

# python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 3 --n-o 3 --train abs --finetune checkpoints/pretrain_64_5_128_3_2.pth.tar data/abs 

# python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 3 --n-o 5 --train plqy --fold 5 --finetune checkpoints/abs_64_5_128_3_2.pth.tar data/plqy    # --val-metric loss