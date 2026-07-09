# python main_og.py --debug --seed --delete --clear-cache --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

python main_og.py --debug --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain --resume checkpoints/pre_64_4_128_1_1_default.pth.tar --start-epoch 100 data/pretrain_s

# python main_og.py --debug --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain_s

# python main_og.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune checkpoints/pre_64_4_128_1_1_default.pth.tar data/abs_s

# python main_og.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --fold 5 --finetune checkpoints/abs_64_4_128_1_1_default.pth.tar data/plqy_s    # --val-metric loss