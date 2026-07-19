python main1.py --debug --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og4 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

# python main.py --debug --seed --delete --workers 1 --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vector image --vec-source graph --train pretrain data/pretrain_s

# python main.py --debug --seed --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --resume ./checkpoint_0.pth.tar --train pretrain data/pretrain_s

# python main.py --seed --delete --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 2 --train abs --finetune checkpoints/pretrain_64_4_128_1_1_default.pth.tar data/abs_s

# python main.py --seed --delete --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --fold 5 --finetune checkpoints/abs_64_4_128_1_2_default.pth.tar data/plqy_s    # --val-metric loss