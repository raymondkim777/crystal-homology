python main2.py --debug --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og5 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s


# python main.py --debug --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 5 --h-fea-len 256 --n-h 1 --n-o 2 --vec-fea-len 64 --n-vec 1 --vector landscape --vec-source graph --train pretrain data/pretrain

# python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 32 --id 1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 3 --n-o 2 --train abs --finetune checkpoints/pretrain_64_5_128_3_2.pth.tar data/abs 

# python main.py --debug --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 2 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --n-h 3 --n-o 5 --train plqy --finetune checkpoints/abs_64_5_128_3_2.pth.tar data/plqy    # --val-metric loss