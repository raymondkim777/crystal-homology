# python main.py --debug --seed --delete --clear-cache --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune checkpoints/pre_64_4_128_1_1.pth.tar data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --fold 5 --finetune checkpoints/abs_64_4_128_1_1.pth.tar data/plqy    # --val-metric loss




# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vector landscape --vec-source custom --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vector landscape --vec-source custom --train abs --finetune checkpoints/pre_64_4_128_1_1_64_1_landc.pth.tar data/abs 

python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vector landscape --vec-source custom --train plqy --fold 5 --finetune checkpoints/abs_64_4_128_1_1_64_1_landc.pth.tar data/plqy    # --val-metric loss