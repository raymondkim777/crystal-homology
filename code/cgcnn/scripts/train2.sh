python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id gate_2 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id gate_3 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 4 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 5 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 2 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 6 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 2 --train pretrain data/pretrain






# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 3 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 2 --vector perslay --vec-source graph --train abs --freeze-vectors both --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 4 --id 5 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 2 --vector perslay --vec-source graph --train plqy --freeze-vectors both --finetune-auto --fold 5 data/plqy    # --val-metric loss