# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 100 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id 101 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune-auto data/abs 

python main.py --delete --seed --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 1000 --lr 0.0005 --batch-size 4 --id 105 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --finetune-auto --fold 5 data/plqy

# python main.py --delete --seed --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 1000 --lr 0.0005 --batch-size 4 --id 104 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --finetune-auto --fold 5 data/plqy

