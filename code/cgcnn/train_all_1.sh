python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 3 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 150 --lr 0.001 --batch-size 256 --id 4 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --train pretrain data/pretrain




# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 01 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 02 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 03 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 04 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 05 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 06 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 07 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 08 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector landscape --freeze-vectors none --train abs --finetune-auto data/abs 




# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 3 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 4 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector landscape --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 5 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector perslay --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 6 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector image --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 7 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector landscape --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 8 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector perslay --freeze-vectors none --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy




# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 9 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector image --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 10 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector landscape --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 11 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source graph --vector perslay --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 12 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector image --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 13 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector landscape --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 14 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source point --vector perslay --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 15 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector image --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 16 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector landscape --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 17 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 0 --vec-source custom --vector perslay --freeze-vectors plqy --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy