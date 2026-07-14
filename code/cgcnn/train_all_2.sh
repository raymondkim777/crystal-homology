python main.py --seed --delete --dims 2 --optim Adam --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 5 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 6 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 7 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector image --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 8 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector landscape --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --wd 0.0001 --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id 9 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector perslay --train pretrain data/pretrain




# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 09 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector landscape --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 10 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector perslay --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 11 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector perslay --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 12 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector image --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 13 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector image --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 14 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector landscape --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 15 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector landscape --freeze-vectors abs --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 16 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector perslay --freeze-vectors none --train abs --finetune-auto data/abs 

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --batch-size 64 --id 17 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector perslay --freeze-vectors abs --train abs --finetune-auto data/abs 




# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 18 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector image --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 19 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 20 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 21 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector image --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 22 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector landscape --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 23 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector perslay --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 24 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector image --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 25 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector landscape --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 26 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector perslay --freeze-vectors abs --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy



# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 27 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector image --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 28 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 29 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 30 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector image --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 31 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector landscape --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 32 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source point --vector perslay --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 33 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector image --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 34 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector landscape --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy

# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0002 --batch-size 4 --id 35 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 64 --n-vec 1 --vec-source custom --vector perslay --freeze-vectors both --train plqy --finetune-auto --val-metric loss --fold 5 data/plqy