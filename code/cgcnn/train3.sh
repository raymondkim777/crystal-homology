python main2.py --debug --seed --seed-val 42 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og_s42 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

python main2.py --debug --seed --seed-val 43 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og_s43 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

python main2.py --debug --seed --seed-val 44 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og_s44 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

python main2.py --debug --seed --seed-val 45 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og_s45 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s

python main2.py --debug --seed --seed-val 46 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id og_s46 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain_s




# python main2.py --seed --seed-val 42 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --print-freq 3 --batch-size 64 --id og_abs_s42 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune out_og_s42/model_best_og_s42.pth.tar data/abs_s

# python main2.py --seed --seed-val 43 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --print-freq 3 --batch-size 64 --id og_abs_s43 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune out_og_s42/model_best_og_s42.pth.tar data/abs_s

# python main2.py --seed --seed-val 44 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --print-freq 3 --batch-size 64 --id og_abs_s44 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune out_og_s42/model_best_og_s42.pth.tar data/abs_s

# python main2.py --seed --seed-val 45 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --print-freq 3 --batch-size 64 --id og_abs_s45 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune out_og_s42/model_best_og_s42.pth.tar data/abs_s

# python main2.py --seed --seed-val 46 --delete --optim SGD --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --print-freq 3 --batch-size 64 --id og_abs_s46 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train abs --finetune out_og_s42/model_best_og_s42.pth.tar data/abs_s



# python main.py --seed --delete --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 64 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 2 --train abs --finetune checkpoints/pretrain_64_4_128_1_1_default.pth.tar data/abs_s

# python main.py --seed --delete --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.0005 --print-freq 3 --batch-size 4 --id 1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 --train plqy --fold 5 --finetune checkpoints/abs_64_4_128_1_2_default.pth.tar data/plqy_s    # --val-metric loss 