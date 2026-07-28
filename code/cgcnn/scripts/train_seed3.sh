# # BL

# python main.py --delete --seed --seed-val 42 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s42 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train abs --finetune out_model_best_p128_4_128_64_1_1_pl_256_64_1.pth.tar data/abs 

# # BI

# python main.py --delete --seed --seed-val 42 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_bi_s42 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train abs --finetune model_best_p_bi_s42.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 43 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_bi_s43 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train abs --finetune model_best_p_bi_s43.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 44 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_bi_s44 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train abs --finetune model_best_p_bi_s44.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 45 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_bi_s45 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train abs --finetune model_best_p_bi_s45.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 46 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_bi_s46 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train abs --finetune model_best_p_bi_s46.pth.tar --fold 5 data/abs 

# # PL

# python main.py --delete --seed --seed-val 42 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s42 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p_pl_s42.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 43 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s43 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p_pl_s43.pth.tar --fold 5 data/abs 

# redo this one 
python main.py --delete --seed --seed-val 44 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s44 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p_pl_s44.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 45 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s45 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p_pl_s45.pth.tar --fold 5 data/abs 

# python main.py --delete --seed --seed-val 46 --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a_pl_s46 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p_pl_s46.pth.tar --fold 5 data/abs 