# python main.py --seed --delete --dims 2 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train pretrain data/pretrain


# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a128_4_128_64_1_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train abs --finetune model_best_p128_4_128_64_1_1.pth.tar data/abs 

# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a64_4_128_64_1_1_gi_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source graph --vector image --train abs --finetune model_best_p64_4_128_64_1_1_gi_256_128_1.pth.tar data/abs 

# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a64_4_128_64_1_1_gl_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source graph --vector landscape --train abs --finetune model_best_p64_4_128_64_1_1_gl_256_128_1.pth.tar data/abs 




# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id aa128_4_128_64_1_1 --attr --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train abs --finetune model_best_pa128_4_128_64_1_1.pth.tar data/abs 

# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id aa64_4_256_64_1_1_gi_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train abs --finetune model_best_pa64_4_256_64_1_1_gi_256_64_1.pth.tar data/abs 

# python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id aa64_4_256_64_1_1_gl_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train abs --finetune model_best_pa64_4_256_64_1_1_gl_256_64_1.pth.tar data/abs 



python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a128_4_128_64_1_1_pi_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train abs --finetune model_best_p128_4_128_64_1_1_pi_256_64_1.pth.tar data/abs 

python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id a128_4_128_64_1_1_pl_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_p128_4_128_64_1_1_pl_256_64_1.pth.tar data/abs 


python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id aa64_4_128_64_1_1_pi_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train abs --finetune model_best_pa64_4_128_64_1_1_pi_256_64_1.pth.tar data/abs 

python main.py --delete --seed --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.0005 --batch-size 64 --id aa64_4_256_64_1_1_pl_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train abs --finetune model_best_pa64_4_256_64_1_1_pl_256_64_1.pth.tar data/abs 