# # no attr no vec step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --train pretrain data/pretrain



# # no attr vec graph img step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_gi_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_gi_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_gi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_gi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_gi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_gi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gi_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gi_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source graph --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gi_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source graph --vector image --train pretrain data/pretrain



# # no attr vec graph land step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_gl_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_gl_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_gl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_gl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_gl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_gl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gl_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gl_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source graph --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gl_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source graph --vector landscape --train pretrain data/pretrain



# # no attr vec point img step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_pi_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_pi_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_pi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_pi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_pi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_pi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pi_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pi_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pi_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source point --vector image --train pretrain data/pretrain



# # no attr vec point landscape step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_pl_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_pl_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_pl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_pl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_pl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_pl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pl_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pl_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pl_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source point --vector landscape --train pretrain data/pretrain




# # no attr vec graph perslay step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_gp_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_gp_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_gp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_gp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_gp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_gp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gp_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gp_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source graph --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_gp_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source graph --vector perslay --train pretrain data/pretrain



# # no attr vec point perslay step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_pp_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_pp_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_pp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_pp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_pp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_pp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pp_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pp_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source point --vector perslay --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_pp_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source point --vector perslay --train pretrain data/pretrain






# # no attr vec both img step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_bi_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_bi_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_bi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_bi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_bi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_bi_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bi_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bi_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector image --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bi_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source both --vector image --train pretrain data/pretrain



# # no attr vec both landscape step1

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_bl_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_bl_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_bl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_bl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_bl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_bl_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bl_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bl_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector landscape --train pretrain data/pretrain

# python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bl_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source both --vector landscape --train pretrain data/pretrain






# no attr vec both perslay step1

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p128_4_128_64_1_1_bp_256_64_1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_5_128_64_1_1_bp_256_64_1 --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_256_64_1_1_bp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_128_1_1_bp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_2_1_bp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_2_bp_256_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bp_128_64_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bp_256_128_1 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id p64_4_128_64_1_1_bp_256_64_2 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source both --vector perslay --train pretrain data/pretrain






# attr vec both perslay step1

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_1_1_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa128_4_128_64_1_1_bp_256_64_1 --attr --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_5_128_64_1_1_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 5 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_256_64_1_1_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 256 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_128_1_1_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 128 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_2_1_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 2 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_1_2_bp_256_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 2 --vec-fea-len 256 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_1_1_bp_128_64_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 128 --cat-fea-len 64 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_1_1_bp_256_128_1 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 128 --n-vec 1 --vec-source both --vector perslay --train pretrain data/pretrain

python main.py --seed --delete --dims 2 --optim Adam --scheduler adapt --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 100 --lr 0.001 --batch-size 256 --id pa64_4_128_64_1_1_bp_256_64_2 --attr --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --o-fea-len 64 --n-h 1 --n-o 1 --vec-fea-len 256 --cat-fea-len 64 --n-vec 2 --vec-source both --vector perslay --train pretrain data/pretrain

