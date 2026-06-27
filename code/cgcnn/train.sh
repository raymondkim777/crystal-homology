# python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --vec-fea-len 64 --n-vec 1 --vector image data/graph_data

# python main.py --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 1 data/graph_data

# python main.py --debug --seed --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 2 --lr 0.001 --id 1 --atom-fea-len 128 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 2 --resume model_best_1.pth.tar --start-epoch 200 data/graph_data

python main.py --debug --seed --delete --workers 4 --dims 3 --optim Adam --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --epochs 200 --lr 0.001 --id 0 --atom-fea-len 64 --n-conv 4 --h-fea-len 128 --n-h 1 --n-o 2 data/pretrain 
