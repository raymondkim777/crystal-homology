import argparse
import os
import pickle
import shutil
import sys
import time
import csv
import warnings
from random import sample
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import Subset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.loss import MultiTaskLoss
from cgcnn.data import GraphData
from cgcnn.data import collate_pool, get_train_val_test_loader, \
    get_fold_indices, get_train_val_test_loader_from_folds
from cgcnn.model import CrystalGraphConvNet, \
    freeze_lower_encoder, set_frozen_encoder_parts_to_eval

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
# parser.add_argument('--task', choices=['regression', 'classification'],
#                     default='regression', help='complete a regression or '
#                                                    'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
parser.add_argument('--n-o', default=1, type=int, metavar='N',
                    help='number of hidden layers in head MLP')
parser.add_argument('--vec-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden vector features per dimension in hidden layers')
parser.add_argument('--n-vec', default=1, type=int, metavar='N',
                    help='number of hidden vector processing layers')


parser.add_argument('--debug', action='store_true',
                    help='prints debug messages')
parser.add_argument('--seed', action='store_true',
                    help='sets torch seed to 42')
# parser.add_argument('--num-classes', default=2, type=int)
parser.add_argument('--delete', action='store_true',
                    help='deletes generated training files before training')
parser.add_argument('--id', default=0, type=int, metavar='N',
                    help='identifier for multiple checkpoint/models')
parser.add_argument('--dims', default=3, type=int,
                    help='number of persistence homology dimensions')
parser.add_argument('--norm-sample', default=2000, type=int,
                    help='number of max samples to use to define normalizers')


parser.add_argument('--fold', default=0, type=int, 
                    help='use k-fold cross validation for val/test')
parser.add_argument('--vec-source', default='graph', type=str,
                    help='choose a vectorization source: graph, point')
parser.add_argument('--vector', default='none', type=str,
                    help='choose a vectorization: none, image, landscape, perslay')
parser.add_argument('--weight', default='none', type=str,
                    help='choose a weight function: none, power, grid, gaussian')
parser.add_argument('--phi', default='none', type=str,
                    help='choose a transformation function: none, image, landscape, betti')
parser.add_argument('--train', default='pretrain', type=str, 
                    help='choose training type: pretrain, abs, plqy')
parser.add_argument('--freeze-vectors', action='store_true', 
                    help='freezes vectorization MLP for abs/plqy fine-tuning')


parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to pretrain checkpoint')
parser.add_argument('--val-metric', default='error', type=str, metavar='PATH',
                    help="validation metric: ['error', 'loss']")


args = parser.parse_args(sys.argv[1:])

# ! setting torch device
args.cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# read in head tasks
task_filepath = os.path.join(args.data_options[0], 'tasks', 'tasks.pkl')
assert os.path.exists(task_filepath), 'Tasks file (tasks.pkl) does not exist!'
with open(task_filepath, 'rb') as f:
    TASK_SPECS = pickle.load(f)

CROSS_VAL = args.fold != 0


def main():
    global args, best_errors

    if args.seed:
        torch.manual_seed(42)
    print("GPU Available", args.cuda)

    if args.delete and args.resume == '':
        file_names = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]

        check_path = f"./checkpoint_{args.id}.pth.tar"
        # model_path = f"./model_best_{args.id}.pth.tar"
        model_paths = [
            f"./{f}" for f in file_names 
            if f.startswith(f"model_best_{args.id}")
            and f.endswith(".pth.tar")
        ]
        # f"model_best_{args.id}_fold_{fold_it}.pth.tar"
        stat_path = f"./test_stats_{args.id}.csv"
        param_paths = f"./test_params_{args.id}.txt"
        result_paths = [
            f"./{f}" for f in file_names
            if f.startswith(f"test_results_{args.id}")
            and f.endswith(".csv")
        ]
        # result_paths = [f"./test_results_{args.id}_{prop}.csv" for prop in TASK_SPECS.keys()]
        
        if os.path.exists(check_path):
            os.remove(check_path)
        for model_path in model_paths:
            if os.path.exists(model_path):
                os.remove(model_path)
        if os.path.exists(stat_path):
            os.remove(stat_path)
        if os.path.exists(param_paths):
            os.remove(param_paths)
        for result_path in result_paths:
            if os.path.exists(result_path):
                os.remove(result_path)
    
    assert args.train in ['pretrain', 'abs', 'plqy'],\
        "Wrong train argument! Should be one of 'pretrain', 'abs', 'plqy'"
    assert (args.train == 'pretrain' and args.finetune == '')or (args.train != 'pretrain' and args.finetune != ''), \
        "No finetune arg if train=pretrain, and need finetune arg if train=abs/plqy"
    assert args.resume == '' or args.finetune == '', "Choose one of resume or finetune!"
    assert args.val_metric in ['error', 'loss']
    assert args.vec_source in ['graph', 'point']
    assert not CROSS_VAL or args.train != 'pretrain',\
        "[STOP] Should not run k-fold cross validation on pretrain dataset!"

    # load data
    if args.debug:
        print("Constructing GraphData")
    dataset = GraphData(
        *args.data_options, 
        vec_source=args.vec_source,
        vector=args.vector,
        dims=args.dims,
        task_specs=TASK_SPECS
    )
    collate_fn = collate_pool

    if args.fold == 0:
        # regular training
        if args.debug:
            print("Constructing train/val/test loaders")
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            return_test=True,
            # ! below causes constantly increasing memory
            # persistent_workers=True
        )
    else:
        if args.debug:
            print(f"Constructing {args.fold} folds")
        # train/val/test loaders are constructed within fold loop
        folds = get_fold_indices(
            dataset=dataset, 
            k=args.fold,
        )
    
    # ! ADJUST FOR FOLD ITERATIONS (should be one pass if fold = 0)
    # average results for all folds (or same as val results for fold = 0)
    best_epochs = []
    total_losses = AverageMeter()
    total_errors = AverageMeter()
    total_stats = dict()
    for prop, value in TASK_SPECS.items():
        total_stats[prop] = dict()
        total_stats[prop]['loss'] = AverageMeter()
        if value['head'] in ['binary', 'multiclass']:
            total_stats[prop]['accuracies'] = AverageMeter()
            total_stats[prop]['precisions'] = AverageMeter()
            total_stats[prop]['recalls'] = AverageMeter()
            total_stats[prop]['fscores'] = AverageMeter()
            total_stats[prop]['auc_scores'] = AverageMeter()
        elif value['head'] == 'regression':
            total_stats[prop]['nrmse_errors'] = AverageMeter()
        else:
            raise ValueError(f"[TRAIN STATS] Unknown task {value['head']}")

    # k-fold cross validation
    fold_num = args.fold if CROSS_VAL else 1
    for fold_it in range(fold_num):

        best_errors = float('inf')

        if CROSS_VAL:
            print(f"\n--------FOLD {fold_it + 1}--------")
            if args.debug:
                print(f"Constructing train/val/test loaders for {fold_num} folds")
            train_loader, val_loader, test_loader = get_train_val_test_loader_from_folds(
                dataset=dataset,
                folds=folds, 
                test_fold_idx=fold_it,
                collate_fn=collate_fn,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=args.cuda,
            )
        
        if args.debug:
            print("Normalizer train sample size 2000")    
        train_indices = list(train_loader.sampler)
        sample_cnt = min(len(train_indices), args.norm_sample)
        sample_indices = sample(train_indices, k=sample_cnt)
        sample_data_list = [dataset[i] for i in tqdm(sample_indices, desc="Normalizers: ")]
        _, _, _, sample_target, sample_mask, _ = collate_pool(sample_data_list)

        normalizers = dict()
        for prop, value in TASK_SPECS.items():
            if value['head'] in ['binary', 'multiclass']:
                normalizer = NormalizerProp(torch.zeros(2))
                normalizer.load_state_dict({'mean': 0., 'std': 1.})
            elif value['head'] in ['regression']:
                normalizer = NormalizerProp(sample_target[prop], mask=sample_mask[prop])
            else:
                raise ValueError(f"[Normalizer] Unknown task {value['head']}")
            normalizers[prop] = normalizer

        # build model
        if args.debug:
            print("Instantiating model")
        structures, _, _, _, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(
            orig_atom_fea_len, nbr_fea_len,
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            h_fea_len=args.h_fea_len,
            n_h=args.n_h,
            # ! vector layer arguments
            vec_fea_len=args.vec_fea_len, 
            n_vec=args.n_vec,
            n_o=args.n_o,
            # ! vectorization arguments
            vec_source=args.vec_source,
            vector=args.vector,
            weight=args.weight,
            phi=args.phi,
            dims=args.dims,
            # ! miscellaneous arguments
            root_dir=args.data_options,
            task_specs=TASK_SPECS,
        )

        # ! if fine-tune, freeze lower encoder layers
        if args.train in ['abs', 'plqy']:
            assert args.finetune != ''

            # load checkpoint encoder weights
            if not os.path.isfile(args.finetune):
                raise ValueError(f"=> no pretrained checkpoint found at '{args.finetune}'")
            print(f"=> Loading pretrained checkpoint '{args.finetune}'")

            # load checkpoint from file
            checkpoint = torch.load(args.finetune, weights_only=True)     # may set weights to Falseff

            # isolate encoder state dict
            if not any(k.startswith("encoder.") for k in checkpoint['state_dict']):
                raise ValueError(f"Pretrained checkpoint model contains no encoder state_dict")
            encoder_state_dict = {
                k[len("encoder."):]: v
                for k, v in checkpoint['state_dict'].items()
                if k.startswith("encoder.")
            }
            if args.debug:
                print("Encoder state dict KEYS:")
                print(encoder_state_dict.keys())

            # load encoder with checkpoint weights
            model.encoder.load_state_dict(
                encoder_state_dict, 
                strict=True
            )
            print(f"=> loaded checkpoint '{args.finetune}' encoder")
            freeze_lower_encoder(model=model, freeze_vectors=args.freeze_vectors)

        # ! updated tensor cuda code
        if args.debug:
            print(f"Moving model to {device}")
        model = model.to(device)

        # ! moving torchPerslay inner params to cuda
        if args.vector == 'perslay':
            for perslay in model.perslays:
                perslay.phi.mu = perslay.phi.mu.to(device)
                perslay.phi.M = tuple(m.to(device) for m in perslay.phi.M)

        # define loss func and optimizer
        # should move loss function outside of fold loop, but whatever
        # ! CUSTOM LOSS FUNCTION 
        if args.debug:
            print("Instantiating custom loss function")
        criterion = MultiTaskLoss(device)
                
        if args.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            optimizer = optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        else:
            raise NameError('Only SGD or Adam is allowed as --optim')

        # optionally resume from a checkpoint
        if not CROSS_VAL and args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                # ! TRUSTED SOURCE
                checkpoint = torch.load(args.resume, weights_only=False)
                args.start_epoch = checkpoint['epoch']
                best_errors = checkpoint['best_errors']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                # normalizer.load_state_dict(checkpoint['normalizer'])
                for prop in normalizers.keys():
                    normalizers[prop].load_state_dict(checkpoint['normalizer'][prop])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                                gamma=0.1)

        if args.debug:
            print("Starting training epochs")
        for epoch in tqdm(range(args.start_epoch, args.epochs), desc='Epochs: ', disable=args.debug):
            # train for one epoch
            if args.debug:
                print(f"Training epoch {epoch}")
            train(train_loader, model, criterion, optimizer, epoch, normalizers)

            # evaluate on validation set
            if args.debug:
                print("Validating")
            val_error, val_loss, _ = validate(val_loader, model, criterion, normalizers)

            if args.val_metric == 'error':
                cur_errors = val_error
            else:
                cur_errors = val_loss

            if cur_errors != cur_errors:
                print('Exit due to NaN')
                sys.exit(1)

            scheduler.step()

            # remember the best error and save checkpoint
            is_best = cur_errors < best_errors and epoch > 10
            best_errors = min(cur_errors, best_errors) if epoch > 10 else best_errors
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_errors': best_errors,
                'optimizer': optimizer.state_dict(),
                'normalizer': {prop: normalizers[prop].state_dict()
                            for prop in normalizers.keys()},
                'args': vars(args)
            }, is_best, fold_it=fold_it)

        # test best model
        if args.debug:
            print('---------------Evaluate Model on Test Set---------------')
        # ! PyTorch 2.6 safety measure: only load model weights -> extra argument
        
        if CROSS_VAL:
            best_checkpoint_path = f"model_best_{args.id}_fold_{fold_it}.pth.tar"
        else:
            best_checkpoint_path = f'model_best_{args.id}.pth.tar'

        best_checkpoint = torch.load(best_checkpoint_path, weights_only=False)
        best_epochs.append(best_checkpoint['epoch'])
        model.load_state_dict(best_checkpoint['state_dict'])

        test_loss, test_error, stat_dict = validate(
            test_loader, model, criterion, normalizers, 
            best_epoch=best_checkpoint['epoch'], 
            test=True
        )
        # saving validation stats to total stats
        total_losses.update(test_loss)
        total_errors.update(test_error)
        for prop, value in TASK_SPECS.items():
            for stat, avg_meter in stat_dict[prop].items():
                total_stats[prop][stat].update(avg_meter.avg)
    
    # ! save results
    save_results(
        best_epochs=best_epochs, 
        total_loss=total_losses.avg, 
        total_error=total_errors.avg,
        stats_dict=total_stats
    )


def train(train_loader, model, criterion, optimizer, epoch, normalizers):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # ! stat trackers for each metric per head
    losses = AverageMeter()     # total batch loss (from loss function)
    stats = dict()
    for prop, value in TASK_SPECS.items():
        stats[prop] = dict()
        stats[prop]['loss'] = AverageMeter()    # loss for each property (intermediate from loss function)
        if value['head'] in ['binary', 'multiclass']:
            stats[prop]['accuracies'] = AverageMeter()
            stats[prop]['precisions'] = AverageMeter()
            stats[prop]['recalls'] = AverageMeter()
            stats[prop]['fscores'] = AverageMeter()
            stats[prop]['auc_scores'] = AverageMeter()
        elif value['head'] == 'regression':
            stats[prop]['nrmse_errors'] = AverageMeter()
        else:
            raise ValueError(f"[TRAIN STATS] Unknown task {value['head']}")

    # switch to train mode
    model.train()

    # ! set frozen layers to eval (undo model.train() for those layers)
    if args.train in ['abs', 'plqy']:
        set_frozen_encoder_parts_to_eval(model)

    end = time.time()
    # for i, (input, target, _) in enumerate(train_loader):
    for i, (
        (atom_fea, nbr_fea, nbr_fea_idx, crys_idx), 
        vectorizations, diagrams, targets, mask, _
    ) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # ! EDTIED to use torch tensors
        atom_fea = atom_fea.to(device, non_blocking=True)
        nbr_fea = nbr_fea.to(device, non_blocking=True)
        nbr_fea_idx = nbr_fea_idx.to(device, non_blocking=True)

        crys_idx = [idx.to(device, non_blocking=True) for idx in crys_idx]

        vectorizations = vectorizations.to(device, non_blocking=True)
        diagrams = [
            diagram.to(device, non_blocking=True)
            for diagram in diagrams
        ]

        # ! updated variables to use torch Tensors & receive additional data
        input_var = (
            atom_fea, 
            nbr_fea, 
            nbr_fea_idx, 
            crys_idx,
            vectorizations, 
            diagrams
        )
    
        # normalize target
        # ! target normalization applied to each property
        targets_normed = {
            prop: (normalizers[prop].norm(targets[prop])).to(device, non_blocking=True)
            for prop in targets.keys()
        }

        # compute output
        output = model(*input_var)  # dictionary[prop]
        loss, loss_dict = criterion(
            input=output, 
            target_dict=targets_normed, 
            mask_dict=mask, 
            task_specs=TASK_SPECS,
        )

        # measure accuracy and record loss
        losses.update(loss.detach().cpu().item())
        for prop, value in TASK_SPECS.items():
            task = value['head']
            stats[prop]['loss'].update(loss_dict[prop]['loss'], int(mask[prop].sum().item()))
            if task in ['binary', 'multiclass']:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output[prop].detach().cpu(), targets[prop])
                stats[prop]['accuracies'].update(accuracy, int(mask[prop].sum().item()))
                stats[prop]['precisions'].update(precision, int(mask[prop].sum().item()))
                stats[prop]['recalls'].update(recall, int(mask[prop].sum().item()))
                stats[prop]['fscores'].update(fscore, int(mask[prop].sum().item()))
                stats[prop]['auc_scores'].update(auc_score, int(mask[prop].sum().item()))
            elif task == 'regression':
                nrmse_error = nrmse(
                    normalizers[prop].denorm(output[prop].detach().cpu()), 
                    targets[prop],
                    normalizers[prop]
                )
                stats[prop]['nrmse_errors'].update(nrmse_error, int(mask[prop].sum().item()))
            else:
                raise ValueError(f"[STAT SAVE] Unknown task {value['head']}")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ! print stats for each property
        if args.debug and i % args.print_freq == 0:
            print('\nEpoch: [{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), loss=losses))
            for prop, values in stats.items():
                if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'PROP {prop}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                        'Prec {prec.val:.3f} ({prec.avg:.3f})\t'
                        'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                        'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                        'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, prop=prop,
                        data_time=data_time, loss=values['loss'], accu=values['accuracies'],
                        prec=values['precisions'], recall=values['recalls'], 
                        f1=values['fscores'],auc=values['auc_scores'])
                    )
                elif TASK_SPECS[prop]['head'] == 'regression':
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'PROP {prop}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'NRMSE {nrmse_errors.val:.3f} ({nrmse_errors.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, prop=prop,
                        data_time=data_time, loss=values['loss'], nrmse_errors=values['nrmse_errors'])
                    )
                else:
                    raise ValueError(f"[STAT PRINT] Unrecognized task {TASK_SPECS[prop]['head']}")
            

def validate(val_loader, model, criterion, normalizers, best_epoch=0, test=False):
    batch_time = AverageMeter()
    # ! stat trackers for each metric per head
    losses = AverageMeter()
    stats = dict()
    for prop, value in TASK_SPECS.items():
        stats[prop] = dict()
        stats[prop]['loss'] = AverageMeter()
        if value['head'] in ['binary', 'multiclass']:
            stats[prop]['accuracies'] = AverageMeter()
            stats[prop]['precisions'] = AverageMeter()
            stats[prop]['recalls'] = AverageMeter()
            stats[prop]['fscores'] = AverageMeter()
            stats[prop]['auc_scores'] = AverageMeter()
        elif value['head'] == 'regression':
            stats[prop]['nrmse_errors'] = AverageMeter()
        else:
            raise ValueError(f"[TRAIN STATS] Unknown task {value['head']}")
    
    if test:
        test_stats = dict()
        for prop, value in TASK_SPECS.items():
            if value['head'] in ['binary', 'multiclass']:
                prop_stats = {
                    'test_targets': [], 
                    'test_preds': [], 
                    'test_probs': [], 
                    'test_cif_ids':[],
                    'test_masks': [],
                }
            elif value['head'] == 'regression':
                prop_stats = {
                    'test_targets': [], 
                    'test_preds': [], 
                    'test_cif_ids':[],
                    'test_masks': [],
                }
            else:
                raise ValueError(f"[TRAIN STATS] Unknown task {value['head']}")
            test_stats[prop] = prop_stats

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # for i, (input, target, batch_cif_ids) in enumerate(val_loader):
    for i, (
        (atom_fea, nbr_fea, nbr_fea_idx, crys_idx), 
        vectorizations, diagrams, targets, mask, batch_cif_ids
    ) in enumerate(val_loader):

        # ! EDTIED to use torch tensors
        atom_fea = atom_fea.to(device, non_blocking=True)
        nbr_fea = nbr_fea.to(device, non_blocking=True)
        nbr_fea_idx = nbr_fea_idx.to(device, non_blocking=True)

        crys_idx = [idx.to(device, non_blocking=True) for idx in crys_idx]

        vectorizations = vectorizations.to(device, non_blocking=True)
        diagrams = [
            diagram.to(device, non_blocking=True)
            for diagram in diagrams
        ]

        # ! updated variables to use torch Tensors & receive additional data
        input_var = (
            atom_fea, 
            nbr_fea, 
            nbr_fea_idx, 
            crys_idx,
            vectorizations, 
            diagrams
        )

        # ! target normalization applied to each property
        targets_normed = {
            key: (normalizers[key].norm(targets[key])).to(device, non_blocking=True)
            for key in targets.keys()
        }

        # compute output
        output = model(*input_var)
        loss, loss_dict = criterion(
            input=output, 
            target_dict=targets_normed, 
            mask_dict=mask, 
            task_specs=TASK_SPECS,
        )

        # measure accuracy and record loss
        losses.update(loss.detach().cpu().item())
        for prop, value in TASK_SPECS.items():
            task = value['head']
            stats[prop]['loss'].update(loss_dict[prop]['loss'], int(mask[prop].sum().item()))
            if task in ['binary', 'multiclass']:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output[prop].detach().cpu(), targets[prop])
                stats[prop]['accuracies'].update(accuracy, int(mask[prop].sum().item()))
                stats[prop]['precisions'].update(precision, int(mask[prop].sum().item()))
                stats[prop]['recalls'].update(recall, int(mask[prop].sum().item()))
                stats[prop]['fscores'].update(fscore, int(mask[prop].sum().item()))
                stats[prop]['auc_scores'].update(auc_score, int(mask[prop].sum().item()))
                
                if test:
                    test_pred = torch.exp(output[prop].detach().cpu())
                    test_target = targets[prop]
                    
                    # if binary classification
                    if TASK_SPECS[prop]['out_dim'] == 1:
                        assert test_pred.ndim == 1
                        test_pred = np.stack([1 - test_pred, test_pred], axis=1)
                    else:
                        assert test_pred.shape[1] == TASK_SPECS[prop]['out_dim']
                    pred_label = np.argmax(test_pred, axis=1)
                    test_stats[prop]['test_preds'] += pred_label.tolist()
                    test_stats[prop]['test_probs'] += test_pred.tolist()
                    test_stats[prop]['test_targets'] += test_target.view(-1).tolist()
                    test_stats[prop]['test_cif_ids'] += batch_cif_ids
                    test_stats[prop]['test_masks'] += mask[prop].view(-1).tolist()

            
            elif task == 'regression':
                nrmse_error = nrmse(
                    normalizers[prop].denorm(output[prop].detach().cpu()), 
                    targets[prop],
                    normalizers[prop]
                )
                stats[prop]['nrmse_errors'].update(nrmse_error, int(mask[prop].sum().item()))

                if test:
                    test_pred = normalizers[prop].denorm(output[prop].detach().cpu())
                    test_target = targets[prop]
                    test_stats[prop]['test_preds'] += test_pred.view(-1).tolist()
                    test_stats[prop]['test_targets'] += test_target.view(-1).tolist()
                    test_stats[prop]['test_cif_ids'] += batch_cif_ids
                    test_stats[prop]['test_masks'] += mask[prop].view(-1).tolist()

            else:
                raise ValueError(f"[STAT SAVE] Unknown task {value['head']}")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ! print stats for each property
        if args.debug and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, len(val_loader), loss=losses))
            for prop, values in stats.items():
                if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
                    print('Test: [{0}/{1}]\t'
                        'PROP {prop}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                        'Prec {prec.val:.3f} ({prec.avg:.3f})\t'
                        'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                        'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                        'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, prop=prop,
                        loss=values['loss'], accu=values['accuracies'],
                        prec=values['precisions'], recall=values['recalls'], 
                        f1=values['fscores'],auc=values['auc_scores'])
                    )
                elif TASK_SPECS[prop]['head'] == 'regression':
                    print('Test: [{0}/{1}]\t'
                        'PROP {prop}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'NRMSE {nrmse_errors.val:.3f} ({nrmse_errors.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, prop=prop,
                        loss=values['loss'], nrmse_errors=values['nrmse_errors'])
                    )
                else:
                    raise ValueError(f"[STAT PRINT] Unrecognized task {TASK_SPECS[prop]['head']}")
    
    # finished validation testing    
    if args.debug:
        print('Final Test\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))
        for prop, values in stats.items():
            if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
                print('Test: [{0}/{1}]\t'
                    'PROP {prop}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                    'Prec {prec.val:.3f} ({prec.avg:.3f})\t'
                    'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                    'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                    'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, prop=prop,
                    loss=values['loss'], accu=values['accuracies'],
                    prec=values['precisions'], recall=values['recalls'], 
                    f1=values['fscores'],auc=values['auc_scores'])
                )
            elif TASK_SPECS[prop]['head'] == 'regression':
                print('Test: [{0}/{1}]\t'
                    'PROP {prop}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'NRMSE {nrmse_errors.val:.3f} ({nrmse_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, prop=prop,
                    loss=values['loss'], nrmse_errors=values['nrmse_errors'])
                )
            else:
                raise ValueError(f"[STAT PRINT] Unrecognized task {TASK_SPECS[prop]['head']}")

    w_cls = 0.5
    w_reg = 0.5
    
    cls_error = AverageMeter()
    reg_error = AverageMeter()

    for prop, values in stats.items():
        if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
            cls_error.update(2 * (1 - values['auc_scores'].avg))
        elif TASK_SPECS[prop]['head'] == 'regression':
            reg_error.update(values['nrmse_errors'].avg)
        else:
            raise ValueError(f"[VAL ERROR] Unrecognized task {TASK_SPECS[prop]['head']}")
    
    overall_error = w_cls * cls_error.avg + w_reg + reg_error.avg

    if not CROSS_VAL and test:
        save_results(
            best_epochs=[best_epoch], 
            total_loss=losses.avg, 
            total_error=overall_error.item(),
            stats_dict=stats,
        )
        # if args.debug:
        #     print("Saving test stats/results")

        # with open(f'test_params_{args.id}.txt', 'w') as f:
        #     f.write(f'Source:\t\t\t{args.vec_source}\nVector:\t\t\t{args.vector}')
        #     f.write(f'\nAtom Len:\t\t{args.atom_fea_len}\nConv Num:\t\t{args.n_conv}')
        #     f.write(f'\nHidden Len:\t\t{args.h_fea_len}\nHidden Num:\t\t{args.n_h}\nHead Layer Num:\t{args.n_o}')
        #     f.write(f'\nVec Len:\t\t{args.vec_fea_len}\nVec Layer Num:\t{args.n_vec}')
        #     f.write(f'\nBest Epoch:\t\t{best_epoch}')
        
        # # ! saving stats for each prop
        # with open(f'test_stats_{args.id}.csv', 'w', newline='', encoding='utf-8') as file_stats:
        #     writer = csv.writer(file_stats)
        #     header = ['head', 'task', 'loss', 'nrmse', 'accuracy', 'precision', 'recall', 'f1', 'auroc']
        #     writer.writerow(header)
        #     writer.writerow(['total', overall_error, losses.avg, '', '', '', '', '', ''])
                    
        #     for prop, values in stats.items():
        #         if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
        #             row = [
        #                 prop, 
        #                 TASK_SPECS[prop]['head'], 
        #                 values['loss'].avg, 
        #                 '', 
        #                 values['accuracies'].avg, 
        #                 values['precisions'].avg, 
        #                 values['recalls'].avg, 
        #                 values['fscores'].avg, 
        #                 values['auc_scores'].avg
        #             ]
        #         elif TASK_SPECS[prop]['head'] == 'regression':
        #             row = [
        #                 prop, 
        #                 TASK_SPECS[prop]['head'], 
        #                 values['loss'].avg, 
        #                 values['nrmse_errors'].avg.item(), 
        #                 '', 
        #                 '', 
        #                 '', 
        #                 '', 
        #                 ''
        #             ]
        #         else:
        #             raise ValueError(f"[STAT CSV] Unrecognized task {TASK_SPECS[prop]['head']}")
        #         writer.writerow(row)
        
        # ! saving results for each prop
        for prop, values in test_stats.items():
            with open(f'test_results_{args.id}_{prop}.csv', 'w', newline='', encoding='utf-8') as file_results:
                writer = csv.writer(file_results)

                if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
                    header = ['mp-id', 'mask', 'target', 'predicted_class']
                    header += [f'prob_class_{i}' for i in range(len(values['test_probs'][0]))]
                    writer.writerow(header)

                    for cif_id, mask, target, pred, probs in zip(
                        values['test_cif_ids'], 
                        values['test_masks'],
                        values['test_targets'], 
                        values['test_preds'], 
                        values['test_probs'],
                    ):
                        writer.writerow([cif_id, mask, target, pred] + probs)

                elif TASK_SPECS[prop]['head'] == 'regression':
                    header = ['mp-id', 'mask', 'target', 'predicted_value']
                    writer.writerow(header)

                    for cif_id, mask, target, pred in zip(
                        values['test_cif_ids'], 
                        values['test_masks'],
                        values['test_targets'], 
                        values['test_preds'], 
                    ):
                        writer.writerow([cif_id, mask, target, pred])
                else:
                    raise ValueError(f"[TEST CSV] Unrecognized task {TASK_SPECS[prop]['head']}")

    if args.debug:
        print(f'\n Model -- \tVector: {args.vector}\tAtom Len: {args.atom_fea_len}\tConv Num: {args.n_conv}\tHidden Len: {args.h_fea_len}\tHidden Num: {args.n_h}')
        print(f'\t\tHead Layer Num: {args.n_o}\tVec Len: {args.vec_fea_len}\tVec Layer Num: {args.n_vec}')
        
        print(' ** ERROR {error:.3f}'.format(error=overall_error))
    return overall_error, losses.avg, stats


def save_results(
        best_epochs: list, 
        total_loss, 
        total_error,
        stats_dict
):
    if args.debug:
        print("Saving test results and stats as CSV")

    with open(f'test_params_{args.id}.txt', 'w') as f:
        f.write(f'Source:\t\t\t{args.vec_source}\nVector:\t\t\t{args.vector}')
        f.write(f'\nAtom Len:\t\t{args.atom_fea_len}\nConv Num:\t\t{args.n_conv}')
        f.write(f'\nHidden Len:\t\t{args.h_fea_len}\nHidden Num:\t\t{args.n_h}\nHead Layer Num:\t{args.n_o}')
        f.write(f'\nVec Len:\t\t{args.vec_fea_len}\nVec Layer Num:\t{args.n_vec}')
        f.write(f'\nBest Epochs:\t\t{", ".join(list(map(str, best_epochs)))}')
    
    import csv
    # ! saving stats for each prop
    with open(f'test_stats_{args.id}.csv', 'w', newline='', encoding='utf-8') as file_stats:
        writer = csv.writer(file_stats)
        header = ['head', 'task', 'loss', 'nrmse', 'accuracy', 'precision', 'recall', 'f1', 'auroc']
        writer.writerow(header)
        writer.writerow(['total', total_error, total_loss, '', '', '', '', '', ''])
                
        for prop, values in stats_dict.items():
            if TASK_SPECS[prop]['head'] in ['binary', 'multiclass']:
                row = [
                    prop, 
                    TASK_SPECS[prop]['head'], 
                    values['loss'].avg, 
                    '', 
                    values['accuracies'].avg, 
                    values['precisions'].avg, 
                    values['recalls'].avg, 
                    values['fscores'].avg, 
                    values['auc_scores'].avg
                ]
            elif TASK_SPECS[prop]['head'] == 'regression':
                row = [
                    prop, 
                    TASK_SPECS[prop]['head'], 
                    values['loss'].avg, 
                    values['nrmse_errors'].avg.item(), 
                    '', 
                    '', 
                    '', 
                    '', 
                    ''
                ]
            else:
                raise ValueError(f"[STAT CSV] Unrecognized task {TASK_SPECS[prop]['head']}")
            writer.writerow(row)
    

class NormalizerProp(object):
    """Normalize a Tensor for one property and restore it later. """

    def __init__(self, tensor, mask=None):
        """
        tensor is taken as a sample to calculate the mean and std.
        mask is applied for targets with invalid labels.
        """
        if mask is None:
            self.mean = torch.mean(tensor)
            self.std = torch.std(tensor)
            self.min, self.max = torch.min(tensor), torch.max(tensor)
            self.bound = self.max - self.min
        else:
            self.mean = torch.mean(tensor[mask].float())
            self.std = torch.std(tensor[mask].float())
            self.min, self.max = torch.min(tensor[mask].float()), torch.max(tensor[mask].float())
            self.bound = self.max - self.min
        if args.debug:
            print('normalprop:', self.mean, self.std, self.min, self.max, self.bound)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    
    def get_bound(self):
        return self.bound

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

        

def nrmse(prediction, target, normalizer):
    """
    Computes the mean squared error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    mse = ((prediction - target) ** 2).mean()
    bound = normalizer.get_bound()
    return torch.sqrt(mse) / bound


# ! Updated for multiclass classification
def class_eval(prediction, target):
    with torch.no_grad():
        target_label = target.detach().cpu().numpy()
        target_label = np.squeeze(target)

        if not target_label.shape:
            target_label = np.asarray([target_label])

        # binary case
        if prediction.ndim == 1 or prediction.shape[-1] == 1:
            logits = prediction.reshape(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            pred_label = (probs >= 0.5).astype(int)
            # if args.debug:
            #     print("CLASS EVAL INPUT")
            #     print("prediction:", prediction)
            #     print("probs", probs)
            #     print("pred", pred_label)
            #     print("targ", target_label)
            prediction = np.stack([1 - probs, probs], axis=1)

            accuracy = metrics.accuracy_score(target_label, pred_label)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                target_label, pred_label, average='binary', zero_division=0)
            # if args.debug:
            #     print("CLASS EVAL DEBUG")
            #     print(precision, recall, fscore)
            auc_score = metrics.roc_auc_score(target_label, probs)

        else:
            prediction = torch.softmax(prediction, dim=-1).detach().cpu().numpy()
            pred_label = np.argmax(prediction, axis=1)

            accuracy = metrics.accuracy_score(target_label, pred_label)
            # multiclass
            # ! average macro (imbalanced?) vs micro (balanced?)
            precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                target_label, pred_label, average='macro', zero_division=0)
            # ! multi_class ovr (balanced) vs ovo (imbalanced)
            auc_score = metrics.roc_auc_score(
                    target_label,
                    prediction,
                    multi_class='ovo',
                    average='macro',
                    labels=list(range(prediction.shape[1]))
                )
            # ! might raise error --> if so, then catch later
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def save_checkpoint(state, is_best, fold_it, filename=f'checkpoint_{args.id}.pth.tar'):
    torch.save(state, filename)
    if is_best:
        if CROSS_VAL:
            best_filename = f"model_best_{args.id}_fold_{fold_it}.pth.tar"
        else:
            best_filename = f"model_best_{args.id}.pth.tar"
        shutil.copyfile(filename, best_filename)


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
