import argparse
import os
import shutil
import sys
import time
import warnings
import pickle
import csv
from random import sample
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from cgcnn.data import CIFData
# from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.data import collate_pool, get_train_val_test_loader, \
    get_fold_indices, get_train_val_test_loader_from_folds
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
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
parser.add_argument('--scheduler', default='normal', type=str,
                    help='choose a scheduler: [normal, adapt], (default: normal)')
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

parser.add_argument('--seed', action='store_true',
                    help='sets torch seed to 42')
parser.add_argument('--num-classes', default=2, type=int)
parser.add_argument('--delete', action='store_true',
                    help='deletes generated training files before training')
parser.add_argument('--id', default='0', type=str, metavar='N',
                    help='identifier for multiple checkpoint/models')
parser.add_argument('--norm-sample', default=2000, type=int,
                    help='number of max samples to use to define normalizers')

parser.add_argument('--fold', default=0, type=int, 
                    help='use k-fold cross validation for val/test')
parser.add_argument('--train', default='pretrain', type=str, 
                    help='choose training type: pretrain, abs, plqy')
parser.add_argument('--freeze', action='store_true', 
                    help="whether to freeze lower encoder layers for finetuning")

parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to pretrain checkpoint')
parser.add_argument('--finetune-auto', action='store_true', 
                    help='finetune, automatically finds path')
parser.add_argument('--val-metric', default='error', type=str,
                    help="validation metric: ['error', 'loss']")
parser.add_argument('--clear-cache', action='store_true',
                    help="whether to clear dataloader cache before training")
parser.add_argument('--save-fold', action='store_true',
                    help="whether to save fold checkpoints")
parser.add_argument('--drop-last', action='store_true',
                    help="whether to drop incomplete batches")


args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()


# read in head tasks
task_filepath = os.path.join(args.data_options[0], 'tasks', 'tasks.pkl')
assert os.path.exists(task_filepath), 'Tasks file (tasks.pkl) does not exist!'
with open(task_filepath, 'rb') as f:
    TASK_SPECS = pickle.load(f)

CROSS_VAL = args.fold != 0

# if args.task == 'regression':
#     best_errors = 1e10
# else:
#     best_errors = 0.


def main():
    global args, best_errors

    # ! CUSTOM
    if args.seed:
        torch.manual_seed(42)
    print("GPU Available", args.cuda)

    if args.delete and args.resume == '':
        if os.path.exists(f'out_{args.id}'):
            shutil.rmtree(f'out_{args.id}')
    open_write_file(f'out_{args.id}', '')
    
    assert args.train in ['pretrain', 'abs', 'plqy'],\
        "Wrong train argument! Should be one of 'pretrain', 'abs', 'plqy'"
    finetune_true = args.finetune != '' or args.finetune_auto
    assert (args.train == 'pretrain' and not finetune_true) or (args.train != 'pretrain' and finetune_true), \
        "No finetune arg if train=pretrain, and need finetune arg if train=abs/plqy"
    assert args.resume == '' or not finetune_true, "Choose one of resume or finetune!"
    assert args.val_metric in ['error', 'loss']
    assert args.scheduler in ['normal', 'adapt']
    assert not CROSS_VAL or args.train != 'pretrain',\
        "[STOP] Should not run k-fold cross validation on pretrain dataset!"

    # load data
    print("Constructing CIFData")
    dataset = CIFData(*args.data_options)
    collate_fn = collate_pool
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
        return_test=True)

    if args.fold == 0:
        # regular training
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
        )
    else:
        print(f"Constructing {args.fold} folds")
        # train/val/test loaders are constructed within fold loop
        folds = get_fold_indices(
            dataset=dataset, 
            k=args.fold,
        )
    
    # ! ADJUST FOR FOLD ITERATIONS (should be one pass if fold = 0)
    # average results for all folds (or same as val results for fold = 0)
    best_epochs = []
    total_val_losses = AverageMeter()
    total_test_losses = AverageMeter()
    total_val_errors = AverageMeter()
    total_test_errors = AverageMeter()
    total_val_stats = dict()
    total_test_stats = dict()

    for stat_dict in [total_val_stats, total_test_stats]:
        for prop, value in TASK_SPECS.items():
            stat_dict[prop] = dict()
            stat_dict[prop]['loss'] = AverageMeter()
            if value['head'] in ['binary', 'multiclass']:
                stat_dict[prop]['accuracies'] = AverageMeter()
                stat_dict[prop]['precisions'] = AverageMeter()
                stat_dict[prop]['recalls'] = AverageMeter()
                stat_dict[prop]['fscores'] = AverageMeter()
                stat_dict[prop]['auc_scores'] = AverageMeter()
            elif value['head'] == 'regression':
                stat_dict[prop]['nrmse'] = AverageMeter()
            else:
                raise ValueError(f"[TRAIN STATS] Unknown task {value['head']}")
            
    # ! time logging for all folds
    total_times = AverageMeter()
    epoch_0_batch_times = AverageMeter()
    epoch_0_times = AverageMeter()
    epoch_n_batch_times = AverageMeter()
    epoch_n_times = AverageMeter()

    # k-fold cross validation
    fold_num = args.fold if CROSS_VAL else 1
    for fold_it in range(fold_num):
        best_errors = float('inf')

        # loss plotting
        loss_t_epochs = []
        loss_t_batches = []
        loss_v_epochs = []

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
                drop_last=args.drop_last,
            )
        
        print("Normalizer train sample size 2000")    
        train_indices = list(train_loader.sampler)
        sample_cnt = min(len(train_indices), args.norm_sample)
        sample_indices = sample(train_indices, k=sample_cnt)
        sample_data_list = [dataset[i] for i in tqdm(sample_indices, desc="Normalizers: ")]
        _, _, _, sample_target, sample_mask, _ = collate_pool(sample_data_list)

    # # obtain target value normalizer
    # if args.task == 'classification':
    #     normalizer = Normalizer(torch.zeros(2))
    #     normalizer.load_state_dict({'mean': 0., 'std': 1.})
    # else:
    #     if len(dataset) < 500:
    #         warnings.warn('Dataset has less than 500 data points. '
    #                       'Lower accuracy is expected. ')
    #         sample_data_list = [dataset[i] for i in range(len(dataset))]
    #     else:
    #         sample_data_list = [dataset[i] for i in
    #                             sample(range(len(dataset)), 500)]
    #     _, sample_target, _ = collate_pool(sample_data_list)
    #     normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_conv=args.n_conv,
                                h_fea_len=args.h_fea_len,
                                n_h=args.n_h,
                                classification=True if args.task ==
                                                       'classification' else False, 
                                num_classes=args.num_classes)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
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
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_errors = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_errors
            best_errors = min(mae_error, best_errors)
        else:
            is_best = mae_error > best_errors
            best_errors = max(mae_error, best_errors)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_errors,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar', weights_only=False)
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_probs = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                
                # ! MODIFIED to account for multiclass
                # assert test_pred.shape[1] == 2
                assert test_pred.shape[1] == args.num_classes
                
                # test_preds += test_pred[:, 1].tolist()
                pred_label = torch.argmax(test_pred, dim = 1)
                test_preds += pred_label.tolist()
                test_probs += test_pred.tolist()
                
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    print('Final C Test\t'
            'Loss ({loss.avg:.4f})\t'
            'Accu ({accu.avg:.3f})\t'
            'Precision ({prec.avg:.3f})\t'
            'Recall ({recall.avg:.3f})\t'
            'F1 ({f1.avg:.3f})\t'
            'AUC ({auc.avg:.3f})'.format(
        i, len(val_loader), batch_time=batch_time, loss=losses,
        accu=accuracies, prec=precisions, recall=recalls,
        f1=fscores, auc=auc_scores))
    
    if test:
        star_label = '**'
        import csv
        # ! MODIFIED to write full multiclass predicted probabilities
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)

            header = ['mp-id', 'target', 'predicted_class']
            header += [f'prob_class_{i}' for i in range(args.num_classes)]
            writer.writerow(header)

            for cif_id, target, pred, probs in zip(test_cif_ids, test_targets, test_preds, test_probs):
                writer.writerow([cif_id, target, pred] + probs)
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval_DEPRECIATED(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


# ! Updated for multiclass classification
def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()

    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)

    if not target_label.shape:
        target_label = np.asarray([target_label])

    accuracy = metrics.accuracy_score(target_label, pred_label)

    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
    else:
        # ! average macro (imbalanced?) vs micro (balanced?)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='macro', zero_division=0)
        # ! multi_class ovr (balanced) vs ovo (imbalanced)
        auc_score = metrics.roc_auc_score(
                target_label,
                prediction,
                multi_class='ovo',
                average='macro',
                labels=list(range(args.num_classes))
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
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def open_write_file(dir_path, file_name):
    """Opens a file for writing, or creates new file if file doesn't exist."""
    file_path = os.path.join(dir_path, file_name)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    return file_path


if __name__ == '__main__':
    main()
