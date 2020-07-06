import argparse
import os
import cv2
import math
import random
import shutil
import time
import warnings
import h5py
import copy
import datetime
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import sys
sys.path.append('./utils')
import general
import help_main
import configuration

from tqdm import tqdm
from models import custom_models
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score

import module.train as train
from module.arguments import get_args          
args = get_args()

from torch.utils.data import Dataset

# model_names = sorted(name for name in custom_models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(custom_models.__dict__[name]))

# if 'resnet' not in args.arch:
#     if args.arch not in model_names:
#         raise Exception('There is no model named : {}'.format(args.arch))

result_path = './result/{}'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S%f'))
args.result_path = result_path
general.make_result_dir(args.result_path, args)

def main():
    ########################################################################################
    data = np.array([])
    result_list = ['train_loss{}', 'train_class_loss{}', 'train_lrp_loss{}', 
                   'val_loss{}', 'val_class_loss{}', 'val_lrp_loss{}', 'val_accuracy{}']
    result_filename = os.path.join(args.result_path, 'result{:02d}.hdf5'.format(args.folder))
    for string in result_list:
        general.save_hdf5(result_filename, string.format(args.trial), data)
    ########################################################################################
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

best_acc1 = 0
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    ################################################################
    # create model
    model = configuration.setup_model(args).cuda(args.gpu)
    print(model)
    ################################################################

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node) # ngpus_per_node is a number of gpus I can use.
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr)
    else:
        raise Exception('No optimizer specified')
    
    
    cudnn.benchmark = True
    
    
    print('#################################### Dataset ####################################')
    train_dataset = configuration.setup_dataset('train', args)
    val_dataset = configuration.setup_dataset('valid', args)
    test_dataset = configuration.setup_dataset('test', args)
    print('args.distributed :', args.distributed)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    print('train_sampler :', train_sampler)
    print('args.workers :', args.workers)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    print('#################################################################################')
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    if args.test_model_path == None:
        # 학습을 시작하기 전의 performance를 먼저 기록
        validate('train', train_loader, model, 0, criterion, args)
        validate('val', val_loader, model, 0, criterion, args)
        args.start_epoch += 1
            
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train.__dict__[args.train_func](train_loader, model, epoch, criterion, optimizer, args)

            # evaluate on validation set
            _ = validate('train', train_loader, model, epoch, criterion, args)
            acc1 = validate('val', val_loader, model, epoch, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, os.path.join(args.result_path, 'model_best_{:02d}.pth.tar'.format(args.trial)))
            
        args.pretrained_model_path = os.path.join(args.result_path, 'model_best_{:02d}.pth.tar'.format(args.trial))
#         print(args.pretrained_model_path, '!!!!!!!!!!!!!!!!!!!!!!!')
    else:
        print('##################################')
        print('######  AFTER TRAINING MODE  #####')
        print('##################################')
        args.pretrained_model_path = args.test_model_path
    ########################################################################################################################
    print('======= start =======')
    if args.arch_for_test == None:
        args.arch_for_test = args.arch
    print('testing model :', args.arch_for_test)
    
    model = configuration.setup_model(args).cuda(args.gpu)
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if 'IoU' in args.metrics:
        print('getting mIoU...')
        if args.loader_IoU == 'train':
            help_main.get_mIoU(train_loader, model, criterion, args)
        elif args.loader_IoU == 'val':
            help_main.get_mIoU(val_loader, model, criterion, args)
        else:
            raise Exception('Wrong loader')
        
        
def validate(name, val_loader, model, epoch, criterion, args): 

    model.eval()
    sum_val_loss, sum_cls_loss, sum_lrp_loss, sum_gain_loss = 0, 0, 0, 0
    label_list, pred_list, pred_bin = [], [], [] # pred_bin is for keeping track of actual predictions.

    with torch.no_grad():
        if args.interp_monitor and ('synthetic' in args.data_path):
            print('mask is been saving!!!!!!!!!!!!')
        for num, (data, label, use_mask) in enumerate(tqdm(val_loader)):
            
            img, mask = data[0].type(torch.FloatTensor).cuda(args.gpu), data[1].type(torch.FloatTensor).cuda(args.gpu)
            label = label.cuda(args.gpu)
            use_mask = use_mask.cuda(args.gpu)
            
            start = time.time()
            
            output, lrp_loss, R = model.forward(img, label, mask, use_mask, args)

            start = time.time()
            
            class_loss = criterion(output, label).sum().item()
            sum_cls_loss += class_loss
            sum_lrp_loss += lrp_loss
            sum_val_loss += class_loss + args.lambda_for_final * lrp_loss
            
            label_list.append(label.cpu().data.numpy())
            pred_list.append(output.cpu().data.numpy())
            pred_bin.append(torch.max(output, 1)[1].detach().cpu().numpy())
            

    total_label = np.concatenate(label_list)
    total_score = np.concatenate(pred_list)
    accuracy= accuracy_score(total_label, total_score.argmax(axis=1))    
    auc = metrics.roc_auc_score(np.squeeze(total_label), total_score[:,1])
    pred_bin = np.hstack(pred_bin)

    mean_val_loss = sum_val_loss / len(val_loader)
    mean_cls_loss = sum_cls_loss / len(val_loader)
    mean_lrp_loss = sum_lrp_loss / len(val_loader)

    print(epoch, 'Average {} cls_loss: {:.4f}, lrp_loss: {:.4f}, acc: {:.4f}, pred: {}/{}'.format(
        name, mean_cls_loss, mean_lrp_loss, accuracy,
        np.sum(pred_bin==0), np.sum(pred_bin==1)))
    
    result_filename = os.path.join(args.result_path, 'result{:02d}.hdf5'.format(args.folder))
    general.save_hdf5(result_filename, '{}_loss{}'.format(name, args.trial), np.array([mean_val_loss.item()]))
    general.save_hdf5(result_filename, '{}_class_loss{}'.format(name, args.trial), np.array([mean_cls_loss]))
    general.save_hdf5(result_filename, '{}_lrp_loss{}'.format(name, args.trial), np.array([mean_lrp_loss.item()]))
    general.save_hdf5(result_filename, '{}_accuracy{}'.format(name, args.trial), np.array([accuracy.item()]))
    general.save_hdf5(result_filename, '{}_auc{}'.format(name, args.trial), np.array([auc.item()]))
    
    return accuracy



if __name__ == '__main__':
    main()