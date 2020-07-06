import module.render as render
import numpy as np
import os
import glob
import torch
from visdom import Visdom
from skimage.io import imsave, imread
from module.arguments import get_args
import pandas as pd
import shutil
import operator
import cv2
from datetime import datetime
import torch
import torch.nn.functional as F
import scipy.stats
import pickle

args = get_args()

import subprocess

def normalize_with_log(C, mean_to_one=False):
    
    use_log = False
    
    # Take log of abs and multiply sign and then normalize to mean as 0
    C = C.detach()
    C_shape = C.shape
    C = C.reshape(C.shape[0],-1)

    
#     # Using log
#     use_log = True
#     sign_C = torch.sign(C).detach()
#     mid_C = (torch.log10(C.abs() +1 ) * sign_C)
    
    # Not use log
    if use_log is True:
        print(check_utils_normalize_with_log)
    else:
        mid_C = C
    
    
#     # Mean-std : NaN occur!!
#     std_ = torch.std(mid_C.abs(),1)[0].detach().reshape(-1,1)
#     std_ = torch.where(std_.abs()==0, (torch.ones(1)*1e+2).cuda(), std_)
#     fin_C = ((mid_C - mid_C.mean().detach().reshape(-1,1)) /(std_))
#     print('std ++++++++++++++++',torch.std(mid_C.abs(),1)[0].detach().mean())
    
#     # Min-Max
#     mid_C = (mid_C - torch.min(mid_C)[0].reshape(-1,1))
#     fin_C = mid_C / torch.max(mid_C)[0].reshape(-1,1)
    
    
    # Only Min
    mid_C = (mid_C - torch.min(mid_C)[0].reshape(-1,1))
    fin_C = mid_C
    
    
    
    if mean_to_one :
        # Add 1 to make mean from 0 to 1
        C = C + torch.ones_like(C)
    
#     fin_C = fin_C * sign_C
#     print('fin_C  - occur or only + ????????????',fin_C.min()[0])
    
    
    # Reshape C
    C = fin_C.reshape(C_shape)

    
    
    
#     # Just divide with max of C
#     C = torch.nn.functional.normalize(C, p=2, dim=1)

#     # Clipping if needed
#     C = torch.where(C.abs()<1e-8, torch.ones(1).cuda(), C)
    
#     # Multiply constant to control scale of C
#     C = C * 1e-8

    return C

def normalize_with_nonlog(C):
    
    # Take log of abs and multiply sign and then normalize to mean as 0
    C_shape = C.shape
    C = C.reshape(C.shape[0],-1)

    sign_C = torch.sign(C)
    mid_C = (C - torch.min(C)[0].reshape(-1,1))
    fin_C = mid_C / torch.max(mid_C)[0].reshape(-1,1) * sign_C
    C = fin_C.reshape(C_shape)
    
#     # Add 1 to make mean from 0 to 1
#     C = C + torch.ones_like(C)
    
#     # Just divide with max of C
#     C = torch.nn.functional.normalize(C, p=2, dim=1)

#     # Clipping if needed
#     C = torch.where(C.abs()<1e-8, torch.ones(1).cuda(), C)
    
#     # Multiply constant to control scale of C
#     C = C * 1e-8

    return C


def get_min_used_gpu():
    import torch.cuda as cutorch
    device = 0
    min_used = 1e+10
    for i in range(cutorch.device_count()):
        if min_used > torch.cuda.memory_allocated(i):
            min_used = torch.cuda.memory_allocated(i)
            device = i
    return device

def __get_min_used_gpu(k):
    import torch.cuda as cutorch
    device = []
    min_used = 1e+10
    for i in range(cutorch.device_count()):
        device.append(torch.cuda.memory_allocated(i))
    print(device)
    _,top_device = torch.topk(torch.tensor(device),k, largest=False)
    top_device = list(top_device)   
    return top_device

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
#     gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def get_gradient_std_dict_and_save(net, grad_dict, save_path):
    # This code receives net and grad_dict as input, and return the std of the gradient of weights for each layers.
    keys = net.state_dict().keys()
    if len(grad_dict) == 0:
        # this case occur when this function called first time.
        for key in keys:
            grad_dict[key] = []

    for key, p in net.named_parameters():
        try:
            grad_dict[key].append(p.grad.std().item())
        except:
            continue
    if len(grad_dict[key]) == 1:
        return grad_dict
    
    f = open(save_path,"wb")
    pickle.dump(grad_dict,f)
    f.close()
    return grad_dict
    

def get_random_model(vgg19, delta):

    # Same amout of change by each layer
    vgg19_keys = list(vgg19.state_dict().keys())

    for i in range(len(vgg19_keys)):
        
        # Random value and sign added to vgg19 weights
        w = vgg19.state_dict()[vgg19_keys[i]][:].detach().cpu().numpy()
#         delta_list = np.random.rand(w.shape)
#         delta_list = delta_list / delta_list.sum() * (delta * np.prod(w.shape)) 
        random_sign = np.random.choice([1,-1],w.shape)
        gaussian_delta = np.random.normal(0, delta, w.shape)
        w += gaussian_delta * random_sign
        vgg19.state_dict()[vgg19_keys[i]][:] = torch.tensor(w, dtype=torch.float32)

    return vgg19


def get_weight_delta(net, vgg19):
    num_weight = 0
    delta = 0
    vgg19_keys = list(vgg19.state_dict().keys())
    net_keys = list(net.state_dict().keys())
    for i in range(len(vgg19_keys)):
        w = net.state_dict()[net_keys[i]][:].detach().cpu().numpy()
        w_ori = vgg19.state_dict()[vgg19_keys[i]][:].numpy()
        delta += ((w - w_ori)**2).sum()
        num_weight += np.prod(w.shape)
    delta = np.sqrt(delta/num_weight) 
    return delta
    
    

def normalize(LRP):
    LRP = LRP - LRP.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].reshape(-1,1,1,1)
    LRP_shape = LRP.shape
#         LRP = torch.nn.functional.normalize(LRP.reshape(LRP_shape[0],-1), p=1, dim=1).reshape(LRP.shape)
    LRP = LRP/(LRP.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].reshape(-1,1,1,1) + 1e-8)
    return LRP


def quanti_metric(R_ori, R, interpreter, quanti_type = args.quanti_type):
    R_shape = R.shape
    l = R.shape[2]
    R_ori_f = torch.tensor(R_ori.reshape(R.shape[0],-1), dtype=torch.float32).cuda()
    R_f = torch.tensor(R.reshape(R.shape[0],-1), dtype=torch.float32).cuda()
    corr_list = []
#     rank_ori = R_ori_f.argsort(axis=1)
#     rank_ = R_f.argsort(axis=1)
    
    
    for i in range(R.shape[0]):
        corr = 0
        if quanti_type == 'rank':
            corr = scipy.stats.spearmanr(R_ori_f[i].detach().cpu().numpy(), R_f[i].detach().cpu().numpy())[0]
        
        elif quanti_type == 'sign':
            mask_ori = ((abs(R_ori_f[i]) - abs(R_ori_f[i]).mean())/abs(R_ori_f[i]).std()) > 1
            R_ori_f_sign = torch.tensor(mask_ori, dtype=torch.float32).cuda() * torch.sign(R_ori_f[i])
            
            mask_ = ((abs(R_f[i]) - abs(R_f[i]).mean())/abs(R_f[i]).std()) > 1
            R_f_sign = torch.tensor(mask_, dtype=torch.float32).cuda() * torch.sign(R_f[i])
            
            corr = np.corrcoef(R_ori_f_sign.detach().cpu().numpy(), R_f_sign.detach().cpu().numpy())[0][1]
        
        elif quanti_type == 'frame':
            
            if len(R_shape) == 3:
                mask = torch.ones(R_shape[0], 1, R_shape[1], R_shape[2])
            else:
                mask = torch.ones(R_shape)
            k=(l//7)

            if interpreter == 'lrp' and args.lrp_target_layer==None:
                mask[:,:,k:-k,k:-k] = -1 * torch.ones((l-2*k,l-2*k), dtype=torch.float32).cuda()
                mask_f = mask.reshape(mask.shape[0],-1)
                R_f_nor = R_f[i]/torch.max(R_f[i])
                corr = np.corrcoef(mask_f[i].detach().cpu().numpy(), R_f_nor.detach().cpu().numpy())[0][1]
            else:
                mask[:,:,k:-k,k:-k] = torch.zeros((l-2*k,l-2*k), dtype=torch.float32).cuda()
                mask_f = mask.reshape(mask.shape[0],-1)
                corr = np.corrcoef(mask_f[i].detach().cpu().numpy(), R_f[i].detach().cpu().numpy())[0][1] 
#             corr = np.corrcoef(mask_f[i], R_f_nor[i])[0][1]
        corr_list.append(corr)
        
    return np.mean(corr_list)


def pixel_flipping(net, R, input, labels, num_grid = (1,1,14,14)):
    result = []
    input = input.clone()
    
    grid_size = input.shape[2] // num_grid[2]
    if R.shape[2] != num_grid[2]:
        R = torch.tensor(R)
        R = R.sum(dim=1, keepdim=True)
        R = F.avg_pool2d(torch.tensor(R), (grid_size, grid_size))
        R = R.cpu().detach().numpy()
        
    # we multiply -1 to R, because argsort is increasing order.
    R_fl = - R.flatten()
    
    
    activation_output = net.prediction(input.cuda())
    softmax = F.softmax(activation_output)
    result.append(softmax[:,labels].item())
    
    input_unfolded = F.unfold(input, (grid_size, grid_size), stride=grid_size)
    
    for index in R_fl.argsort()[:100]:
        input_unfolded[:,:,index] = 0
        input_folded = F.fold(input_unfolded, input.shape[2:], (grid_size, grid_size), stride=grid_size)
        
        activation_output = net.prediction(input_folded.cuda())
        softmax = F.softmax(activation_output)
        result.append(softmax[:,labels].item())
        
    return result

def aopc_batch(net, R, input, labels, num_grid = (1,1,14,14)):
    result = []
    input = input.clone()
    
    # 1. if interpreter == LRP, it resize R to be 14*14
    grid_size = input.shape[2] // num_grid[2]
    if R.shape[2] != num_grid[2]:
        R = R.sum(dim=1, keepdim=True)
        R = F.avg_pool2d(torch.tensor(R), (grid_size, grid_size))
    R = R.cpu().detach().numpy()
        
    # 2. we multiply -1 to R, because argsort is increasing order.
    R_fl = - R.reshape(R.shape[0],-1)
    
    # first evaludation
    activation_output = net.prediction(input)
    prediction = torch.argmax(activation_output, 1)
#     c0 = (prediction == labels).float().squeeze()
    a0 = activation_output[:,labels]
    
    aopc_sum_K = 0
    result.append(aopc_sum_K)

    
    # 3. unfold
    input_unfolded = F.unfold(input, (grid_size, grid_size), stride=grid_size)
    
    for index in R_fl.argsort()[:,:100].T:
        # gaussian
        input_unfolded[np.arange(len(input_unfolded)), :, index] = torch.tensor(np.random.normal(loc=0.0, scale=0.3, size=input_unfolded[np.arange(len(input_unfolded)), :, index].shape),dtype=torch.float32).cuda()
        input_folded = F.fold(input_unfolded, input.shape[2:], (grid_size, grid_size), stride=grid_size)
        
        activation_output = net.prediction(input_folded.cuda())
        prediction = torch.argmax(activation_output, 1)

#         c = (prediction == labels).float().squeeze()
        a = activation_output[:,labels]   
        aopc_sum_K = aopc_sum_K + (a0 - a).mean().item()
        
        result.append(aopc_sum_K / (len(result)+1))
        
    return result

def aopc_one_image(net, R, input, labels, num_grid = (1,1,14,14)):
    result = []
    input = input.clone()
    
    # 1. if interpreter == LRP, it resize R to be 14*14
    grid_size = input.shape[2] // num_grid[2]
    if R.shape[2] != num_grid[2]:
        R = R.sum(dim=1, keepdim=True)
        R = F.avg_pool2d(torch.tensor(R), (grid_size, grid_size))
    R = R.cpu().detach().numpy()
        
    # 2. we multiply -1 to R, because argsort is increasing order.
    R_fl = - R.flatten()
    
    # first evaludation
    activation_output = net.prediction(input)
    prediction = torch.argmax(activation_output, 1)
#     c0 = (prediction == labels).float().squeeze()
    a0 = activation_output[:,labels]
    
    aopc_sum_K = 0
    result.append(aopc_sum_K)

    
    # 3. unfold
    input_unfolded = F.unfold(input, (grid_size, grid_size), stride=grid_size)
    
    for index in R_fl.argsort()[:100]:
        # gaussian
        input_unfolded[:,:,index] = torch.tensor(np.random.normal(loc=0.0, scale=0.3, size=input_unfolded[:,:,index].shape),dtype=torch.float32).cuda()
        input_folded = F.fold(input_unfolded, input.shape[2:], (grid_size, grid_size), stride=grid_size)
        
        activation_output = net.prediction(input_folded.cuda())
        prediction = torch.argmax(activation_output, 1)

#         c = (prediction == labels).float().squeeze()
        a = activation_output[:,labels] 
        aopc_sum_K = aopc_sum_K + (a0 - a.item())
        
        result.append(aopc_sum_K / (len(result)+1))
        
    return result
   

def target_quanti(interpreter, R_ori, R):
    
    rcorr_sign = []
    rcorr_rank = []

    if len(R.shape) == 4:
        R = R.sum(axis=1)
        R_ori = R_ori.sum(axis=1)

        
    rank_corr_sign = quanti_metric(R_ori, R, interpreter, 'sign')
    rank_corr_rank = quanti_metric(R_ori, R, interpreter, 'rank')

        
    rcorr_sign.append(rank_corr_sign)
    rcorr_rank.append(rank_corr_rank)
            
    
    rcorr_sign_mean = np.array(rcorr_sign).mean(axis=0)
    rcorr_rank_mean = np.array(rcorr_rank).mean(axis=0)

    
    return rcorr_rank_mean, rcorr_sign_mean
    
def passive_quanti(interpreter, R_ori, R):
    
    rcorr_sign = []
    rcorr_rank = []
    rcorr_frame = []

    if len(R.shape) == 4:
        R = R.sum(axis=1)
        R_ori = R_ori.sum(axis=1)

        
    rank_corr_sign = quanti_metric(R_ori, R, interpreter, 'sign')
    rank_corr_rank = quanti_metric(R_ori, R, interpreter, 'rank')
    rank_corr_frame = quanti_metric(R_ori, R, interpreter, 'frame')

        
    rcorr_sign.append(rank_corr_sign)
    rcorr_rank.append(rank_corr_rank)
    rcorr_frame.append(rank_corr_frame)
            
    
    rcorr_sign_mean = np.array(rcorr_sign).mean(axis=0)
    rcorr_rank_mean = np.array(rcorr_rank).mean(axis=0)
    rcorr_frame_mean = np.array(rcorr_frame).mean(axis=0)

    
    return rcorr_rank_mean, rcorr_sign_mean, rcorr_frame_mean    
    
def __target_quanti(net, to_loader, interpreter, R_ori, label_class):
    
    rcorr_sign = []
    rcorr_rank = []
    rcorr_frame = []
    
    for j, data in enumerate(to_loader):
            
        if j % 5 == 0:
            print(j)
        inputs, labels = data
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()+label_class
            
        activation_output = net.prediction(inputs)
        if interpreter == 'grad_cam':
            R = net.grad_cam(activation_output, labels, '34')
        elif interpreter == 'lrp34':
            R = net.lrp(activation_output, 0, labels=labels, target_layer='34')
        elif interpreter == 'lrp':
            R = net.lrp(activation_output, 0, labels=labels, target_layer=None)
        
        if len(R.shape) != 4:
            R = R.unsqueeze(1)
        R = R.sum(dim=1, keepdim=True)
        
        
        rank_corr_sign = quanti_metric(R_ori[j], R, interpreter, 'sign')
        rank_corr_rank = quanti_metric(R_ori[j], R, interpreter, 'rank')
        rank_corr_frame = 0
        
        rcorr_sign.append(rank_corr_sign)
        rcorr_rank.append(rank_corr_rank)
            
        
        if j % 10 == 9:
            break
    
    rcorr_sign_mean = np.array(rcorr_sign).mean(axis=0)
    rcorr_rank_mean = np.array(rcorr_rank).mean(axis=0)
    rcorr_frame_mean = rank_corr_frame
    
    return rcorr_rank_mean, rcorr_sign_mean, rcorr_frame_mean

    
def get_Rori(net, test_loader, interpreter, label_class=None):
    test_acc=0
    Rori = []
    for j, data in enumerate(test_loader):
        
        if j % 5 == 0:
            print(j)
        inputs, labels = data
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        if label_class is not None:
            labels = labels + label_class
            
        activation_output = net.prediction(inputs)
        if interpreter == 'grad_cam':
            R = net.grad_cam(activation_output, labels, '34')
        elif interpreter == 'lrp34':
            R = net.lrp(activation_output, 0, labels=labels, target_layer='34')
        elif interpreter == 'lrp':
            R = net.lrp(activation_output, 0, labels=labels, target_layer=None)
        elif interpreter == 'simple_grad':
            R = net.simple_grad(activation_output, labels, '0')
        
        if len(R.shape) != 4:
            R = R.unsqueeze(1)
        R = R.sum(dim=1, keepdim=True)   
        Rori.append(R)
        
        if j % 10 == 9:
            break

    return Rori

def get_aopc(net, test_loader, interpreter):
    test_acc=0
    result = []
    rcorr_sign = []
    rcorr_rank = []
    rcorr_frame = []
    for j, data in enumerate(test_loader):
        
        if j % 5 == 0:
            print(j)
        inputs, labels = data
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        activation_output = net.prediction(inputs)
        if interpreter == 'grad_cam':
            R = net.grad_cam(activation_output, labels, '34')
        elif interpreter == 'lrp34':
            R = net.lrp(activation_output, 0, labels=labels, target_layer='34')
        elif interpreter == 'lrp':
            R = net.lrp(activation_output, 0, labels=labels, target_layer=None)
        elif interpreter == 'simple_grad':
            R = net.simple_grad(activation_output, labels, '0')
       
          
        result.append(aopc_batch(net, R, inputs, labels))
        if len(R.shape) != 4:
            R = R.unsqueeze(1)
        R = R.sum(dim=1, keepdim=True)   
        
        if j % 10 == 9:
            break

    return np.array(result).mean(axis=0)


    
def __get_aopc_rcorr(net, test_loader, interpreter, R_ori):
    test_acc=0
    result = []
    rcorr_sign = []
    rcorr_rank = []
    rcorr_frame = []
    for j, data in enumerate(test_loader):
        
        if j % 5 == 0:
            print(j)
        inputs, labels = data
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        activation_output = net.prediction(inputs)
        if interpreter == 'grad_cam':
            R = net.grad_cam(activation_output, labels, '34')
        elif interpreter == 'lrp34':
            R = net.lrp(activation_output, 0, labels=labels, target_layer='34')
        elif interpreter == 'lrp':
            R = net.lrp(activation_output, 0, labels=labels, target_layer=None)
        elif interpreter == 'simple_grad':
            R = net.simple_grad(activation_output, labels, '0')
       
          
        result.append(aopc_batch(net, R, inputs, labels))
        if len(R.shape) != 4:
            R = R.unsqueeze(1)
        R = R.sum(dim=1, keepdim=True)   
        
        if R_ori is not None:
            rank_corr_sign = quanti_metric(R_ori[j], R, interpreter, 'sign')
            rank_corr_rank = quanti_metric(R_ori[j], R, interpreter, 'rank')
            rank_corr_frame = quanti_metric(R_ori[j], R, interpreter, 'frame')
            
            rcorr_sign.append(rank_corr_sign)
            rcorr_rank.append(rank_corr_rank)
            rcorr_frame.append(rank_corr_frame)
        
        
        if j % 10 == 9:
            break
    if R_ori is None:
        return np.array(result).mean(axis=0), 0
    
    res = np.array(result).mean(axis=0)
    rcorr_sign = np.array(rcorr_sign).mean(axis=0)
    rcorr_frame = np.array(rcorr_frame).mean(axis=0)
    rcorr_rank = np.array(rcorr_rank).mean(axis=0)
    
    return res, rcorr_rank, rcorr_sign, rcorr_frame

def get_accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#                 res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res


def test_accuracy(net, test_loader, num_data = 50000, label = 0):
    
    test_acc1=0.0
    test_acc5=0.0
    net.eval()
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            inputs, labels = data
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda() + label
            
            if j % 100 == 99:
                n = (1+j)*inputs.shape[0]
                print('j: %d\t Top1: %d/%d\t, %.4f%% \t Top5: %d/%d\t, %.4f%%' %(j, int(test_acc1), n, test_acc1/n*100, int(test_acc5), n, test_acc5/n*100))
            activation_output = net.prediction(inputs)
            acc1, acc5 = get_accuracy(activation_output, labels, topk=(1, 5))

            test_acc1 = test_acc1 + acc1.item()
            test_acc5 = test_acc5 + acc5.item()
            
#             # For debugging
#             n = (1+j)*inputs.shape[0]
#             prediction = torch.argmax(activation_output, 1)
#             print('prediction: ', prediction, ', labels: ', labels)
#             print('j: %d\t acc1: %d/%d\t, %.4f%% \t acc5: %d/%d\t, %.4f%%' %(j, int(test_acc1), n, test_acc1/n, int(test_acc5), n, test_acc5/n))

    return (test_acc1 / num_data), (test_acc5 / num_data)

#=======================================================================================================================
def classification_val_accuracy(net, test_loader, label = 0):
    
    loss_sum = 0.0
    test_acc1=0.0
    test_acc5=0.0
    net.eval()
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            inputs, labels = data
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda() + label
            
            if j % 100 == 99:
                n = (1+j)*inputs.shape[0]
                print('j: %d\t Top1: %d/%d\t, %.4f%% \t Top5: %d/%d\t, %.4f%%' %(j, int(test_acc1), n, test_acc1/n*100, int(test_acc5), n, test_acc5/n*100))
            activation_output = net.prediction(inputs)
            
            loss = net.class_loss(activation_output, labels)
            acc1 = get_accuracy(activation_output, labels)

            test_acc1 = test_acc1 + acc1.item()
            loss_sum += loss
            
#             # For debugging
#             n = (1+j)*inputs.shape[0]
#             prediction = torch.argmax(activation_output, 1)
#             print('prediction: ', prediction, ', labels: ', labels)
#             print('j: %d\t acc1: %d/%d\t, %.4f%% \t acc5: %d/%d\t, %.4f%%' %(j, int(test_acc1), n, test_acc1/n, int(test_acc5), n, test_acc5/n))

    return (loss_sum / len(test_loader.dataset)) (test_acc1 / len(test_loader.dataset))
#=======================================================================================================================

def visualize3(R_lrp, R_lrp34, R_grad, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    heatmaps_lrp = []
    heatmaps_lrp34 = []
    heatmaps_grad = []
    
    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)
    
#     images = images - images.min()
#     images = images / images.max()


    
    # 1. LRP
    for h, heat in enumerate(R_lrp):
        heat = heat.permute(1,2,0).detach().cpu().numpy()
        maps = render.heatmap(heat,reduce_axis=-1)
        heatmaps_lrp.append(maps)
        
    # 2. LRP34
    
    R_lrp34 = R_lrp34.squeeze(1).permute(1,2,0)
    R_lrp34 = R_lrp34.detach().cpu().detach().numpy()
    R_lrp34 = cv2.resize(R_lrp34, (224, 224))
    R_lrp34.reshape(input_shape[2], input_shape[3], input_shape[0])
    
    for i in range(input_shape[0]):
        heatmap = np.float32(cv2.applyColorMap(np.uint8((1-R_lrp34[:,:,i])*255), cv2.COLORMAP_JET))/255
        cam = heatmap + np.float32(images[i])
        cam = cam / np.max(cam)
        heatmaps_lrp34.append(cam)
    
    # 3. Grad_CAM
    
    R_grad = R_grad.squeeze(1).permute(1,2,0)
    R_grad = R_grad.cpu().detach().numpy()
    R_grad = cv2.resize(R_grad, (224, 224))
    R_grad.reshape(input_shape[2], input_shape[3], input_shape[0])
    
    for i in range(input_shape[0]):
        heatmap = np.float32(cv2.applyColorMap(np.uint8((1-R_grad[:,:,i])*255), cv2.COLORMAP_JET))/255
        cam = heatmap + np.float32(images[i])
        cam = cam / np.max(cam)
        heatmaps_grad.append(cam)
        
    
    R_lrp = np.array(heatmaps_lrp,dtype=np.float32)
    R_lrp34 = np.array(heatmaps_lrp34,dtype=np.float32)
    R_grad = np.array(heatmaps_grad,dtype=np.float32)
    
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    for i in range(l):
        file_name = 'lrp_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        prediction_dic['lrp_epoch' + str(epoch) + '_no-' + str(i+l*j)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R_lrp[i], plugin='pil')
        
    for i in range(l):
        file_name = 'lrp34_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        prediction_dic['lrp34_epoch' + str(epoch) + '_no-' + str(i+l*j)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R_lrp34[i], plugin='pil')
        
    for i in range(l):
        file_name = 'grad_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        prediction_dic['grad_epoch' + str(epoch) + '_no-' + str(i+l*j)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R_grad[i], plugin='pil')
        
    for i in range(l):
        file_name = 'ori_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        prediction_dic['ori_epoch' + str(epoch) + '_no-' + str(i+l*j)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, images[i], plugin='pil')
    
    prediction_dic = sorted(prediction_dic.items(), key=operator.itemgetter(0))
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic) 
    #print('prediction_dic  : ',prediction_dic)

def lrp_visualize(R_lrp, gamma = args.gamma):
    heatmaps_lrp = []
    for h, heat in enumerate(R_lrp):
        heat = heat.permute(1,2,0).detach().cpu().numpy()
        maps = render.heatmap(heat,reduce_axis=-1, gamma_ = gamma)
        heatmaps_lrp.append(maps)
    return heatmaps_lrp    

def grad_visualize(R_grad, images):
    R_grad = R_grad.squeeze(1).permute(1,2,0)
    R_grad = R_grad.cpu().detach().numpy()
    R_grad = cv2.resize(R_grad, (224, 224))
    R_grad.reshape(224, 224, images.shape[0])
    heatmaps_grad = []
    
    for i in range(images.shape[0]):
        heatmap = np.float32(cv2.applyColorMap(np.uint8((1-R_grad[:,:,i])*255), cv2.COLORMAP_JET))/255
        cam = heatmap + np.float32(images[i])
        cam = cam / np.max(cam)
        heatmaps_grad.append(cam)
    return heatmaps_grad

def save_visualize(l, R, pre_name, epoch, j, image_path):
    for i in range(l):
        file_name = str(pre_name) + '_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        path = os.path.join(image_path+args.img_name+str('/'), file_name)##
        imsave(path, R[i], plugin='pil')

def visualize_bitargeted(R_lrp_to, R_lrp_from, R_lrp34_to, R_lrp34_from, R_grad_to, R_grad_from, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

#     heatmaps_lrp_to = []
#     heatmaps_lrp_from = []
    
#     heatmaps_lrp34_to = []
#     heatmaps_lrp34_from = []
    
#     heatmaps_grad_to = []
#     heatmaps_grad_from = []
    
    
    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)
    
#     images = images - images.min()
#     images = images / images.max()

    # 0. Ready for save images
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    
    
    # 1. LRP
    heatmaps_lrp_to = lrp_visualize(R_lrp_to)
    heatmaps_lrp_from = lrp_visualize(R_lrp_from)
    
    R_lrp_to = np.array(heatmaps_lrp_to,dtype=np.float32)
    R_lrp_from = np.array(heatmaps_lrp_from,dtype=np.float32)
    
    save_visualize(l, R_lrp_to, 'R_lrp_to', epoch, j, image_path)
    save_visualize(l, R_lrp_from, 'R_lrp_from', epoch, j, image_path)

        
    # 2. LRP34
    
    heatmaps_lrp34_to = grad_visualize(R_lrp34_to, images)
    heatmaps_lrp34_from = grad_visualize(R_lrp34_from, images)
    
    R_lrp34_to = np.array(heatmaps_lrp34_to,dtype=np.float32)
    R_lrp34_from = np.array(heatmaps_lrp34_from,dtype=np.float32)
    
    save_visualize(l, R_lrp34_to, 'R_lrp34_to', epoch, j, image_path)
    save_visualize(l, R_lrp34_from, 'R_lrp34_from', epoch, j, image_path)
    
    # 3. Grad_CAM
    
    heatmaps_grad_to = grad_visualize(R_grad_to, images)
    heatmaps_grad_from = grad_visualize(R_grad_from, images)
    
    R_grad_to = np.array(heatmaps_grad_to,dtype=np.float32)
    R_grad_from = np.array(heatmaps_grad_from,dtype=np.float32)
    
    save_visualize(l, R_grad_to, 'R_grad_to', epoch, j, image_path)
    save_visualize(l, R_grad_from, 'R_grad_from', epoch, j, image_path)
        
    # 4. save original image
    save_visualize(l, images, 'ori', epoch, j, image_path)
    
    # 5. save prediction
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic)


def visualize4(R_lrp, R_lrp34, R_grad, R_simple, R_simple34, R_smooth, R_ig, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    
    
    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)
    
#     images = images - images.min()
#     images = images / images.max()

    # 0. Ready for save images
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    
    
    # 1. LRP
    heatmaps_lrp= lrp_visualize(R_lrp)
    R_lrp = np.array(heatmaps_lrp,dtype=np.float32)
    save_visualize(l, R_lrp, 'R_lrp', epoch, j, image_path)

    # 2. LRP34
    
    heatmaps_lrp34= grad_visualize(R_lrp34, images)
    R_lrp34 = np.array(heatmaps_lrp34,dtype=np.float32)
    save_visualize(l, R_lrp34, 'R_lrp34', epoch, j, image_path)
    
    # 3. Grad_CAM
    
    heatmaps_grad= grad_visualize(R_grad, images)
    R_grad = np.array(heatmaps_grad,dtype=np.float32)
    save_visualize(l, R_grad, 'R_grad', epoch, j, image_path)
    
    # 4. Simple gradient 34
    
    heatmaps_simple34= grad_visualize(R_simple34, images)
    R_simple34 = np.array(heatmaps_simple34,dtype=np.float32)
    save_visualize(l, R_simple34, 'R_simple34', epoch, j, image_path)
    
    # 5. Simple gradient 
    heatmaps_simple= lrp_visualize(R_simple, gamma = 1.4)
    R_simple = np.array(heatmaps_simple,dtype=np.float32)
    save_visualize(l, R_simple, 'R_simple', epoch, j, image_path)
    
    # 6. Smooth gradient
    
    heatmaps_smooth= lrp_visualize(R_smooth)
    R_smooth = np.array(heatmaps_smooth,dtype=np.float32)
    save_visualize(l, R_smooth, 'R_smooth', epoch, j, image_path)
        
    # 7. integrated gradient
    
    heatmaps_ig= lrp_visualize(R_ig)
    R_ig = np.array(heatmaps_ig,dtype=np.float32)
    save_visualize(l, R_ig, 'R_ig', epoch, j, image_path)
        
    # 8. save original image
    save_visualize(l, images, 'ori', epoch, j, image_path)
    
    # 9. save prediction
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic)    
    
    

def visualize5(R_34s, R_inputs, R_34s_key, R_inputs_key, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)
    
#     images = images - images.min()
#     images = images / images.max()

    # 0. Ready for save images
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    
    
    # 1. input friends
    for key, R_input in zip(R_inputs_key, R_inputs):
        heatmaps_R= lrp_visualize(R_input, gamma = args.gamma)
        R_input = np.array(heatmaps_R,dtype=np.float32)
        save_visualize(l, R_input, key, epoch, j, image_path)

    # 1. 34 friends
    for key, R_34 in zip(R_34s_key, R_34s):
        print(key, R_34.shape)
        if R_34.shape[0] == 1:
            R_34 = R_34.detach().cpu().numpy()
            R_34 = np.concatenate((R_34, R_34), axis=0)
            R_34 = torch.tensor(R_34, dtype=torch.float32)
            
        if images.shape[0]==1:
            images = np.concatenate((images, images), axis=0)
            
        print(R_34.shape)
        heatmaps_R34= grad_visualize(R_34, images)
        R_34 = np.array(heatmaps_R34,dtype=np.float32)
        save_visualize(l, R_34, key, epoch, j, image_path)
    
    # 8. save original image
    save_visualize(l, images, 'ori', epoch, j, image_path)
    
    # 9. save prediction
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic)   
    return





    
    
def visualize(relevances, epoch, prediction,images_tensor=None, shape=None, reduce_op='sum', reduce_axis=-1):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass


    n, dim, h, w  = relevances.shape
    heatmaps = []
    oris = []
    # import pdb;pdb.set_trace()
    if args.model in ['CNN8', 'VGG19'] :
        images_tensor = images_tensor.reshape([args.batch_size_test, 3, 224, 224])
    else:
        images_tensor = images_tensor.reshape([args.batch_size_test, 1, 28, 28])
    if images_tensor is not None:
        assert relevances.shape == images_tensor.shape, 'Relevances shape != Images shape'
    for h, heat in enumerate(relevances):
        if images_tensor is not None:
            input_image = images_tensor[h].permute(1,2,0).cpu().numpy()
            heat = heat.permute(1,2,0).cpu().numpy()
            print(3, heat.shape)
            maps = render.hm_to_rgb(heat, input_image, scaling=3, sigma=2, shape=shape, reduce_op=reduce_op, reduce_axis=reduce_axis)
            print(4, maps.shape)
#             ori = input_image - input_image.min()
#             ori = ori / ori.max()
            
        else:
            heat = heat.cpu().numpy()
            maps = render.hm_to_rgb(heat, scaling=3, sigma=2, shape=shape, reduce_op=reduce_op, reduce_axis=reduce_axis)
#             ori = input_image - input_image.min()
#             ori = ori / ori.max()
        heatmaps.append(maps)
#         oris.append(ori)
    R = np.array(heatmaps,dtype=np.float32)
#     ori = np.array(oris,dtype=np.float32)

    # save R to img file
    # print('---- R shape---',R.shape)
    # print('--- img saved in utils.py in def visualize. change file name every time----')
    # imsave('../results/test/180225_lrp_module.png', R[1])
    # np.save('../results/test/180225_R.npy',R)
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    for i in range(min(args.batch_size_test, args.num_visualize_plot)):
        #path = os.path.join(image_path, 'results{}.png'.format(i))
        #file_name = img_name + str(i) + '.png'
        file_name = '_epoch' + str(epoch) + '_no-' + str(i) + '.png'
        prediction_dic['_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R[i], plugin='pil')
        
#     for i in range(min(args.batch_size_test, args.num_visualize_plot)):
#         #path = os.path.join(image_path, 'results{}.png'.format(i))
#         #file_name = img_name + str(i) + '.png'
#         file_name = 'ori_epoch' + str(epoch) + '_no-' + str(i) + '.png'
#         prediction_dic['ori_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
#         path = os.path.join(image_path+img_name+str('/'), file_name)##
#         imsave(path, ori[i], plugin='pil')    
        
    prediction_dic = sorted(prediction_dic.items(), key=operator.itemgetter(0))
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic) 
    #print('prediction_dic  : ',prediction_dic)
    
def visualize2(relevances, epoch, prediction,images_tensor=None, shape=None):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    print('relevance shape', relevances.shape)
    n, dim, h, w  = relevances.shape
    heatmaps = []
    oris = []
    # import pdb;pdb.set_trace()
    if args.model in ['CNN8', 'VGG19'] :
        images_tensor = images_tensor.reshape([args.batch_size_test, 3, 224, 224])
    else:
        images_tensor = images_tensor.reshape([args.batch_size_test, 1, 28, 28])
    if images_tensor is not None:
        assert relevances.shape == images_tensor.shape, 'Relevances shape != Images shape'
#     heatmaps = render.heatmap(relevances.cpu().numpy(),reduce_axis=1)
#     oris = render.image(images_tensor.cpu().numpy())

#     ori = oris - oris.min(axis=0)
#     ori = ori / ori.max(axis=0)

    for h, heat in enumerate(relevances):
        if images_tensor is not None:
            print(2, heat.shape)
            input_image = images_tensor[h].permute(1,2,0).detach().cpu().numpy()
            heat = heat.permute(1,2,0).detach().cpu().numpy()
            #maps = render.hm_to_rgb(heat, input_image, scaling=3, sigma=2, shape=shape)
            maps = render.heatmap(heat,reduce_axis=-1)
            #ori = render.image(input_image)
            ori = input_image - input_image.min()
            ori = ori / ori.max()
        else:
            heat = heat.cpu().numpy()
            #maps = render.hm_to_rgb(heat, scaling=3, sigma=2, shape=shape)
            maps = render.heatmap(heat,reduce_axis=0)
            #ori = render.image(input_image)
            ori = input_image - input_image.min()
            ori = ori / ori.max()
            
            
        heatmaps.append(maps)
        oris.append(ori)
        
    R = np.array(heatmaps,dtype=np.float32)
    #R = np.transpose(R,(0,3,2,1))
    ori = np.array(oris,dtype=np.float32)
    #ori= np.transpose(ori,(0,3,2,1))

    # save R to img file
    # print('---- R shape---',R.shape)
    # print('--- img saved in utils.py in def visualize. change file name every time----')
    # imsave('../results/test/180225_lrp_module.png', R[1])
    # np.save('../results/test/180225_R.npy',R)
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    for i in range(min(args.batch_size_test, args.num_visualize_plot)):
        #path = os.path.join(image_path, 'results{}.png'.format(i))
        #file_name = img_name + str(i) + '.png'
        file_name = 'lrp_epoch' + str(epoch) + '_no-' + str(i) + '.png'
        prediction_dic['lrp_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R[i], plugin='pil')
        
    for i in range(min(args.batch_size_test, args.num_visualize_plot)):
        #path = os.path.join(image_path, 'results{}.png'.format(i))
        #file_name = img_name + str(i) + '.png'
        file_name = 'ori_epoch' + str(epoch) + '_no-' + str(i) + '.png'
        prediction_dic['ori_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, ori[i], plugin='pil')    
    
    prediction_dic = sorted(prediction_dic.items(), key=operator.itemgetter(0))
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic) 
    #print('prediction_dic  : ',prediction_dic)

    
    
def visualize_grad_cam(mask, epoch, prediction,images_tensor):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    mask = mask.squeeze(1).permute(1,2,0)
    print('mask shape', mask.shape)
    mask = mask.cpu().detach().numpy()
    
    # n*k*w*h -> n*w*h*k
    images = images_tensor.permute(0,2,3,1).cpu().detach().numpy()
#     images = images - images.min(axis=(1,2,3), keepdims=True)
#     images = images / images.max(axis=(1,2,3), keepdims=True)
    
#     images = images - images.min()
#     images = images / images.max()
    
    images -= np.min(images, axis=(1,2,3), keepdims=True)
    images = np.minimum(images, 255)
    
    input_shape = images_tensor.shape
    if args.model == 'VGG19':
        mask = cv2.resize(mask, (224, 224))
    elif args.model == 'CNN8':
        mask = cv2.resize(mask, (32, 32))
    mask.reshape(input_shape[2], input_shape[3], input_shape[0])
    
    cam_list = []
    
    # Using for-loop is necessary because cv2.applyColorMap does not offer multiple input images.
    for i in range(input_shape[0]):
        heatmap = np.float32(cv2.applyColorMap(np.uint8((1-mask[:,:,i])*255), cv2.COLORMAP_JET))/255
        cam = heatmap + np.float32(images[i])
        cam = cam / np.max(cam)
        cam_list.append(cam)
    
    R = np.array(cam_list)
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    print(len(prediction))
    
    for i in range(min(args.batch_size_test, args.num_visualize_plot, len(prediction))):
        #path = os.path.join(image_path, 'results{}.png'.format(i))
        #file_name = img_name + str(i) + '.png'
        file_name = 'grad_cam_epoch' + str(epoch) + '_no-' + str(i) + '.png'
        prediction_dic['grad_cam_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
        path = os.path.join(image_path+img_name+str('/'), file_name)##
        imsave(path, R[i], plugin='pil')
        
#     for i in range(args.num_visualize_plot):
#         #path = os.path.join(image_path, 'results{}.png'.format(i))
#         #file_name = img_name + str(i) + '.png'
#         file_name = 'original_epoch' + str(epoch) + '_no-' + str(i) + '.png'
#         prediction_dic['original_epoch' + str(epoch) + '_no-' + str(i)] = prediction[i]
#         path = os.path.join(image_path+img_name+str('/'), file_name)##
#         imsave(path, images[i], plugin='pil')
    
    prediction_dic = sorted(prediction_dic.items(), key=operator.itemgetter(0))
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic) 
    #print('prediction_dic  : ',prediction_dic)
    
    

def load_data(epoch, file_path = args.img_dir):
    data = []
    print(file_path+args.img_name)
    infiles = glob.glob(os.path.join(file_path+args.img_name+str('/'), str('*epoch'+str(epoch)+'*')))##
    infiles.sort()
    for inf in infiles: 
        file = imread(inf)        
        data.append(file)

    return data



def visdom_plot(viz, epoch, data_path, win, img_number=None):
    images = load_data(epoch, file_path=data_path)
    images = np.array(images)
    if images is None:
        return win
    if img_number == None:
        images = np.asarray(images)
        print('oooo',images.shape)
        images = torch.from_numpy(images).permute(0, 3,1,2)
        return viz.images(images, nrow=8)
    else:
        print(images.shape)
        images = np.asarray(images)[img_number]
        images = np.expand_dims(images, axis=0)
        images = torch.from_numpy(images).permute(0, 3,1,2)
        return viz.images(images)
    # Show it in visdom
    #images = torch.from_numpy(images).permute(0, 3,1,2)
    #return viz.images(images[:args.num_visualize_plot], nrow=5)#args.vis_img_num
    #return viz.images(images, nrow=10)



def visdom_line(viz,data_path,win, line, line_name):
    return viz.line(X=np.arange(len(line)), Y=line, win=win, opts=dict(title=line_name))

def visdom_line_original(viz, data_path, win1,win2,win3, loss_line, acc_line,num_zero):
    
    return viz.line(
        X=np.arange(len(loss_line)),
        Y=loss_line,
        win=win1,
        opts=dict(title='loss function')), viz.line(
        X=np.arange(len(loss_line)),
        Y=acc_line,
        win=win2,
        opts=dict(title='acc function')), viz.line(
        X=np.arange(len(num_zero)),
        Y=num_zero,
        win=win3,
        opts=dict(title='num_zero'))

def visdom_R_sort(viz,data_path,win,R,epoch,img_index):
    #R = abs(R)
    #R.sort()
    return viz.line(
        X=np.arange(len(R)),
        Y=R,
        win=win,opts=dict(title='R_sort_'+str(epoch)+'_img_index_'+str(img_index)))
        


class logger(object):
    def __init__(self, file_name='mnist_result', resume=False, path=args.log_dir, data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))