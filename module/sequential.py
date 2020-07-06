import os
import sys
sys.path.append('../utils')
import general

import h5py
import warnings
from collections import OrderedDict, Iterable, Mapping
from itertools import islice
import operator
import scipy.stats as sc
import h5py
import torch
from module.module import Module

import numpy as np
import time
from module.arguments import get_args
args = get_args()
import pickle


class Container(Module):

    def __init__(self, **kwargs):
        super(Container, self).__init__()
        # DeprecationWarning is ignored by default <sigh>
        warnings.warn("nn.Container is deprecated. All of it's functionality "
                      "is now implemented in nn.Module. Subclass that instead.")
        for key, value in kwargs.items():
            self.add_module(key, value)


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
                
    """180913 15:11 becuase of error, changed!, main.py def weight init""" 
    def reset_parameters(self):
        pass
    
    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys


    
    def prediction(self, input):
        # classic forward
        activation_output = input
        for module in self._modules.values():
            activation_output = module.forward(activation_output)
        return activation_output
    
    
    def feature_extraction(self, input):
        # classic forward
        activation_output = input
        for key, module in self._modules.items():
            activation_output = module.forward(activation_output)
            if key == '36':
                break
        return activation_output


    def class_loss(self, activation_output, labels):
        criterion = torch.nn.CrossEntropyLoss()
        diff = criterion(activation_output, labels)
        loss = torch.mean(diff)
        return loss
    
    def variance(self, LRP):
        entropy = torch.var(LRP.reshape(LRP.shape[0],-1),dim=1).mean()
        return entropy
    
    def normalize(self, LRP):
        LRP = LRP - LRP.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].reshape(-1,1,1,1)
        LRP_shape = LRP.shape
#         LRP = torch.nn.functional.normalize(LRP.reshape(LRP_shape[0],-1), p=1, dim=1).reshape(LRP.shape)
        LRP = LRP/(LRP.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].reshape(-1,1,1,1) + 1e-8)
        return LRP
        

    ##############################################################################################
    def mask(self, LRP, mask): 
        if LRP[0,:,:,:].shape != mask.shape:
            raise Exception('The shape of LRP and mask are different!!')
            
        mask = mask.unsqueeze(0)
        LRP_within_the_mask = LRP * mask
        loss = torch.nn.functional.mse_loss(LRP, LRP_within_the_mask)
        return loss
    
    def mask_LRPxmask(self, LRP, mask): 
        if LRP[0,:,:,:].shape != mask.shape:
            raise Exception('The shape of LRP and mask are different!!')
            
        mask = torch.stack([mask for i in range(LRP.shape[0])], dim=0)
        LRP_within_the_mask = LRP * mask
        loss = torch.nn.functional.mse_loss(mask, LRP_within_the_mask)
        return loss
    
    def mask_LRP(self, LRP, mask):
        if LRP[0,:,:,:].shape != mask.shape:
            raise Exception('The shape of LRP and mask are different!!')
            
        mask = torch.stack([mask for i in range(LRP.shape[0])], dim=0)
        loss = torch.nn.functional.mse_loss(LRP, mask)
        return loss
    
    def mask_LRP_no_stacking(self, LRP, mask, label, args):
        if LRP.shape != mask.shape:
            print('LRP.shape : {} / mask.shape : {}'.format(LRP.shape, mask.shape))
            raise Exception('The shape of LRP and mask are different!!')
        
        general.save_hdf5(os.path.join(args.result_path, 'mask_check.hdf5'), 'mask', mask.cpu().detach().numpy())
        general.save_hdf5(os.path.join(args.result_path, 'mask_check.hdf5'), 'LRP', LRP.cpu().detach().numpy())
        
        loss = torch.nn.functional.mse_loss(LRP, mask)
        return loss
    
    def mask_LRP_seg(self, LRP, mask, loss_filter, args):
        if LRP.shape != mask.shape:
            raise Exception('The shape of LRP and mask are different!! | LRP.shape : {} / mask.shape : {}'.format(LRP.shape, mask.shape))
        new_mask = mask * loss_filter.reshape(loss_filter.shape[0], 1, 1, 1).detach() \
                   + LRP * (1-loss_filter.reshape(loss_filter.shape[0], 1, 1, 1))
#         general.save_hdf5('data_test.hdf5', 'new_mask', new_mask.cpu().detach().numpy())
#         print('LRP :', LRP)
#         print('new_mask :', new_mask)
        loss = torch.nn.functional.mse_loss(LRP, new_mask)
        return loss
    ##############################################################################################
# #     def __call__(self, x):
#         target_activations = []
#         for name, module in self.model._modules.items():
#             if module == self.feature_module:
#                 target_activations, x = self.feature_extractor(x)
#             elif "avgpool" in name.lower():
#                 x = module(x)
#                 x = x.view(x.size(0),-1)
#             else:
#                 x = module(x)
        
#         return target_activations, x
    
#     for key, module in reversed(list(self._modules.items())):
#         requires_activation = (key == args.lrp_target_layer)
# #                     print('=====',target_layer, key, module, requires_activation)
#         # x: feature map, dx: dL/dx
#         R = module.lrp(R, labels, args.r_method, 1e-8)
#         if requires_activation:
#             break
    
        
    def forward(self, input, labels, mask, use_mask, args=None):
        # classic forward
        ############################################################### $$$$
        activation_output = input
        target_feature = None
        for key, module in list(self._modules.items()):
            activation_output = module.forward(activation_output)

#             if int(key) == (int(args.lrp_target_layer)-1):
#                 target_feature = activation_output
            
            if args.check_activ_output_shape:
                print('module : {} / output_shape : {}'.format(module, activation_output.shape))
        
        if args.check_activ_output_shape:
            print('sleep...')
            time.sleep(1000)
        ###############################################################
        ################################################
        if args.interpreter == 'None' or args.lambda_for_final == 0:
#             print('!!!!!args.interpreter is None!!!!!')
            return activation_output, torch.tensor(0).type(torch.FloatTensor).cuda(args.gpu), None
        ################################################


        ############################
        ###  Interpretation 선택  ###
        ############################
        LRP = None
        if args.no_lambda_for_each_layer == True:
            if args.interpreter == 'lrp':
                if args.label_oppo_status:
                    labels = 1 - labels
                LRP = self.lrp(activation_output, labels, args)
            
                if args.label_oppo_status:
                    labels = 1 - labels
            elif args.interpreter == 'grad_cam':
                ######################################################################
#                 # .retain_grad()는 non-leaf tensor에 .grad를 만들어준다.
#                 target_feature.retain_grad()
#                 print('self :', self)
#                 LRP = self.grad_cam(activation_output, labels, target_feature, self)
                ######################################################################
    
                LRP = self.grad_cam(activation_output, labels, args)
            
#                 elif args.arch == 'CNN8':
#                     LRP = self.grad_cam(activation_output, labels, '17')
            
#             elif args.interpreter == 'simple_grad':
#                 LRP = self.simple_grad(activation_output, labels, args.lrp_target_layer)
                
#             elif args.interpreter == 'smooth_grad':
#                 LRP = self.smooth_grad(net, input, labels, args.lrp_target_layer)
                
#             elif args.interpreter == 'integrated_grad':
#                 LRP = self.integrated_grad(net, input, labels, args.lrp_target_layer)
            else:
                print('wrong interpreter!!',tt)
                

            
            
            # label이 tumor(1)인 것 중에 pred가 맞췄을 때를 1로 나머지는 0으로 표시해 true_pos에 담는다(제발 조심)
            pred = torch.argmax(activation_output, dim=1)
            assert args.loss_filter in ['true_pos', 'pos', 'all']
            if args.loss_filter == 'true_pos':
                loss_filter = labels.type(torch.cuda.FloatTensor) * pred.type(torch.cuda.FloatTensor) * use_mask.type(torch.cuda.FloatTensor)
            elif args.loss_filter == 'pos':
                loss_filter = labels.type(torch.cuda.FloatTensor) * use_mask.type(torch.cuda.FloatTensor)
            elif args.loss_filter == 'all':
                loss_filter = use_mask.type(torch.cuda.FloatTensor)
        

            if args.loss_type == 'mask_LRP_seg':
                lrp_loss = self.mask_LRP_seg(LRP, mask, loss_filter, args)
            elif args.loss_type == 'None':
                lrp_loss = torch.tensor([0])
            else:
                print('wrong loss type!!',tt)
                

        ###############################################
        return activation_output, lrp_loss, LRP
        ###############################################
    
    
    
            
            
#     #181210 added
#     def grad_cam(self, output, labels, target_feature, model):
#         label_score = output[torch.arange(output.shape[0]), labels].sum() # output : (1, 1000)
# #         print('label_score :', label_score)
# #         target_grad = target_feature.grad # grads_val = (1, 1024, 14, 14)
# #         if target_grad is None:
# #             print('target_grad 1 :', target_grad)
# #         else:
# #             print('target_grad 1 :', target_grad[0, 0, :5, :5])
#         model.zero_grad()
    
#         ################################################################################
# #         label_score.backward(retain_graph=True, create_graph=True)
# #         target_grad = target_feature.grad # grads_val = (1, 1024, 14, 14)
# #         print('target_grad 2 :', target_grad[0, 0, :5, :5])

#         target_grad = torch.autograd.grad(label_score, target_feature, retain_graph=True, create_graph=True)[0]
# #         print('target_grad :', target_grad.shape)
#         ################################################################################
    
#         weights = torch.mean(target_grad, axis=(2, 3), keepdims=True)

#         cam = target_feature * weights
#         cam = torch.sum(cam, dim=1, keepdim=True)
        
# #         cam = torch.nn.functional.threshold(cam, threshold=0, value=0)
#         cam = torch.nn.functional.relu(cam)
#         cam_min = torch.min(torch.min(cam, 2, keepdim=True)[0], 3, keepdim=True)[0]
#         cam = cam - cam_min
#         cam_max = torch.max(torch.max(cam, 2, keepdim=True)[0], 3, keepdim=True)[0]
#         cam = cam / cam_max
        
#         return cam
    
    
    def grad_cam(self, output, labels, args, by_label=True):
#         if args.arch == 'CNN8':
#             num_target = 10
            
        eye = torch.eye(args.num_classes, dtype=torch.float32)
        eye = eye.cuda()
        if by_label:
            dx = eye[labels]
#         else:
#             dx = eye[torch.argmax(output, dim=1)]
        
        dx = torch.ones_like(output)*dx

        for key, module in reversed(list(self._modules.items())):
            requires_activation = (key == args.lrp_target_layer)
            
            # x: feature map, dx: dL/dx
            dx, x = module.grad_cam(dx, requires_activation)
            if requires_activation:
                break
                
        weights = dx.mean(3, keepdim = True)
        weights = weights.mean(2, keepdim = True)
        
#         with h5py.File('Results/res.h5', 'w') as f:
#             f['weights'] = weights.cpu().detach().numpy()
#             f['x'] = x.cpu().detach().numpy()
        
        cam = torch.sum(x*weights , dim=1, keepdim=True)
        cam = torch.nn.functional.threshold(cam, threshold=0, value=0)
        
        cam_max,_ = cam.max(1, keepdim=True)
        cam_max,_ = cam_max.max(2, keepdim=True)
        cam_min,_ = cam.min(1, keepdim=True)
        cam_min,_ = cam_min.min(2, keepdim=True)
        

        cam = cam / (cam_max + 1e-8)
        
        
        #cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)  # Normalize between 0-1

        return cam
    

    def lrp(self, R, labels, args):
            
        if args.no_lambda_for_each_layer == True:
            res_list = []
#             print('target_layer :', target_layer, '!%@^$%#&^$@^#%$^%#&^$*&%#&$@^%&#^*$&%#&$', type(target_layer))
            if args.lrp_target_layer == None: ###$$$%%%^^^&&&
                for module in reversed(list(self.children())):
                    res_list.append(R.detach().cpu().numpy())
                    R = module.lrp(R, labels, args.r_method, 1e-8)
                return R
        
            else:
                ### 190101 target layer #####
                for key, module in reversed(list(self._modules.items())):
                    requires_activation = (key == args.lrp_target_layer)
                    R = module.lrp(R, labels, args.r_method, 1e-8)
#                     print('=====',args.lrp_target_layer, key, module, R.shape, requires_activation)
                    if requires_activation:
                        break
                
                if args.R_process == 'thr_maxmin_norm':
#                     print('thr_maxmin_norm')
                    # IoU를 양수 음수 모두 포함해서 max로만 나눠서 normalization을 해봤을 때 충분히 결과가 좋다면 양수,음수 모두 표현 가능
                    R = torch.sum(R, dim=1, keepdim=True)
#                     print('after sum R : {}'.format(R.shape))
                    R = torch.nn.functional.threshold(R, threshold=0, value=0)
#                     print('after threshold R : {}'.format(R.shape))
                    R_max,_ = R.max(2, keepdim=True)
                    R_max,_ = R_max.max(3, keepdim=True)
                    R_min,_ = R.min(2, keepdim=True)
                    R_min,_ = R_min.min(3, keepdim=True)
#                     print('R_max : {}, R_min : {}'.format(R_max, R_min))

                    R = (R - R_min) / (R_max - R_min + 1e-8)
                    
                elif args.R_process == 'thr_max_norm':
#                     print('thr_max_norm')
                    R = torch.sum(R, dim=1, keepdim=True)
#                     print('after sum R : {}'.format(R.shape))
                    R = torch.nn.functional.threshold(R, threshold=0, value=0)
#                     print('after threshold R : {}'.format(R.shape))
                    R_max,_ = R.max(2, keepdim=True)
                    R_max,_ = R_max.max(3, keepdim=True)
#                     print('R_max : {}'.format(R_max))

                    R = R / (R_max + 1e-8)
                    
                elif args.R_process == 'max_norm':
#                     print('max_norm')
                    R = torch.sum(R, dim=1, keepdim=True)
#                     print('after sum R : {}'.format(R.shape))
                    R_abs = torch.abs(R)
#                     print('R_abs : {}'.format(R_abs.shape))
                    R_max,_ = R_abs.max(2, keepdim=True)
                    R_max,_ = R_max.max(3, keepdim=True)

                    R = R / (R_max + 1e-8)
                
                elif args.R_process is None:
#                     print('None')
                    R = torch.sum(R, dim=1, keepdim=True)
                else:
                    raise Exception('Not right args.R_process : {}'.format(args.R_process))
            
#             if args.r_method == 'new_composite':
#                 net.zero_grad()
            
            
            return R
            
            
                
#         if args.no_lambda_for_each_layer == False:
#             lrp_loss_added = torch.cuda.DoubleTensor(np.zeros([1]))
#             lambda_dic = torch.cuda.FloatTensor(lambda_dic)

#             i = len(lambda_dic)-1
            
#             for module in reversed(list(self.children())):
#                 R = module.lrp(R, labels, lrp_var, param)
#                 ld = torch.tensor(lambda_dic[i], dtype=torch.float32).cuda()
                
#                 if args.loss_type == 'absR':
#                     lrp_loss_added = lrp_loss_added + (ld * torch.mean(torch.abs(R)))
                
#                 elif args.loss_type == 'uniformR_abs':      
#                     k = (torch.abs(R)/torch.sum(torch.abs(R)))
#                     k = k + 1e-8 * torch.ones_like(k, dtype = torch.float32)
#                     S = (k * torch.log(k)).sum()
#                     lrp_loss_added = lrp_loss_added + (ld * S)
                
#                 elif args.loss_type == 'uniformR_positive':       
#                     LRP_p = R - torch.min(R)
#                     p = LRP_p / torch.sum(LRP_p)
#                     p = p + 1e-8 * torch.ones_like(p, dtype = torch.float32)
#                     sum_p = (p * torch.log(p)).sum()
#                     lrp_loss_added = lrp_loss_added + (ld * sum_p)                  
                
#                 #elif args.loss_type == 'uniformR_variance':
                    
                    
#                 i = i-1

#             return R, lrp_loss_added

    # ------------------------------------------------------
    """
    lrp_gradients
    """

    def lrp_gradients(self, g_a_lrp):

        # save g w/a lrp in the dic
        g_w_lrp_dic = {}
        g_a_lrp_dic = {}
        layer_num = 1

        for module in self._modules.values()[::-1]:
            g_w_lrp, g_a_lrp, weights_var = module.lrp_gradients(g_a_lrp)
            g_w_lrp_dic[weights_var] = g_w_lrp
            g_a_lrp_dic[layer_num] = g_a_lrp
            layer_num = layer_num + 1

        # last g_a_lrp = "lrp_flow_loss"
        return g_a_lrp, g_w_lrp_dic, g_a_lrp_dic

    """
    original_gradients
    """

    def ori_gradients(self, total_loss, g_a_lrp_dic):
        layer_num = 1
        g_w_ori_dic = {}
        loss = total_loss

        for module in self._modules.values()[::-1]:
            g_w_ori, g_a_ori, weights_var = module.ori_gradients(loss)
            g_w_ori_dic[weights_var] = g_w_ori
            loss = g_a_lrp_dic[layer_num] + g_a_ori
            layer_num = layer_num + 1

        return g_w_ori_dic

    # ------------------------------------------------------

    def lrp_layerwise(self, m, R, lrp_var=None, param=None):
        R = m.lrp(R, lrp_var, param)
        m.clean()
        return R

    """def fit(self, output=None, ground_truth=None, last_lrp=None loss = 'CE', optimizer = 'Adam', opt_params = []):
        return Train(output, ground_truth, last_lrp, loss, optimizer, opt_params)"""
    
    def set_lrp_parameters(self, lrp_var=None, param=None):
        for module in self._modules.values():
            module.set_lrp_parameters(lrp_var=lrp_var, param=param)
            
            
    def integrated_grad(self, net, inputs, labels, target_layer, by_label = True, num_target = 1000):
        iterations=args.smooth_num
        inputs_zero = torch.zeros_like(inputs).cuda()
        for i in range(iterations):
            alpha = float(i) / iterations
            inputs_interpolation = (1-alpha) * inputs_zero + alpha * inputs
            activation_output = net.prediction(inputs_interpolation)
            if i == 0:
                R = net.simple_grad(activation_output, labels, target_layer, integrated_grad=True).detach()
            else:
                # If you want to train the model by using LR with IG, then you need to remove detach() function.
                R += net.simple_grad(activation_output, labels, target_layer, integrated_grad=True).detach()
            R = R / iterations
            #R = (inputs - inputs_zero) * R / iterations
        
        return R
    
    def forward_new_lrp(self, input, lambda_dic,labels, lambda_for_final, target_layer = '34', net=None):
        
        # classic forward
        activation_output = input
        for module in self._modules.values():
            activation_output = module.forward(activation_output)
        
        """
        20180827 new sequential flow
        """
        # Class loss and score
        _, prediction = torch.max(activation_output, 1)
        criterion = torch.nn.CrossEntropyLoss()
        diff = criterion(activation_output, labels)
        
        class_loss = torch.mean(diff)
        
        # Backward of Class Loss for getting Fisher in every self.weight.grad
        
        class_loss.backward(retain_graph = True)

        # when backpropagation, LRP flow loss + class loss, otherwise, identity
        if args.no_lambda_for_each_layer == True:
            
            #181210 added
            if args.interpreter == 'lrp':
                # get LRP loss
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method,1e-8, None, target_layer = args.lrp_target_layer)
                elif args.arch == 'CNN8':
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method, 1e-8, None, '17')
                
                
                
            elif args.interpreter == 'grad_cam':
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.grad_cam(activation_output, labels, target_layer)
                
                elif args.arch == 'CNN8':
                    LRP = self.grad_cam(activation_output, labels, '17')
            
            elif args.interpreter == 'simple_grad':
                LRP = self.simple_grad(activation_output, labels, args.lrp_target_layer)
                
            elif args.interpreter == 'smooth_grad':
                LRP = self.smooth_grad(net, input, labels, args.lrp_target_layer)
                
            elif args.interpreter == 'integrated_grad':
                LRP = self.integrated_grad(net, input, labels, args.lrp_target_layer)
            else:
                print('wrong interpreter!!',tt)
                
            if len(LRP.shape) != 4:
                LRP = LRP.unsqueeze(1)

            #190106 train with entropy of R (n, 1, 224,224), not (n, 3, 224,224)
            LRP = LRP.sum(dim=1, keepdim=True)
            
            if args.loss_type == 'absR':
                lrp_loss = torch.mean(torch.abs(LRP)) 
            
            elif args.loss_type == 'uniformR':
                if args.uniformR_type == 'variance':
                    entropy = self.variance(LRP)
                else:
                    entropy = self.entropy(LRP)
                lrp_loss = entropy
                
            elif args.loss_type == 'sparseR':
                if args.sparseR_type == 'variance':
                    entropy = -self.variance(LRP)
                elif args.sparseR_type == 'L1':
                    entropy =  self.L1(LRP)
                else:
                    entropy = -self.entropy(LRP)
                lrp_loss = entropy 
                
            elif args.loss_type == 'frame':
                lrp_loss = self.frame(LRP)
                print('frame loss ----------------',lrp_loss)
                
            elif args.loss_type == 'corner':
                lrp_loss = self.corner(LRP)
            else:
                print('wrong loss type!!',tt)
                
                
            # Get total loss, we already get gradient of class loss in self.weight.grad
            total_loss = lambda_for_final * lrp_loss
            check_total_loss = total_loss
            #self.total_loss =  (lambda_for_final * self.lrp_loss)


        if args.no_lambda_for_each_layer == False:
            # get LRP loss
            if args.loss_type == 'absR':
                LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'absR',labels, args.r_method, 1e-8)

            elif args.loss_type == 'uniformR':
                # abs R
                if args.entropy_type == 'abs':
                    LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'uniformR_abs',labels, args.r_method, 1e-8)

                # all positive R
                elif args.entropy_type == 'all_positive': 
                    LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'uniformR_positive',labels, args.r_method, 1e-8)

            LRP, lrp_loss = self.lrp(activation_output, lambda_dic, labels, args.r_method, 1e-8)

            # get total loss
            check_total_loss = lambda_for_final * lrp_loss
            total_loss = class_loss + (lambda_for_final *lrp_loss)
            #self.total_loss =  (self.lrp_loss)
        
        return total_loss, class_loss, lrp_loss, activation_output, check_total_loss, LRP    

    
    def FGSM_forward(self, input, lambda_dic,labels, lambda_for_final, target_layer = '34', net=None, r_targeted = None):
        
        # classic forward
        activation_output = input
        for module in self._modules.values():
            activation_output = module.forward(activation_output)
        
        """
        20180827 new sequential flow
        """
        # Class loss and score
        _, prediction = torch.max(activation_output, 1)
        criterion = torch.nn.CrossEntropyLoss()
        diff = criterion(activation_output, labels)
        
        class_loss = torch.mean(diff)
        #print('========*********************************')
        #print(activation_output[0])
        # when backpropagation, LRP flow loss + class loss, otherwise, identity
        if args.no_lambda_for_each_layer == True:
            
            #181210 added
            if args.interpreter == 'lrp':
                # get LRP loss
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method,1e-8, None, target_layer = args.lrp_target_layer, loss = class_loss, net = net)
                elif args.arch == 'CNN8':
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method, 1e-8, None, '17')
                
                
                
            elif args.interpreter == 'grad_cam':
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.grad_cam(activation_output, labels, target_layer)
                
                elif args.arch == 'CNN8':
                    LRP = self.grad_cam(activation_output, labels, '17')
            
            elif args.interpreter == 'simple_grad':
                LRP = self.simple_grad(activation_output, labels, args.lrp_target_layer)
                
            elif args.interpreter == 'smooth_grad':
                LRP = self.smooth_grad(net, input, labels, args.lrp_target_layer)
                
            elif args.interpreter == 'integrated_grad':
                LRP = self.integrated_grad(net, input, labels, args.lrp_target_layer)
            else:
                print('wrong interpreter!!',tt)
                
            if len(LRP.shape) != 4:
                LRP = LRP.unsqueeze(1)

            #190106 train with entropy of R (n, 1, 224,224), not (n, 3, 224,224)
            LRP = LRP.sum(dim=1, keepdim=True)
            
            if args.FGSM_loss_type == 'topk':
                lrp_loss = self.FGSM_topk(LRP)
                
            elif args.FGSM_loss_type == 'mass_center':
                lrp_loss = self.FGSM_mass_center(LRP, r_targeted)
                
            elif args.FGSM_loss_type == 'return_p1':
                LRP = LRP.sum(dim=1, keepdim=True)
                r_targeted = r_targeted.sum(dim=1, keepdim=True)
                lrp_loss = torch.abs(LRP.cuda() - r_targeted)
                
            elif args.FGSM_loss_type == 'return_p2':
                LRP = LRP.sum(dim=1, keepdim=True)
                r_targeted = r_targeted.sum(dim=1, keepdim=True)
                lrp_loss = torch.nn.functional.mse_loss(LRP.cuda(), r_targeted) 
                
            elif args.FGSM_loss_type == 'frame':
                lrp_loss = self.frame(LRP)
                
            
            
            else:
                print('wrong loss type!!',tt)
                
                
            #self.lrp_loss = torch.div(torch.sum( torch.abs(LRP) ), args.img_size)

            # get total loss

            total_loss = class_loss + (lambda_for_final * lrp_loss)
            check_total_loss = total_loss


        if args.no_lambda_for_each_layer == False:
            # get LRP loss
            if args.loss_type == 'absR':
                LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'absR',labels, args.r_method, 1e-8)

            elif args.loss_type == 'uniformR':
                # abs R
                if args.entropy_type == 'abs':
                    LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'uniformR_abs',labels, args.r_method, 1e-8)

                # all positive R
                elif args.entropy_type == 'all_positive': 
                    LRP, lrp_loss = self.lrp(activation_output, lambda_dic, 'uniformR_positive',labels, args.r_method, 1e-8)

            LRP, lrp_loss = self.lrp(activation_output, lambda_dic, labels, args.r_method, 1e-8)

            # get total loss
            check_total_loss = class_loss + lrp_loss
            """ 
            check point!
            if you give optimizer a loss that class_loss + lrp_loss*a, then don't use this.
            """
            total_loss = class_loss + lrp_loss
            #self.total_loss =  (self.lrp_loss)
        
        return total_loss, class_loss, lrp_loss, activation_output, check_total_loss, LRP
    
    def smooth_grad(self, net, inputs, labels, target_layer, by_label = True, num_target = 1000, alpha = 0.01):
        iterations = args.smooth_num
        alpha = args.smooth_std
        for i in range(iterations):
            inputs_noise = inputs + alpha*torch.randn(inputs.shape).cuda()
            activation_output = net.prediction(inputs_noise)
            if i == 0:
                R = net.simple_grad(activation_output, labels, target_layer).detach()
            else:
                # If you want to train the model by using LR with smooth grad, then you need to remove detach() function.
                R += net.simple_grad(activation_output, labels, target_layer).detach()
        
        return R / iterations
    


    def simple_grad(self, output, labels, target_layer, by_label = True, num_target = 1000, integrated_grad = False):
        if args.arch == 'CNN8':
            num_target = 10
            
        eye = torch.eye(num_target, dtype=torch.float32)
        eye = eye.cuda()
        if by_label:
            dx = eye[labels]
#         else:
#             dx = eye[torch.argmax(output, dim=1)]
        
        dx = torch.ones_like(output)*dx

        for key, module in reversed(list(self._modules.items())):

            requires_activation = (key == target_layer)
            # x: feature map, dx: dL/dx
            dx, x = module.grad_cam(dx, requires_activation)
            if requires_activation:
                break
                
                
        if integrated_grad:
            return dx

        #R = torch.pow(dx, 2)
        R = torch.abs(dx)
        w = torch.ones_like(R)*2
        
        if target_layer is not None:
            R = torch.sum( R*w , dim = 1)
            # thresholding is useless
            R = torch.nn.functional.threshold(R, threshold=0, value=0) 
            R_max,_ = R.max(1, keepdim=True)
            R_max,_ = R_max.max(2, keepdim=True)
            R_min,_ = R.min(1, keepdim=True)
            R_min,_ = R_min.min(2, keepdim=True)

            # R \in [0,1]
            R= (R - R_min) / (R_max - R_min + 1e-8)
            
            return R
        
        return R/torch.sum(R, dim=[1,2,3], keepdim=True)
        # return R/torch.sum(dx, dim=[1,2,3], keepdim=True)
            
            

        
    def corner(self, LRP):
        LRP = self.normalize(LRP)
        l = LRP.shape[2]
        mask = torch.zeros_like(LRP)
        k = (l//7) *2
        mask[:,:,:k,:k] = torch.ones((k,k), dtype=torch.float32).cuda()
        loss = torch.nn.functional.mse_loss(LRP, mask)
#         loss = torch.mean(torch.abs(mask - LRP))
        return loss
    
    def L1(self, LRP):
        return abs(LRP.reshape(LRP.shape[0], -1)).sum(dim=1).mean()
    
    def entropy(self, LRP):
    
        if torch.is_tensor(LRP) == False:
            LRP = torch.tensor(LRP)
        
        LRP = LRP.reshape(LRP.shape[0],-1)    
        if len(LRP.shape) == 3:
            LRP = LRP.unsqueeze(1)            
        if args.entropy_type == 'abs':
            LRP = torch.abs(LRP)
        elif args.entropy_type == 'all_positive':
            LRP_min, _ = LRP.min(dim=1, keepdim=True)
            LRP = LRP - LRP_min
        elif args.entropy_type == 'softmax':
            LRP = torch.nn.functional.softmax(LRP, dim=1)
        # first 1e-8: stabilizer for log function, second 1e-8: stabilizer for division
        prob = 1e-8 + LRP / (LRP.sum(dim=(1), keepdim=True) + 1e-8)
        entropy = (prob*torch.log(prob)).sum(dim=(1)).mean()
        
        return entropy
    
    def forward_targeted(self, input, lambda_dic, labels, lambda_for_final, class1_label, class2_label, target_layer = '34', r_targeted1 = None, r_targeted2 = None):
        
        # classic forward
        activation_output = input
        for module in self._modules.values():
            activation_output = module.forward(activation_output)
        
        # Class loss and score
        _, prediction = torch.max(activation_output, 1)
        
        
#         m = torch.nn.LogSoftmax(dim=0)
#         softmax_output = m(activation_output)
#         criterion1 = torch.nn.NLLLoss()
#         criterion2 = torch.nn.NLLLoss()
        
        
        #181231 added
        class_loss = 0
        
        label = torch.zeros(len(activation_output), dtype=torch.long).cuda()
        
#         #TODO
#         diff = criterion1(softmax_output, label + class1_label)
#         class_loss += torch.mean(diff)
#         diff = criterion2(softmax_output, label + class2_label)
#         class_loss += torch.mean(diff)
        
                
#         for pred in prediction:
#             if not pred.item() in [class1_label, class2_label]:
#                 diff = criterion(activation_output, torch.tensor(class1_label, dtype=torch.float32))
#                 class_loss += torch.mean(diff)
#                 diff = criterion(activation_output, torch.tensor(class2_label, dtype=torch.float32))
#                 class_loss += torch.mean(diff)

        class_loss = class_loss / len(prediction)
        
        
        if args.loss_type == 'bitargeted':
          
            if args.interpreter == 'grad_cam':
                LRP1 = self.grad_cam(activation_output, labels, target_layer)
                LRP2 = self.grad_cam(activation_output, labels - args.class_to + args.class_from, target_layer)
            
            if args.interpreter == 'lrp':
                LRP1 = self.lrp(activation_output, lambda_dic, None, labels , args.r_method, 1e-8, None, target_layer = args.lrp_target_layer)
                LRP2 = self.lrp(activation_output, lambda_dic, None, labels - args.class_to + args.class_from, args.r_method, 1e-8, None, target_layer = args.lrp_target_layer)

            if len(LRP1.shape) != 4:
                LRP1.unsqueeze(1)
            if len(LRP2.shape) != 4:
                LRP2.unsqueeze(1)
            
            LRP1 = LRP1.sum(dim=1, keepdim=True)
            LRP2 = LRP2.sum(dim=1, keepdim=True)
            r_targeted1 = r_targeted1.sum(dim=1, keepdim=True)
            r_targeted2 = r_targeted2.sum(dim=1, keepdim=True)  
                
            lrp_loss = (torch.nn.functional.mse_loss(LRP1.cuda(), r_targeted1) + torch.nn.functional.mse_loss(LRP2.cuda(), r_targeted2)) * (lambda_for_final/2)
            total_loss = class_loss + lrp_loss
            check_total_loss = total_loss
            return total_loss, class_loss, lrp_loss, activation_output, check_total_loss, LRP1
            
        
        if args.no_lambda_for_each_layer == True:
            
            #181210 added
            if args.interpreter == 'lrp':
                # get LRP loss
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method, 1e-8, None, target_layer = args.lrp_target_layer)
                elif args.arch == 'CNN8':
                    LRP = self.lrp(activation_output, lambda_dic, None, labels, args.r_method, 1e-8, None, '17')
                
                
            else:
                if args.arch in ['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet']:
                    LRP = self.grad_cam(activation_output, labels, target_layer)
                
                elif args.arch == 'CNN8':
                    LRP = self.grad_cam(activation_output, labels, '17')
            
            
                
            if len(LRP.shape) != 4:
                LRP.unsqueeze(1)
            #190106 train with entropy of R (n, 1, 224,224), not (n, 3, 224,224)
            LRP = LRP.sum(dim=1, keepdim=True)
            r_targeted1 = r_targeted1.sum(dim=1, keepdim=True)
            if args.loss_type == 'targeted':

                #LRP = self.normalize(LRP)
                #r_targeted1 = self.normalize(r_targeted1)
                lrp_loss = torch.nn.functional.mse_loss(LRP.cuda(), r_targeted1) * lambda_for_final
              
                

            #self.lrp_loss = torch.div(torch.sum( torch.abs(LRP) ), args.img_size)

            # get total loss

            total_loss = class_loss + lrp_loss
            check_total_loss = total_loss
            #self.total_loss =  (lambda_for_final * self.lrp_loss)

        return total_loss, class_loss, lrp_loss, activation_output, check_total_loss, LRP
   
            
    

class ModuleList(Module):
    r"""Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = operator.index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self


    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self



class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    ModuleDict can be indexed like a regular Python dictionary, but modules it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key/value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()


    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v


    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()


    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()


    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()


    def update(self, modules):
        r"""Update the ModuleDict with the key/value pairs from a mapping or
        an iterable, overwriting existing keys.

        Arguments:
            modules (iterable): a mapping (dictionary) of (string: :class:`~torch.nn.Module``) or
                an iterable of key/value pairs of type (string, :class:`~torch.nn.Module``)
        """
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, Mapping):
            if isinstance(modules, OrderedDict):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]



class ParameterList(Module):
    r"""Holds parameters in a list.

    ParameterList can be indexed like a regular Python list, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.nn.Parameter`` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterList, self).__init__()
        if parameters is not None:
            self += parameters

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ParameterList(list(self._parameters.values())[idx])
        else:
            idx = operator.index(idx)
            if not (-len(self) <= idx < len(self)):
                raise IndexError('index {} is out of range'.format(idx))
            if idx < 0:
                idx += len(self)
            return self._parameters[str(idx)]

    def __setitem__(self, idx, param):
        idx = operator.index(idx)
        return self.register_parameter(str(idx), param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def __iadd__(self, parameters):
        return self.extend(parameters)

    def __dir__(self):
        keys = super(ParameterList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, parameter):
        """Appends a given parameter at the end of the list.

        Arguments:
            parameter (nn.Parameter): parameter to append
        """
        self.register_parameter(str(len(self)), parameter)
        return self


    def extend(self, parameters):
        """Appends parameters from a Python iterable to the end of the list.

        Arguments:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParameterList.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)
        offset = len(self)
        for i, param in enumerate(parameters):
            self.register_parameter(str(offset + i), param)
        return self


    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr



class ParameterDict(Module):
    r"""Holds parameters in a dictionary.

    ParameterDict can be indexed like a regular Python dictionary, but parameters it
    contains are properly registered, and will be visible by all Module methods.

    Arguments:
        parameters (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.nn.Parameter``) or an iterable of key,value pairs
            of type (string, :class:`~torch.nn.Parameter``)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ParameterDict({
                        'left': nn.Parameter(torch.randn(5, 10)),
                        'right': nn.Parameter(torch.randn(5, 10))
                })

            def forward(self, x, choice):
                x = self.params[choice].mm(x)
                return x
    """

    def __init__(self, parameters=None):
        super(ParameterDict, self).__init__()
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, parameter):
        self.register_parameter(key, parameter)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.keys())

    def __contains__(self, key):
        return key in self._parameters

    def clear(self):
        """Remove all items from the ParameterDict.
        """
        self._parameters.clear()


    def pop(self, key):
        r"""Remove key from the ParameterDict and return its parameter.

        Arguments:
            key (string): key to pop from the ParameterDict
        """
        v = self[key]
        del self[key]
        return v


    def keys(self):
        r"""Return an iterable of the ParameterDict keys.
        """
        return self._parameters.keys()


    def items(self):
        r"""Return an iterable of the ParameterDict key/value pairs.
        """
        return self._parameters.items()


    def values(self):
        r"""Return an iterable of the ParameterDict values.
        """
        return self._parameters.values()


    def update(self, parameters):
        r"""Update the ParameterDict with the key/value pairs from a mapping or
        an iterable, overwriting existing keys.

        Arguments:
            parameters (iterable): a mapping (dictionary) of
                (string : :class:`~torch.nn.Parameter``) or an iterable of
                key/value pairs of type (string, :class:`~torch.nn.Parameter``)
        """
        if not isinstance(parameters, Iterable):
            raise TypeError("ParametersDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(parameters).__name__)

        if isinstance(parameters, Mapping):
            if isinstance(parameters, OrderedDict):
                for key, parameter in parameters.items():
                    self[key] = parameter
            else:
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                if not isinstance(p, Iterable):
                    raise TypeError("ParameterDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(p).__name__)
                if not len(p) == 2:
                    raise ValueError("ParameterDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(p)) +
                                     "; 2 is required")
                self[p[0]] = p[1]


    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(
                torch.typename(p.data), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr