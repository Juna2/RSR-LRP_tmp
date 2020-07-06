# coding=utf-8
import math
from math import ceil
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from module.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
from module.arguments import get_args
from module.utils import get_gpu_memory_map, get_min_used_gpu, normalize_with_log, normalize_with_nonlog
args=get_args()

import time

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
            
        """This is pytorch original one"""
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.activations_shape = None
#         self.padding_mode = padding_mode
        if transposed:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

        #181210 added
    def _grad_cam(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer.
        '''
        
        grad_input = torch.nn.grad.conv2d_input(self.input_tensor.shape, self.weight, grad_output, self.stride, self.padding, self.dilation, self.groups)
        if requires_activation:
            return grad_input, self.input_tensor
        else:
            return grad_input, None
        
    
    def _simple_lrp(self, R, labels):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : simple lrp')
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        
        # Variables
        
        def f(layer,X):
            # Get activations of full positive or negative part.
            
            Zs = F.conv2d(X, layer, bias, stride,
                        padding, dilation, self.groups)

            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)


            tmp = torch.nn.grad.conv2d_input(X.shape, layer, grad_output=RdivZs, padding=1) * X

            
            return tmp
        r = f(weight,input_patches)
        return r
    

    # 성환이형한테 받은 original _composite_lrp
    def _composite_lrp(self, R, labels):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : alpha beta lrp')
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        alpha = 1 - args.beta
        beta = args.beta
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        def f(layer1, layer2, X1, X2):
            # Get activations of full positive or negative part.
            
            Z1 = F.conv2d(X1, layer1, bias, stride,
                        padding, dilation, self.groups)

            Z2 = F.conv2d(X2, layer2, bias, stride,
                        padding, dilation, self.groups)
            Zs = Z1 + Z2
            
#             stabilizer = 1e-3*torch.tensor(torch.eq(Zs,0), dtype=torch.float32).cuda()
            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)

#             tmp1 = self.grad_object(RdivZs, Zs, X1, layer1) * X1
#             tmp2 = self.grad_object(RdivZs, Zs, X2, layer2) * X2

            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            
            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, input_p, input_n)
        
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, input_p, input_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
        
        return re
    
    
    # 성환이형한테 받고 준하가 수정한 _composite_lrp
    def _composite_lrp_corrected(self, R, labels):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : alpha beta lrp')
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        alpha = 1 - args.beta
        beta = args.beta
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        ########################
        bias_p = F.relu(bias)
        bias_n = bias - bias_p
        ########################
        
        def f(layer1, layer2, X1, X2, Y):
            # Get activations of full positive or negative part.
            Z1 = F.conv2d(X1, layer1, Y, stride,
                        padding, dilation, self.groups)

            Z2 = F.conv2d(X2, layer2, Y, stride,
                        padding, dilation, self.groups)
            Zs = Z1 + Z2

            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)
            
            # conv2d_input에 대한 gradient를 구한다.
            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2
            
            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, input_p, input_n, bias_p)
        
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, input_p, input_n, bias_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
        
        return re
    
    
    def _flrp_acw(self, R, labels):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : alpha beta lrp')

        try:
            # Use fisher : gradient ^2
            c = self.weight.grad.abs().clone().detach()
            
#             # Use Gradient 
#             k = w.grad.clone()
            
#             # Normalize fisher
#             c = normalize_with_log(c.detach())
        
        except:
            c = torch.ones_like(self.weight).clone().detach()

            
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
        
        weight = weight * c
        
        alpha = 1 - args.beta
        beta = args.beta
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        def f(layer1, layer2, X1, X2):
            # Get activations of full positive or negative part.
            
            Z1 = F.conv2d(X1, layer1, bias, stride,
                        padding, dilation, self.groups)

            Z2 = F.conv2d(X2, layer2, bias, stride,
                        padding, dilation, self.groups)
            Zs = Z1 + Z2
            
#             stabilizer = 1e-3*torch.tensor(torch.eq(Zs,0), dtype=torch.float32).cuda()
            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)

#             tmp1 = self.grad_object(RdivZs, Zs, X1, layer1) * X1
#             tmp2 = self.grad_object(RdivZs, Zs, X2, layer2) * X2

            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            
            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, input_p, input_n)
        
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, input_p, input_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
        
        return re
        
        
    def _flrp_aw(self, R, labels):

        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        '''
#         print('cnn : alpha beta lrp')

        try:
            # Use fisher : gradient ^2
            c = self.weight.grad.clone().abs()
            
#             # Use Gradient 
#             k = w.grad.clone()
            
#             # Normalize fisher
#             c = normalize_with_log(c.detach())
        
        except:
            c = torch.ones_like(self.weight)

        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        
        weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
        
        weight = weight * c
        
        alpha = 1 - args.beta
        beta = args.beta
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        def f(layer1, layer2, X1, X2):
            # Get activations of full positive or negative part.
            
            Z1 = F.conv2d(X1, layer1, bias, stride,
                        padding, dilation, self.groups)

            Z2 = F.conv2d(X2, layer2, bias, stride,
                        padding, dilation, self.groups)
            Zs = Z1 + Z2
            
#             stabilizer = 1e-3*torch.tensor(torch.eq(Zs,0), dtype=torch.float32).cuda()
            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
            RdivZs = R/(Zs+stabilizer)

#             tmp1 = self.grad_object(RdivZs, Zs, X1, layer1) * X1
#             tmp2 = self.grad_object(RdivZs, Zs, X2, layer2) * X2

            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            
            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, input_p, input_n)
        
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, input_p, input_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
        
        return re
    
    
    
    def _composite_new_lrp(self, R, labels):
        '''
        LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
        True code.
        '''
#         print('cnn : alpha beta lrp')
#         print(all_linear_finished)
        R = self.check_shape(R)

        input_patches = self.input_tensor
#         print('R shape:',R.shape)
        
        
        weight, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
        alpha = 1 - args.beta
        beta = args.beta
        
        # Compute C
        # Compute P

        x = self.input_tensor
        w = weight
        print(w.shape)
        try:
            # Use fisher : gradient ^2
            k = w.grad.clone()
            
#             # Use Gradient 
#             k = w.grad.clone()
            
            # Normalize fisher
            k = normalize_with_log(k.detach())
        
        except:
            k = torch.ones_like(w)

        _x = torch.where(x==0, torch.zeros(1).cuda(), 1/x)
        p = torch.nn.functional.conv2d(_x,k, padding=padding)
        n, out_channels, width, height = p.shape
        aj_shape = p.shape
        p = p.unsqueeze(2) / (w.shape[1] * w.shape[2] * w.shape[3] - 1)

        C = torch.zeros_like(w)
        x_padd = torch.nn.functional.pad(x, (padding[0],padding[0],padding[0],padding[0]))
        in_channels = w.shape[1]
        
        

        

        for i in range(k.shape[2]):
            for j in range(k.shape[3]):
                
               
                
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&',i, j)
                # Compute k/a
                # This code is only for conv with stride=1
                x_padd_ = (x_padd[:,:,i:i+width, j:j+height])
                
                x_sign = torch.sign(x_padd_).detach()
                _x_cut = torch.where(x_padd_.abs()<1e-8, (torch.ones(1)*1e+8).cuda(), 1/x_padd_.abs())
                _x_cut = _x_cut * x_sign
                _x_cut = _x_cut.detach()

                
                device_list = []
                for gg in range(4):
                    device_list.append(torch.device("cuda:"+str(gg)))
                
#                 # Gpu for gathering all C
#                 gpu_gather = get_min_used_gpu()
                
                for gpu in range(2):
                    
                    # Use the gpu used the smallest memory so far
                    gpu_used = get_min_used_gpu()
                    
                    # Separate calculation as 16, dim=out_channels
                    c = w.shape[0]//2
                    
                    # Variable 
                    w_gpu = w[c*gpu:(gpu+1)*c,:,i,j].reshape(-1,1,1,1).to(device_list[gpu_used]).contiguous()
                    k_gpu = k[c*gpu:(gpu+1)*c,:,i,j].reshape(-1,1,1,1).to(device_list[gpu_used]).contiguous()
                    R_gpu = R[:,c*gpu:(gpu+1)*c,:,:].to(device_list[gpu_used]).contiguous()
                    p_gpu = p[:,c*gpu:(gpu+1)*c,:,:].to(device_list[gpu_used]).contiguous()
                    
                    # To Avoid devided by 0 or close to 0
                    w_sign = torch.sign(w_gpu).detach()
                    r_sign = torch.sign(R_gpu).detach()
                    
                    w_gpu_abs = w_gpu.abs()
                    R_gpu_abs = R_gpu.abs()
                    
                    _w_gpu = torch.where(w_gpu_abs<1e-8, (torch.ones(1)*1e+8).to(device_list[gpu_used]), 1/w_gpu_abs)
                    _R_gpu = torch.where(R_gpu_abs<1e-8, (torch.ones(1)*1e+8).to(device_list[gpu_used]), 1/R_gpu_abs)
                    
                    _w_gpu = _w_gpu * w_sign
                    _R_gpu = _R_gpu * r_sign
                    
                    _w_gpu = _w_gpu.detach()
                    _R_gpu = _R_gpu.detach()
                   
                    
                    
                    
                    k_a_gpu = torch.nn.functional.conv2d(_x_cut.to(device_list[gpu_used]), k_gpu,groups=in_channels).contiguous()
                    k_a_gpu = k_a_gpu.reshape(n, c, in_channels, k_a_gpu.shape[2], k_a_gpu.shape[-1]).contiguous()
                    
                    
                    _aw_gpu = torch.nn.functional.conv2d(_x_cut.to(device_list[gpu_used]), 1/_w_gpu,groups=in_channels).contiguous()
                    _aw_gpu = _aw_gpu.reshape(n, c, in_channels, _aw_gpu.shape[2], _aw_gpu.shape[-1]).contiguous()
                    
                    
                    p_k_a_gpu = p_gpu - k_a_gpu
                    
                
                    c_gpu = (p_k_a_gpu * _aw_gpu  * args.new_lrp_proportion * (_R_gpu.unsqueeze(2)))
                    
                    # Sum C becasue one weight is related to many activations
                    c_gpu = c_gpu.sum(dim=(0,3,4)).contiguous()
                
#                     # Mean C becasue one weight is related to many activations
#                     c_gpu = c_gpu.mean(0).mean(2).mean(2).contiguous()
                    
                    
                     
                    new_lrp_device = int(np.argmax(get_gpu_memory_map()))

                    # Gather all C to the gpu used the smallest memory so far
                    new_lrp_device = get_min_used_gpu()
                    C[c*gpu:(gpu+1)*c,:, i, j] = c_gpu.detach().to(new_lrp_device).contiguous()
        
        
        # Scale Normalize C
        C = normalize_with_log(C.detach())
        print('------------ C after normalization',C.mean(), C.min(), C.max())
        
        # Variables
        weight_p = F.relu(weight)
        weight_n = weight - weight_p
        
        Cweight = weight * C.detach()
        Cweight_p = F.relu(Cweight)
        Cweight_n = Cweight - Cweight_p
        
        input_p = F.relu(input_patches)
        input_n = input_patches - input_p
        
        def f(layer1, layer2, clayer1, clayer2, X1, X2):
            # Get activations of full positive or negative part.
            
            Z1 = F.conv2d(X1, clayer1, bias, stride,
                        padding, dilation, self.groups)

            Z2 = F.conv2d(X2, clayer2, bias, stride,
                        padding, dilation, self.groups)
            Zs = Z1 + Z2
            
            # Add stabilizer to Zs by avoiding devided by 0
            stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1)).cuda()
            Zs = (Zs+stabilizer)
            
            # Clipping Zs when really close to 0 for blocking R to be inf or NaN
            Zs_sign = torch.sign(Zs)
            Zs = torch.where(Zs.abs()<1e-8, (torch.ones(1)*1e-8).cuda(), Zs.abs())
            Zs = Zs * Zs_sign
            
            
            RdivZs =  R/(Zs)
            
            tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
            tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

#             # Get acw values.
#             import time
#             import pickle
#             R_num = time.time()
#             save_path_zs = args.img_dir+'/acw_conv_zs' + str(R_num) +'.pkl' 
#             save_path_acw = args.img_dir+ '/acw_conv_acw' + str(R_num) +'.pkl' 
#             save_path_c = args.img_dir+'/acw_conv_c' + str(R_num) +'.pkl' 
#             save_path_xrw_mean = args.img_dir+'/acw_conv_x' + str(R_num) +'.pkl' 
#             save_path_xrw_max = args.img_dir+'/acw_conv_r' + str(R_num) +'.pkl' 
#             save_path_xrw_min = args.img_dir+'/acw_conv_w' + str(R_num) +'.pkl' 
#     #         save_path_zs = args.img_dir+args.img_name+'/acw_zs' + str(R_num) +'.pkl' 
#     #         save_path_acw = args.img_dir+args.img_name+ '/acw_acw' + str(R_num) +'.pkl' 
#     #         save_path_c = args.img_dir+args.img_name+'/acw_c' + str(R_num) +'.pkl' 

#             try:
#                 with open(save_path_zs,"rb") as f:
#                     data_zs = pickle.load(f)

#             except:
#                 data_zs = []

#             try:
#                 with open(save_path_acw,"rb") as f:
#                     data_acw = pickle.load(f)

#             except:
#                 data_acw = []
#             try:
#                 with open(save_path_c,"rb") as f:
#                     data_c = pickle.load(f)
#             except:
#                 data_c = []   
#             try:
#                 with open(save_path_xrw_mean,"rb") as f:
#                     data_xrw_mean = pickle.load(f)
#                 with open(save_path_xrw_max,"rb") as f:
#                     data_xrw_max = pickle.load(f)
#                 with open(save_path_xrw_min,"rb") as f:
#                     data_xrw_min = pickle.load(f)    
                    
#             except:
#                 data_xrw_mean = []
#                 data_xrw_max = []   
#                 data_xrw_min = []   

#             data_zs.append(Zs.detach())
#             data_acw.append(tmp1.detach() + tmp2.detach())
#             data_c.append(C.detach())
#             data_xrw_max.append(R.detach())
#             data_xrw_min.append(weight.detach())
#             data_xrw_mean.append(x.detach())
#             with open(save_path_zs,"wb") as f:
#                 pickle.dump(data_zs,f)

#             with open(save_path_acw,"wb") as f:
#                 pickle.dump(data_acw,f)

#             with open(save_path_c,"wb") as f:
#                 pickle.dump(data_c,f) 
                
#             with open(save_path_xrw_max,"wb") as f:
#                 pickle.dump(data_xrw_max,f) 
                
#             with open(save_path_xrw_min,"wb") as f:
#                 pickle.dump(data_xrw_min,f) 
#             with open(save_path_xrw_mean,"wb") as f:
#                 pickle.dump(data_xrw_mean,f)     
            
            return tmp1 + tmp2
        
        re_alpha = f(weight_p, weight_n, Cweight_p, Cweight_n, input_p, input_n)
        
        
        if  beta != 0:
            re_beta = f(weight_n, weight_p, Cweight_n, Cweight_p, input_p, input_n)
            re = (re_alpha * alpha) + (re_beta * beta)
        else:
            re = re_alpha
            
        
#         # Normalize R
#         re = normalize_with_log(re)
        
        return re

    def check_shape(self, R):
        self.R = R
        R_shape = self.R.shape
        if len(R_shape) != 4:
            self.R = torch.reshape(self.R, self.activations_shape)
        N, NF, self.Hout, self.Wout = self.R.shape
        return self.R
    
    
    def extract_patches(self):

        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation)
        image_patches = unfold(self.input_tensor)

        '''
        <im2col>
        Apply sliding window to input volumes of convolution and make it as a matrix (each column represents receptive field)
        length of rows are the same as the stretched size of output feature map size. i.e. Hout x Wout
        (N, C_in, H_in, W_in) with filter size (Kernel_H, Kernel_W) -> (N, C_in x Kernel_H x Kernel_W, L)
        where L = Hout x Wout 
        '''

        return image_patches
    
    
#     def ____simple_lrp(self, R, labels):
#         '''
#         LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
#         '''
#         R = self.check_shape(R)

#         image_patches = self.extract_patches()
#         input_R = self.simple_lrp_object( R, self.weight, self.input_tensor, self.out_channels, image_patches, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias)

#         return input_R
    
#     def __composite_lrp(self, R, labels):
#         '''
#         LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
#         '''
# #         print('cnn : alpha beta lrp')
#         R = self.check_shape(R)

#         input_patches = self.extract_patches()
# #         print('R shape:',R.shape)
        
#         weight, out_channels,  kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
#         alpha = 1 - args.beta
#         beta = args.beta
        
#         # Variables
#         input_patches = input_patches.permute(0,2,1)
#         input = torch.unsqueeze(input_patches, -1)
#         weight = self.weight.reshape(weight.size(0),-1).t()
        
#         # Unfold RdivZs
#         output_shape = R.shape
#         unfold = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         R_unfold = unfold(R)


#         R_unfold = R_unfold.unsqueeze(1)
#         R_unfold = R_unfold.permute(0,3,1,2)
        
        
#         weight_zero = torch.zeros(weight.shape).cuda()
#         weight_p = torch.where(weight>0, weight, weight_zero)
#         weight_n = torch.where(weight<0, weight, weight_zero)
        
#         input_p = torch.where(input>0, weight, weight_zero)
#         input_n = torch.where(input>0, weight, weight_zero)
        
#         def f(layer1, layer2, X1, X2):
#             # Get activations of full positive or negative part.
#             Z1 = torch.matmul(X1, layer1)
#             Z2 = torch.matmul(X2, layer2)
#             Zs = Z1 + Z2
            
#             RdivZs = R/Zs
            
#             tmp1 = self.grad_object(RdivZs, Zs, X1, layer1) * X1
#             tmp2 = self.grad_object(RdivZs, Zs, X2, layer2) * X2
            
#             return tmp1 + tmp2
        
#         re_alpha = f(weight_p, weight_n, input_p, input_n)
        
#         if  beta != 0:
#             re_beta = f(weight_n, weight_p, input_p, input_n)
#             re = (re_alpha * alpha) + (re_beta * beta)
#         else:
#             re = re_alpha
        
#         re = re.permute(0,2,1)


#         # fold : back to the input shape 
#         fold = torch.nn.Fold(output_size=(in_h, in_w), kernel_size=kernel_size,
#                              padding=padding, dilation=dilation, stride=stride)

#         input_R = fold(re)
        
#         return input_R  
   
#     def __composite_new_lrp(self, R, labels):
#         '''
#         LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
#         '''
# #         print('cnn : alpha beta lrp')
#         R = self.check_shape(R)

#         input_patches = self.input_tensor
# #         print('R shape:',R.shape)
        
        
#         weight, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
#         alpha = 1 - args.beta
#         beta = args.beta
        
#         # Compute C
#         # Compute P

#         x = self.input_tensor
#         w = weight
#         print(w.shape)
#         k = w.grad.clone()

#         p = torch.nn.functional.conv2d(x,w, padding=padding)
#         n, out_channels, width, height = p.shape
#         p = p.unsqueeze(2) / (w.shape[1] * w.shape[2] * w.shape[3] - 1)

#         C = torch.zeros_like(w)
#         x_padd = torch.nn.functional.pad(x, (padding[0],padding[0],padding[0],padding[0]))
#         in_channels = w.shape[1]

#         for i in range(k.shape[2]):
#             for j in range(k.shape[3]):
#                 print(i, j)
#                 # Compute k/a
#                 # This code is only for conv with stride=1
#                 k_ij = k[:,:,i,j].reshape(-1,1,1,1)
#                 _x_cut = 1/x_padd[:,:,i:i+width, j:j+height]

#                 k_a = torch.nn.functional.conv2d(_x_cut,k_ij,groups=in_channels)
#                 k_a = k_a.reshape(n, out_channels, in_channels, k_a.shape[2], k_a.shape[-1])


#                 p_k_a = p - k_a

#             #     w = w.reshape(w.shape[0]*w.shape[1], 1, w.shape[2],w.shape[3])

#                 w_ij = w[:,:,i, j].reshape(-1,1,1,1)
#                 _aw = torch.nn.functional.conv2d(_x_cut,1/w_ij,groups=in_channels)
#                 _aw = _aw.reshape(n, out_channels, in_channels, _aw.shape[2], _aw.shape[-1])

#                 C[:,:, i, j] = (p_k_a * _aw  * args.new_lrp_proportion / R.unsqueeze(2)).sum(dim=(0,3,4))
#                 del k_a, p_k_a, _aw, w_ij, k_ij, _x_cut
                
        
#         # Variables
#         weight_p = F.relu(weight)
#         weight_n = weight - weight_p
        
#         Cweight = weight * C
#         Cweight_p = F.relu(Cweight)
#         Cweight_n = Cweight - Cweight_p
        
#         input_p = F.relu(input_patches)
#         input_n = input_patches - input_p
        
#         def f(layer1, layer2, clayer1, clayer2, X1, X2):
#             # Get activations of full positive or negative part.
            
#             Z1 = F.conv2d(X1, clayer1, bias, stride,
#                         padding, dilation, self.groups)

#             Z2 = F.conv2d(X2, clayer2, bias, stride,
#                         padding, dilation, self.groups)
#             Zs = Z1 + Z2
            
# #             stabilizer = 1e-3*torch.tensor(torch.eq(Zs,0), dtype=torch.float32).cuda()
#             stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
#             RdivZs = R/(Zs+stabilizer)


#             tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
#             tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            
#             return tmp1 + tmp2
        
#         re_alpha = f(weight_p, weight_n, Cweight_p, Cweight_n, input_p, input_n)
        
        
#         if  beta != 0:
#             re_beta = f(weight_n, weight_p, Cweight_n, Cweight_p, input_p, input_n)
#             re = (re_alpha * alpha) + (re_beta * beta)
#         else:
#             re = re_alpha
        
#         return re
    
#     def __composite_new_lrp(self, R, labels):
#         return self._composite_lrp(R, labels)
    
#     def __composite_new_lrp(self, R, labels):
#         '''
#         LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
#         '''
# #         print('cnn : alpha beta lrp')
#         R = self.check_shape(R)

#         input_patches = self.input_tensor
# #         print('R shape:',R.shape)
        
        
#         weight, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, bias = self.weight, self.out_channels, self.kernel_size, self.padding, self.dilation, self.stride, self.in_h, self.in_w, self.bias
 
#         alpha = 1 - args.beta
#         beta = args.beta
        
#         # Compute C
#         # Compute P

#         x = self.input_tensor
#         w = weight
#         print(w.shape)
#         try:
#             k = w.grad.clone()
#         except:
#             k = torch.ones_like(w)
#         p = torch.nn.functional.conv2d(x,w, padding=padding)
#         n, out_channels, width, height = p.shape
#         aj_shape = p.shape
#         p = p.unsqueeze(2) / (w.shape[1] * w.shape[2] * w.shape[3] - 1)

#         C = torch.zeros_like(w)
#         x_padd = torch.nn.functional.pad(x, (padding[0],padding[0],padding[0],padding[0]))
#         in_channels = w.shape[1]
        
        
#         for c in range(k.shape[1]):
#             for i in range(k.shape[2]):
#                 for j in range(k.shape[3]):
#                     # Compute k/a
#                     # This code is only for conv with stride=1
#                     k_ij = k[c:c+1,:,i,j]
#                     k_ij = k_ij.reshape(-1,1,1,1)
#                     _x_cut = 1/x_padd[:,:,i:i+width, j:j+height]

#                     k_a = torch.nn.functional.conv2d(_x_cut,k_ij,groups=in_channels)
#                     k_a = k_a.reshape(n, in_channels, k_a.shape[2], k_a.shape[-1])


#                     p_k_a = p[:,c] - k_a


#                     w_ij = w[c,:,i, j].reshape(-1,1,1,1)
#                     _aw = torch.nn.functional.conv2d(_x_cut,1/w_ij,groups=in_channels)
#                     _aw = _aw.reshape(n, in_channels, _aw.shape[2], _aw.shape[-1])

#                     C[:,:, i, j] = (p_k_a * _aw  * args.new_lrp_proportion / R.unsqueeze(2)).sum(dim=(0,2,3))
#                     del k_a, p_k_a, _aw, w_ij, k_ij, _x_cut
                
        
#         # Variables
#         weight_p = F.relu(weight)
#         weight_n = weight - weight_p
        
#         Cweight = weight * C
#         Cweight_p = F.relu(Cweight)
#         Cweight_n = Cweight - Cweight_p
        
#         input_p = F.relu(input_patches)
#         input_n = input_patches - input_p
        
#         def f(layer1, layer2, clayer1, clayer2, X1, X2):
#             # Get activations of full positive or negative part.
            
#             Z1 = F.conv2d(X1, clayer1, bias, stride,
#                         padding, dilation, self.groups)

#             Z2 = F.conv2d(X2, clayer2, bias, stride,
#                         padding, dilation, self.groups)
#             Zs = Z1 + Z2
            
# #             stabilizer = 1e-3*torch.tensor(torch.eq(Zs,0), dtype=torch.float32).cuda()
#             stabilizer = 1e-2*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
            
#             RdivZs = R/(Zs+stabilizer)


#             tmp1 = torch.nn.grad.conv2d_input(X1.shape, layer1, grad_output=RdivZs, stride = stride, padding=padding) * X1
#             tmp2 = torch.nn.grad.conv2d_input(X2.shape, layer2, grad_output=RdivZs, stride = stride, padding=padding) * X2

            
#             return tmp1 + tmp2
        
#         re_alpha = f(weight_p, weight_n, Cweight_p, Cweight_n, input_p, input_n)
        
        
#         if  beta != 0:
#             re_beta = f(weight_n, weight_p, Cweight_n, Cweight_p, input_p, input_n)
#             re = (re_alpha * alpha) + (re_beta * beta)
#         else:
#             re = re_alpha
        
#         return re
    

    
# class grad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, RdivZs, zs, x, w):
        
#         re = torch.tensor(torch.autograd.grad(zs, x, grad_outputs=RdivZs, retain_graph=True,allow_unused=True)[0].clone(), requires_grad=True)
#         ctx.save_for_backward(RdivZs, zs, x, w, re)
#         return re
            

#     @staticmethod
#     def backward(ctx, grad_output):
#         RdivZs, zs, x, w, re = ctx.saved_tensors
#         print(33333333,re.shape, zs.shape, RdivZs.shape, grad_output.shape)
#         x_ = torch.ones_like(zs)
#         w_re = torch.tensor(torch.autograd.grad(zs, x, grad_outputs = x_, retain_graph=True,allow_unused=True)[0].clone(), requires_grad=True)
        
#         dRdivZs = torch.matmul(grad_output, w_re)
#         print(555, dRdivZs.shape)
        
#         dw = torch.matmul(grad_output.transpose(1,0), RdivZs)
       
#         return dRdivZs, None, None,dw
    
# class Alphabeta_lrp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, R, weight, input_tensor, out_channels, input_patches, kernel_size, padding, dilation, stride, in_h, in_w, bias,beta):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
        
        
        
        
        
#         alpha = 1 - beta
        
#         # Variables
#         input_patches = input_patches.permute(0,2,1)
#         input = torch.unsqueeze(input_patches, -1)
#         weight = weight.reshape(weight.size(0),-1).t()
        
#         # Unfold RdivZs
#         output_shape = R.shape
#         unfold = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         R_unfold = unfold(R)


#         R_unfold = R_unfold.unsqueeze(1)
#         R_unfold = R_unfold.permute(0,3,1,2)
        
        
#         weight_zero = torch.zeros(weight.shape)
#         weight_p = torch.where(weight>0, weight, weight_zero)
#         weight_n = torch.where(weight<0, weight, weight_zero)
#         input_p = torch.where(input>0, weight, weight_zero)
#         input_n = torch.where(input>0, weight, weight_zero)
        
#         def f(layer1, layer2, X1, X2):
#             # Get activations of full positive or negative part.
#             Z1 = torch.matmul(layer1, X1)
#             Z2 = torch.matmul(layer2, X2)
#             Zs = Z1 + Z2
            
#             RdivZs = R/Zs
            
#             tmp1 = grad_object(RdivZs, Zs, X1, layer1) * X1
#             tmp2 = grad_object(RdivZs, Zs, X2, layer2) * X2
            
#             return tmp1 + tmp2
        
#         re_alpha = f(weight_p, weight_n, input_p, input_n)
        
#         if  beta != 0:
#             re_beta = f(weight_n, weight_p, input_p, input_n)
#             re = (re_alpha * alpha) + (re_beta * beta)
#         else:
#             re = re_alpha
        
#         re = re.permute(0,2,1)


#         # fold : back to the input shape 
#         fold = torch.nn.Fold(output_size=(in_h, in_w), kernel_size=kernel_size,
#                              padding=padding, dilation=dilation, stride=stride)

#         input_R = fold(re)
#         return input_R

 
        
        
#         #RdivZs : output_R/Zs, Index mask of positive forward predictions
#         Zplus = torch.tensor(Z > 0, dtype=torch.float32)
        
        
#         if alpha * beta != 0 :
#             Zp = Z * Zplus #Zp = torch.masked_select(Z, Zplus) 
#             Zsp = torch.sum(Zp, 2, keepdim=True) + torch.unsqueeze(bias.cpu(), 0)+ 1e-16

#             Zn = Z - Zp
#             Zsn = torch.sum(Zn, 2, keepdim=True) + torch.unsqueeze(bias.cpu(), 0)- 1e-16
#             RdivZs = (alpha * (Zp/Zsp) + beta * (Zn/Zsn))
            
            

      
#         elif alpha: #only alpha is not 0 -> alpha = 1, beta = 0
#             Zp = Z * Zplus
#             Zsp = torch.sum(Zp, 2, keepdim=True) + torch.unsqueeze(bias.cpu(), 0)+ 1e-16
#             RdivZs = ((Zp/Zsp))


            
#         elif beta: # only beta is not 0 -> alpha = 0, beta = 1
#             Zn = Z * -1 * (Zplus-1)
#             Zsn = torch.sum(Zn, 2, keepdim=True) + torch.unsqueeze(bias.cpu(), 0)- 1e-16
#             RdivZs = ((Zn/Zsn))


#         else:
#             raise Exception('This case should never occur: alpha={}, beta={}.'.format(alpha, beta))

        
        
#         # Unfold RdivZs
#         output_shape = RdivZs.shape
#         unfold = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         RdivZs_unfold = unfold(RdivZs)


#         RdivZs_unfold = RdivZs_unfold.unsqueeze(1)
#         RdivZs_unfold = RdivZs_unfold.permute(0,3,1,2)
#         RdivZs_unfold = RdivZs_unfold.cpu()
        
#         re = grad_object(RdivZs, Zs, input_patches, weight_cpu)
        
#         re * input_patches
        
#         # result : self.compute_result(Z, Zs)
        
        
#         re = torch.sum(Zs * R_cpu,3)
#         result = re.cuda()
#         #result = torch.sum((Z / Zs) * R_unfold, 3)

#         result = result.permute(0,2,1)


#         # fold : back to the input shape 
#         fold = torch.nn.Fold(output_size=(in_h, in_w), kernel_size=kernel_size,
#                              padding=padding, dilation=dilation, stride=stride)

#         input_R = fold(result)

        
#         # save Variables in ctx      
#         input_patches = torch.autograd.Variable(input_patches, requires_grad=False) # stop gradient from input_pathces to input_tensors
        
#         out_channels = torch.tensor(out_channels)
#         padding = torch.tensor(padding)
#         dilation = torch.tensor(dilation)
#         stride = torch.tensor(stride)
#         in_h = torch.tensor(in_h)
#         in_w = torch.tensor(in_w)
#         kernel_size = torch.tensor(kernel_size)
#         output_shape = torch.tensor(output_shape)
#         ctx.save_for_backward(output_R, weight, input_patches, input_R, Z, Zs, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, output_shape,Zplus)
#         del Z,Zs, R_cpu
        

#         return input_R    
            

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         # Get Variable from ctx
#         output_R, weight, input_patches, input_R,  Z, Zs, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, output_shape, Zplus = ctx.saved_tensors
        

#         output_R.matmul(output_R)
        
        
#         out_channels = out_channels.item()
#         padding = padding.numpy().tolist()
#         dilation = dilation.numpy().tolist()
#         stride = stride.numpy().tolist()
#         in_h = in_h.item()
#         in_w = in_w.item()
#         kernel_size = kernel_size.numpy().tolist()
#         output_shape = output_shape.numpy().tolist()
        
#         # dimension of elements
#         unfold_1 = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=padding, stride=1, dilation=1)
        
#         weight = weight.cpu()
#         input_R = input_R.cpu()
#         output_R = output_R.cpu()
#         input_patches = input_patches.cpu()
#         grad_output = grad_output.cpu()
        
        
#         w = weight.reshape(weight.size(0), -1)
#         w_t = torch.t(w)
        

#         input_R = unfold(input_R).permute(0,2,1)
#         grad_output = unfold(grad_output).permute(0,2,1)
        
#         out = Zs.squeeze(2)
#         output_R = unfold_1(output_R).permute(0,2,1)
        
#         RdivZs = output_R / out
#         RdivZs2 = RdivZs / out
#         nabla_mul_X = (grad_output * input_patches)
#         nabla_x_w = nabla_mul_X.matmul(w_t)
#         nabla_mul_X = nabla_mul_X
        
#         # 1. D loss/D w
        
#         M1 = torch.sum(nabla_mul_X.unsqueeze(3) * RdivZs.unsqueeze(2), (0,1))
#         M2 = torch.sum(input_patches.unsqueeze(3) * (nabla_x_w * RdivZs2).unsqueeze(2), (0,1))
#         Dloss_Dw = (M1 - M2).t()
#         Dloss_Dw = Dloss_Dw.reshape(weight.shape)
        
        
#         # 2. D loss/D r_out (= next dy_)
        
#         dy_ = nabla_x_w/out
#         fold_output = torch.nn.Fold(output_size=output_shape[2:], kernel_size=(1,1),padding=0, dilation=dilation, stride=stride)
#         dy_ = fold_output(dy_.permute(0,2,1))
        
        
#         # 3. D loss/D input
#         M1 = grad_output * RdivZs.matmul(w)
#         M2 = (nabla_x_w * RdivZs2).matmul(w)
#         Dloss_Dinput = M1 - M2
        
#         fold_input = torch.nn.Fold(output_size=(in_h,in_w), kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
#         Dloss_Dinput = fold_input(Dloss_Dinput.permute(0,2,1))
        
        
        
#         return dy_.cuda(),Dloss_Dw.cuda(),Dloss_Dinput.cuda(),None,None,None,None,None,None,None,None,None
        
        
# class Simple_lrp(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, R, weight, input_tensor, out_channels, input_patches, kernel_size, padding, dilation, stride, in_h, in_w, bias):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         output_R = R
        
        
#         input_patches = input_patches.permute(0,2,1)
#         input_cpu = torch.unsqueeze(input_patches, -1).cpu()
#         weight_cpu = weight.reshape(weight.size(0),-1).t().cpu()
#         Z = torch.mul(input_cpu, torch.unsqueeze(weight_cpu, 0))
       
        
#         # Zs : self.compute_zs(Z)
#         stabilizer=True
#         epsilon=1e-12
        
#         Zs = torch.sum(Z, 2, keepdim=True) + torch.unsqueeze(bias.cpu(), 0)
#         if stabilizer == True:
#             stabilizer = epsilon * (torch.where(torch.ge(Zs, 0), torch.ones_like(Zs, dtype=torch.float32),
#                                              torch.ones_like(Zs, dtype=torch.float32) * -1))
#             Zs += stabilizer
    
        
#         # result : self.compute_result(Z, Zs)
#         output_shape = output_R.shape
#         unfold = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         R_unfold = unfold(output_R)


#         R_unfold = R_unfold.unsqueeze(1)
#         R_unfold = R_unfold.permute(0,3,1,2)
#         #Z_cpu = Z.cpu()
#         #Zs_cpu = Zs.cpu()
#         R_cpu = R_unfold.cpu()
#         re = torch.sum((Z / Zs) * R_cpu,3)
        
#         result = re.cuda()
#         #result = torch.sum((Z / Zs) * R_unfold, 3)

#         result = result.permute(0,2,1)
        
        
#         # fold : back to the input shape 
#         fold = torch.nn.Fold(output_size=(in_h, in_w), kernel_size=kernel_size,
#                              padding=padding, dilation=dilation, stride=stride)

#         input_R = fold(result)

        
#         # save Variables in ctx      
#         input_patches = torch.autograd.Variable(input_patches, requires_grad=False) # stop gradient from input_pathces to input_tensors
        
#         out_channels = torch.tensor(out_channels)
#         padding = torch.tensor(padding)
#         dilation = torch.tensor(dilation)
#         stride = torch.tensor(stride)
#         in_h = torch.tensor(in_h)
#         in_w = torch.tensor(in_w)
#         kernel_size = torch.tensor(kernel_size)
#         output_shape = torch.tensor(output_shape)
#         ctx.save_for_backward(output_R, weight, input_patches, input_R, Z, Zs, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, output_shape)
#         del Z, Zs, R_cpu
        

#         return input_R    
            

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         # Get Variable from ctx
#         output_R, weight, input_patches, input_R,  Z, Zs, out_channels, kernel_size, padding, dilation, stride, in_h, in_w, output_shape = ctx.saved_tensors
        

#         output_R.matmul(output_R)
        
        
#         out_channels = out_channels.item()
#         padding = padding.numpy().tolist()
#         dilation = dilation.numpy().tolist()
#         stride = stride.numpy().tolist()
#         in_h = in_h.item()
#         in_w = in_w.item()
#         kernel_size = kernel_size.numpy().tolist()
#         output_shape = output_shape.numpy().tolist()
        
#         # dimension of elements
#         unfold_1 = torch.nn.Unfold(kernel_size=(1,1), padding=0, stride=1, dilation=1)
#         unfold = torch.nn.Unfold(kernel_size=kernel_size, padding=padding, stride=1, dilation=1)
        
#         weight = weight.cpu()
#         input_R = input_R.cpu()
#         output_R = output_R.cpu()
#         input_patches = input_patches.cpu()
#         grad_output = grad_output.cpu()
        
        
#         w = weight.reshape(weight.size(0), -1)
#         w_t = torch.t(w)
        

#         input_R = unfold(input_R).permute(0,2,1)
#         grad_output = unfold(grad_output).permute(0,2,1)
        
#         out = Zs.squeeze(2)
#         output_R = unfold_1(output_R).permute(0,2,1)
        
#         RdivZs = output_R / out
#         RdivZs2 = RdivZs / out
#         nabla_mul_X = (grad_output * input_patches)
#         nabla_x_w = nabla_mul_X.matmul(w_t)
#         nabla_mul_X = nabla_mul_X
        
#         # 1. D loss/D w
        
#         M1 = torch.sum(nabla_mul_X.unsqueeze(3) * RdivZs.unsqueeze(2), (0,1))
#         M2 = torch.sum(input_patches.unsqueeze(3) * (nabla_x_w * RdivZs2).unsqueeze(2), (0,1))
#         Dloss_Dw = (M1 - M2).t()
#         Dloss_Dw = Dloss_Dw.reshape(weight.shape)
        
        
#         # 2. D loss/D r_out (= next dy_)
        
#         dy_ = nabla_x_w/out
#         fold_output = torch.nn.Fold(output_size=output_shape[2:], kernel_size=(1,1),padding=0, dilation=dilation, stride=stride)
#         dy_ = fold_output(dy_.permute(0,2,1))
        
        
#         # 3. D loss/D input
#         M1 = grad_output * RdivZs.matmul(w)
#         M2 = (nabla_x_w * RdivZs2).matmul(w)
#         Dloss_Dinput = M1 - M2
        
#         fold_input = torch.nn.Fold(output_size=(in_h,in_w), kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
#         Dloss_Dinput = fold_input(Dloss_Dinput.permute(0,2,1))
        
        
        
#         return dy_.cuda(),Dloss_Dw.cuda(),Dloss_Dinput.cuda(),None,None,None,None,None,None,None,None,None

    
class Conv1d(_ConvNd):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, L)` and output :math:`(N, C_{out}, L_{out})` can be
    precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both sides
      for :attr:`padding` number of points.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor \frac{\text{out_channels}}{\text{in_channels}} \right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid
         `cross-correlation`_, and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, L_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            (out_channels, in_channels, kernel_size)
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels)

    Examples::

        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)


    def check_input_shape(self):
        inp_shape = self.input_tensor.shape
        try:
            if len(inp_shape)!=4:
                mod_shape = [self.batch_size, self.input_depth, self.input_dim, self.input_dim]
                self.input_tensor = torch.reshape(self.input_tensor, mod_shape)
        except:
            raise ValueError('Expected dimension of input tensor: 4')


    def forward(self, input):
        self.input_tensor = input
        self.check_input_shape()
        self.in_N, _, self.in_h, self.in_w = self.input_tensor.shape
        activations = F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        self.activations_shape = activations.shape
        
        return activations


class Conv3d(_ConvNd):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::

        \begin{equation*}
        \text{out}(N_i, C_{out_j}) = \text{bias}(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{out_j}, k) \star \text{input}(N_i, k)
        \end{equation*},

    where :math:`\star` is the valid 3D `cross-correlation`_ operator

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

         The configuration when `groups == in_channels` and `out_channels == K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.

         In other words, for an input of size :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(\text{in_channels}=C_{in}, \text{out_channels}=C_{in} * K, ..., \text{groups}=C_{in})`

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class _ConvTransposeMixin(object):

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        func = self._backend.ConvNd(
            self.stride, self.padding, self.dilation, self.transposed,
            output_padding, self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)

    def _output_padding(self, input, output_size):
        if output_size is None:
            return self.output_padding

        output_size = list(output_size)
        k = input.dim() - 2
        if len(output_size) == k + 2:
            output_size = output_size[-2:]
        if len(output_size) != k:
            raise ValueError(
                "output_size must have {} or {} elements (got {})"
                .format(k, k + 2, len(output_size)))

        def dim_size(d):
            return ((input.size(d + 2) - 1) * self.stride[d] -
                    2 * self.padding[d] + self.kernel_size[d])

        min_sizes = [dim_size(d) for d in range(k)]
        max_sizes = [min_sizes[d] + self.stride[d] - 1 for d in range(k)]
        for size, min_size, max_size in zip(output_size, min_sizes, max_sizes):
            if size < min_size or size > max_size:
                raise ValueError((
                    "requested an output size of {}, but valid sizes range "
                    "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.size()[2:]))

        return tuple([output_size[d] - min_sizes[d] for d in range(k)])


class ConvTranspose1d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = (L_{in} - 1) \times \text{stride} - 2 \times \text{padding}
                    + \text{kernel_size} + \text{output_padding}

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose1d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



class ConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = torch.randn(1, 16, 12, 12)
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



class ConvTranspose3d(_ConvTransposeMixin, _ConvNd):
    r"""Applies a 3D transposed convolution operator over an input image composed of several input
    planes.
    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.

    This module can be seen as the gradient of Conv3d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for ``kernel_size - 1 - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\left\lfloor\frac{\text{out_channels}}{\text{in_channels}}\right\rfloor`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::
        The :attr:`padding` argument effectively adds ``kernel_size - 1 - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): ``kernel_size - 1 - padding`` zero-padding
            will be added to both sides of each dimension in the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side
            of each dimension in the output shape. Default: 0
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where

          .. math::
              D_{out} = (D_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0]
                    + \text{kernel_size}[0] + \text{output_padding}[0]

              H_{out} = (H_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1]
                    + \text{kernel_size}[1] + \text{output_padding}[1]

              W_{out} = (W_{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2]
                    + \text{kernel_size}[2] + \text{output_padding}[2]

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super(ConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias)

    def forward(self, input, output_size=None):
        output_padding = self._output_padding(input, output_size)
        return F.conv_transpose3d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)