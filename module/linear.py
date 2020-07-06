import math
import gc
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from module.module import Module
from module.arguments import get_args
from module.utils import normalize_with_log, normalize_with_nonlog
import numpy as np
import pickle
args=get_args()


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, lastLayer=False, bias=True, whichScore = None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.whichScore  = whichScore
        self.lastLayer = lastLayer
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.simple_lrp_object = Simple_lrp.apply
        self.grad_object = grad.apply

    """180913 15:11 becuase of error, changed!, main.py def weight init"""    
    def type_(self):
        return str('Linear')

    """180913 15:19 for the he normalization, changed!, main.py def weight init""" 
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         #self.weight.data.uniform_(-stdv, stdv)
#         torch.nn.init.kaiming_normal_(self.weight.data, nonlinearity='relu')
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
            
    """This is the original weight init"""
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def check_input_shape(self, input):
        inp_shape = input.shape
        if len(inp_shape)!=2:
            input = input.view(input.size(0),-1)
            
        return input

    def forward(self, input):
        self.input_tensor = self.check_input_shape(input)
        
        activation = F.linear(self.input_tensor, self.weight, self.bias)
        self.activation_shape = torch.Tensor(activation.shape)
        
        return activation

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    #181210 added
    def _grad_cam(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer. In linear case, this variable is always false
        '''
        grad_input = grad_output.mm(self.weight)
        if requires_activation:
            return grad_input, self.input_tensor
        else:
            return grad_input, None

    def ______simple_lrp(self, R):
        re = self.simple_lrp_object( R, self.activation_shape, self.weight, self.input_tensor,  self.bias)
        return re

    def ____simple_lrp(self, R, labels):
        re = self.simple_lrp_object( R, self.activation_shape, self.weight, self.input_tensor,  self.bias, self.whichScore, self.lastLayer, labels)
        return re
    
    def _____simple_lrp(self, R, labels):
        print('linear : simple lrp222')


        input_patches = self.input_tensor

        
        
        # Variables
       
        
        Zs = F.linear(self.input_tensor, self.weight, self.bias)+ args.epsilon
        RdivZs = R/Zs

        tmp1 = self.grad_object(RdivZs, Zs, self.input_tensor, self.weight) * self.input_tensor
            
            
        return tmp1 
   
    def _simple_lrp(self, R, labels):
        print('linear : simple lrp222')      
        # Variables
             
#         Zs = F.linear(self.input_tensor, self.weight, None)
        Zs = F.linear(self.input_tensor, self.weight, self.bias) 
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs += stabilizer

        if self.lastLayer and self.whichScore is None:
            self.whichScore = labels   
        if self.lastLayer and self.whichScore is not None:
            print('---last layer---')
            mask = torch.zeros_like(R)
            index = torch.range(0,R.shape[0]-1,dtype=torch.long).cuda()
            mask[index,self.whichScore] = 1
            R = R * mask


        RdivZs = R/Zs

        tmp1 = RdivZs.mm(self.weight)* self.input_tensor
#         tmp1 = self.grad_object(RdivZs, Zs, self.input_tensor, self.weight) * self.input_tensor
        
        
        return tmp1 

    
    
    ## 190105 even though _alpha_beta_lrp() is called, _simple_lrp is runned ## 
    def _composite_lrp(self, R, labels):
        Zs = F.linear(self.input_tensor, self.weight, self.bias) 
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs = Zs + stabilizer
        
        if self.lastLayer:
#             print('---last layer---')
            mask = torch.zeros_like(R)
            
#             index = torch.range(0,R.shape[0]-1,dtype=torch.long).cuda() # torch.range() is deprecated
            index = torch.arange(0, R.shape[0], dtype=torch.long).cuda()
            mask[index,labels] = 1 
            # interpretation은 label이 가리키는 output에서부터 내린다. 
            # (Segmentation 정보가 label에 맞게 setting되어 있기 때문이라고 생각하면 기억하기 쉬움)
            R = R * mask
        
        RdivZs = R/Zs
        tmp1 = RdivZs.mm(self.weight)* self.input_tensor

#         tmp1 = self.grad_object(RdivZs, Zs, self.input_tensor, self.weight) * self.input_tensor
        
        
        return tmp1
    
    
    def _flrp_acw(self, R, labels):
        try:
            c = self.weight.grad.clone().detach().abs()
            
#             # Use Gradient 
#             fisher = self.weight.grad.clone().detach().transpose(0,1)
            
#             # Normalize fisher
#             c = normalize_with_log(c)

        except:
            c = torch.ones_like(self.weight).clone().detach()
       
        
        wc = self.weight * c.detach()
        
        Zs = F.linear(self.input_tensor, wc, self.bias)
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs = Zs + stabilizer
        
        if self.lastLayer:
            print('---last layer---')
            mask = torch.zeros_like(R)
            
            index = torch.range(0,R.shape[0]-1,dtype=torch.long).cuda()
            mask[index,labels] = 1
            R = R * mask

        RdivZs = R/Zs

        tmp1 = RdivZs.mm(wc)* self.input_tensor
        
        
        return tmp1
    
    def _flrp_aw(self, R, labels):
        try:
            c = self.weight.grad.clone().detach().abs()
#             # Use Gradient 
#             fisher = self.weight.grad.clone().detach().transpose(0,1)
            
#             # Normalize fisher
#             c = normalize_with_log(c)
        except:
            c = torch.ones_like(self.weight).clone().detach().pow(2)
       
    
        wc = self.weight * c.detach()
        
        Zs = F.linear(self.input_tensor, wc, self.bias)
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs = Zs + stabilizer
        
        if self.lastLayer:
            print('---last layer---')
            mask = torch.zeros_like(R)
            
            index = torch.range(0,R.shape[0]-1,dtype=torch.long).cuda()
            mask[index,labels] = 1
            R = R * mask

        RdivZs = R/Zs

        tmp1 = RdivZs.mm(self.weight)* self.input_tensor
        
        
        return tmp1
    
    
    def _composite_new_lrp(self, R, labels):
        #Assume
        '''
        input : (N , d1)
        weight: (d2, d1)
        output: (N , d2)
        '''
        # Composite C
        weight = self.weight.transpose(0,1)
        M = args.new_lrp_proportion

        try:
            # Use fisher : gradient ^2
            fisher = self.weight.grad.clone().detach().transpose(0,1)
            
#             # Use Gradient 
#             fisher = self.weight.grad.clone().detach().transpose(0,1)
            
            # Normalize fisher
            fisher = normalize_with_log(fisher.detach())
            
            
        except:
            fisher = torch.ones_like(self.weight).clone().detach().transpose(0,1)
            
        # To Avoid devided by 0 or close to 0
        x_sign = torch.sign(self.input_tensor).detach()
        w_sign = torch.sign(weight).detach()
        r_sign = torch.sign(R).detach()
        
        _x = torch.where(self.input_tensor.abs()<1e-8, (torch.ones(1)*1e+8).cuda(), 1/self.input_tensor.abs())
        _w = torch.where(weight.abs()<1e-8, (torch.ones(1)*1e+8).cuda(), 1/weight.abs())
        _R = torch.where(R.abs()<1e-8, (torch.ones(1)*1e+8).cuda(), 1/R.abs())
        
        _x = _x * x_sign
        _w = _w * w_sign
        _R = _R * r_sign
        
        _x = _x.detach()
        _w = _w.detach()
        _R = _R.detach()
        
        # Comput P and C
        P = (1/(self.input_tensor.shape[1]-1) * torch.matmul(_x, fisher)).unsqueeze(1).detach()
        
        print('P +++++++++++',P.mean(), P.min(), P.max())
        C = M * (P - (fisher.unsqueeze(0)*_x.unsqueeze(-1))) * (_x.unsqueeze(-1) * _w.unsqueeze(0) * _R.unsqueeze(1))
        
        

#         # Make C as zero when divided by 0
#         zero_index = (self.input_tensor != 0).float().unsqueeze(-1)
#         C = C * zero_index + (1-zero_index)
        
        # Sum C for all N (batch size)
        C = C.sum(dim=0).detach()
        
        # Scale Normalize C
        print('C before +++++++++++',C.mean(), C.min(), C.max())
        C = normalize_with_log(C.detach())

        # Multiply C to weight
        wc = weight * C.detach()
        wc = wc.transpose(0,1)
        
        
        # Calculate Zs
        Zs = F.linear(self.input_tensor, wc, self.bias)
        
#         # Set sign of sum(acw) same as sum(aw)
#         w_for_sign = weight.transpose(0,1)
#         Zs_for_sign = torch.sign(F.linear(self.input_tensor, w_for_sign, self.bias).detach())
#         Zs = Zs * Zs_for_sign
        
        # Add stabilizer when Zs is 0
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))
        Zs = (Zs + stabilizer) 
        
        # Clipping Zs when really close to 0 for blocking R to be inf or NaN
        Zs_sign = torch.sign(Zs).detach()
        Zs = torch.where(Zs.abs()<1e-8, (torch.ones(1)*1e-8).cuda(), Zs.abs())
        Zs = Zs * Zs_sign

        
        


        
        
        print('fisher +++++++++++',fisher.mean(), fisher.min(), fisher.max())
        print('C +++++++++++++++++',C.mean(), C.min(), C.max())
       
        if self.lastLayer:
            mask = torch.zeros_like(R)
            
            index = torch.range(0,R.shape[0]-1,dtype=torch.long).cuda()
            mask[index,labels] = 1
            R = R * mask

        RdivZs = R/Zs

        tmp1 = RdivZs.mm(self.weight)* self.input_tensor
        print('tmp1++++++++++++++++++++',tmp1.mean(), tmp1.min(), tmp1.max(), tmp1)
        
#         # Normalize R
#         tmp1 = normalize_with_log(tmp1)

#         tmp1 = self.grad_object(RdivZs, Zs, self.input_tensor, self.weight) * self.input_tensor
        
        
#         # Get acw values.
#         import time
#         import pickle
#         R_num = time.time()
#         save_path_zs = args.img_dir+'/acw_linear_zs' + str(R_num) +'.pkl' 
#         save_path_acw = args.img_dir+ '/acw_linear_acw' + str(R_num) +'.pkl' 
#         save_path_c = args.img_dir+'/acw_linear_c' + str(R_num) +'.pkl' 
#         save_path_xrw_mean = args.img_dir+'/acw_linear_Zs' + str(R_num) +'.pkl' 
#         save_path_xrw_max = args.img_dir+'/acw_linear_r' + str(R_num) +'.pkl' 
#         save_path_xrw_min = args.img_dir+'/acw_linear_RdivZs' + str(R_num) +'.pkl' 
# #         save_path_zs = args.img_dir+args.img_name+'/acw_zs' + str(R_num) +'.pkl' 
# #         save_path_acw = args.img_dir+args.img_name+ '/acw_acw' + str(R_num) +'.pkl' 
# #         save_path_c = args.img_dir+args.img_name+'/acw_c' + str(R_num) +'.pkl' 

#         try:
#             with open(save_path_zs,"rb") as f:
#                 data_zs = pickle.load(f)

#         except:
#             data_zs = []

#         try:
#             with open(save_path_acw,"rb") as f:
#                 data_acw = pickle.load(f)

#         except:
#             data_acw = []
#         try:
#             with open(save_path_c,"rb") as f:
#                 data_c = pickle.load(f)
#         except:
#             data_c = []   
#         try:
#             with open(save_path_xrw_mean,"rb") as f:
#                 data_xrw_mean = pickle.load(f)
#             with open(save_path_xrw_max,"rb") as f:
#                 data_xrw_max = pickle.load(f)
#             with open(save_path_xrw_min,"rb") as f:
#                 data_xrw_min = pickle.load(f)    

#         except:
#             data_xrw_mean = []
#             data_xrw_max = []   
#             data_xrw_min = []   

#         data_zs.append(Zs.cpu().detach())
#         data_acw.append(tmp1.cpu().detach())
#         data_c.append(C.cpu().detach())
#         data_xrw_max.append(R.cpu().detach())
#         data_xrw_min.append(RdivZs.cpu().detach())
# #         data_xrw_mean.append(Zs.detach())
#         with open(save_path_zs,"wb") as f:
#             pickle.dump(data_zs,f)

#         with open(save_path_acw,"wb") as f:
#             pickle.dump(data_acw,f)

#         with open(save_path_c,"wb") as f:
#             pickle.dump(data_c,f) 

#         with open(save_path_xrw_max,"wb") as f:
#             pickle.dump(data_xrw_max,f) 

#         with open(save_path_xrw_min,"wb") as f:
#             pickle.dump(data_xrw_min,f) 
#         with open(save_path_xrw_mean,"wb") as f:
#             pickle.dump(data_xrw_mean,f)
            
            
        return tmp1 
    
    
#     def __composite_lrp(self, R, labels):
# #         re = self.simple_lrp_object( R, self.activation_shape, self.weight, self.input_tensor,  self.bias, self.whichScore, self.lastLayer, labels)        

#         R, activation_shape, weight, input_tensor,  biases, whichScore, lastLayer, labels = R, self.activation_shape, self.weight, self.input_tensor,  self.bias, self.whichScore, self.lastLayer, labels
        
#         R_shape = list(R.shape)
#         if len(R_shape)!=2:
#             R = torch.reshape(R, activation_shape)

#         Z = torch.unsqueeze(input_tensor, -1)*torch.unsqueeze(weight.t(), 0) 
#         Zs = torch.unsqueeze(torch.sum(Z, 1), 1) + torch.unsqueeze(torch.unsqueeze(biases, 0), 0)
#         stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))#1e-8
#         Zs += stabilizer
        
#         if lastLayer and whichScore == None:
#             whichScore = labels.cpu()
        
#         if lastLayer and whichScore is not None:
#             print('---last layer---')
#             Z_cpu = Z.cpu()
#             Zs_cpu = Zs.cpu()
#             R_cpu = R.cpu()
            
#             mask = torch.zeros_like(R_cpu)
#             mask[:,whichScore] = 1
            
           
#             R_cpu = R_cpu*mask
            
#             re = torch.sum((Z_cpu / Zs_cpu) * torch.unsqueeze(R_cpu, 1), 2)
#             re = re.cuda()
            
            
#         else:
#             print('---not last layer---')
#             Z_cpu = Z.cpu()
#             Zs_cpu = Zs.cpu()
#             R_cpu = R.cpu()
#             re = torch.sum((Z_cpu / Zs_cpu) * torch.unsqueeze(R_cpu, 1), 2)
#             re = re.cuda()
    
#         if not args.no_bias:
#             R_bias = torch.sum((bias / Zs), dim=2)/re.size(1)
#             re = re + R_biasself.lastLayer, labels


#         return re
    
    
      
class grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, RdivZs, zs, x, w):
        
        re = torch.tensor(torch.autograd.grad(zs, x, grad_outputs=RdivZs, retain_graph=True,allow_unused=True)[0].clone(), requires_grad=True)
        ctx.save_for_backward(RdivZs, zs, x, w, re)
        return re
            

    @staticmethod
    def backward(ctx, grad_output):
        RdivZs, zs, x, w, re = ctx.saved_tensors
        dRdivZs = torch.matmul(grad_output, w.transpose(1,0))
        dw = torch.matmul(RdivZs.transpose(1,0), grad_output)
       
        return dRdivZs, None, None,dw
        
class Simple_lrp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, activation_shape, weight, input_tensor,  biases, whichScore, lastLayer, labels):
        

        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        R_shape = list(R.shape)
        if len(R_shape)!=2:
            R = torch.reshape(R, activation_shape)

        Z = torch.unsqueeze(input_tensor, -1)*torch.unsqueeze(weight.t(), 0) 
        Zs = torch.unsqueeze(torch.sum(Z, 1), 1) + torch.unsqueeze(torch.unsqueeze(biases, 0), 0)
        stabilizer = args.epsilon*(torch.where(torch.ge(Zs,0), torch.ones_like(Zs), torch.ones_like(Zs)*-1))#1e-8
        Zs += stabilizer
        
        if lastLayer and whichScore == None:
            whichScore = labels.cpu()

#         try:
#             for obj in gc.get_objects():
#                 if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                     print(type(obj), obj.size())
#         except:
#             pass
        
        if lastLayer and whichScore is not None:
            print('---last layer---')
            Z_cpu = Z.cpu()
            Zs_cpu = Zs.cpu()
            R_cpu = R.cpu()
            
            mask = torch.zeros_like(R_cpu)
            mask[:,whichScore] = 1
            
            #R_cpu = torch.ones_like(R_cpu)*1000
            
            #mask[:,[]] = 1
            R_cpu = R_cpu*mask
#             re = (Z_cpu / Zs_cpu) * torch.unsqueeze(R_cpu, 1)
#             re = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(re, whichScore.cpu())]).squeeze()
            
            re = torch.sum((Z_cpu / Zs_cpu) * torch.unsqueeze(R_cpu, 1), 2)
            re = re.cuda()
            
            #re = re[:,:,whichScore]
            print(re.shape)
            print(whichScore)
            print(re)
            
            #re = torch.cat([torch.index_select(a, 1, i).unsqueeze(0) for a, i in zip(re, self.whichScore)]).squeeze()
        else:
            print('---not last layer---')
            Z_cpu = Z.cpu()
            Zs_cpu = Zs.cpu()
            R_cpu = R.cpu()
            re = torch.sum((Z_cpu / Zs_cpu) * torch.unsqueeze(R_cpu, 1), 2)
            re = re.cuda()

#             re = torch.sum((Z / Zs) * torch.unsqueeze(R, 1), 2)
    
    
        if not args.no_bias:
            R_bias = torch.sum((bias / Zs), dim=2)/re.size(1)
            re = re + R_bias
        
        
        #re = torch.sum((Z / Zs) * torch.unsqueeze(R, 1),2)
        ctx.save_for_backward(R, activation_shape, weight, input_tensor,  biases, Z, Zs)


        
        return re

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        R, activation_shape, weight, input_tensor,  biases, Z, Zs = ctx.saved_tensors
        a = input_tensor
        r = R 
        z = Z 
        zs = Zs
        grad = grad_output
        w = weight
        
        # expand_dims
        a = torch.unsqueeze(a, -1)
        r = torch.unsqueeze(r, 1)
        w = torch.t(w)
        w = torch.unsqueeze(w, 0)
            
        # w_gradient computation
        V1 = torch.matmul(a, r)
        V2 = torch.matmul( torch.unsqueeze(grad, 1),  z)
        V3 = torch.matmul( torch.unsqueeze(grad, -1), zs)
        V4 = zs * zs
        w_grad = torch.div( V1 * (-V2+V3), V4 )
            
        # a_gradient computation
        V5 = r * w
        a_grad = torch.sum( torch.div( V5 * (-V2+V3), V4), -1)
        
        # r_gradient computation
        r_next_grad = torch.sum( torch.div( V2, zs ), 1)
            
        dy_ = r_next_grad    
        w_grad=w_grad.sum(0).t()
        
        return dy_, None,w_grad,a_grad,None,None,None, None
    