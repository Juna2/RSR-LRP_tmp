import math
import gc
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from module.module import Module
from module.arguments import get_args
import numpy as np
args=get_args()



class Identity(Module):
    def __init__(self, x):
        self.x = x
        
    def forward(self, inputs):
        self.inputs = inputs
        self.out = inputs + self.x
        return self.out
   
    def _simple_lrp(self, R, labels):
        self.Rout = (self.out/(self.out + self.x)) * R
        self.Rx = (self.x/(self.out + self.x)) * R
        return self.Rout, self.Rx
        
    def _composite_lrp(self, R):
        return self._simple_lrp(R)
    
    def _grad_cam(self, grad_output, requires_activation):
        if requires_activation:
            return grad_output, self.inputs
        else:
            return grad_output, None