import warnings
import torch
#from torch.nn.parameter import Parameter

from module.module import Module
from torch.nn import functional as F


class Threshold(Module):
    r"""Thresholds each element of the input Tensor

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, threshold, value, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

        # TODO: check in THNN (if inplace == True, then assert value <= threshold)

    """180913 15:11 becuase of error, changed!, main.py def weight init"""    
    def type_(self):
        return 'ReLu'
    def reset_parameters(self):
        pass

    def forward(self, input):
        self.activations = F.threshold(input, self.threshold, self.value, self.inplace)
        return self.activations

    def clean(self):
        self.activations = None
    
    #181210 added
    def _grad_cam(self, grad_output, requires_activation):
        '''
        dx: derivative of previous layer
        requires_activation: True if current layer is target layer.
        '''
        shape = self.activations.shape
        multiplier = torch.sign(self.activations)
        grad_input = grad_output.reshape(shape) * multiplier
        return grad_input, None

    def lrp(self, R, *args, **kwargs):
        return R
    
    

    def extra_repr(self):
        inplace_str = ', inplace' if self.inplace else ''
        return 'threshold={}, value={}{}'.format(
            self.threshold, self.value, inplace_str
        )


class ReLU(Threshold):
    r"""Applies the rectified linear unit function element-wise
    :math:`\text{ReLU}(x)= \max(0, x)`

    .. image:: scripts/activation_images/ReLU.png

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 0, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str