from module.sequential import Sequential
from module.linear import Linear
from module.softmax import Softmax
from module.relu import ReLU
from module.convolution import _ConvNd
from module.pool import _MaxPoolNd
from module.module import Module
from module.arguments import get_args
args = get_args()


from module.convolution import Conv2d

import torch
import torch.nn.functional as F


class model(Module):
#     def forward(self):
    
#         ### --------- Explaination ------------: 
#         ### -------- kernel_shape = [3,3,79,8] => [kernel_size, kernel_size, input_depth, output_depth]
#         ## --------- input shape => [batch_size, input_dim, input_dim, input_depth] 'NHWC'


#         return Sequential(Linear(784, 300),
#                            ReLU(),
#                        Linear(300, 300),
#                            ReLU(),
#                        Linear(300, 300),
#                            ReLU(),
#                        Linear(300, 10, whichScore = args.whichScore)
#                           )
    
    
    def forward(self):
        return Sequential(Conv2d(1, 5, 3),
                           ReLU(),
                       Conv2d(5, 5, 3),
                           ReLU(),
    
                       Linear(24*24*5, 10, lastLayer = True, whichScore = args.whichScore)
                          )

    
class model_(Module):

    def __init__(self):
        super(model, self).__init__()

        self.fc1 = Linear(784, 1296)
        self.fc2 = Linear(1296, 1296)
        self.fc3 = Linear(1296, 1296)
        self.fc4 = Linear(1296, 10)
        self.relu1 = ReLU.apply
        self.relu2 = ReLU.apply
        self.relu3 = ReLU.apply

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)

        return x
