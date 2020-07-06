import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import h5py
import numpy as np

import torchvision
import torch.nn.functional as F

from module.sequential import Sequential
from module.linear import Linear
from module.relu import ReLU
from module.convolution import _ConvNd
from module.pool import _MaxPoolNd
from module.module import Module
from module.convolution import Conv2d
from module.batchnorm import BatchNorm2d
from module.pool import MaxPool2d
from module.adaptiveAvgPool2d import AdaptiveAvgPool2d, AvgPool2d

from module.arguments import get_args
args = get_args()



class juyeon(nn.Module):
    def __init__(self, checkpoint_path='None'):
        super(juyeon, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(79, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*5*4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        if checkpoint_path != 'None':
            self.weight_initialize(checkpoint_path)
        
    def weight_initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        juyeon_state_dict = checkpoint['state_dict']
        
        juyeon_keys = list(juyeon_state_dict.keys())
        features_keys = list(self.features.state_dict().keys())
        classifier_keys = list(self.classifier.state_dict().keys())

        for key in range(len(features_keys)):
            self.features.state_dict()[features_keys[key]][:] = juyeon_state_dict['features.'+features_keys[key]][:]
        for key in range(len(classifier_keys)):
            self.classifier.state_dict()[classifier_keys[key]][:] = juyeon_state_dict['classifier.'+classifier_keys[key]][:]
        return

    def forward(self, x):
        x = self.features(x)
        shape = np.array(x.size())
        x = x.view(shape[0], np.prod(shape[1:]).item())
        x = self.classifier(x)
        return x

class juyeon_test(nn.Module):
    def __init__(self, checkpoint_path='None'):
        super(juyeon_test, self).__init__()
        self.conv1 = nn.Conv2d(79, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*4, 128)
        self.fc2 = nn.Linear(128, 2)
        
        if checkpoint_path != 'None':
            self.weight_initialize(checkpoint_path)
        
    def forward(self, x, epoch=None):
        x = self.conv1(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'w') as f:
                f.create_dataset('{:02d}'.format(0), data=x.cpu().detach().numpy())
        x = self.relu(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(1), data=x.cpu().detach().numpy())
        x = self.maxpool(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(2), data=x.cpu().detach().numpy())
        x = self.conv2(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(3), data=x.cpu().detach().numpy())
        x = self.relu(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(4), data=x.cpu().detach().numpy())
        x = self.maxpool(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(5), data=x.cpu().detach().numpy())
        x = self.conv3(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(6), data=x.cpu().detach().numpy())
        x = self.relu(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(7), data=x.cpu().detach().numpy())
        x = self.maxpool(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(8), data=x.cpu().detach().numpy())
        x = self.conv4(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(9), data=x.cpu().detach().numpy())
        x = self.relu(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(10), data=x.cpu().detach().numpy())
        x = self.maxpool(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(11), data=x.cpu().detach().numpy())
        shape = np.array(x.size())
        x = x.view(shape[0], np.prod(shape[1:]).item())
        x = self.fc1(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(12), data=x.cpu().detach().numpy())
        x = self.relu(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(13), data=x.cpu().detach().numpy())
        x = self.fc2(x)
        if epoch != None:
            with h5py.File(os.path.join(args.test_data_path, 
                                        'output{:02d}_{:02d}.hdf5'.format(args.test_num, epoch)), 'a') as f:
                f.create_dataset('{:02d}'.format(14), data=x.cpu().detach().numpy())
        return x
    
    def weight_initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        juyeon_state_dict = checkpoint['state_dict']
        
        self.conv1.state_dict()['weight'][:] = juyeon_state_dict['features.0.weight']
        self.conv1.state_dict()['bias'][:] = juyeon_state_dict['features.0.bias']
        self.conv2.state_dict()['weight'][:] = juyeon_state_dict['features.3.weight']
        self.conv2.state_dict()['bias'][:] = juyeon_state_dict['features.3.bias']
        self.conv3.state_dict()['weight'][:] = juyeon_state_dict['features.6.weight']
        self.conv3.state_dict()['bias'][:] = juyeon_state_dict['features.6.bias']
        self.conv4.state_dict()['weight'][:] = juyeon_state_dict['features.9.weight']
        self.conv4.state_dict()['bias'][:] = juyeon_state_dict['features.9.bias']
        
        self.fc1.state_dict()['weight'][:] = juyeon_state_dict['classifier.0.weight']
        self.fc1.state_dict()['bias'][:] = juyeon_state_dict['classifier.0.bias']
        self.fc2.state_dict()['weight'][:] = juyeon_state_dict['classifier.2.weight']
        self.fc2.state_dict()['bias'][:] = juyeon_state_dict['classifier.2.bias']
        return

    
    
class juyeon_4classes(nn.Module):
    def __init__(self, ):
        super(juyeon_4classes, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(79, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*5*4, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        
        # Use this when you want to customize your initial weight.
#         if init_weights:
#             self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        shape = np.array(x.size())
        x = x.view(shape[0], np.prod(shape[1:]).item())
        x = self.classifier(x)
        return x
    
    
class conv3d_07092244(nn.Module):
    def __init__(self, ):
        super(conv3d_07092244, self).__init__() #### check this
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*4*5*4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        # Use this when you want to customize your initial weight.
#         if init_weights:
#             self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        shape = np.array(x.size())
        x = x.view(shape[0], np.prod(shape[1:]).item())
        x = self.classifier(x)
        return x

    
class juyeon_lrp(Module):
    def forward(self, pretrained=False, checkpoint_path='None'):
        self.features = Sequential(
            Conv2d(79, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = Sequential(
            Linear(64*5*4, 128),
            ReLU(),
            Linear(128, 2, whichScore = args.whichScore,lastLayer=True)
        )
        if pretrained:
            if checkpoint_path == 'None':
                raise Exception('checkpoint path is : {}'.format(checkpoint_path))
            else:
                self.weight_initialize(checkpoint_path)
        self.net = Sequential(*self.features, *self.classifier)
        
        return self.net
    
    def weight_initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        juyeon_state_dict = checkpoint['state_dict']

        juyeon_keys = list(juyeon_state_dict.keys())
        features_keys = list(self.features.state_dict().keys())
        classifier_keys = list(self.classifier.state_dict().keys())

        for key in range(len(features_keys)):
            self.features.state_dict()[features_keys[key]][:] = juyeon_state_dict['features.'+features_keys[key]][:]
        for key in range(len(classifier_keys)):
            self.classifier.state_dict()[classifier_keys[key]][:] = juyeon_state_dict['classifier.'+classifier_keys[key]][:]
        return

class juyeon_lrp_from_lrp_model(Module):
    def forward(self, pretrained=False, checkpoint_path='None'):
        self.features = Sequential(
            Conv2d(79, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = Sequential(
            Linear(64*5*4, 128),
            ReLU(),
            Linear(128, 2, whichScore = args.whichScore, lastLayer=True)
        )
        self.net = Sequential(*self.features, *self.classifier)
        if pretrained:
            if checkpoint_path == 'None':
                raise Exception('checkpoint path is : {}'.format(checkpoint_path))
            else:
                self.weight_initialize(checkpoint_path)
        
        return self.net
    
    def weight_initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        juyeon_state_dict = checkpoint['state_dict']

        juyeon_keys = list(juyeon_state_dict.keys())
        net_keys = list(self.net.state_dict().keys())
        
        for num in range(len(net_keys)):
            self.net.state_dict()[net_keys[num]][:] = juyeon_state_dict[juyeon_keys[num]][:]
        return
    
    
class cifar10(nn.Module):
    def __init__(self):
        super(cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class cifar10_lrp(Module):
    def __init__(self, checkpoint_path='None'):
        super(cifar10_lrp, self).__init__()
        self.features = Sequential(
            Conv2d(3, 6, 5),
            ReLU(),
            MaxPool2d(2, 2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2, 2),
        )
        self.classifier = Sequential(
            Linear(16 * 5 * 5, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, 10, whichScore = args.whichScore,lastLayer=True)
        )
        if checkpoint_path != 'None':
            self.weight_initialize(checkpoint_path)
        self.net = Sequential(*self.features, *self.classifier)
        
    def weight_initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        juyeon_state_dict = checkpoint['state_dict']

        juyeon_keys = list(juyeon_state_dict.keys())
        features_keys = list(self.features.state_dict().keys())
        classifier_keys = list(self.classifier.state_dict().keys())

        for key in range(len(features_keys)):
            self.features.state_dict()[features_keys[key]][:] = juyeon_state_dict['features.'+features_keys[key]][:]
        for key in range(len(classifier_keys)):
            self.classifier.state_dict()[classifier_keys[key]][:] = juyeon_state_dict['classifier.'+classifier_keys[key]][:]
        return
                
    def forward(self):
        return self.net




    
    
class juyeon_lrp_4classes(Module):
    def __init__(self, pretrained=False):
        super(juyeon_lrp_4classes, self).__init__()
        self.features = Sequential(
            Conv2d(79, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = Sequential(
            Linear(64*5*4, 128),
            ReLU(),
            Linear(128, 4, whichScore = args.whichScore, lastLayer=True)
        )
        self.weight_initialize(pretrained)
        self.net = Sequential(*self.features, *self.classifier)
        
        
    def weight_initialize(self, pretrained):
        if pretrained:
            checkpoint = torch.load('../juyeon_checkpoint_lr0.001_conv2d.pth.tar')
            juyeon_state_dict = checkpoint['state_dict']
        else:
            if 'juyeon_4classes_model.h5' in os.listdir('../'):
                juyeon_state_dict = torch.load('../juyeon_4classes_model.h5')
            else:
                return
        juyeon_keys = list(juyeon_state_dict.keys())
        features_keys = list(self.features.state_dict().keys())
        classifier_keys = list(self.classifier.state_dict().keys())
        
        for key in range(len(features_keys)):
            self.features.state_dict()[features_keys[key]][:] = juyeon_state_dict['features.'+features_keys[key]][:]
        for key in range(len(classifier_keys)):
            self.classifier.state_dict()[classifier_keys[key]][:] = juyeon_state_dict['classifier.'+classifier_keys[key]][:]
        return
    
    def forward(self):
        return self.net

    
    
    
    
###############################################################################################################################
###############################################################################################################################
#####################################################          VGG          ###################################################
###############################################################################################################################
###############################################################################################################################


    
class vgg19(Module):
    def forward(self, pretrained=False, checkpoint_path=None, batch_norm=args.vgg_bn, input_channel_num=3):
        layers = self.make_layers(cfg['E'], input_channel_num)
        
        layers = layers + [Linear(512 * 7 * 7, 4096),
                           ReLU(),
                           Linear(4096, 4096),
                           ReLU(),
                           Linear(4096, 2, whichScore = args.whichScore, lastLayer=True)]
        
        if pretrained == False:
            return Sequential(*layers)
        
        net = Sequential(*layers)
        
        
        if checkpoint_path == None: 
            if batch_norm == True:
                vgg19 = torchvision.models.vgg19_bn(pretrained=True)
            else:
                vgg19 = torchvision.models.vgg19(pretrained=True)
            vgg19_keys = list(vgg19.state_dict().keys())
            net_keys = list(net.state_dict().keys())

            for i in range(len(vgg19_keys)):
                if net.state_dict()[net_keys[i]].shape == vgg19.state_dict()[vgg19_keys[i]].shape:
                    try:
                        net.state_dict()[net_keys[i]][:] = vgg19.state_dict()[vgg19_keys[i]][:]
                    except:
                        net.state_dict()[net_keys[i]] = vgg19.state_dict()[vgg19_keys[i]]
                else:
                    print('skip copying layer {} (different shape)'.format(net_keys[i]))
        else: 
            print(os.getcwd())
            #vgg19 = torch.load()
            net.load_state_dict(torch.load(checkpoint_path)['state_dict'])
            #vgg19.eval()
        
        return net



    def make_layers(self, cfg, input_channel_num, batch_norm = args.vgg_bn):
        layers = []
        in_channels = input_channel_num
        for v in cfg:
            if v == 'M':
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, BatchNorm2d(v), ReLU()]
                else:
                    layers += [conv2d, ReLU()]
                in_channels = v
        return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}



###############################################################################################################################
###############################################################################################################################
#####################################################        ResNet         ###################################################
###############################################################################################################################
###############################################################################################################################

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(Module):
    expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = BatchNorm2d(planes)
#         self.relu = ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
        
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        # 190321: init할 때 self에 layer가 안 들어가 있으면 model.state_dict에 안 잡혀서 아래 추가함.
        self.downsample = downsample
        if downsample is not None and len(downsample) != 0:
            self.down_conv3 = downsample[0]
            self.down_bn3 = downsample[1]
        self.stride = stride


    def forward(self, x):

        # For foward output, For LRP, save layers in list 'layers'
        layers = []
        identity = x

        out = self.conv1(x)
        layers.append(self.conv1)
        
        out = self.bn1(out)
        layers.append(self.bn1)
        
        out = self.relu(out)
        layers.append(self.relu)

        out = self.conv2(out)
        layers.append(self.conv2)
        
        out = self.bn2(out)
        layers.append(self.bn2)

        if self.downsample is not None and len(self.downsample) != 0:
#             identity = self.downsample(x)
            identity = self.down_conv3(x)
            identity = self.down_bn3(identity)

    
        self.out = out
        out += identity
        out = self.relu(out)
        layers.append(self.relu)
        
        self.x = identity
        self.layers = layers

        return out
    
    def _simple_lrp(self, R, labels):
        
        target_layer = None
        lrp_var = args.r_method
        param=None
        whichScore=None
        
        if target_layer == None:
            for key, module in enumerate(reversed(self.layers)):
                R = module.lrp(R, labels, lrp_var, param)
                if key == 0 : # when identity order, after last relu
                    
                    # Skip connection alpha beta
                    out_p = torch.where(self.out<0, torch.zeros(1).cuda(), self.out)
                    out_n = self.out - out_p
                    
                    x_p = torch.where(self.x<0, torch.zeros(1).cuda(), self.x)
                    x_n = self.x - x_p
                    
                    Rout_p = (out_p/(out_p + x_p + 1e-12)) * R
                    Rx_p = (x_p/(out_p + x_p + 1e-12)) * R
                    
                    Rout_n = (out_n/(out_p + x_n + 1e-12)) * R
                    Rx_n = (x_n/(out_n + x_n + 1e-12)) * R
                    
                    self.Rout = (1-args.beta) * Rout_p + (args.beta * Rout_n)
                    self.Rx = (1-args.beta) * Rx_p + (args.beta * Rx_n)
                    
                    R = self.Rout

                    if self.downsample is not None and len(self.downsample) != 0:
                        Rx = self.Rx
#                         print('&&&&&&&&&&&&& downsample before : ', Rx.shape)
                        for key, module in enumerate(reversed(self.downsample)):
                            Rx = module.lrp(Rx, labels, lrp_var, param)
#                         print('&&&&&&&&&&&&& after', Rx.shape)
                        self.Rx = Rx
            
#             # If you want to not consider x with downsample
#             if self.downsample is not None and len(self.downsample) != 0:
#                 return self.Rx 
#             return R
            
            return R + self.Rx

        else:
            ### 190101 target layer #####
            stop_before_x = False
            for key, module in enumerate(reversed(self.layers)):
                print('block', key, module)
                requires_activation = (str(key) == target_layer)

                # x: feature map, dx: dL/dx
                R = module.lrp(R, labels, lrp_var, param)



                if key == 0 : # when identity order, after last relu
                    self.Rout = (self.out/(self.out + self.x)) * R
                    self.Rx = (self.x/(self.out + self.x)) * R
                    R = self.Rout

                    if self.downsample is not None and len(self.downsample) != 0:
                        Rx = self.Rx
                        for key, module in enumerate(reversed(self.downsample)):
                            Rx = module.lrp(Rx, labels, lrp_var, param)
                        self.Rx = Rx

                if requires_activation and key is not len(self.layers)-1:
                    stop_before_x = True
                    break

            if stop_before_x == False: 
                R = R + self.Rx

        return R
        
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        for key, module in enumerate(reversed(self.layers)):
            # x: feature map, dx: dL/dx
            dx, x = module.grad_cam(dx, requires_activation)
            if key == 0 : # when identity order, after last relu, +x used only downsample is None
                dx_iden = dx
                if self.downsample is not None:
                    for key, module in enumerate(reversed(self.downsample)):
                        # x: feature map, dx: dL/dx
                        dx_iden, x = module.grad_cam(dx_iden, requires_activation)

        # Add dx_iden for block that has identity layer
        dx = dx + dx_iden
 
        if requires_activation:
            return dx, x

        return dx, None
    
    
        

class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.relu1 = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=True)
        self.relu3 = ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None and len(downsample) != 0:
            self.down_conv4 = downsample[0]
            self.down_bn4 = downsample[1]
        self.stride = stride

    def forward(self, x):
        
        
        # For LRP, save layers in list 'layers'
        layers = []
        identity = x
#         self.x = x



        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        layers.append(self.conv1)
        layers.append(self.bn1)
        layers.append(self.relu1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        layers.append(self.conv2)
        layers.append(self.bn2)
        layers.append(self.relu2)
        
        out = self.conv3(out)
        out = self.bn3(out)
        layers.append(self.conv3)
        layers.append(self.bn3)
        
        if self.downsample is not None and len(self.downsample) != 0:
            identity = self.down_conv4(x)
            identity = self.down_bn4(identity)
            
        

        self.out = out
        out += identity
        out = self.relu3(out)
        layers.append(self.relu3)

        self.x = identity
        self.layers = layers
            
#         layers.append(self.relu)
#         self.layers = layers
        

#         out += identity
#         out = self.relu(out)
#         self.out = out

        return out
                      
    
    def _simple_lrp(self, R, labels):

        target_layer = None
        lrp_var = args.r_method
        param=None
        whichScore=None
        
        if target_layer == None:
            for key, module in enumerate(reversed(self.layers)):
                R = module.lrp(R, labels, lrp_var, param)
                if key == 0 : # when identity order, after last relu
                    
                    # Skip connection alpha beta
                    out_p = torch.where(self.out<0, torch.zeros(1).cuda(), self.out)
                    out_n = self.out - out_p
                    
                    x_p = torch.where(self.x<0, torch.zeros(1).cuda(), self.x)
                    x_n = self.x - x_p
                    
                    Rout_p = (out_p/(out_p + x_p + 1e-12)) * R
                    Rx_p = (x_p/(out_p + x_p + 1e-12)) * R
                    
                    Rout_n = (out_n/(out_p + x_n + 1e-12)) * R
                    Rx_n = (x_n/(out_n + x_n + 1e-12)) * R
                    
                    self.Rout = (1-args.beta) * Rout_p + (args.beta * Rout_n)
                    self.Rx = (1-args.beta) * Rx_p + (args.beta * Rx_n)
                    
                    R = self.Rout

                    if self.downsample is not None and len(self.downsample) != 0:
                        Rx = self.Rx
#                         print('&&&&&&&&&&&&& downsample before : ', Rx.shape)
                        for key, module in enumerate(reversed(self.downsample)):
                            Rx = module.lrp(Rx, labels, lrp_var, param)
#                         print('&&&&&&&&&&&&& after', Rx.shape)
                        self.Rx = Rx
            
#             # If you want to not consider x with downsample
#             if self.downsample is not None and len(self.downsample) != 0:
#                 return self.Rx 
#             return R
            
            return R + self.Rx

        else:
            ### 190101 target layer #####
            stop_before_x = False
            for key, module in enumerate(reversed(self.layers)):
                print('block', key, module)
                requires_activation = (str(key) == target_layer)

                # x: feature map, dx: dL/dx
                R = module.lrp(R, labels, lrp_var, param)



                if key == 0 : # when identity order, after last relu
                    self.Rout = (self.out/(self.out + self.x)) * R
                    self.Rx = (self.x/(self.out + self.x)) * R
                    R = self.Rout

                    if self.downsample is not None and len(self.downsample) != 0:
                        Rx = self.Rx
                        for key, module in enumerate(reversed(self.downsample)):
                            Rx = module.lrp(Rx, labels, lrp_var, param)
                        self.Rx = Rx

                if requires_activation and key is not len(self.layers)-1:
                    stop_before_x = True
                    break

            if stop_before_x == False: 
                R = R + self.Rx

        return R
        
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        for key, module in enumerate(reversed(self.layers)):
            # x: feature map, dx: dL/dx
            dx, x = module.grad_cam(dx, requires_activation)
            if key == 0 : # when identity order, after last relu, +x used only downsample is None
                dx_iden = dx
                if self.downsample is not None:
                    for key, module in enumerate(reversed(self.downsample)):
                        # x: feature map, dx: dL/dx
                        dx_iden, x = module.grad_cam(dx_iden, requires_activation)

        # Add dx_iden for block that has identity layer
        dx = dx + dx_iden
 
        if requires_activation:
            return dx, x

        return dx, None


class ResNet(Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes, whichScore = args.whichScore,lastLayer=True)

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
#         layers_down = []
        downsample = []
        if stride != 1 or self.inplanes != planes * block.expansion:
#             layers_down.append(conv1x1(self.inplanes, planes * block.expansion, stride),
#                                BatchNorm2d(planes * block.expansion))
            
            downsample=downsample+[conv1x1(self.inplanes, planes * block.expansion, stride), BatchNorm2d(planes * block.expansion)]


        layers = []
        layers = layers + [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers = layers + [block(self.inplanes, planes)]
        
        return layers

    
    def forward(self):
        layers = []
        layers.append(self.conv1)
        layers.append(self.bn1)
        layers.append(self.relu)
        layers.append(self.maxpool)
        layers = layers + self.layer1
        layers = layers + self.layer2
        layers = layers + self.layer3
        layers = layers + self.layer4
        layers.append(self.avgpool) 
        layers.append(self.fc)
                      
        return Sequential(*layers)


def resnet18(pretrained=False, checkpoint_path=None, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs).forward()
    if checkpoint_path == None: 
        dummy_model = torchvision.models.resnet18(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == dummy_model.state_dict()[key2].shape:
                if model.state_dict()[key1].shape == torch.tensor(1).shape:
                    model.state_dict()[key1] = dummy_model.state_dict()[key2]
                else:
                    model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, dummy_model.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def resnet34(pretrained=False, checkpoint_path=None, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs).forward()
    if checkpoint_path == None: 
        dummy_model = torchvision.models.resnet34(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == dummy_model.state_dict()[key2].shape:
                if model.state_dict()[key1].shape == torch.tensor(1).shape:
                    model.state_dict()[key1] = dummy_model.state_dict()[key2]
                else:
                    model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, dummy_model.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def resnet50(pretrained=False, checkpoint_path=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs).forward()

    if pretrained == False:
        return model
        
    if checkpoint_path == None: 
        dummy_model = torchvision.models.resnet50(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == dummy_model.state_dict()[key2].shape:
                if model.state_dict()[key1].shape == torch.tensor(1).shape:
                    model.state_dict()[key1] = dummy_model.state_dict()[key2]
                else:
                    model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, dummy_model.state_dict()[key2].shape))
    else:
#         print(checkpoint_path, '$^%#&^*$&(%^*$%&#$^@%&#^*$')
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def resnet101(pretrained=False, checkpoint_path=None, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs).forward()
    if pretrained == False:
        return model
        
    if checkpoint_path == None: 
        dummy_model = torchvision.models.resnet101(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == dummy_model.state_dict()[key2].shape:
                if model.state_dict()[key1].shape == torch.tensor(1).shape:
                    model.state_dict()[key1] = dummy_model.state_dict()[key2]
                else:
                    model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, dummy_model.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
                                                                                    
    return model


def resnet152(pretrained=False, checkpoint_path=None, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if checkpoint_path == None: 
        dummy_model = model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

        for key1, key2 in zip(model.state_dict().keys(), dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == dummy_model.state_dict()[key2].shape:
                if model.state_dict()[key1].shape == torch.tensor(1).shape:
                    model.state_dict()[key1] = dummy_model.state_dict()[key2]
                else:
                    model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, dummy_model.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model










###############################################################################################################################
###############################################################################################################################
#####################################################       DenseNet        ###################################################
###############################################################################################################################
###############################################################################################################################


__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.layers = []
        self.add_module('1',BatchNorm2d(num_input_features))
        self.add_module('2',ReLU(inplace=False))
        self.add_module('3',Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)) # L*k -> 4*k
        self.add_module('4',BatchNorm2d(bn_size * growth_rate))
        self.add_module('5',ReLU(inplace=False))
        self.add_module('6',Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)) # 4*k -> k
        self.drop_rate = drop_rate
        self.growth_rate = growth_rate
        
    def forward(self, x):
        """
        <Chennel dim>
        X : L*k
        out : k (from L*k -> 4*k -> k)
        return : (L+1)*k
        """
        out = x
        for module in self._modules.values():
            out = module(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)
    
    def _simple_lrp(self, R, labels):
        """
        <Chennel dim>
        input R : (L+1)*k 
        R_init : L*k
        R0 : k
        R : L*k  (from k -> 4*k -> L*k)
        return : L*k (L*k + L*k)
        """
        
        R_init, R = R[:,:-self.growth_rate,:,:], R[:,-self.growth_rate:,:,:]
        for key, module in enumerate(reversed(self._modules.values())):
            R = module.lrp(R, labels, args.r_method)
            
        return R_init + R
 
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _composite_new_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        dx_init, dx = dx[:,:-self.growth_rate,:,:], dx[:,-self.growth_rate:,:,:]
        for key, module in enumerate(reversed(self._modules.values())):
            dx, x = module.grad_cam(dx, requires_activation)
        return dx_init + dx, x


def _DenseBlock(num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    layers = []
    for i in range(num_layers):
        layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        
    return layers
    
class _Transition(Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.layers = []
        self.add_module('1',BatchNorm2d(num_input_features))
        self.add_module('2',ReLU(inplace=False))
        self.add_module('3',Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('4',AvgPool2d(kernel_size=2, stride=2))
    
    def forward(self, x):
        """
        <Channel dim>
        out : about 1/2 of X (because of conv)
        
        <H, W dim>
        out : about 1/4 of X (because of AvgPool2d)
        """
        out = x
        for module in self._modules.values():
            out = module(out)
        return out
    
    def _simple_lrp(self, R, labels):
        for key, module in enumerate(reversed(self._modules.values())):
            R = module.lrp(R, labels, args.r_method)
        return R
 
    def _composite_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _composite_new_lrp(self, R, labels):
        return self._simple_lrp(R, labels)
    
    def _grad_cam(self, dx, requires_activation):
        for key, module in enumerate(reversed(self._modules.values())):
            dx, x = module.grad_cam(dx, requires_activation)
        if requires_activation:
            return dx, x
        return dx, None

class DenseNet(Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):

        super(DenseNet, self).__init__()

        # First convolution
        self.layers = []
        self.layers.append(Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))
        self.layers.append(BatchNorm2d(num_init_features))
        self.layers.append(ReLU(inplace=False))
        self.layers.append(MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.layers += block
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.layers.append(trans)
#                 self.layers+=trans
                num_features = num_features // 2

        # Final batch norm
        self.layers.append(BatchNorm2d(num_features))

        # Linear layer
        self.layers.append(ReLU(inplace=False))
        self.layers.append(AdaptiveAvgPool2d((1, 1)))
        self.layers.append(Linear(num_features, num_classes, whichScore = args.whichScore,lastLayer=True))

#         # Official init from torch repo.
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)

    def forward(self):
        return Sequential(*self.layers)


# def _load_state_dict(model, model_url, checkpoint_path):
#     # '.'s are no longer allowed in module names, but previous _DenseLayer
#     # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
#     # They are also in the checkpoints in model_urls. This pattern is used
#     # to find such keys.
    
#     if checkpoint_path == 'None': 
#         model_pretrained = model.load_state_dict(model_zoo.load_url(model_url))

#         for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
#             try:
#                 model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
#             except:
#                 model.state_dict()[key1] = model_pretrained.state_dict()[key2]
#     else:
#         model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        
        


def densenet121(pretrained=False, checkpoint_path=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=([6, 12, 24, 16]), **kwargs).forward()
    if checkpoint_path == None: 
        model_pretrained = torchvision.models.densenet121(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
            if model.state_dict()[key1].shape == model_pretrained.state_dict()[key2].shape:
                try:
                    model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
                except:
                    model.state_dict()[key1] = model_pretrained.state_dict()[key2]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, model_pretrained.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def densenet169(pretrained=False, checkpoint_path=None, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs).forward()
    if checkpoint_path == None: 
        model_pretrained = torchvision.models.densenet169(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
            if model.state_dict()[key1].shape == model_pretrained.state_dict()[key2].shape:
                try:
                    model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
                except:
                    model.state_dict()[key1] = model_pretrained.state_dict()[key2]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, model_pretrained.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def densenet201(pretrained=False, checkpoint_path=None, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs).forward()
    if checkpoint_path == None: 
        model_pretrained = torchvision.models.densenet201(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
            if model.state_dict()[key1].shape == model_pretrained.state_dict()[key2].shape:
                try:
                    model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
                except:
                    model.state_dict()[key1] = model_pretrained.state_dict()[key2]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, model_pretrained.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model


def densenet161(pretrained=False, checkpoint_path=None, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs).forward()
    if checkpoint_path == None: 
        model_pretrained = torchvision.models.densenet161(pretrained=True)

        for key1, key2 in zip(model.state_dict().keys(), model_pretrained.state_dict().keys()):
            if model.state_dict()[key1].shape == model_pretrained.state_dict()[key2].shape:
                try:
                    model.state_dict()[key1][:] = model_pretrained.state_dict()[key2][:]
                except:
                    model.state_dict()[key1] = model_pretrained.state_dict()[key2]
            else:
                print('skip copying layer {}, {} / {}, {} (different shape)'.format(key1, key2, model.state_dict()[key1].shape, model_pretrained.state_dict()[key2].shape))
    else:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model





































