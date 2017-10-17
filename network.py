import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from algorithm import ExtractPatchIndexs
from pyinn import im2col, col2im
from pyinn.im2col import Im2Col, Col2Im
import visdom
# testing
import matplotlib.pyplot as plt
import numpy as np

vis = visdom.Visdom(port=80)

class WeightedSum(nn.Module):
    def __init__(self, channels, positive=True):
        super(WeightedSum, self).__init__()

        self.channels = channels
        self.positive = positive
        self.params = nn.ParameterList()
        self.params.extend([nn.Parameter(torch.ones(1)) for i in xrange(channels)])


    def forward(self, x):
        assert x.data.size()[1] == self.channels, "Wrong number of channels!"

        tempx = x[:, 0] * self.params[0].expand_as(x[:,0])
        for i in xrange(1, self.channels):
            if self.positive:
                tempx = tempx + x[:,i] * torch.abs(self.params[i]).expand_as(x[:,i])
            else:
                tempx = tempx + x[:,i] * self.params[i].expand_as(x[:,i])
        x = tempx / self.channels
        return x

class DownTransition(nn.Module):
    def __init__(self, inchan, outchan, kernsize, padding=True):
        super(DownTransition, self).__init__()

        self.inchan = inchan
        self.outchan = outchan
        self.padding = padding
        self.kernsize = kernsize

        if padding:
            pad = np.int(kernsize/2.)
        else:
            pad = 0

        self.conv1 = nn.Conv2d(inchan, outchan, kernel_size=kernsize,padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(outchan)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class UpTransition(nn.Module):
    def __init__(self, upscaleFactor):
        super(UpTransition, self).__init__()

        self.ps = nn.PixelShuffle(upscaleFactor)
        self.pool = nn.AvgPool2d(upscaleFactor)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.pool(self.ps(x))
        x = self.bn(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.kernsize = 7

        self.convsModules = nn.ModuleList()
        self.psModules = nn.ModuleList()
        self.fcModules = nn.ModuleList()
        self.bnModules = nn.ModuleList()
        self.poolingLayers = nn.ModuleList()
        self.linearModules = nn.ModuleList()
        self.miscParams = nn.ParameterList()
        self.bnModules['init'] = nn.BatchNorm2d(1)


        self.base_conv_channels = 32
        self.sub_conv_channels = [16, 25, 36, 49, 64]

        self.baseconv = DownTransition(1, 32, self.kernsize)
        self.down = [DownTransition(32, i, self.kernsize) for i in self.sub_conv_channels]
        self.up   = [UpTransition(np.int(np.sqrt(i))) for i in self.sub_conv_channels]
        self.w    = WeightedSum(len(self.sub_conv_channels))
        self.linearModules.append(self.w)
        self.convsModules.extend(self.up)
        self.convsModules.extend(self.down)
        self.convsModules.append(self.baseconv)

        self.CircularMask = None

        self.current_step = 0
        self.current_epoch = 0
        self.loss_list = []

    def forward(self, x1, x2):
        """
        x2 should have better resolution
        :param x1:
        :param x2:
        :return:
        """
        assert x1.is_cuda and x2.is_cuda, "Inputs are not in GPU!"

        orix = x2 - x1
        x = self.bnModules['init'].cuda()(orix.unsqueeze(1))

        x = self.baseconv.forward(x)
        xx = [m.forward(x) for m in self.down]
        xx = [self.up[i].forward(xx[i]) for i in xrange(len(self.sub_conv_channels))]
        xx = torch.cat(xx, 1)
        x = self.w.forward(xx)
        s = x.data.size()

        if self.CircularMask is None:
            self.CircularMask = []
            for i in xrange(s[-2]):
                for j in xrange(s[-1]):
                    if (i - s[-2]/2.) ** 2 + (j - s[-1] / 2.) ** 2 > ((s[-2]/2.)**2.) + 1:
                        self.CircularMask.append([i,j])

        for c in self.CircularMask:
            x[:,c[0],c[1]] = 10

        x = x2 - x
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


